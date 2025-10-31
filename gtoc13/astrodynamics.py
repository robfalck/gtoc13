import jax
import jax.numpy as jnp
from jax import jit
from diffrax import diffeqsolve, ODETerm, Dopri5, SaveAt, PIDController
from typing import Tuple
import numpy as np

from gtoc13.bodies import bodies_data
from gtoc13.orbital_elements import OrbitalElements
from gtoc13.cartesian_state import CartesianState
from gtoc13.constants import (
    KMPAU, MU_ALTAIRA, YEAR,
    SPTU
)


@jit
def solve_kepler(M: float, e: float, tol: float = 1e-10, max_iter: int = 50) -> float:
    """
    Solve Kepler's equation M = E - e*sin(E) for eccentric anomaly E
    using Newton-Raphson iteration.
    """
    # Initial guess: use jax.lax.cond for JIT compatibility
    E = jax.lax.cond(e < 0.8, lambda: M, lambda: jnp.pi)

    def body_fn(carry):
        E, i = carry
        f = E - e * jnp.sin(E) - M
        fp = 1.0 - e * jnp.cos(E)
        E_new = E - f / fp
        return (E_new, i + 1)

    def cond_fn(carry):
        E, i = carry
        E_prev = E - (E - e * jnp.sin(E) - M) / (1.0 - e * jnp.cos(E))
        return (jnp.abs(E - E_prev) > tol) & (i < max_iter)

    E_final, _ = jax.lax.while_loop(cond_fn, body_fn, (E, 0))
    return E_final


@jit
def elements_to_cartesian(elements: OrbitalElements, t: float) -> CartesianState:
    """
    Convert orbital elements to Cartesian state at time t.
    t is time since epoch in seconds.
    """
    a, e, i, Omega, omega, M0 = elements.a, elements.e, elements.i, elements.Omega, elements.omega, elements.M0
    mu = MU_ALTAIRA
    
    # Mean motion
    n = jnp.sqrt(mu / a**3)
    
    # Mean anomaly at time t
    M = M0 + n * t
    
    # Solve for eccentric anomaly
    E = solve_kepler(M, e)
    
    # True anomaly
    theta = 2.0 * jnp.arctan2(
        jnp.sqrt(1.0 + e) * jnp.sin(E / 2.0),
        jnp.sqrt(1.0 - e) * jnp.cos(E / 2.0)
    )
    
    # Distance
    r_mag = a * (1.0 - e**2) / (1.0 + e * jnp.cos(theta))
    
    # Velocity magnitude
    v_mag = jnp.sqrt(2.0 * mu / r_mag - mu / a)
    
    # Flight path angle
    gamma = jnp.arctan2(e * jnp.sin(theta), 1.0 + e * jnp.cos(theta))
    
    # Position in orbital plane
    cos_theta_omega = jnp.cos(theta + omega)
    sin_theta_omega = jnp.sin(theta + omega)
    cos_Omega = jnp.cos(Omega)
    sin_Omega = jnp.sin(Omega)
    cos_i = jnp.cos(i)
    sin_i = jnp.sin(i)
    
    x = r_mag * (cos_theta_omega * cos_Omega - sin_theta_omega * cos_i * sin_Omega)
    y = r_mag * (cos_theta_omega * sin_Omega + sin_theta_omega * cos_i * cos_Omega)
    z = r_mag * sin_theta_omega * sin_i
    
    # Velocity in orbital plane
    cos_theta_omega_gamma = jnp.cos(theta + omega - gamma)
    sin_theta_omega_gamma = jnp.sin(theta + omega - gamma)
    
    vx = v_mag * (-sin_theta_omega_gamma * cos_Omega - cos_theta_omega_gamma * cos_i * sin_Omega)
    vy = v_mag * (-sin_theta_omega_gamma * sin_Omega + cos_theta_omega_gamma * cos_i * cos_Omega)
    vz = v_mag * cos_theta_omega_gamma * sin_i
    
    return CartesianState(r=jnp.array([x, y, z]), v=jnp.array([vx, vy, vz]))


@jit
def compute_v_infinity(v_sc: jnp.ndarray, v_body: jnp.ndarray) -> Tuple[jnp.ndarray, float]:
    """
    Compute hyperbolic excess velocity vector and magnitude.
    
    Returns:
        (v_infinity_vector, v_infinity_magnitude)
    """
    v_inf = v_sc - v_body
    v_inf_mag = jnp.linalg.norm(v_inf)
    return v_inf, v_inf_mag


@jit
def patched_conic_flyby(
    v_inf_in: jnp.ndarray,
    v_inf_out: jnp.ndarray,
    mu_body: float,
    r_body: float
) -> Tuple[float, bool]:
    """
    Validate and compute flyby altitude using patched conic approximation.
    
    Args:
        v_inf_in: incoming hyperbolic excess velocity (km/s)
        v_inf_out: outgoing hyperbolic excess velocity (km/s)
        mu_body: GM of the body (km^3/s^2)
        r_body: radius of the body (km)
    
    Returns:
        (flyby_altitude, is_valid)
    """
    v_inf_in_mag = jnp.linalg.norm(v_inf_in)
    v_inf_out_mag = jnp.linalg.norm(v_inf_out)
    
    # Check magnitude equality
    mag_diff = jnp.abs(v_inf_in_mag - v_inf_out_mag)
    
    # Compute turn angle
    cos_delta = jnp.dot(v_inf_in, v_inf_out) / (v_inf_in_mag * v_inf_out_mag + 1e-20)
    cos_delta = jnp.clip(cos_delta, -1.0, 1.0)
    delta = jnp.arccos(cos_delta)
    
    # Compute flyby altitude from turn angle
    sin_half_delta = jnp.sin(delta / 2.0)
    
    # rp = (mu_body / v_inf^2) * (1/sin(delta/2) - 1)
    rp = (mu_body / (v_inf_in_mag**2 + 1e-20)) * (1.0 / (sin_half_delta + 1e-20) - 1.0)
    h_p = rp - r_body
    
    # Check constraints
    is_valid = (mag_diff < 1e-4) & (h_p >= 0.1 * r_body) & (h_p <= 100.0 * r_body)
    
    return h_p, is_valid


def initial_arc_defects(
    r0: jnp.ndarray,
    rf: jnp.ndarray,
    t0: float,
    tf: float,
    distance_units: str,
    time_units: str
):
    """
    Compute the defects associated with the initial coast arc.

    Propagate a lambert arc from r0 to rf in the alloted time.
    The defects associated with this arc are:
    1. The initial position [r0_x, r0_y, r0_z] must have an r0_x component of -200 AU.
    2. The initial velocity resulting from the lambert solve must only
       have a nonzero component in the x direction.
    3. rf should be zero if it is at the same location as a body at tf.
       If it is between bodies then its value should be positive, and it should be
       some form of distance metric based on how close it is to the nearest bodies.
       In essence, this should provide some gradient that points to an rf that is at the location
       of some body.

    Args:
        r0: Initial position vector [x, y, z] in distance_units
        rf: Final position vector [x, y, z] in distance_units
        t0: Initial time in time_units
        tf: Final time in time_units
        distance_units: Units for position ('km', 'AU', 'DU')
        time_units: Units for time ('s', 'year', 'TU')

    Returns:
        Tuple of defects:
        - r0_x_defect: Defect in initial x position (should be -200 AU)
        - v0_y_defect: Initial velocity y component (should be 0)
        - v0_z_defect: Initial velocity z component (should be 0)
        - rf_defect: 3-vector of distance to nearest body [dx, dy, dz] (should be zero)
        - converged: Whether Lambert solver converged
    """
    from lamberthub import vallado2013_jax

    # Convert to canonical units (km and seconds) for Lambert solver
    if distance_units.lower() in ('au', 'du'):
        r0_km = r0 * KMPAU
        rf_km = rf * KMPAU
    else:  # km
        r0_km = r0
        rf_km = rf

    if time_units.lower() in ('year', 'years'):
        tof_s = (tf - t0) * YEAR
    elif time_units.lower() == 'tu':
        tof_s = (tf - t0) * SPTU
    else:  # seconds
        tof_s = tf - t0

    # Solve Lambert's problem
    v0, vf, lam_resid = vallado2013_jax(
        mu=MU_ALTAIRA,
        r1=r0_km,
        r2=rf_km,
        tof=tof_s,
        M=0,  # Zero revolutions
        prograde=True,
        low_path=True,
        full_output=False
    )

    # Defect 1: r0_x should be -200 AU
    r0_x_au = r0_km[0] / KMPAU
    r0_x_defect = r0_x_au + 200.0  # Should be zero when r0_x = -200 AU

    # Defect 2: v0_y should be zero
    v0_y_defect = v0[1]

    # Defect 3: v0_z should be zero
    v0_z_defect = v0[2]

    # Defect 4: rf should be at the location of a body
    # Compute body positions at time tf

    bodies_positions = []
    for body in bodies_data.values():
        state = body.get_state(tf, time_units=time_units, distance_units=distance_units)
        bodies_positions.append(state.r)

    bodies_positions = jnp.stack(bodies_positions)  # Shape: (n_bodies, 3)

    # Compute distance to each body
    deltas = bodies_positions - rf  # Shape: (n_bodies, 3)
    distances_squared = jnp.sum(deltas**2, axis=1)  # Shape: (n_bodies,)

    # Use softmin to compute weighted average of deltas
    # This provides a smooth, differentiable approximation to "nearest body"
    # Smaller alpha = sharper focus on nearest body
    alpha = 10.0  # Sharpness parameter
    weights = jnp.exp(-alpha * distances_squared)
    weights = weights / (jnp.sum(weights) + 1e-20)  # Normalize

    # Weighted average of position deltas points toward nearest body
    rf_defect = jnp.sum(weights[:, None] * deltas, axis=0)

    return r0_x_defect, v0_y_defect, v0_z_defect, rf_defect, lam_resid


def flyby_defects(
    v_in: jnp.ndarray,
    v_out: jnp.ndarray,
    v_body: jnp.ndarray,
    mu_body: float,
    r_body: float
) -> Tuple[float, bool]:
    """
    Compute parameters which determine whether a flyby is valid.

    Returns the body-centric difference between the incoming and
    outgoing v-infinity magnitudes, and the violation associated
    with the flyby altitude.
    
    The flyby altitude is the units of radii above or below the allowable flyby
    normalized altitude from 0.1 to 100.0.

    Therefore a flyby altitude of 110 body radii would return an h_p_defect of 10.
    A flyby altitude of 0.01 body radii would return an h_p_defect of 0.99.
    
    Args:
        v_in: incoming inertial velocity (km/s, DU/TU, or AU/year)
        v_out: outgoing inertial velocity (km/s, DU/TU, or AU/year)
        v_body: body inertial velocity (km/s, DU/TU, or AU/year)
        mu_body: GM of the body (km^3/s^2)
        r_body: radius of the body (km)
    
    Returns:
        (v_inf_mag_defect, h_p_defect)
    """
    v_inf_in = v_in - v_body
    v_inf_out = v_out - v_body

    # V_inf magnitude defect
    # Need to be the same from the body frame
    v_inf_in_mag = jnp.linalg.norm(v_inf_in)
    v_inf_out_mag = jnp.linalg.norm(v_inf_out)
    v_inf_mag_defect = v_inf_out_mag - v_inf_in_mag

    # Turn angle calculation
    # For this purpose, use the average of the two v_inf magnitudes
    # When defects are satisfied, they will be the same anyway.
    v_inf_mag_avg = (v_inf_in_mag + v_inf_out_mag) / 2
    cos_delta = jnp.dot(v_inf_out, v_inf_in) / v_inf_mag_avg
    cos_delta = jnp.clip(cos_delta, -1.0, 1.0)
    delta = jnp.arccos(cos_delta)

    # Compute flyby altitude from turn angle
    sin_half_delta = jnp.sin(delta / 2.0)

    # rp = (mu_body / v_inf^2) * (1/sin(delta/2) - 1)
    rp = (mu_body / (v_inf_in_mag**2)) * (1.0 / sin_half_delta) - 1.0)
    h_p_norm = (rp - r_body) / r_body

    # Compute altitude constraint defect (double-sided ReLU)
    # h_p_defect > 0 means altitude too high (> 100 radii) or too low (< 0.1 radii)
    # h_p_defect = 0 means altitude is valid (between 0.1 and 100 radii)
    h_p_defect = jnp.maximum(h_p_norm - 100.0, 0.0) + jnp.maximum(0.1 - h_p_norm, 0.0)

    return v_inf_mag_defect, h_p_defect
    

@jit
def seasonal_penalty(r_hat_current: jnp.ndarray, r_hat_previous: jnp.ndarray) -> float:
    """
    Compute seasonal penalty term S for a single previous flyby.
    
    Args:
        r_hat_current: unit heliocentric position vector at current flyby
        r_hat_previous: array of unit heliocentric position vectors at previous flybys (shape: (n_prev, 3))
    
    Returns:
        S: seasonal penalty factor
    """
    # Handle first flyby case
    def first_flyby():
        return 1.0
    
    def subsequent_flyby():
        # Compute angles with all previous flybys
        dot_products = jnp.dot(r_hat_previous, r_hat_current)
        dot_products = jnp.clip(dot_products, -1.0, 1.0)
        angles_deg = jnp.arccos(dot_products) * 180.0 / jnp.pi
        
        # Sum of exponential terms
        exp_sum = jnp.sum(jnp.exp(-angles_deg**2 / 50.0))
        
        S = 0.1 + 0.9 / (1.0 + 10.0 * exp_sum)
        return S
    
    n_prev = r_hat_previous.shape[0]
    return jax.lax.cond(n_prev == 0, first_flyby, subsequent_flyby)


@jit
def flyby_velocity_penalty(v_infinity: float) -> float:
    """
    Compute flyby velocity penalty term F.
    
    Args:
        v_infinity: hyperbolic excess velocity magnitude (km/s)
    
    Returns:
        F: velocity penalty factor
    """
    F = 0.2 + jnp.exp(-v_infinity / 13.0) / (1.0 + jnp.exp(-5.0 * (v_infinity - 1.5)))
    return F


@jit
def time_bonus(t_submission_days: float) -> float:
    """
    Compute time bonus term c based on submission time.
    
    Args:
        t_submission_days: days elapsed from competition start
    
    Returns:
        c: time bonus factor
    """
    return jax.lax.cond(
        t_submission_days <= 7.0,
        lambda: 1.13,
        lambda: -0.005 * t_submission_days + 1.165
    )


def compute_score(
    flybys: list,
    body_weights: dict,
    grand_tour_achieved: bool,
    submission_time_days: float
) -> float:
    """
    Compute total mission score.
    
    Args:
        flybys: list of flyby data, each containing:
                {'body_id', 'r_hat', 'v_infinity', 'is_scientific', 'r_hat_previous'}
        body_weights: dict mapping body_id to scientific weight
        grand_tour_achieved: whether grand tour bonus applies
        submission_time_days: days from competition start
    
    Returns:
        J: total score
    """
    b = 1.2 if grand_tour_achieved else 1.0
    c = time_bonus(submission_time_days)
    
    # Group flybys by body
    body_flybys = {}
    for fb in flybys:
        if not fb['is_scientific']:
            continue
        
        body_id = fb['body_id']
        if body_id not in body_flybys:
            body_flybys[body_id] = []
        body_flybys[body_id].append(fb)
    
    total_score = 0.0
    
    for body_id, fb_list in body_flybys.items():
        w_k = body_weights.get(body_id, 0.0)
        
        for i, fb in enumerate(fb_list[:13]):  # Max 13 scientific flybys per body
            # Get previous flyby positions for this body
            prev_list = [jnp.asarray(fb_list[j]['r_hat']) for j in range(i)]
            if prev_list:
                r_hat_prev = jnp.stack(prev_list, axis=0)
            else:
                r_hat_prev = jnp.zeros((0, 3))

            S = seasonal_penalty(fb['r_hat'], r_hat_prev)
            F = flyby_velocity_penalty(fb['v_infinity'])

            total_score += w_k * S * F
    
    J = b * c * total_score
    
    return J


# Example usage demonstration
if __name__ == "__main__":
    from .odes import solar_sail_acceleration

    print("GTOC13 JAX Simulation Framework")
    print("=" * 50)
    
    # Example: Create orbital elements for a planet
    example_planet = OrbitalElements(
        a=13.0 * KMPAU,
        e=0.05,
        i=jnp.deg2rad(2.0),
        Omega=jnp.deg2rad(45.0),
        omega=jnp.deg2rad(90.0),
        M0=jnp.deg2rad(0.0)
    )
    
    # Convert to Cartesian at t=0
    state = elements_to_cartesian(example_planet, 0.0)
    print(f"\nExample planet state at t=0:")
    print(f"Position (AU): {state.r / KMPAU}")
    print(f"Velocity (km/s): {state.v}")
    
    # Test solar sail acceleration
    r_test = jnp.array([13.0 * KMPAU, 0.0, 0.0])
    u_n_test = jnp.array([1.0, 0.0, 0.0])  # Facing sun
    a_sail = solar_sail_acceleration(r_test, u_n_test)
    print(f"\nSolar sail acceleration at 13 AU (mm/s²): {jnp.linalg.norm(a_sail) * 1e6}")
    
    # Test scoring functions
    v_inf_test = 10.0  # km/s
    F_test = flyby_velocity_penalty(v_inf_test)
    print(f"\nFlyby velocity penalty at V∞={v_inf_test} km/s: {F_test:.6f}")
    
    print("\nFramework ready for trajectory optimization!")
