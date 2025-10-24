import jax
import jax.numpy as jnp
from jax import jit
from diffrax import diffeqsolve, ODETerm, Dopri5, SaveAt, PIDController
from typing import Tuple
import numpy as np

from .orbital_elements import OrbitalElements
from .spacecraft_state import SpacecraftState

# Constants
KMPAU = 149597870.691  # km
MU_ALTAIRA = 139348062043.343  # km^3/s^2
DAY = 86400.0  # seconds
YEAR = 365.25 * DAY  # seconds

# Solar sail parameters
C_FLUX = 5.4026e-6  # N/m^2 at 1 AU
R0 = 1.0 * KMPAU  # Reference distance (1 AU in km)
SAIL_AREA = 15000.0  # m^2
SPACECRAFT_MASS = 500.0  # kg


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
def elements_to_cartesian(elements: OrbitalElements, t: float) -> SpacecraftState:
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
    
    return SpacecraftState(r=jnp.array([x, y, z]), v=jnp.array([vx, vy, vz]))


@jit
def solar_sail_acceleration(r: jnp.ndarray, u_n: jnp.ndarray) -> jnp.ndarray:
    """
    Calculate solar sail acceleration.
    
    Args:
        r: spacecraft position relative to Altaira (km)
        u_n: sail normal unit vector (must point towards sun, cone angle in [0, 90])
    
    Returns:
        acceleration vector (km/s^2)
    """
    r_mag = jnp.linalg.norm(r)
    u_r = r / r_mag  # unit vector from spacecraft to sun
    
    cos_alpha = jnp.dot(u_n, u_r)
    
    # Ensure cone angle is valid (should be enforced by caller)
    cos_alpha = jnp.clip(cos_alpha, 0.0, 1.0)
    
    # Acceleration magnitude
    coeff = -2.0 * C_FLUX * SAIL_AREA / SPACECRAFT_MASS
    a_mag = coeff * (R0 / r_mag)**2 * cos_alpha**2
    
    return a_mag * u_n


@jit
def keplerian_derivatives(t: float, y: jnp.ndarray, args: None) -> jnp.ndarray:
    """
    Derivatives for pure Keplerian motion (no solar sail).
    y = [x, y, z, vx, vy, vz]
    """
    r = y[:3]
    v = y[3:]
    r_mag = jnp.linalg.norm(r)
    
    a = -MU_ALTAIRA * r / r_mag**3
    
    return jnp.concatenate([v, a])


@jit
def solar_sail_derivatives(t: float, y: jnp.ndarray, args: Tuple) -> jnp.ndarray:
    """
    Derivatives including solar sail acceleration.
    y = [x, y, z, vx, vy, vz]
    args = (u_n,) where u_n is the sail normal unit vector
    """
    r = y[:3]
    v = y[3:]
    r_mag = jnp.linalg.norm(r)
    
    u_n = args[0]
    
    # Gravitational acceleration
    a_grav = -MU_ALTAIRA * r / r_mag**3
    
    # Solar sail acceleration
    a_sail = solar_sail_acceleration(r, u_n)
    
    a_total = a_grav + a_sail
    
    return jnp.concatenate([v, a_total])


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
            r_hat_prev = jnp.array([fb_list[j]['r_hat'] for j in range(i)])
            
            S = seasonal_penalty(fb['r_hat'], r_hat_prev)
            F = flyby_velocity_penalty(fb['v_infinity'])
            
            total_score += w_k * S * F
    
    J = b * c * total_score
    
    return J


# Example usage demonstration
if __name__ == "__main__":
    print("GTOC13 JAX Simulation Framework")
    print("=" * 50)
    
    # Example: Create orbital elements for a planet
    example_planet = OrbitalElements(
        a=13.0 * KMPAU,
        e=0.05,
        i=jnp.deg2rad(2.0),
        Omega=jnp.deg2rad(45.0),
        omega=jnp.deg2rad(90.0),
        M0=jnp.deg2rad(0.0),
        mu_body=1e8,
        radius=70000.0,
        weight=10.0
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