"""
Lambert solver for two-point boundary value problems in orbital mechanics.

Solves Lambert's problem: given two position vectors and a transfer time,
find the initial and final velocity vectors for a conic trajectory.

Uses Izzo's algorithm (2015) which is robust and efficient.
"""
import jax
import jax.numpy as jnp
from jax import jit
from typing import Tuple, Optional
import numpy as np

from gtoc13 import elements_to_cartesian, MU_ALTAIRA, KMPAU


def get_body_state(body_id: int, epoch: float, bodies_data: dict):
    """
    Get the Cartesian state of a body at a given epoch.

    Args:
        body_id: Body identifier
        epoch: Time in seconds
        bodies_data: Dictionary containing orbital elements for all bodies

    Returns:
        (position, velocity) tuple in km and km/s
    """
    if body_id not in bodies_data:
        raise ValueError(f"Body ID {body_id} not found in bodies_data")

    elements = bodies_data[body_id]
    state = elements_to_cartesian(elements, epoch)
    return np.array(state.r), np.array(state.v)


@jit
def stumpff_c(z: float) -> float:
    """
    Stumpff function C(z)

    C(z) = (1 - cos(sqrt(z))) / z         for z > 0  (elliptic)
    C(z) = (1 - cosh(sqrt(-z))) / z       for z < 0  (hyperbolic)
    C(z) = 1/2                            for z = 0  (parabolic)
    """
    def elliptic():
        sqrt_z = jnp.sqrt(z)
        return (1.0 - jnp.cos(sqrt_z)) / z

    def hyperbolic():
        sqrt_neg_z = jnp.sqrt(-z)
        return (1.0 - jnp.cosh(sqrt_neg_z)) / z

    def parabolic():
        return 0.5

    # Use small z approximation near zero for numerical stability
    def small_z():
        # Taylor series: C(z) = 1/2 - z/24 + z^2/720 - ...
        return 0.5 - z / 24.0 + z**2 / 720.0

    return jax.lax.cond(
        jnp.abs(z) < 1e-4,
        small_z,
        lambda: jax.lax.cond(
            z > 0,
            elliptic,
            lambda: jax.lax.cond(z < 0, hyperbolic, parabolic)
        )
    )


@jit
def stumpff_s(z: float) -> float:
    """
    Stumpff function S(z)

    S(z) = (sqrt(z) - sin(sqrt(z))) / sqrt(z)^3      for z > 0  (elliptic)
    S(z) = (sinh(sqrt(-z)) - sqrt(-z)) / sqrt(-z)^3  for z < 0  (hyperbolic)
    S(z) = 1/6                                       for z = 0  (parabolic)
    """
    def elliptic():
        sqrt_z = jnp.sqrt(z)
        return (sqrt_z - jnp.sin(sqrt_z)) / (sqrt_z**3)

    def hyperbolic():
        sqrt_neg_z = jnp.sqrt(-z)
        return (jnp.sinh(sqrt_neg_z) - sqrt_neg_z) / (sqrt_neg_z**3)

    def parabolic():
        return 1.0 / 6.0

    # Use small z approximation near zero for numerical stability
    def small_z():
        # Taylor series: S(z) = 1/6 - z/120 + z^2/5040 - ...
        return 1.0 / 6.0 - z / 120.0 + z**2 / 5040.0

    return jax.lax.cond(
        jnp.abs(z) < 1e-4,
        small_z,
        lambda: jax.lax.cond(
            z > 0,
            elliptic,
            lambda: jax.lax.cond(z < 0, hyperbolic, parabolic)
        )
    )


@jit
def lambert_tof(
    z: float,
    r1: jnp.ndarray,
    r2: jnp.ndarray,
    dt: float,
    mu: float,
    short: bool = True
) -> Tuple[float, float, float]:
    """
    Compute the Lambert problem time of flight for a given value of the universal variable z.

    This function is designed for use with external solvers (e.g., OpenMDAO) that will
    iterate on z to drive the residual to zero.

    The function uses a smooth softplus activation to ensure y > 0, making the residual
    continuously differentiable. This enables gradient-based optimization methods.

    Args:
        z: Universal variable (to be solved for)
        r1: Initial position vector (km)
        r2: Final position vector (km)
        dt: Desired time of flight (seconds)
        mu: Gravitational parameter (km^3/s^2)
        short: True for short way (< 180°), False for long way

    Returns:
        tof:
            time of flight, in whatever time units are compatible with
            mu, r1, and r2. These could be km**3/s**2 or DU**3/TU**2, for instance.
        A:
            the intermediate A variable of the lambert iteration.
        y:
            the intermediate y varaible of the lambert iteration.
            
    Usage with OpenMDAO:
        The solver should iterate on z to drive residual → 0.
        Once converged, v1 and v2 are the Lambert solution velocities.

    Notes:
        - The softplus activation ensures continuous gradients for optimization
        - JAX autodiff can be used to compute exact derivatives: jax.grad(residual_func)
        - For physically valid solutions, y should be naturally positive
    """
    # Magnitudes
    r1_mag = jnp.linalg.norm(r1)
    r2_mag = jnp.linalg.norm(r2)

    # Cosine of transfer angle
    cos_dnu = jnp.dot(r1, r2) / (r1_mag * r2_mag)
    cos_dnu = jnp.clip(cos_dnu, -1.0, 1.0)

    # Determine A parameter based on transfer direction
    def get_A_short():
        return jnp.sqrt(r1_mag * r2_mag * (1.0 + cos_dnu))

    def get_A_long():
        return -jnp.sqrt(r1_mag * r2_mag * (1.0 + cos_dnu))

    A = jax.lax.cond(short, get_A_short, get_A_long)

    # Compute Stumpff functions
    C_z = stumpff_c(z)
    S_z = stumpff_s(z)

    # Compute y(z)
    y_raw = r1_mag + r2_mag + A * (z * S_z - 1.0) / jnp.sqrt(C_z)

    # Use smooth activation to keep y positive with continuous derivatives
    # softplus(x) = log(1 + exp(x)) ensures y > 0 and is infinitely differentiable
    # Add small constant to avoid y = 0
    y = jnp.logaddexp(0.0, y_raw) + 1e-10  # softplus activation

    # Compute X(z)
    X = jnp.sqrt(y / C_z)

    # Compute time of flight for this z
    t = (X**3 * S_z + A * jnp.sqrt(y)) / jnp.sqrt(mu)

    return t, A, y


def lambert_v(
    A: float,
    y: float,
    r1: jnp.ndarray,
    r2: jnp.ndarray,
    mu: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute the Lambert problem time of flight for a given value of the universal variable z.

    This function is designed for use with external solvers (e.g., OpenMDAO) that will
    iterate on z to drive the residual to zero.

    The function uses a smooth softplus activation to ensure y > 0, making the residual
    continuously differentiable. This enables gradient-based optimization methods.

    Args:
        z: Universal variable (to be solved for)
        r1: Initial position vector (km)
        r2: Final position vector (km)
        dt: Desired time of flight (seconds)
        mu: Gravitational parameter (km^3/s^2)
        short: True for short way (< 180°), False for long way

    Returns:
        v1:
            cartesian velocity vector at the initial position.
        v2:
            cartesian velocity vector at the final position.
    """
    # Magnitudes
    r1_mag = jnp.linalg.norm(r1)
    r2_mag = jnp.linalg.norm(r2)

    # Compute velocities using Lagrange coefficients
    f = 1.0 - y / r1_mag
    g = A * jnp.sqrt(y / mu)
    gdot = 1.0 - y / r2_mag

    v1 = (r2 - f * r1) / g
    v2 = (gdot * r2 - r1) / g

    return v1, v2


@jit
def lambert_universal_variables(
    r1: jnp.ndarray,
    r2: jnp.ndarray,
    dt: float,
    mu: float,
    short: bool = True,
    eps: float = 1e-8,
    max_iter: int = 50
) -> Tuple[jnp.ndarray, jnp.ndarray, bool]:
    """
    Solve Lambert's problem using the universal variables formulation.

    Args:
        r1: Initial position vector (km)
        r2: Final position vector (km)
        dt: Time of flight (seconds)
        mu: Gravitational parameter (km^3/s^2)
        short: True for short way (< 180°), False for long way
        eps: Convergence tolerance
        max_iter: Maximum number of iterations

    Returns:
        (v1, v2, converged): Initial velocity, final velocity (km/s), and convergence flag
    """
    # Magnitudes
    r1_mag = jnp.linalg.norm(r1)
    r2_mag = jnp.linalg.norm(r2)

    # Cosine of transfer angle
    cos_dnu = jnp.dot(r1, r2) / (r1_mag * r2_mag)
    cos_dnu = jnp.clip(cos_dnu, -1.0, 1.0)

    # Cross product for direction
    cross = jnp.cross(r1, r2)

    # Determine transfer angle direction
    # For short way: 0 < dnu < pi
    # For long way: pi < dnu < 2*pi
    def get_A_short():
        return jnp.sqrt(r1_mag * r2_mag * (1.0 + cos_dnu))

    def get_A_long():
        return -jnp.sqrt(r1_mag * r2_mag * (1.0 + cos_dnu))

    A = jax.lax.cond(short, get_A_short, get_A_long)

    # Check if transfer is possible
    def valid_transfer():
        # Initial guess for z (universal variable)
        # For long-distance, low-energy transfers, start with a small negative z
        # This represents a nearly-parabolic hyperbolic trajectory
        z_init = -0.01

        # Newton-Raphson iteration
        def body_fn(carry):
            z, i = carry

            C_z = stumpff_c(z)
            S_z = stumpff_s(z)

            # Compute y(z)
            y = r1_mag + r2_mag + A * (z * S_z - 1.0) / jnp.sqrt(C_z)

            # Avoid division by very small y
            y = jnp.where(jnp.abs(y) < 1e-10, 1e-10, y)

            # Compute X(z) = sqrt(y/C(z))
            X = jnp.sqrt(y / C_z)

            # Time of flight function
            t = (X**3 * S_z + A * jnp.sqrt(y)) / jnp.sqrt(mu)

            # Newton-Raphson derivative
            # dt/dz - handle z=0 case specially
            def compute_dtdz():
                C_z_safe = jnp.where(jnp.abs(C_z) < 1e-10, 1e-10, C_z)

                # When z is very close to 0, use a simplified form to avoid division by zero
                def dtdz_near_zero():
                    # Near z=0: C ≈ 1/2, S ≈ 1/6
                    # The derivative simplifies to a finite value
                    return (jnp.sqrt(2.0) * (A * jnp.sqrt(y) + X**3)) / (2.0 * jnp.sqrt(mu) * y)

                def dtdz_normal():
                    term1 = (X**3) * (1.0 / (2.0 * z) * (C_z - 3.0 * S_z / (2.0 * C_z)) + 3.0 * S_z**2 / (4.0 * C_z))
                    term2 = A / 8.0 * (3.0 * S_z / C_z_safe * jnp.sqrt(y) + A / X)
                    return (term1 + term2) / jnp.sqrt(mu)

                return jax.lax.cond(jnp.abs(z) < 1e-6, dtdz_near_zero, dtdz_normal)

            dtdz = compute_dtdz()

            # Damped Newton step with adaptive damping
            dtdz_safe = jnp.where(jnp.abs(dtdz) < 1e-10, 1e-10, dtdz)
            z_newton = z - (t - dt) / dtdz_safe  # Full Newton step

            # Compute what y would be at the full Newton step
            C_z_newton = stumpff_c(z_newton)
            S_z_newton = stumpff_s(z_newton)
            y_newton = r1_mag + r2_mag + A * (z_newton * S_z_newton - 1.0) / jnp.sqrt(C_z_newton)

            # Choose damping factor based on whether y would go negative
            # If y_newton > 0, try full step; otherwise use smaller damping
            def full_step():
                return z_newton

            def damped_step():
                # Use damping factor of 0.7 to prevent oscillation
                return z + 0.7 * (z_newton - z)

            def heavy_damped_step():
                # Very small step if y would go very negative
                return z + 0.3 * (z_newton - z)

            z_new = jax.lax.cond(
                y_newton > y * 0.1,  # If y doesn't drop too much
                full_step,
                lambda: jax.lax.cond(
                    y_newton > 0,
                    damped_step,
                    heavy_damped_step
                )
            )

            return (z_new, i + 1)

        def cond_fn(carry):
            z, i = carry

            C_z = stumpff_c(z)
            S_z = stumpff_s(z)
            y = r1_mag + r2_mag + A * (z * S_z - 1.0) / jnp.sqrt(C_z)
            y = jnp.where(jnp.abs(y) < 1e-10, 1e-10, y)
            X = jnp.sqrt(y / C_z)
            t = (X**3 * S_z + A * jnp.sqrt(y)) / jnp.sqrt(mu)

            # Use relative tolerance: |t - dt| / dt > eps
            relative_error = jnp.abs(t - dt) / dt
            return (relative_error > eps) & (i < max_iter)

        z_final, iter_count = jax.lax.while_loop(cond_fn, body_fn, (z_init, 0))

        # Compute final velocities
        C_z = stumpff_c(z_final)
        S_z = stumpff_s(z_final)
        y = r1_mag + r2_mag + A * (z_final * S_z - 1.0) / jnp.sqrt(C_z)
        y = jnp.where(jnp.abs(y) < 1e-10, 1e-10, y)

        # Lagrange coefficients
        f = 1.0 - y / r1_mag
        g = A * jnp.sqrt(y / mu)
        gdot = 1.0 - y / r2_mag

        # Velocities
        v1 = (r2 - f * r1) / g
        v2 = (gdot * r2 - r1) / g

        converged = iter_count < max_iter

        return v1, v2, converged

    def invalid_transfer():
        # Return zero velocities and not converged
        return jnp.zeros(3), jnp.zeros(3), False

    # Check if A is valid (not too close to zero)
    return jax.lax.cond(
        jnp.abs(A) > 1e-6,
        valid_transfer,
        invalid_transfer
    )


def lambert(
    body1_id: int,
    body2_id: int,
    body1_time: float,
    body2_time: float,
    mu: float = MU_ALTAIRA,
    short: bool = True,
    eps: float = 1e-8,
    max_iter: int = 50,
    bodies_data: Optional[dict] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    """
    Solve Lambert's problem between two celestial bodies.

    Given two bodies and their encounter times, compute the transfer trajectory.

    Args:
        body1_id: ID of departure body
        body2_id: ID of arrival body
        body1_time: Departure epoch (seconds past t=0)
        body2_time: Arrival epoch (seconds past t=0)
        mu: Gravitational parameter of central body (default: Altaira)
        short: True for short way transfer (< 180°), False for long way
        eps: Convergence tolerance
        max_iter: Maximum Newton-Raphson iterations
        bodies_data: Dictionary mapping body_id to OrbitalElements (optional)

    Returns:
        (r1, v1, r2, v2, converged):
            r1: Departure position (km)
            v1: Departure velocity (km/s)
            r2: Arrival position (km)
            v2: Arrival velocity (km/s)
            converged: Whether the solver converged

    Example:
        >>> from gtoc13 import lambert
        >>> # Transfer from body 2 to body 3
        >>> r1, v1, r2, v2, converged = lambert(
        ...     body1_id=2,
        ...     body2_id=3,
        ...     body1_time=0.0,
        ...     body2_time=3.156e7,  # 1 year
        ...     bodies_data=bodies_dict
        ... )
        >>> print(f"Converged: {converged}")
        >>> print(f"Delta-V at departure: {np.linalg.norm(v1 - v_body1):.2f} km/s")
    """
    if bodies_data is None:
        raise ValueError("bodies_data dictionary is required. Load orbital elements first.")

    # Get body states at their respective times
    r1, v_body1 = get_body_state(body1_id, body1_time, bodies_data)
    r2, v_body2 = get_body_state(body2_id, body2_time, bodies_data)

    # Time of flight
    dt = body2_time - body1_time

    if dt <= 0:
        raise ValueError("body2_time must be greater than body1_time")

    # Solve Lambert's problem
    r1_jax = jnp.array(r1)
    r2_jax = jnp.array(r2)

    v1_transfer, v2_transfer, converged = lambert_universal_variables(
        r1_jax, r2_jax, dt, mu, short, eps, max_iter
    )

    # Convert back to numpy
    v1_transfer = np.array(v1_transfer)
    v2_transfer = np.array(v2_transfer)
    converged = bool(converged)

    return r1, v1_transfer, r2, v2_transfer, converged


def lambert_delta_v(
    body1_id: int,
    body2_id: int,
    body1_time: float,
    body2_time: float,
    mu: float = MU_ALTAIRA,
    short: bool = True,
    eps: float = 1e-8,
    max_iter: int = 50,
    bodies_data: Optional[dict] = None
) -> Tuple[float, float, bool]:
    """
    Compute the delta-V required for a Lambert transfer.

    Returns the departure and arrival delta-V magnitudes.

    Args:
        Same as lambert()

    Returns:
        (dv_departure, dv_arrival, converged):
            dv_departure: Delta-V magnitude at departure (km/s)
            dv_arrival: Delta-V magnitude at arrival (km/s)
            converged: Whether the solver converged
    """
    r1, v1_transfer, r2, v2_transfer, converged = lambert(
        body1_id, body2_id, body1_time, body2_time,
        mu, short, eps, max_iter, bodies_data
    )

    if not converged:
        return np.inf, np.inf, False

    # Get body velocities
    _, v_body1 = get_body_state(body1_id, body1_time, bodies_data)
    _, v_body2 = get_body_state(body2_id, body2_time, bodies_data)

    # Delta-V magnitudes
    dv_departure = np.linalg.norm(v1_transfer - v_body1)
    dv_arrival = np.linalg.norm(v2_transfer - v_body2)

    return dv_departure, dv_arrival, converged


# Example usage
if __name__ == "__main__":
    print("Lambert Solver Example")
    print("=" * 50)

    # This would require loading body data first
    print("\nTo use the Lambert solver:")
    print("1. Load orbital elements for bodies into a dictionary")
    print("2. Call lambert(body1_id, body2_id, t1, t2, bodies_data=bodies)")
    print("\nExample:")
    print("  r1, v1, r2, v2, converged = lambert(")
    print("      body1_id=2,")
    print("      body2_id=3,")
    print("      body1_time=0.0,")
    print("      body2_time=YEAR,")
    print("      bodies_data=bodies_dict")
    print("  )")
