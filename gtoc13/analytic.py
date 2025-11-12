"""
Analytic trajectory propagation functions for GTOC13.
"""
import jax.numpy as jnp
from jax import lax
from gtoc13.constants import MU_ALTAIRA


def propagate_ballistic(r0: jnp.ndarray, v0: jnp.ndarray, times: jnp.ndarray,
                        mu: float = MU_ALTAIRA) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Propagate a ballistic (Keplerian) trajectory using f and g functions.

    This function uses the classical Lagrange coefficients (f, g, fdot, gdot) to
    analytically propagate an orbit. This is more accurate and efficient than
    numerical integration for pure two-body motion.

    JAX-compatible for automatic differentiation.

    Args:
        r0: Initial position vector [x, y, z] in km
        v0: Initial velocity vector [vx, vy, vz] in km/s
        times: Array of times in seconds at which to evaluate the trajectory.
               times[0] is treated as the initial time (t0).
        mu: Gravitational parameter in km^3/s^2 (default: MU_ALTAIRA)

    Returns:
        positions: Array of shape (n_times, 3) with positions in km
        velocities: Array of shape (n_times, 3) with velocities in km/s

    References:
        Vallado, D. A. (2013). Fundamentals of Astrodynamics and Applications (4th ed.).
        Section 2.3: Position and Velocity as a Function of Time
    """
    # Ensure inputs are JAX arrays
    r0 = jnp.asarray(r0, dtype=float)
    v0 = jnp.asarray(v0, dtype=float)
    times = jnp.asarray(times, dtype=float)

    # Initial orbital parameters
    r0_mag = jnp.linalg.norm(r0)
    v0_mag = jnp.linalg.norm(v0)

    # Radial velocity
    vr0 = jnp.dot(r0, v0) / r0_mag

    # Specific angular momentum
    h_vec = jnp.cross(r0, v0)
    h = jnp.linalg.norm(h_vec)

    # Semi-parameter (semi-latus rectum)
    p = h**2 / mu

    # Eccentricity vector and magnitude
    e_vec = ((v0_mag**2 - mu/r0_mag) * r0 - jnp.dot(r0, v0) * v0) / mu
    e = jnp.linalg.norm(e_vec)

    # Semi-major axis (handle near-parabolic case)
    a = jnp.where(e < 0.999999, p / (1 - e**2), 1e10)

    # Reference time (first time in array)
    t0 = times[0]

    # Vectorized propagation using vmap
    def propagate_single_time(t):
        dt = t - t0

        # Handle initial time case
        def initial_state():
            return r0, v0

        def propagated_state():
            # Compute universal anomaly using Newton's method
            chi = _solve_universal_kepler(r0_mag, vr0, a, dt, mu)

            # Compute Stumpff functions
            z = jnp.where(a > 0, chi**2 / a, -chi**2 / a)
            c = _stumpff_c(z)
            s = _stumpff_s(z)

            # Lagrange coefficients
            f = 1 - (chi**2 / r0_mag) * c
            g = dt - (chi**3 / jnp.sqrt(mu)) * s

            # New position
            r = f * r0 + g * v0
            r_mag = jnp.linalg.norm(r)

            # Time derivatives of Lagrange coefficients
            fdot = (jnp.sqrt(mu) / (r_mag * r0_mag)) * chi * (z * s - 1)
            gdot = 1 - (chi**2 / r_mag) * c

            # New velocity
            v = fdot * r0 + gdot * v0

            return r, v

        return lax.cond(dt == 0, initial_state, propagated_state)

    # Use vmap to vectorize over all times
    from jax import vmap
    positions, velocities = vmap(propagate_single_time)(times)

    return positions, velocities


def _solve_universal_kepler(r0_mag: float, vr0: float, a: float, dt: float,
                            mu: float, tol: float = 1e-12, max_iter: int = 50) -> float:
    """
    Solve the universal Kepler equation using Newton-Raphson iteration.

    JAX-compatible using lax.fori_loop for fixed iteration count.

    Args:
        r0_mag: Initial position magnitude (km)
        vr0: Initial radial velocity (km/s)
        a: Semi-major axis (km), can be negative for hyperbolic orbits
        dt: Time since epoch (s)
        mu: Gravitational parameter (km^3/s^2)
        tol: Convergence tolerance
        max_iter: Maximum number of iterations

    Returns:
        chi: Universal anomaly
    """
    sqrt_mu = jnp.sqrt(mu)

    # Initial guess for universal anomaly
    chi_elliptic = sqrt_mu * dt / a

    # Hyperbolic guess (safeguarded)
    safe_arg = jnp.maximum(
        (-2 * mu * dt) / (a * (vr0 + jnp.sign(dt) * jnp.sqrt(jnp.abs(-mu / a)) * (1 - r0_mag / a))),
        1e-10
    )
    chi_hyperbolic = jnp.sign(dt) * jnp.sqrt(-a) * jnp.log(safe_arg)

    chi = jnp.where(a > 0, chi_elliptic, chi_hyperbolic)

    # Newton-Raphson iteration using lax.fori_loop
    def newton_step(i, chi):
        z = jnp.where(a > 0, chi**2 / a, -chi**2 / a)
        c = _stumpff_c(z)
        s = _stumpff_s(z)

        # Universal Kepler equation
        f = (r0_mag * vr0 / sqrt_mu) * chi**2 * c \
            + (1 - r0_mag / a) * chi**3 * s \
            + r0_mag * chi \
            - sqrt_mu * dt

        # Derivative
        fp = (r0_mag * vr0 / sqrt_mu) * chi * (1 - z * s) \
             + (1 - r0_mag / a) * chi**2 * c \
             + r0_mag

        # Newton step
        chi_new = chi - f / fp

        return chi_new

    chi = lax.fori_loop(0, max_iter, newton_step, chi)

    return chi


def _stumpff_c(z: float) -> float:
    """
    Compute the Stumpff C function.

    C(z) = (1 - cos(sqrt(z))) / z for z > 0
    C(z) = (cosh(sqrt(-z)) - 1) / (-z) for z < 0
    C(0) = 1/2

    JAX-compatible with smooth transitions.

    Args:
        z: Argument (unitless)

    Returns:
        Value of C(z)
    """
    # Series expansion for small z
    c_series = 0.5 - z/24 + z**2/720 - z**3/40320

    # Positive z case
    sqrt_z = jnp.sqrt(jnp.abs(z))
    c_pos = (1 - jnp.cos(sqrt_z)) / z

    # Negative z case
    c_neg = (jnp.cosh(sqrt_z) - 1) / (-z)

    # Use series near zero, otherwise use appropriate formula
    c = jnp.where(z > 1e-6, c_pos,
                  jnp.where(z < -1e-6, c_neg, c_series))

    return c


def _stumpff_s(z: float) -> float:
    """
    Compute the Stumpff S function.

    S(z) = (sqrt(z) - sin(sqrt(z))) / (sqrt(z))^3 for z > 0
    S(z) = (sinh(sqrt(-z)) - sqrt(-z)) / (sqrt(-z))^3 for z < 0
    S(0) = 1/6

    JAX-compatible with smooth transitions.

    Args:
        z: Argument (unitless)

    Returns:
        Value of S(z)
    """
    # Series expansion for small z
    s_series = 1/6 - z/120 + z**2/5040 - z**3/362880

    # Positive z case
    sqrt_z = jnp.sqrt(jnp.abs(z))
    s_pos = (sqrt_z - jnp.sin(sqrt_z)) / (sqrt_z**3)

    # Negative z case
    s_neg = (jnp.sinh(sqrt_z) - sqrt_z) / (sqrt_z**3)

    # Use series near zero, otherwise use appropriate formula
    s = jnp.where(z > 1e-6, s_pos,
                  jnp.where(z < -1e-6, s_neg, s_series))

    return s
