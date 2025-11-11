
from typing import Tuple
import jax.numpy as jnp

from gtoc13.constants import C_FLUX, SAIL_AREA, SPACECRAFT_MASS, R0


def solar_sail_acceleration(r: jnp.ndarray, u_n: jnp.ndarray, r0: float) -> jnp.ndarray:
    """
    Calculate solar sail acceleration.
    
    Args:
        r: spacecraft position relative to Altaira (km)
        u_n: sail normal unit vector (must point towards sun, cone angle in [0, 90])
        r0 : reference distance for solar sail. (should be the same units as r)
    
    Returns:
        acceleration vector (km/s^2)
    """
    r_mag = jnp.linalg.norm(r, axis=-1, keepdims=True)
    u_r = -r / r_mag
    
    cos_alpha = jnp.dot(u_n, u_r)
    
    # Ensure cone angle is valid.
    # Note that this cosine should only be from 0.0 to 1.0, but rather than clipping it here
    # We'll enforce that constraint externally.
    cos_alpha = jnp.clip(cos_alpha, -1.0, 1.0)
    
    # Acceleration magnitude
    coeff = -2.0 * C_FLUX * SAIL_AREA / SPACECRAFT_MASS # N / kg = kg * m / s**2 /kg = m/s**2
    coeff = coeff / 1000.0  # km/s**2
    a_mag = coeff * (r0 / r_mag)**2 * cos_alpha**2
    
    return a_mag * u_n, cos_alpha


def ballistic_ode(t: float, y: jnp.ndarray, args: None) -> jnp.ndarray:
    """
    Derivatives for pure Keplerian motion (no solar sail).
    y = [x, y, z, vx, vy, vz]
    args = (mu,)
    """
    mu = args[0]
    r = y[:3]
    v = y[3:]
    r_mag = jnp.linalg.norm(r)
    
    a = -mu * r / r_mag**3
    
    return jnp.concatenate([v, a])


def solar_sail_ode(r: jnp.ndarray, v: jnp.ndarray, u_n: jnp.ndarray, mu: float, r0: float) -> jnp.ndarray:
    """
    Derivatives for pure Keplerian motion (no solar sail).
    r = [x, y, z]
    v = [vx, vy, vz]
    dt_dtau = transfer time to nondimensional time conversion factor.
    u_n = [u_x, u_y, u_z]
    mu = gravitational parameter in consistent units.
    r0 = reference solar array distance in consistent units.

    We're going to assume that a single arc of the trajectory requires dt units of time.
    Each arc also traverses from [-1, 1] in nondimensional time.
    Argument dt_dtau provides a factor that we can use to convert the result of this ODE
    from "per unit of time" to "per unit of tau".
    """
    r_mag = jnp.linalg.norm(r)
    
    a_grav = -mu * r / r_mag**3    
    a_sail, cos_alpha = solar_sail_acceleration(r, u_n, r0)
    a_total = a_grav + a_sail
    
    return v, a_total, a_grav, a_sail, cos_alpha
