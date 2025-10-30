
from typing import Tuple
import jax.numpy as jnp

from gtoc13.constants import C_FLUX, SAIL_AREA, SPACECRAFT_MASS, R0


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


def solar_sail_ode(t: float, y: jnp.ndarray, args: None) -> jnp.ndarray:
    """
    Derivatives for pure Keplerian motion (no solar sail).
    y = [x, y, z, vx, vy, vz]
    args = (mu, u_n)
    """
    mu, u_n = args
    r = y[:3]
    v = y[3:]
    r_mag = jnp.linalg.norm(r)
    
    a_grav = -mu * r / r_mag**3
    # TODO: solar_sail_acceleration requires specific units
    a_sail = solar_sail_acceleration(r, u_n)
    a_total = a_grav + a_sail
    
    return jnp.concatenate([v, a_total])
