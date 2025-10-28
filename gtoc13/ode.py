"""
GTOC13 Ordinary Differential Equations (ODEs).

This module contains the equations of motion for spacecraft dynamics in the GTOC13 problem,
including both ballistic (gravity-only) and controlled (solar sail) trajectories.
"""
import jax.numpy as jnp
from jax import jit

from .astrodynamics import MU_ALTAIRA, solar_sail_acceleration


@jit
def gtoc13_ballistic_ode(t: float, y: jnp.ndarray, args: None) -> jnp.ndarray:
    """
    GTOC13 ballistic Cartesian equations of motion (Keplerian two-body dynamics).

    Implements the equations of motion for a spacecraft under the gravitational
    influence of the central body (Altaira) only, with no solar sail acceleration.

    Args:
        t: Time (seconds)
        y: State vector [x, y, z, vx, vy, vz] in km and km/s
        args: Additional arguments (unused, for compatibility)

    Returns:
        State derivative [vx, vy, vz, ax, ay, az]
    """
    r = y[:3]
    v = y[3:]
    r_mag = jnp.linalg.norm(r)

    a = -MU_ALTAIRA * r / r_mag**3

    return jnp.concatenate([v, a])


@jit
def gtoc13_ode(t: float, y: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
    """
    GTOC13 Cartesian equations of motion with solar sail control.

    Implements the full equations of motion including gravitational acceleration
    from the central body (Altaira) and solar sail acceleration.

    Args:
        t: Time (seconds)
        y: State vector [x, y, z, vx, vy, vz] in km and km/s
        u: Control vector (solar sail unit normal vector), shape (3,)

    Returns:
        State derivative [vx, vy, vz, ax, ay, az]
    """
    r = y[:3]
    v = y[3:]
    r_mag = jnp.linalg.norm(r)

    # Gravitational acceleration
    a_grav = -MU_ALTAIRA * r / r_mag**3

    # Solar sail acceleration
    a_sail = solar_sail_acceleration(r, u)

    a_total = a_grav + a_sail

    return jnp.concatenate([v, a_total])
