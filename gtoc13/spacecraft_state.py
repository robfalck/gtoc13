"""
Spacecraft state representation in Cartesian coordinates.
"""
from typing import NamedTuple
import jax.numpy as jnp


class SpacecraftState(NamedTuple):
    """
    Cartesian state of a spacecraft or celestial body.

    This represents the position and velocity in heliocentric (sun-centered)
    Cartesian coordinates. The state is compatible with JAX transformations.

    Attributes:
        r: Position vector [x, y, z] in km
        v: Velocity vector [vx, vy, vz] in km/s

    Note:
        - Both r and v are JAX arrays (jnp.ndarray)
        - This is JAX-compatible and can be used with jax.jit, jax.vmap, etc.
        - The coordinate system is typically heliocentric ecliptic

    Examples:
        >>> import jax.numpy as jnp
        >>> state = SpacecraftState(
        ...     r=jnp.array([1.5e8, 0.0, 0.0]),  # 1 AU from sun
        ...     v=jnp.array([0.0, 30.0, 0.0])     # ~30 km/s orbital velocity
        ... )
        >>> print(state.r)
        [1.5e8 0.0 0.0]
    """
    r: jnp.ndarray  # position [x, y, z] (km)
    v: jnp.ndarray  # velocity [vx, vy, vz] (km/s)
