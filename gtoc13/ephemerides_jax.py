import jax.numpy as jnp

from gtoc13.astrodynamics import solve_kepler


def keplerian_state(elements: jnp.ndarray, mu_star: float, t: float) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute heliocentric Cartesian state from orbital elements (JAX friendly)."""
    a, e, inc, Omega, omega, M0 = elements
    n = jnp.sqrt(mu_star / (a ** 3))
    M = M0 + n * t
    E = solve_kepler(M, e)

    theta = 2.0 * jnp.arctan2(
        jnp.sqrt(1.0 + e) * jnp.sin(E / 2.0),
        jnp.sqrt(1.0 - e) * jnp.cos(E / 2.0),
    )
    r_mag = a * (1.0 - e ** 2) / (1.0 + e * jnp.cos(theta))
    v_mag = jnp.sqrt(2.0 * mu_star / r_mag - mu_star / a)
    gamma = jnp.arctan2(e * jnp.sin(theta), 1.0 + e * jnp.cos(theta))

    cos_t_w = jnp.cos(theta + omega)
    sin_t_w = jnp.sin(theta + omega)
    cos_O = jnp.cos(Omega)
    sin_O = jnp.sin(Omega)
    cos_i = jnp.cos(inc)
    sin_i = jnp.sin(inc)

    x = r_mag * (cos_t_w * cos_O - sin_t_w * cos_i * sin_O)
    y = r_mag * (cos_t_w * sin_O + sin_t_w * cos_i * cos_O)
    z = r_mag * sin_t_w * sin_i

    cos_t_w_gamma = jnp.cos(theta + omega - gamma)
    sin_t_w_gamma = jnp.sin(theta + omega - gamma)

    vx = v_mag * (-sin_t_w_gamma * cos_O - cos_t_w_gamma * cos_i * sin_O)
    vy = v_mag * (-sin_t_w_gamma * sin_O + cos_t_w_gamma * cos_i * cos_O)
    vz = v_mag * cos_t_w_gamma * sin_i

    return jnp.array([x, y, z], dtype=jnp.float64), jnp.array([vx, vy, vz], dtype=jnp.float64)

