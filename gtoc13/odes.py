
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


def modeq_ode(p: float, f: float, g: float, h: float, k: float, L: float,
              u_n: jnp.ndarray, mu: float, r0: float) -> Tuple[float, float, float, float, float, float, float, jnp.ndarray, jnp.ndarray]:
    """
    Modified Equinoctial Element (MEE) equations of motion with solar sail perturbations.

    The modified equinoctial elements avoid singularities for near-circular and near-equatorial
    orbits. They are defined as:
        p = a(1 - e²)      : semi-latus rectum
        f = e·cos(ω + Ω)   : eccentricity vector component 1
        g = e·sin(ω + Ω)   : eccentricity vector component 2
        h = tan(i/2)·cos(Ω): inclination vector component 1
        k = tan(i/2)·sin(Ω): inclination vector component 2
        L = Ω + ω + ν      : true longitude

    where:
        a = semi-major axis
        e = eccentricity
        i = inclination
        Ω = right ascension of ascending node (RAAN)
        ω = argument of periapsis
        ν = true anomaly

    Parameters
    ----------
    p : float
        Semi-latus rectum in km
    f : float
        Eccentricity vector x-component (dimensionless)
    g : float
        Eccentricity vector y-component (dimensionless)
    h : float
        Inclination vector x-component (dimensionless)
    k : float
        Inclination vector y-component (dimensionless)
    L : float
        True longitude in radians
    u_n : jnp.ndarray, shape (3,)
        Sail normal unit vector (dimensionless)
    mu : float
        Gravitational parameter in km³/s²
    r0 : float
        Reference distance for solar sail in km

    Returns
    -------
    tuple
        (pdot, fdot, gdot, hdot, kdot, Ldot, cos_alpha, r_vec, v_vec) where:
        - pdot : Rate of change of semi-latus rectum (km/s)
        - fdot : Rate of change of f (1/s)
        - gdot : Rate of change of g (1/s)
        - hdot : Rate of change of h (1/s)
        - kdot : Rate of change of k (1/s)
        - Ldot : Rate of change of true longitude (rad/s)
        - cos_alpha : Cosine of sail cone angle (for constraint enforcement)
        - r_vec : Cartesian position vector (km)
        - v_vec : Cartesian velocity vector (km/s)

    Notes
    -----
    The equations of motion are derived from Gauss's variational equations adapted
    to modified equinoctial elements. The perturbation acceleration from the solar
    sail is decomposed into radial, tangential, and normal components in the
    local vertical local horizontal (LVLH) frame.

    References
    ----------
    Betts, J. T. (2010). Practical methods for optimal control and estimation using
    nonlinear programming (2nd ed.). SIAM.

    Falck, R. D., & Dankanich, J. W. (2012). Optimization of low-thrust spiral
    trajectories by collocation. AIAA/AAS Astrodynamics Specialist Conference.
    """
    # Compute auxiliary variables
    s2 = 1.0 + h**2 + k**2
    w = 1.0 + f * jnp.cos(L) + g * jnp.sin(L)
    r = p / w  # radius magnitude

    # Compute position and velocity in inertial frame from MEE
    cos_L = jnp.cos(L)
    sin_L = jnp.sin(L)

    # Auxiliary variables
    alpha2 = h**2 - k**2
    s_sq = 1.0 + h**2 + k**2

    # Position vector (corrected formulation from Betts/Falck)
    r_x = (r / s_sq) * (cos_L + alpha2 * cos_L + 2.0 * h * k * sin_L)
    r_y = (r / s_sq) * (sin_L - alpha2 * sin_L + 2.0 * h * k * cos_L)
    r_z = (2.0 * r / s_sq) * (h * sin_L - k * cos_L)

    r_vec = jnp.array([r_x, r_y, r_z])

    # Velocity vector (corrected formulation from Betts/Falck)
    sqrt_mu_p = jnp.sqrt(mu / p)

    v_x = -(sqrt_mu_p / s_sq) * (sin_L + alpha2 * sin_L - 2.0 * h * k * cos_L + g - 2.0 * f * h * k + alpha2 * g)
    v_y = -(sqrt_mu_p / s_sq) * (-cos_L + alpha2 * cos_L + 2.0 * h * k * sin_L - f + 2.0 * g * h * k + alpha2 * f)
    v_z = (2.0 * sqrt_mu_p / s_sq) * (h * cos_L + k * sin_L + f * h + g * k)

    v_vec = jnp.array([v_x, v_y, v_z])

    # Compute solar sail acceleration
    a_sail, cos_alpha = solar_sail_acceleration(r_vec, u_n, r0)

    # Transform acceleration to RSW (radial, tangential, normal) frame
    # Radial unit vector
    u_r = r_vec / r

    # Normal unit vector (angular momentum direction)
    h_vec = jnp.cross(r_vec, v_vec)
    h_mag = jnp.linalg.norm(h_vec)
    u_h = h_vec / h_mag

    # Tangential unit vector (in-plane, perpendicular to radial)
    u_theta = jnp.cross(u_h, u_r)

    # Project acceleration onto RSW frame
    a_r = jnp.dot(a_sail, u_r)
    a_theta = jnp.dot(a_sail, u_theta)
    a_h = jnp.dot(a_sail, u_h)

    # Modified equinoctial element equations of motion
    # Reference: Betts (2010) "Practical Methods for Optimal Control..." or
    # Falck & Dankanich (2012) "Optimization of Low-Thrust Spiral Trajectories..."
    sqrt_p_mu = jnp.sqrt(p / mu)

    # Gauss variational equations in modified equinoctial elements
    pdot = (2.0 * p / w) * sqrt_p_mu * a_theta

    fdot = sqrt_p_mu * (a_r * sin_L + a_theta * ((w + 1.0) * cos_L + f) / w - a_h * (g / w) * (h * sin_L - k * cos_L))

    gdot = sqrt_p_mu * (-a_r * cos_L + a_theta * ((w + 1.0) * sin_L + g) / w + a_h * (f / w) * (h * sin_L - k * cos_L))

    hdot = sqrt_p_mu * (s2 / (2.0 * w)) * a_h * cos_L

    kdot = sqrt_p_mu * (s2 / (2.0 * w)) * a_h * sin_L

    Ldot = jnp.sqrt(mu * p) / (r**2) + (sqrt_p_mu / w) * a_h * (h * sin_L - k * cos_L)

    return pdot, fdot, gdot, hdot, kdot, Ldot, cos_alpha, r_vec, v_vec
