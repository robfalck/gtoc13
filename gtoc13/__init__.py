# Configure JAX to use double precision (64-bit floats) throughout the package
import jax
jax.config.update("jax_enable_x64", True)

from .orbital_elements import OrbitalElements
from .cartesian_state import CartesianState

from .constants import (
    # Constants
    KMPAU,
    KMPDU,
    SPTU,
    YPTU,
    MU_ALTAIRA,
    DAY,
    YEAR,
    C_FLUX,
    R0,
    SAIL_AREA,
    SPACECRAFT_MASS,
)

from .astrodynamics import (
    # Functions
    solve_kepler,
    elements_to_cartesian,
    compute_v_infinity,
    patched_conic_flyby,
    seasonal_penalty,
    flyby_velocity_penalty,
    time_bonus,
    compute_score,
)

from .odes import (
    # ODE functions
    solar_sail_acceleration,
    ballistic_ode,
    solar_sail_ode,
)

from .solution import (
    # Solution models
    StatePoint,
    FlybyArc,
    ConicArc,
    PropagatedArc,
    GTOC13Solution,
)

from .lambert import (
    # Lambert solver
    lambert,
    lambert_universal_variables,
    lambert_tof,
    lambert_v,
    lambert_delta_v,
    get_body_state,
)

from .bodies import (
    # Body class
    Body,
    load_bodies_data,
    bodies_data
)

import openmdao.utils.units as om_units
import numpy as np

# Add our specific DU and TU to OpenMDAO's recognized units.
om_units.add_unit('DU', f'{KMPDU}*1000*m')
period = 2 * np.pi * np.sqrt(KMPDU**3 / MU_ALTAIRA)
om_units.add_unit('TU', f'{period}*s')

# Add GTOC13's year definition (365.25 days * 86400 s/day = 31557600 s)
# This differs from OpenMDAO's default 'year' which uses 31556925.99 s
from gtoc13.constants import YEAR as GTOC_YEAR
om_units.add_unit('gtoc_year', f'{GTOC_YEAR}*s')

# Alias for compatibility with existing code
AU = KMPAU

__all__ = [
    # Constants
    "AU",
    "KMPAU",
    "MU_ALTAIRA",
    "DAY",
    "YEAR",
    "C_FLUX",
    "R0",
    "SAIL_AREA",
    "SPACECRAFT_MASS",

    # Named tuples
    "OrbitalElements",
    "CartesianState",

    # Functions
    "solve_kepler",
    "elements_to_cartesian",
    "compute_v_infinity",
    "patched_conic_flyby",
    "seasonal_penalty",
    "flyby_velocity_penalty",
    "time_bonus",
    "compute_score",

    # ODE functions
    "solar_sail_acceleration",
    "ballistic_ode",
    "solar_sail_ode",

    # Solution models
    "StatePoint",
    "FlybyArc",
    "ConicArc",
    "PropagatedArc",
    "GTOC13Solution",
    "MissionPlan",

    # Lambert solver
    "lambert",
    "lambert_universal_variables",
    "lambert_residual",
    "lambert_tof",
    "lambert_v",
    "lambert_tof_vmap",
    "lambert_v_vmap",
    "lambert_delta_v",
    "get_body_state",

    # Bodies
    "Body",
    "load_bodies_data",
    "bodies_data"
]
