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

    # Convenience functions
    create_flyby,
    create_conic,
    create_propagated,
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

    # Convenience functions
    "create_flyby",
    "create_conic",
    "create_propagated",

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
