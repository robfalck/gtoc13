import csv
from pathlib import Path
import numpy as np

import jax.numpy as jnp
import pydantic
from pydantic import ConfigDict

from gtoc13.orbital_elements import OrbitalElements


class Body(pydantic.BaseModel):
    """
    Represents a celestial body in the GTOC13 problem.

    Attributes:
        name: Name of the body (e.g., "PlanetX", "Vulcan")
        id: Unique identifier for the body
        mu: Gravitational parameter GM (km^3/s^2)
        radius: Physical radius of the body (km)
        weight: Scientific weight for scoring
        elements: Orbital elements of the body
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)  # Allow OrbitalElements (NamedTuple)

    name: str
    id: int
    mu: float
    radius: float
    weight: float
    elements: OrbitalElements

    def get_state(self, epoch: float, time_units: str = 's', distance_units: str = 'km'):
        """
        Get the Cartesian state (position and velocity) of the body at a given epoch.

        This method is JAX-compatible and can be used with jax.jit, jax.vmap, etc.

        Args:
            epoch: Time past t=0 in the units specified by time_units (can be a JAX array)
            time_units: Units of the input epoch. Options:
                - 's' or 'seconds': epoch in seconds (default)
                - 'year' or 'years': epoch in years
                - 'TU': epoch in canonical time units
            distance_units: Units for the output position and velocity. Options:
                - 'km': position in km, velocity in km/time_units (default)
                - 'AU': position in AU, velocity in AU/time_units
                - 'DU': position in DU, velocity in DU/time_units (same as AU)

        Returns:
            CartesianState with position and velocity vectors in the specified units
            (JAX arrays if epoch is a JAX array)
            Note: velocity is returned in distance_units/time_units

        Examples:
            >>> # Single epoch in seconds (default)
            >>> state = body.get_state(0.0)  # velocity in km/s

            >>> # Epoch in years
            >>> state = body.get_state(5.0, time_units='year')  # velocity in km/year

            >>> # Get state in AU
            >>> state = body.get_state(0.0, distance_units='AU')  # velocity in AU/s

            >>> # Canonical units
            >>> state = body.get_state(1.0, time_units='TU', distance_units='DU')  # velocity in DU/TU

            >>> # JAX array of epochs
            >>> import jax.numpy as jnp
            >>> epochs = jnp.linspace(0, 1, 100)  # 0 to 1 year
            >>> states = jax.vmap(lambda t: body.get_state(t, time_units='year'))(epochs)

            >>> # JIT-compiled version
            >>> get_state_jit = jax.jit(lambda t: body.get_state(t, time_units='year'))
            >>> state = get_state_jit(5.0)
        """
        from gtoc13 import elements_to_cartesian
        from gtoc13.constants import YEAR, SPTU, KMPAU

        # Convert epoch to seconds
        if time_units in ('s', 'seconds'):
            epoch_seconds = epoch
            time_factor = 1.0  # velocity already in units/s
        elif time_units in ('year', 'years'):
            epoch_seconds = epoch * YEAR
            time_factor = YEAR  # convert velocity from units/s to units/year
        elif time_units == 'TU':
            epoch_seconds = epoch * SPTU
            time_factor = SPTU  # convert velocity from units/s to units/TU
        else:
            raise ValueError(f"Invalid time_units '{time_units}'. Must be one of: 's', 'year', 'TU'")

        # Get state in km and km/s
        state = elements_to_cartesian(self.elements, epoch_seconds)

        # Convert to requested distance and time units
        if distance_units == 'km':
            # Position already in km
            # Convert velocity from km/s to km/time_units
            v_converted = state.v * time_factor
            return state._replace(v=v_converted)
        elif distance_units in ('AU', 'DU'):
            # Convert position from km to AU/DU
            # Convert velocity from km/s to (AU or DU)/time_units
            r_converted = state.r / KMPAU
            v_converted = (state.v / KMPAU) * time_factor
            return state._replace(r=r_converted, v=v_converted)
        else:
            raise ValueError(f"Invalid distance_units '{distance_units}'. Must be one of: 'km', 'AU', 'DU'")

    def get_period(self, units: str = 'year') -> float:
        """
        Compute the orbital period of the body.

        The period is calculated using Kepler's third law:
        T = 2π√(a³/μ)

        Args:
            units: Units for the returned period. Options:
                - 's' or 'seconds': Period in seconds
                - 'year' or 'years': Period in years (default)
                - 'TU': Period in canonical time units

        Returns:
            Orbital period in the specified units

        Examples:
            >>> body = bodies_data[1]  # Vulcan
            >>> period_years = body.get_period('year')
            >>> period_seconds = body.get_period('s')
            >>> period_TU = body.get_period('TU')
        """
        from gtoc13.constants import MU_ALTAIRA, YEAR, SPTU

        # Calculate period in seconds using Kepler's third law
        a = self.elements.a
        period_seconds = 2.0 * np.pi * np.sqrt(a**3 / MU_ALTAIRA)

        # Convert to requested units
        units_lower = units.lower()
        if units_lower in ('s', 'seconds'):
            return period_seconds
        elif units_lower in ('year', 'years'):
            return period_seconds / YEAR
        elif units_lower == 'tu':
            return period_seconds / SPTU
        else:
            raise ValueError(f"Invalid units '{units}'. Must be one of: 's', 'year', 'TU'")

    def is_planet(self) -> bool:
        """Check if this body is a planet (has non-zero mass)"""
        return self.mu > 0.0

    def is_small_body(self) -> bool:
        """Check if this body is an asteroid or comet (negligible mass)"""
        return self.mu == 0.0

    def __repr__(self) -> str:
        return f"Body(id={self.id}, name='{self.name}', weight={self.weight})"

    def __str__(self) -> str:
        return f"{self.name} (ID: {self.id})"


def load_bodies_data() -> dict[int, Body]:
    """
    Load all bodies (planets, asteroids, comets) from CSV files.

    Returns:
        Dictionary mapping body ID to Body object
    """
    # Hardcode data directory to be in the same directory as this file
    data_dir = Path(__file__).parent / 'data'
    bodies = {}

    # Configuration for each body type
    body_configs = [
        {
            'filename': 'gtoc13_planets.csv',
            'id_key_options': ['#Planet ID', '# Planet ID', 'Planet ID'],
            'body_type': 'planet',
            'has_mass': True,
            'name_key': 'Name',
        },
        {
            'filename': 'gtoc13_asteroids.csv',
            'id_key_options': ['#Asteroid ID', '# Asteroid ID', 'Asteroid ID'],
            'body_type': 'asteroid',
            'has_mass': False,
            'name_prefix': 'Asteroid_',
        },
        {
            'filename': 'gtoc13_comets.csv',
            'id_key_options': ['#Comet ID', '# Comet ID', 'Comet ID'],
            'body_type': 'comet',
            'has_mass': False,
            'name_prefix': 'Comet_',
        },
    ]

    # Loop through each body type
    for config in body_configs:
        filepath = data_dir / config['filename']

        # Skip if file doesn't exist (only planets file is required)
        if not filepath.exists():
            continue

        with open(filepath, 'r', encoding='latin-1') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Find the ID column (handle various naming conventions)
                id_key = next((key for key in config['id_key_options'] if key in row), None)
                if id_key is None:
                    continue

                body_id = int(row[id_key])

                # Handle different column names for Mean Anomaly
                m0_key = 'Mean Anomaly at t=0 (deg)' if 'Mean Anomaly at t=0 (deg)' in row else 'Mean Anomaly at t=0'

                # Get mass and radius (0 for small bodies)
                if config['has_mass']:
                    mu = float(row['GM (km3/s2)'])
                    radius = float(row['Radius (km)'])
                    name = row[config['name_key']]
                else:
                    mu = 0.0
                    radius = 0.0
                    name = f"{config['name_prefix']}{body_id}"

                # Create orbital elements (only the 6 classical elements)
                elements = OrbitalElements(
                    a=float(row['Semi-Major Axis (km)']),
                    e=float(row['Eccentricity ()']),
                    i=np.deg2rad(float(row['Inclination (deg)'])),
                    Omega=np.deg2rad(float(row['Longitude of the Ascending Node (deg)'])),
                    omega=np.deg2rad(float(row['Argument of Periapsis (deg)'])),
                    M0=np.deg2rad(float(row[m0_key]))
                )

                # Create Body object
                body = Body(
                    name=name,
                    id=body_id,
                    mu=mu,
                    radius=radius,
                    weight=float(row['Weight ()']),
                    elements=elements
                )
                bodies[body_id] = body

    return bodies


bodies_data = load_bodies_data()
