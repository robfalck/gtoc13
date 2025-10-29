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

    def get_state(self, epoch: float):
        """
        Get the Cartesian state (position and velocity) of the body at a given epoch.

        This method is JAX-compatible and can be used with jax.jit, jax.vmap, etc.

        Args:
            epoch: Time in seconds past t=0 (can be a JAX array)

        Returns:
            SpacecraftState with position (km) and velocity (km/s) vectors
            (JAX arrays if epoch is a JAX array)

        Examples:
            >>> # Single epoch
            >>> state = body.get_state(0.0)

            >>> # JAX array of epochs
            >>> import jax.numpy as jnp
            >>> epochs = jnp.linspace(0, YEAR, 100)
            >>> states = jax.vmap(body.get_state)(epochs)

            >>> # JIT-compiled version
            >>> get_state_jit = jax.jit(body.get_state)
            >>> state = get_state_jit(5.0 * YEAR)
        """
        from gtoc13 import elements_to_cartesian
        return elements_to_cartesian(self.elements, epoch)

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

                # Create orbital elements
                elements = OrbitalElements(
                    a=float(row['Semi-Major Axis (km)']),
                    e=float(row['Eccentricity ()']),
                    i=np.deg2rad(float(row['Inclination (deg)'])),
                    Omega=np.deg2rad(float(row['Longitude of the Ascending Node (deg)'])),
                    omega=np.deg2rad(float(row['Argument of Periapsis (deg)'])),
                    M0=np.deg2rad(float(row[m0_key])),
                    mu_body=mu,
                    radius=radius,
                    weight=float(row['Weight ()'])
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
