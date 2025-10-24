# GTOC13 - Global Trajectory Optimization Competition Round 13

A trajectory optimization framework for GTOC13 using JAX for high-performance computation.

## Features

- **Orbital Mechanics**: JAX-based orbital propagation with Kepler solvers
- **Solar Sail Dynamics**: Solar sail acceleration modeling
- **Scoring Functions**: Complete GTOC13 scoring implementation
- **3D Visualization**: Interactive animated visualization of orbits
- **Solution Format**: Pydantic models for GTOC13 submission files

## Installation

Install the package in development mode:

```bash
python -m pip install -e .
```

## Quick Start

### Orbital Mechanics

```python
from gtoc13 import OrbitalElements, elements_to_cartesian, AU
import jax.numpy as jnp

# Define orbital elements
planet = OrbitalElements(
    a=13.0 * AU,
    e=0.05,
    i=jnp.deg2rad(2.0),
    Omega=jnp.deg2rad(45.0),
    omega=jnp.deg2rad(90.0),
    M0=jnp.deg2rad(0.0),
    mu_body=1e8,
    radius=70000.0,
    weight=10.0
)

# Convert to Cartesian state at time t
state = elements_to_cartesian(planet, t=0.0)
print(f"Position: {state.r}")
print(f"Velocity: {state.v}")
```

### 3D Orbit Visualization

Launch the interactive 3D orbit animation:

```bash
python -m gtoc13.anim
```

**Controls:**
- **Mouse wheel**: Zoom in/out
- **Mouse drag**: Rotate 3D view
- **+ or =**: Speed up time (doubles rate, max 64x)
- **- or _**: Slow down time (halves rate, min 0.125x)
- **0**: Reset time rate to 1x

The animation displays:
- Current epoch (in seconds)
- Time in years and days
- Current playback rate
- Planets (blue circles)
- Asteroids (green dots)
- Comets (red triangles)
- Altaira star (yellow star at origin)

### Creating Solutions

```python
from gtoc13 import GTOC13Solution, create_conic, create_flyby, create_propagated

# Create a conic arc (ballistic coast)
conic = create_conic(
    epoch_start=0.0,
    epoch_end=1000.0,
    position_start=(1e8, 0, 0),
    position_end=(1.1e8, 1e7, 0),
    velocity_start=(0, 30, 0),
    velocity_end=(0, 29, 0)
)

# Create a flyby arc
flyby = create_flyby(
    body_id=2,
    epoch=1000.0,
    position=(1.1e8, 1e7, 0),
    velocity_in=(0, 29, 0),
    velocity_out=(5, 28, 1),
    v_inf_in=(-5, 2, 0),
    v_inf_out=(5, 1, 1),
    is_science=True
)

# Create a propagated arc (solar sail)
propagated = create_propagated(
    epochs=[1000.0, 2000.0, 3000.0],
    positions=[(1.1e8, 1e7, 0), (1.2e8, 1.5e7, 0), (1.3e8, 2e7, 0)],
    velocities=[(5, 28, 1), (4, 27, 1), (3, 26, 1)],
    controls=[(0.707, 0.707, 0), (0.8, 0.6, 0), (0.9, 0.436, 0)]
)

# Build complete solution
solution = GTOC13Solution(
    arcs=[conic, flyby, propagated],
    comments=["My GTOC13 solution", "Team: Example"]
)

# Write to submission file
solution.write_solution_file("submission.txt", precision=8)

# Read solution back
loaded = GTOC13Solution.from_file("submission.txt")
```

See [SOLUTION_FORMAT.md](SOLUTION_FORMAT.md) for detailed documentation on solution models.

## Project Structure

```
gtoc13/
├── __init__.py           # Main package exports
├── jax.py                # Orbital mechanics and scoring functions
├── solution.py           # Pydantic models for solutions
├── anim.py               # 3D visualization
└── data/                 # Orbital data
    ├── gtoc13_planets.csv
    ├── gtoc13_asteroids.csv
    └── gtoc13_comets.csv
```

## Constants

Available physical constants:
- `AU`: Astronomical unit (km)
- `MU_ALTAIRA`: Gravitational parameter of Altaira (km³/s²)
- `DAY`: Seconds in a day
- `YEAR`: Seconds in a year
- `C_FLUX`: Solar radiation pressure coefficient
- `SAIL_AREA`: Solar sail area (m²)
- `SPACECRAFT_MASS`: Spacecraft mass (kg)

## Key Functions

### Orbital Mechanics
- `solve_kepler(M, e)`: Solve Kepler's equation for eccentric anomaly
- `elements_to_cartesian(elements, t)`: Convert orbital elements to Cartesian state
- `solar_sail_acceleration(r, u_n)`: Calculate solar sail acceleration
- `keplerian_derivatives(t, y, args)`: Keplerian motion derivatives
- `solar_sail_derivatives(t, y, args)`: Solar sail motion derivatives

### Flyby & Scoring
- `compute_v_infinity(v_sc, v_body)`: Compute hyperbolic excess velocity
- `patched_conic_flyby(...)`: Validate flyby using patched conic approximation
- `seasonal_penalty(r_hat_current, r_hat_previous)`: Compute seasonal penalty
- `flyby_velocity_penalty(v_infinity)`: Compute velocity penalty
- `time_bonus(t_submission_days)`: Compute time bonus
- `compute_score(flybys, body_weights, grand_tour, submission_time)`: Total score

## Examples

See the [examples/](examples/) directory for complete working examples:
- `solution_example.py`: Creating and writing solution files

## Development

Run tests:
```bash
pytest
```

Format code:
```bash
black gtoc13/
isort gtoc13/
```

Type checking:
```bash
mypy gtoc13/
```

## License

Apache License 2.0 - See [LICENSE](LICENSE) file for details.

## References

- GTOC13 Problem Statement
- GTOC13 Solution File Format Specification (see `gtoc13_submission_format.pdf`)

## Contributing

This is a competition repository. Please follow the competition rules and guidelines.
