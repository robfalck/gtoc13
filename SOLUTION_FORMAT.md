# GTOC13 Solution Format

This document describes the Pydantic-based solution models for GTOC13 submissions.

## Overview

The solution module ([gtoc13/solution.py](gtoc13/solution.py)) provides Pydantic models that represent GTOC13 trajectory solutions according to the official submission format specification.

## Key Classes

### `StatePoint`
Represents a single row in the solution file with 12 fields:
- `body_id`: Body identifier (0 for heliocentric, >0 for flyby)
- `flag`: Arc type flag (0 or 1)
- `epoch`: Time in seconds past t=0
- `position`: 3D position vector [x, y, z] in km
- `velocity`: 3D velocity vector [vx, vy, vz] in km/s
- `control`: 3D control vector (arc-dependent)

### `FlybyArc`
Represents a gravity assist maneuver (2 state points):
- Always has `body_id > 0`
- Contains incoming and outgoing velocities
- Contains v∞ vectors for scoring
- `is_science` flag determines if flyby counts for scoring

### `ConicArc`
Represents ballistic Keplerian motion (2 state points):
- Always has `body_id = 0` and `flag = 0`
- Control vectors are always [0, 0, 0]
- Defined by start and end states

### `PropagatedArc`
Represents solar sail propulsion or numerical integration (≥2 state points):
- Always has `body_id = 0` and `flag = 1`
- Control vectors are unit normals to the sail
- Minimum time step of 60 seconds (except for control discontinuities)

### `GTOC13Solution`
Complete trajectory solution containing a sequence of arcs:
- List of `FlybyArc`, `ConicArc`, and `PropagatedArc` objects
- Optional comments for the file header
- Methods to read/write solution files

## Usage

### Creating a Solution

```python
from gtoc13 import (
    GTOC13Solution,
    create_conic,
    create_flyby,
    create_propagated
)

# Create individual arcs
conic = create_conic(
    epoch_start=0.0,
    epoch_end=1000.0,
    position_start=(1e8, 0, 0),
    position_end=(1.1e8, 1e7, 0),
    velocity_start=(0, 30, 0),
    velocity_end=(0, 29, 0)
)

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

propagated = create_propagated(
    epochs=[1000.0, 2000.0, 3000.0],
    positions=[(1.1e8, 1e7, 0), (1.2e8, 1.5e7, 0), (1.3e8, 2e7, 0)],
    velocities=[(5, 28, 1), (4, 27, 1), (3, 26, 1)],
    controls=[(0.707, 0.707, 0), (0.8, 0.6, 0), (0.9, 0.436, 0)]
)

# Create solution
solution = GTOC13Solution(
    arcs=[conic, flyby, propagated],
    comments=["My GTOC13 solution", "Team: Example"]
)
```

### Writing to File

```python
# Write solution to file
solution.write_solution_file("my_solution.txt", precision=8)
```

Output format:
```
# GTOC13 Solution File
# Format: body_id flag epoch x y z vx vy vz cx cy cz
# Units: epoch(s), position(km), velocity(km/s)
# My GTOC13 solution
# Team: Example
#
0 0 0.00000000 100000000.00000000 0.00000000 0.00000000 0.00000000 30.00000000 0.00000000 0.00000000 0.00000000 0.00000000
0 0 1000.00000000 110000000.00000000 10000000.00000000 0.00000000 0.00000000 29.00000000 0.00000000 0.00000000 0.00000000 0.00000000
2 1 1000.00000000 110000000.00000000 10000000.00000000 0.00000000 0.00000000 29.00000000 0.00000000 -5.00000000 2.00000000 0.00000000
2 1 1000.00000000 110000000.00000000 10000000.00000000 0.00000000 5.00000000 28.00000000 1.00000000 5.00000000 1.00000000 1.00000000
...
```

### Reading from File

```python
# Read solution from file
solution = GTOC13Solution.from_file("my_solution.txt")

# Access arcs
for i, arc in enumerate(solution.arcs):
    print(f"Arc {i}: {type(arc).__name__}")

# Get all state points
state_points = solution.to_state_points()
print(f"Total state points: {len(state_points)}")
```

## Validation

The Pydantic models automatically validate:
- Body ID and flag values are in valid ranges
- Time ordering (must be non-decreasing)
- Minimum time steps (60 seconds for propagated arcs)
- Arc structure requirements
- First arc must be heliocentric

## Example

See [examples/solution_example.py](examples/solution_example.py) for a complete working example.

## File Format Details

The solution file format follows the GTOC13 Solution File Format Specification:
- ASCII text file
- Lines starting with `#` or `!` are comments
- Each data row contains exactly 12 fields
- Fields can be separated by commas and/or whitespace
- Units: seconds, kilometers, km/s

## Arc Types

| Arc Type | body_id | flag | Control Vector |
|----------|---------|------|----------------|
| Conic | 0 | 0 | [0, 0, 0] |
| Propagated | 0 | 1 | Unit normal to sail |
| Flyby (non-science) | >0 | 0 | v∞ vector |
| Flyby (science) | >0 | 1 | v∞ vector |
