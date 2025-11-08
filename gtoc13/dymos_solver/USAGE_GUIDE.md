# Using Solution Files as Initial Guess

## Overview

When you load a mission plan (.pln file), the system automatically looks for a solution file (.txt) with the same name and uses it as an initial guess for the optimization.

## Key Features

1. **Automatic Detection**: Just name files the same (e.g., `mission.pln` and `mission.txt`)
2. **Partial Matching**: The solution can have **fewer** bodies than the plan
3. **Body Sequence Validation**: The first N bodies in the plan must match the solution's bodies in order
4. **Flyby Time Updates**: Times from the solution file take precedence and update the plan

## How It Works

When loading a mission plan:
1. The system checks for a `.txt` file with the same base name
2. If found, it extracts the body sequence and flyby times from the solution
3. It validates that the first N bodies of the plan match the solution
4. For matched bodies, it uses the solution's states, velocities, and controls as initial guess
5. For unmatched bodies (if plan has more), it uses the default body-centered guess

## Example 1: Full Match (Solution has same bodies as plan)

**Files:**
- `test_mission.pln` - bodies: `[10, 9]`
- `test_mission.txt` - bodies: `[10, 9]`

**Usage:**
```python
from gtoc13.mission_plan import MissionPlan

plan = MissionPlan.load('solutions/test_mission.pln')
```

**Output:**
```
Found solution file: solutions/test_mission.txt
Loaded solution with 2 bodies matching first 2 of 2 plan bodies
Updated flyby times for bodies [10, 9]: [0.66, 2.39]
```

**Result:**
- ✓ All arcs use solution guess (states, velocities, controls)
- ✓ Flyby times updated from solution

## Example 2: Partial Match (Solution has fewer bodies than plan)

**Files:**
- `example_partial.pln` - bodies: `[10, 9, 8, 7]`
- `example_partial.txt` - bodies: `[10, 9, 8]` (only 3 bodies)

**Usage:**
```python
from gtoc13.mission_plan import MissionPlan

plan = MissionPlan.load('solutions/example_partial.pln')
```

**Output:**
```
Found solution file: solutions/example_partial.txt
Loaded solution with 3 bodies matching first 3 of 4 plan bodies
Updated flyby times for bodies [10, 9, 8]: [98.16, 141.33, 179.84]
Remaining 1 bodies will use default guess
```

**Result:**
- ✓ Arcs 0-2 (bodies 10, 9, 8) use solution guess
- ✓ Arc 3 (body 7) uses default body-centered guess
- ✓ First 3 flyby times updated from solution
- ✓ Last flyby time keeps plan value

## Example 3: Using with the dymos_solver Command Line

### Scenario A: Exact Match (Re-optimize existing trajectory)

```python
from gtoc13.mission_plan import MissionPlan

# Create a plan matching the solution
plan = MissionPlan(
    bodies=[10, 9, 8],
    flyby_times=[100.0, 140.0, 180.0],  # Initial guess (will be updated from solution)
    t0=0.0
)
plan.save('solutions/10_9_8.pln')

# Load it - will automatically use 10_9_8.txt as guess
plan = MissionPlan.load('solutions/10_9_8.pln')
```

Then run the solver:
```bash
python -m gtoc13.dymos_solver --plan solutions/10_9_8.pln --num-nodes 20
```

### Scenario B: Extend the Trajectory (Add more bodies)

```python
from gtoc13.mission_plan import MissionPlan

# Create a plan with additional bodies beyond the solution
plan = MissionPlan(
    bodies=[10, 9, 8, 7, 6],  # Add bodies 7 and 6
    flyby_times=[100.0, 140.0, 180.0, 190.0, 195.0],
    t0=0.0
)
plan.save('solutions/10_9_8_extended.pln')

# Copy the solution file to match
# cp solutions/10_9_8.txt solutions/10_9_8_extended.txt

# Load it - first 3 bodies use solution guess, bodies 7 & 6 use default
plan = MissionPlan.load('solutions/10_9_8_extended.pln')
```

Output:
```
Loaded solution with 3 bodies matching first 3 of 5 plan bodies
Updated flyby times for bodies [10, 9, 8]: [98.16, 141.33, 179.84]
Remaining 2 bodies will use default guess
```

## What Data Is Used From the Solution?

### For Matched Arcs (first N bodies):
- **Position (r)**: Initial and final positions from solution's PropagatedArc
- **Velocity (v)**: Initial and final velocities from solution's PropagatedArc
- **Control (u_n)**: Control vectors from solution's PropagatedArc
- **Flyby times**: Extracted from FlybyArc epochs and updated in the plan

### For Unmatched Arcs (additional bodies beyond solution):
- **Position/velocity**: Computed from body ephemeris at the flyby time specified in plan
- **Control**: Zero vector (ballistic trajectory guess)
- **Flyby times**: Original values from the mission plan

## Error Cases

### Case 1: Body Sequence Mismatch
```
Plan bodies:     [10, 8, 9]
Solution bodies: [10, 9, 8]
```
**Result:**
```
Warning: Solution body sequence [10, 9, 8] does not match
first 3 bodies of plan [10, 8, 9]. Ignoring solution.
```

### Case 2: Solution Has More Bodies Than Plan
```
Plan bodies:     [10, 9]
Solution bodies: [10, 9, 8]
```
**Result:**
```
Warning: Solution has more bodies (3) than plan (2). Ignoring solution.
```

### Case 3: Solution File Not Found
```
# Only plan file exists, no matching .txt file
```
**Result:**
- No warning printed
- Uses default body-centered guess for all arcs

## Tips and Best Practices

1. **File Naming**: Always use the same base name for `.pln` and `.txt` files
   - Good: `mission1.pln` + `mission1.txt`
   - Bad: `mission1.pln` + `mission_solution.txt`

2. **Extending Trajectories**: When adding bodies to an existing solution:
   - Keep the existing bodies in the same order at the start
   - Add new bodies at the end
   - Provide reasonable initial guess times for the new bodies

3. **Debugging**: If the solution isn't being loaded:
   - Check that body sequences match
   - Verify the `.txt` file is in the same directory as the `.pln` file
   - Look for warning messages during loading

4. **Performance**: Using solution guess can significantly improve:
   - Convergence speed
   - Solution quality
   - Optimizer stability

5. **Automatic IPOPT Warm-Start**: When a solution guess is detected, the system automatically enables IPOPT warm-start settings for faster convergence:
   - `warm_start_init_point = 'yes'`
   - `warm_start_bound_push = 1e-9`
   - `warm_start_bound_frac = 1e-9`
   - `warm_start_slack_bound_push = 1e-9`
   - `warm_start_slack_bound_frac = 1e-9`
   - `warm_start_mult_bound_push = 1e-9`
   - `mu_init = 1e-4`

## Command Line Interface

The dymos_solver CLI automatically uses solution guess when loading plans:

```bash
# Load a plan file (automatically loads matching .txt if present)
python -m gtoc13.dymos_solver --plan solutions/mission.pln --num-nodes 20

# The output will show if a guess was loaded:
# "Found solution file: solutions/mission.txt"
# "Enabling IPOPT warm-start settings for solution guess"
# "Using guess_solution for initial guess"
# "Using solution guess for first N of M arcs"
# "Set detailed initial guess with 20 time points per arc"

# After optimization, three files are created with the same base name:
# - dymos_solution_N.txt (solution file)
# - dymos_solution_N.png (trajectory plot)
# - dymos_solution_N.pln (mission plan with optimized times)
```

## Programmatic Usage

```python
from gtoc13.mission_plan import MissionPlan

# Load mission plan with automatic solution guess loading
plan = MissionPlan.load('solutions/my_mission.pln')

# Check if guess was loaded
if plan.guess_solution is not None:
    print("Solution guess loaded successfully")

# Solve the problem (uses guess automatically)
prob = plan.solve(num_nodes=20, run_driver=True)
```

## Implementation Details

The guess loading happens in two methods:

1. **`MissionPlan.load()`** (mission_plan.py:286-357):
   - Loads the `.pln` file
   - Checks for matching `.txt` file
   - Validates body sequence
   - Updates flyby times
   - Stores solution in `guess_solution` field

2. **`MissionPlan.solve()`** (mission_plan.py:579-693):
   - Checks if `guess_solution` is present
   - Extracts states and controls from PropagatedArcs
   - Uses solution data for first N arcs
   - Falls back to default for remaining arcs
   - Sets initial values in the dymos phase

The `guess_solution` field is marked with `exclude=True` in the Pydantic model, so it is not saved when you call `plan.save()`.