import jax.numpy as jnp
import numpy as np
from functools import partial
from pathlib import Path
from scipy.optimize import brentq

from gtoc13 import (
    bodies_data, lambert_tof, lambert_v,
    GTOC13Solution, create_conic
)
from gtoc13.constants import KMPDU, SPTU, YPTU, YEAR

print("Solving Lambert's problem for multiple target times")
print("=" * 70)
print()

# Problem setup
r0 = jnp.array([-200.0, 50.0, 0.0]) # in DU
planet_x_id = 10  # Planet X

# Target times of flight (years)
dt_target = 5.0
dt_target_TU = dt_target / YPTU

print(f"Initial position: r0 = [-200, 50, 0] AU")
print(f"Target: Planet X (ID={planet_x_id})")
print(f"Target transfer times: {dt_target} years")
print()

# Define residual function for a single Lambert problem
def residual_fn(z, args):
    """
    Residual function: difference between computed TOF and target TOF.

    Args:
        z: Universal variable
        args: (r0, rf, dt_target, mu)

    Returns:
        Scalar residual: t_computed - dt_target
    """
    r0, rf, dt_target, mu = args
    t, A, y = lambert_tof(z, r0, rf, mu)
    return t - dt_target


rf = bodies_data[planet_x_id].get_state(dt_target).r / KMPDU
print(residual_fn(-0.3, (r0, rf, dt_target_TU, 1.0)))
print(residual_fn(-0.25, (r0, rf, dt_target_TU, 1.0)))

# solver = optx.Newton(rtol=1e-8, atol=1e-8)

# # Solve with bounded bisection
# solution = optx.root_find(
#     residual_fn,
#     solver,
#     y0=-0.265,
#     args=(r0, rf, dt_target_TU, 1.0),
#     max_steps=10000
# )

# Scipy Brent solver using functools.partial
print("\nSolving with scipy.optimize.brentq:")
print("-" * 70)

# Create partial function with fixed args
residual_partial = partial(residual_fn, args=(r0, rf, dt_target_TU, 1.0))

# Solve using Brent's method (robust hybrid root-finding)
z_solution = brentq(residual_partial, -0.3, -0.25, xtol=1e-10, rtol=1e-10)
print(f"Solution: z = {z_solution}")

# Verify the solution
residual_value = residual_partial(z_solution)
print(f"Residual at solution: {residual_value:.2e}")

# Compute the velocities
t, A, y = lambert_tof(z_solution, r0, rf, 1.0)
v1, v2 = lambert_v(A, y, r0, rf, 1.0)
print(f"\nTransfer time: {float(t) * YPTU:.4f} years (target: {dt_target} years)")
print(f"Initial velocity: v1 = {np.array(v1)} DU/TU")
print(f"Final velocity: v2 = {np.array(v2)} DU/TU")

# Create GTOCSolution and save to file
print("\nCreating GTOC13 solution...")
print("-" * 70)

# Convert positions and velocities from DU/TU to km and km/s
r0_km = np.array(r0) * KMPDU  # DU to km
rf_km = np.array(rf) * KMPDU  # DU to km
v1_km_s = np.array(v1) * KMPDU / SPTU  # DU/TU to km/s
v2_km_s = np.array(v2) * KMPDU / SPTU  # DU/TU to km/s

# Time in seconds
epoch_start = 0.0
epoch_end = dt_target * YEAR  # years to seconds

print(f"Start epoch: {epoch_start} s (t = 0 years)")
print(f"End epoch: {epoch_end} s (t = {dt_target} years)")
print(f"Start position: {r0_km} km")
print(f"End position: {rf_km} km")
print(f"Start velocity: {v1_km_s} km/s")
print(f"End velocity: {v2_km_s} km/s")

# Create a conic arc
arc = create_conic(
    epoch_start=epoch_start,
    epoch_end=epoch_end,
    position_start=tuple(r0_km),
    position_end=tuple(rf_km),
    velocity_start=tuple(v1_km_s),
    velocity_end=tuple(v2_km_s)
)

# Create solution
solution = GTOC13Solution(
    arcs=[arc],
    comments=[
        "Solution 0: Single Lambert transfer to Planet X",
        f"Departure: r0 = [-200, 50, 0] DU at t=0",
        f"Arrival: Planet X (ID={planet_x_id}) at t={dt_target} years",
        f"Transfer type: Conic arc (Lambert ballistic)",
        f"Universal variable: z = {z_solution}",
        f"Residual: {residual_value:.2e}"
    ]
)

# Save to file
output_dir = Path(__file__).parent.parent / 'solutions'
output_dir.mkdir(exist_ok=True)
output_file = output_dir / 'solution0.txt'
solution.write_solution_file(output_file, precision=12)

print(f"\nSolution saved to: {output_file}")
print(f"Number of state points: {len(solution.to_state_points())}")

