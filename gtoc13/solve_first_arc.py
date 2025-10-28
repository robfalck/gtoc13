import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
from functools import partial
from pathlib import Path
from scipy.optimize import brentq
from diffrax import diffeqsolve, ODETerm, Dopri5, SaveAt, PIDController

from gtoc13 import (
    KMPDU, SPTU, YPTU, YEAR,
    bodies_data, lambert_tof, lambert_v,
    GTOC13Solution, create_conic,
    gtoc13_ballistic_ode
)

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

# Propagate the initial conditions forward to verify Lambert solution
print("\n" + "=" * 70)
print("Propagating initial conditions with diffrax (Keplerian dynamics)")
print("=" * 70)

# Initial state: position and velocity in km and km/s
y0 = jnp.concatenate([r0_km, v1_km_s])

# Set up the ODE term
term = ODETerm(gtoc13_ballistic_ode)

# Set up the solver
solver = Dopri5()

# Time span: 0 to dt_target years (in seconds)
t0 = 0.0
t1 = dt_target * YEAR

# Save only at final time
saveat = SaveAt(t1=True)

# Solve the ODE
print(f"Propagating from t={t0} s to t={t1} s ({dt_target} years)")
print(f"Initial state: r = {r0_km} km")
print(f"               v = {v1_km_s} km/s")

solution_ode = diffeqsolve(
    term,
    solver,
    t0=t0,
    t1=t1,
    dt0=60.0,  # Initial timestep of 60 seconds
    y0=y0,
    saveat=saveat,
    stepsize_controller=PIDController(rtol=1e-9, atol=1e-9),
)

# Extract final state
r_final_propagated = solution_ode.ys[0, :3]
v_final_propagated = solution_ode.ys[0, 3:]

print(f"\nPropagated final state:")
print(f"  Position: {r_final_propagated} km")
print(f"  Velocity: {v_final_propagated} km/s")

print(f"\nLambert solution final state:")
print(f"  Position: {rf_km} km")
print(f"  Velocity: {v2_km_s} km/s")

# Compute errors
r_error = r_final_propagated - rf_km
v_error = v_final_propagated - v2_km_s
r_error_mag = jnp.linalg.norm(r_error)
v_error_mag = jnp.linalg.norm(v_error)

print(f"\nErrors (Propagated - Lambert):")
print(f"  Position error: {r_error} km")
print(f"  Position error magnitude: {r_error_mag:.6e} km ({r_error_mag/KMPDU:.6e} AU)")
print(f"  Velocity error: {v_error} km/s")
print(f"  Velocity error magnitude: {v_error_mag:.6e} km/s")

# Also compare with actual Planet X position at t=5 years
rf_planetX = bodies_data[planet_x_id].get_state(dt_target).r
print(f"\nPlanet X actual position at t={dt_target} years:")
print(f"  Position: {rf_planetX} km")
print(f"  Error from propagated: {jnp.linalg.norm(r_final_propagated - rf_planetX):.6e} km")

# Use optimistix to optimize initial conditions
print("\n" + "=" * 70)
print("Optimizing initial conditions with optimistix")
print("=" * 70)
print("Varying: y0, z0, vx0")
print("Fixed: x0 = -200 DU, vy0, vz0 from Lambert solution")

# Define residual function for optimizer
# We'll vary [y0, z0, vx0] and compute position error at final time
def residual_for_optimizer(vars, args):
    """
    Residual function: difference between propagated and target position.

    Args:
        vars: [y0_DU, z0_DU, vx0_DU_TU] - variables to optimize
        args: (target_position_km,) - target position at t=5 years

    Returns:
        3D residual vector: propagated_position - target_position
    """
    y0_DU, z0_DU, vx0_DU_TU = vars
    target_position_km = args[0]

    # Fixed x0 from original problem
    x0_DU = -200.0

    # Use vy0, vz0 from Lambert solution (in DU/TU)
    vy0_DU_TU = -0.4959434  # From Lambert solution
    vz0_DU_TU = -0.24021977

    # Construct full initial state in km and km/s
    r0_km_opt = jnp.array([x0_DU, y0_DU, z0_DU]) * KMPDU
    v0_km_s_opt = jnp.array([vx0_DU_TU, vy0_DU_TU, vz0_DU_TU]) * KMPDU / SPTU

    y0_opt = jnp.concatenate([r0_km_opt, v0_km_s_opt])

    # Propagate forward
    term = ODETerm(gtoc13_ballistic_ode)
    solver = Dopri5()
    saveat = SaveAt(t1=True)

    sol = diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=dt_target * YEAR,
        dt0=60.0,
        y0=y0_opt,
        saveat=saveat,
        stepsize_controller=PIDController(rtol=1e-9, atol=1e-9),
    )

    r_final = sol.ys[0, :3]

    # Return position error
    return r_final - target_position_km

# Initial guess: use Lambert solution values
y0_initial = 50.0  # DU
z0_initial = 0.0   # DU
vx0_initial = 2.105799  # DU/TU (from Lambert)

# Target: Planet X position at t=5 years
target_pos = rf_planetX

print(f"\nInitial guess:")
print(f"  y0 = {y0_initial} DU")
print(f"  z0 = {z0_initial} DU")
print(f"  vx0 = {vx0_initial} DU/TU")
print(f"\nTarget position: {target_pos} km")

# Set up the solver - use BFGS which supports reverse-mode autodiff
optimizer = optx.BFGS(rtol=1e-8, atol=1e-8)

# Convert to scalar objective (sum of squares)
def scalar_objective(vars, args):
    residual = residual_for_optimizer(vars, args)
    return jnp.sum(residual**2)

# Solve
print("\nSolving with BFGS (gradient-based with reverse-mode autodiff)...")
print("This may take a minute as each iteration requires a full 5-year propagation...")
optimized_solution = optx.minimise(
    scalar_objective,
    optimizer,
    y0=jnp.array([y0_initial, z0_initial, vx0_initial]),
    args=(target_pos,),
    max_steps=500,
    throw=False
)

print(f"Result: {optimized_solution.result}")
print(f"Steps: {optimized_solution.stats}")

# Extract optimized values
y0_opt, z0_opt, vx0_opt = optimized_solution.value
print(f"\nOptimized initial conditions:")
print(f"  y0 = {y0_opt} DU (was {y0_initial})")
print(f"  z0 = {z0_opt} DU (was {z0_initial})")
print(f"  vx0 = {vx0_opt} DU/TU (was {vx0_initial})")

# Verify the solution
final_residual = residual_for_optimizer(optimized_solution.value, (target_pos,))
print(f"\nFinal residual: {final_residual} km")
print(f"Final position error: {jnp.linalg.norm(final_residual):.6e} km ({jnp.linalg.norm(final_residual)/KMPDU:.6e} AU)")

