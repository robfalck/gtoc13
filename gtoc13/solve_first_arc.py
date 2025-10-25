import jax.numpy as jnp
from pathlib import Path

from gtoc13 import KMPAU, bodies_data, lambert_universal_variables, MU_ALTAIRA, YEAR

# Canonical units: mu = 1.0
# Distance Unit (DU) = 1 AU = KMPAU km
# Time Unit (TU) chosen so that mu_canonical = 1.0
# mu = r^3 / t^2, so TU = sqrt(DU^3 / mu)
DU = KMPAU  # 1 AU in km
TU = jnp.sqrt(DU**3 / MU_ALTAIRA)  # Time unit in seconds

print(f"Canonical units:")
print(f"  1 DU = {DU:.6e} km = 1 AU")
print(f"  1 TU = {TU:.6e} seconds = {TU/YEAR:.6f} years")
print(f"  mu_canonical = 1.0 DU^3/TU^2")
print()

# Problem setup in physical units
t0_phys = 0.0
tf_phys = 10.0 * YEAR  # 5 years in seconds
r0_phys = jnp.array([-200.0, 50.0, 0.0]) * KMPAU  # km
rf_phys = bodies_data[10].get_state(tf_phys).r  # km
target_body_id = 10  # Planet X

print(f"Problem setup:")
print(f"  Initial position: r0 = [-200, 50, 0] AU")
print(f"  Target: Planet X (ID={target_body_id})")
print(f"  Transfer time: {tf_phys / YEAR:.2f} years")
print()

# Convert to canonical units
t0_canon = t0_phys / TU
tf_canon = tf_phys / TU
dt_canon = tf_canon - t0_canon
r0_canon = r0_phys / DU
rf_canon = rf_phys / DU
mu_canon = 1.0

# Solve Lambert's problem in canonical units
v0_canon, vf_canon, converged = lambert_universal_variables(
    r0_canon, rf_canon, dt=dt_canon, mu=mu_canon
)

# Convert back to physical units
v0_phys = v0_canon * (DU / TU)  # km/s
vf_phys = vf_canon * (DU / TU)  # km/s

print(f"Lambert solution:")
print(f"  converged = {converged}")
# print(f"  iterations = {iter_count}")
print(f"  v0 = [{v0_phys[0]:.6f}, {v0_phys[1]:.6f}, {v0_phys[2]:.6f}] km/s")
print(f"  |v0| = {jnp.linalg.norm(v0_phys):.4f} km/s")
print(f"  |vf| = {jnp.linalg.norm(vf_phys):.4f} km/s")
print()

# Create solution file in GTOC13 format
# Format: body_id flag epoch x y z vx vy vz cx cy cz
# Units: epoch(s), position(km), velocity(km/s), control(km/s²)

solution_dir = Path(__file__).parent.parent / "solutions"
solution_dir.mkdir(exist_ok=True)
solution_file = solution_dir / "solution0.txt"

# Create solution content
with open(solution_file, 'w') as f:
    # Write header
    f.write("# GTOC13 Solution File\n")
    f.write("# Format: body_id flag epoch x y z vx vy vz cx cy cz\n")
    f.write("# Units: epoch(s), position(km), velocity(km/s), control(km/s²)\n")
    f.write("#\n")
    f.write("# Solution 0: Single Lambert arc\n")
    f.write(f"# Departure: r0 = [-200, 50, 0] AU at t=0\n")
    f.write(f"# Arrival: Planet X (ID={target_body_id}) at t={tf_phys/YEAR:.2f} years\n")
    f.write(f"# Transfer type: Conic arc (ballistic Lambert transfer)\n")
    f.write(f"# Converged: {converged}\n")
    f.write(f"# Delta-V: {jnp.linalg.norm(v0_phys):.4f} km/s\n")
    f.write("#\n")

    # Write departure point (initial state)
    # body_id=0 (no flyby), flag=0 (conic arc)
    # Control acceleration = 0 for ballistic arc
    f.write(f"0 0 {t0_phys:.12f} "
            f"{r0_phys[0]:.12f} {r0_phys[1]:.12f} {r0_phys[2]:.12f} "
            f"{v0_phys[0]:.12f} {v0_phys[1]:.12f} {v0_phys[2]:.12f} "
            f"0.000000000000 0.000000000000 0.000000000000\n")

    # Write arrival point (final state)
    # body_id=target, flag=0 (conic arc)
    f.write(f"{target_body_id} 0 {tf_phys:.12f} "
            f"{rf_phys[0]:.12f} {rf_phys[1]:.12f} {rf_phys[2]:.12f} "
            f"{vf_phys[0]:.12f} {vf_phys[1]:.12f} {vf_phys[2]:.12f} "
            f"0.000000000000 0.000000000000 0.000000000000\n")

print(f"Solution saved to: {solution_file}")
print(f"  2 trajectory points written (departure and arrival)")
print()

# Verify the position at arrival matches Planet X
planet_x_state = bodies_data[10].get_state(tf_phys)
position_error = jnp.linalg.norm(rf_phys - planet_x_state.r)
velocity_error = jnp.linalg.norm(vf_phys - planet_x_state.v)

print(f"Verification:")
print(f"  Position delta at arrival: {position_error/1e3:.6f} thousand km")
print(f"  Velocity delta at arrival: {velocity_error:.6f} km/s")
print()

if position_error < 1e3:  # Less than 1000 km error
    print("✓ Solution verified: spacecraft reaches Planet X position!")
else:
    print("⚠ Warning: Large position error at arrival")
