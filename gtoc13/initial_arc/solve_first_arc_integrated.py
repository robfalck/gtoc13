"""
Solve the initial arc problem using integration and optimization.

The spacecraft starts at x=-200 AU with velocity only in the x direction.
We integrate the ballistic ODE and optimize to match the final position
with a target body's position.
"""
import jax
import jax.numpy as jnp
from diffrax import diffeqsolve, ODETerm, Dopri5, SaveAt, PIDController
from optimistix import minimise, BFGS

from gtoc13.odes import ballistic_ode
from gtoc13.bodies import bodies_data
from gtoc13.constants import KMPAU, KMPDU, SPTU, YEAR, MU_ALTAIRA


def solve_first_arc(
    body_id: int,
    tf: float,
    time_units: str = 'TU',
    initial_vx_guess: float = None,
    desired_flyby_velocity: float = None,
    optimize_t0: bool = False,
    t0_guess: float = None
) -> dict:
    """
    Solve for the initial ballistic arc from x=-200 AU to a target body.

    The spacecraft starts at position [-200, 0, 0] AU with velocity [vx, 0, 0].
    We optimize vx (and optionally t0) such that after propagating for time tf,
    the spacecraft arrives at the target body's position (and optionally matches
    a desired flyby velocity).

    Args:
        body_id: ID of the target body
        tf: Final time in time_units
        time_units: Time units ('s', 'year', 'TU')
        initial_vx_guess: Initial guess for vx in km/s (if None, uses a heuristic)
        desired_flyby_velocity: Optional desired relative velocity magnitude at flyby in DU/TU.
                                If provided, tf becomes a design variable and optimization will
                                match both position and the magnitude of velocity relative to the target body.
        optimize_t0: If True, initial time t0 becomes a design variable
        t0_guess: Initial guess for t0 in time_units (only used if optimize_t0=True)

    Returns:
        dict with keys:
            - 'success': bool, whether optimization converged
            - 'vx_optimal': optimal initial x velocity in km/s
            - 't0_optimal': optimal initial time (if optimize_t0=True)
            - 'final_position': final spacecraft position [x, y, z] in DU
            - 'final_velocity': final spacecraft velocity [vx, vy, vz] in DU/TU
            - 'target_position': target body position [x, y, z] in DU
            - 'target_velocity': target body velocity [vx, vy, vz] in DU/TU
            - 'position_error': distance between final and target positions in DU
            - 'relative_velocity_magnitude': magnitude of v_spacecraft - v_target (if desired_flyby_velocity provided)
            - 'velocity_magnitude_error': error in relative velocity magnitude (if desired_flyby_velocity provided)
            - 'trajectory': full trajectory state history (if requested)
    """
    # Get target body
    if body_id not in bodies_data:
        raise ValueError(f"Body {body_id} not found in bodies_data")

    target_body = bodies_data[body_id]

    # Convert tf to canonical time units (TU) for integration
    if time_units.lower() in ('year', 'years'):
        tf_tu = tf * YEAR / SPTU
    elif time_units.lower() == 'tu':
        tf_tu = tf
    else:  # seconds
        tf_tu = tf / SPTU

    # Set up t0 guess
    if optimize_t0:
        if t0_guess is None:
            t0_guess_tu = 0.0
        else:
            # Convert t0_guess to TU
            if time_units.lower() in ('year', 'years'):
                t0_guess_tu = t0_guess * YEAR / SPTU
            elif time_units.lower() == 'tu':
                t0_guess_tu = t0_guess
            else:  # seconds
                t0_guess_tu = t0_guess / SPTU
    else:
        t0_guess_tu = 0.0

    # Canonical gravitational parameter (mu = 1 in canonical units)
    mu_canon = 1.0

    # Determine if we're matching velocity as well
    match_velocity = desired_flyby_velocity is not None

    # Define the objective function to minimize
    def objective(design_vars, args):
        """
        Objective function: minimize position (and optionally velocity) error.

        Args:
            design_vars: array with:
                - If match_velocity=True and optimize_t0=True: [vx, y0, z0, t0, tf]
                - If match_velocity=True: [vx, y0, z0, tf]
                - If optimize_t0=True: [vx, y0, z0, t0]
                - Otherwise: [vx, y0, z0]
            where:
                - vx: initial x-velocity in DU/TU
                - y0: initial y-position in DU
                - z0: initial z-position in DU
                - t0: initial time in TU (if optimize_t0=True)
                - tf: flight time in TU (if match_velocity=True)
            args: unused, required by optimistix interface

        Returns:
            scalar error (weighted sum of position and velocity errors)
        """
        # Extract design variables based on what we're optimizing
        idx = 0
        vx = design_vars[idx]; idx += 1
        y0 = design_vars[idx]; idx += 1
        z0 = design_vars[idx]; idx += 1

        if optimize_t0:
            t0 = design_vars[idx]; idx += 1
        else:
            t0 = 0.0

        if match_velocity:
            tf_current = design_vars[idx]; idx += 1
        else:
            tf_current = tf_tu

        # Get target body state at final time (t0 + tf_current)
        # We need to convert back to original time units for get_state
        t_final_tu = t0 + tf_current
        if time_units.lower() in ('year', 'years'):
            t_final = t_final_tu * SPTU / YEAR
        elif time_units.lower() == 'tu':
            t_final = t_final_tu
        else:  # seconds
            t_final = t_final_tu * SPTU

        target_state = target_body.get_state(t_final, time_units=time_units, distance_units='DU')
        r_target = target_state.r
        v_target = target_state.v  # In DU/TU

        # Initial position: x is fixed at -200 DU, y and z are free
        r0 = jnp.array([-200.0, y0, z0])

        # Initial velocity: only x component, y and z are zero
        v0 = jnp.array([vx, 0.0, 0.0])

        # Initial state
        y0_state = jnp.concatenate([r0, v0])

        # Integrate the ballistic ODE in canonical units
        term = ODETerm(ballistic_ode)
        solver = Dopri5()
        stepsize_controller = PIDController(rtol=1e-6, atol=1e-9)

        solution = diffeqsolve(
            term,
            solver,
            args=(mu_canon,),
            t0=t0,
            t1=t_final_tu,
            dt0=0.01 * tf_current,
            y0=y0_state,
            stepsize_controller=stepsize_controller,
            saveat=SaveAt(t1=True),  # Only save final state
            max_steps=10000  # Should be plenty in canonical units
        )

        # Extract final state in DU and DU/TU
        r_final = solution.ys[0][:3]
        v_final = solution.ys[0][3:]

        # Compute position error (distance to target)
        pos_error_vec = r_final - r_target
        pos_error = jnp.sum(pos_error_vec**2)  # Squared distance for smooth gradients

        if match_velocity:
            # Compute relative velocity magnitude error
            v_rel = v_final - v_target
            v_rel_mag = jnp.linalg.norm(v_rel)
            vel_mag_error = (v_rel_mag - desired_flyby_velocity)**2
            # Combine position and velocity magnitude errors
            error = pos_error + vel_mag_error
        else:
            error = pos_error

        return error

    # Initial guess for design variables
    if initial_vx_guess is None:
        # Heuristic: approximate velocity needed to cover ~200 DU in time tf_tu TU
        # For circular orbit at 1 DU: v = sqrt(mu/r) = sqrt(1/1) = 1 DU/TU
        # For larger radius, velocity is smaller
        initial_vx_guess = jnp.sqrt(mu_canon / 200.0)  # Roughly circular orbit velocity at 200 DU
    else:
        # Convert provided guess from km/s to DU/TU
        initial_vx_guess = initial_vx_guess * SPTU / KMPDU

    # Initial guess: build array based on what we're optimizing
    # Start with y0=0, z0=0 as initial guess
    design_vars_list = [initial_vx_guess, 0.0, 0.0]

    if optimize_t0:
        design_vars_list.append(t0_guess_tu)

    if match_velocity:
        design_vars_list.append(tf_tu)  # Use provided tf as initial guess

    design_vars_0 = jnp.array(design_vars_list)

    # Optimize using BFGS
    result = minimise(
        fn=objective,
        solver=BFGS(rtol=1e-9, atol=1e-9),
        y0=design_vars_0,
        args=None,  # No additional args needed
        max_steps=100
    )

    # Extract optimal design variables
    idx = 0
    vx_optimal_canon = result.value[idx]; idx += 1
    y0_optimal = result.value[idx]; idx += 1
    z0_optimal = result.value[idx]; idx += 1

    if optimize_t0:
        t0_optimal_tu = result.value[idx]; idx += 1
        # Convert t0 back to original time units
        if time_units.lower() in ('year', 'years'):
            t0_optimal = t0_optimal_tu * SPTU / YEAR
        elif time_units.lower() == 'tu':
            t0_optimal = t0_optimal_tu
        else:  # seconds
            t0_optimal = t0_optimal_tu * SPTU
    else:
        t0_optimal_tu = 0.0
        t0_optimal = 0.0

    if match_velocity:
        tf_optimal_tu = result.value[idx]; idx += 1
        # Convert tf back to original time units
        if time_units.lower() in ('year', 'years'):
            tf_optimal = tf_optimal_tu * SPTU / YEAR
        elif time_units.lower() == 'tu':
            tf_optimal = tf_optimal_tu
        else:  # seconds
            tf_optimal = tf_optimal_tu * SPTU
    else:
        tf_optimal_tu = tf_tu
        tf_optimal = tf

    # Convert to km/s for output
    vx_optimal_kms = vx_optimal_canon * KMPDU / SPTU

    # Get target body state at optimal final time
    t_final_tu = t0_optimal_tu + tf_optimal_tu
    if time_units.lower() in ('year', 'years'):
        t_final = t_final_tu * SPTU / YEAR
    elif time_units.lower() == 'tu':
        t_final = t_final_tu
    else:  # seconds
        t_final = t_final_tu * SPTU

    target_state_final = target_body.get_state(t_final, time_units=time_units, distance_units='DU')
    r_target = target_state_final.r
    v_target = target_state_final.v

    # Compute final trajectory with optimal design variables
    r0_optimal = jnp.array([-200.0, y0_optimal, z0_optimal])
    v0_optimal = jnp.array([vx_optimal_canon, 0.0, 0.0])
    y0_state_optimal = jnp.concatenate([r0_optimal, v0_optimal])

    term = ODETerm(ballistic_ode)
    solver = Dopri5()
    stepsize_controller = PIDController(rtol=1e-6, atol=1e-9)

    final_solution = diffeqsolve(
        term,
        solver,
        args=(mu_canon,),
        t0=t0_optimal_tu,
        t1=t_final_tu,
        dt0=0.01 * tf_optimal_tu,
        y0=y0_state_optimal,
        stepsize_controller=stepsize_controller,
        saveat=SaveAt(ts=jnp.linspace(t0_optimal_tu, t_final_tu, 100)),  # Save full trajectory
        max_steps=10000
    )

    r_final = final_solution.ys[-1, :3]  # In DU
    v_final = final_solution.ys[-1, 3:]  # In DU/TU
    position_error = jnp.linalg.norm(r_final - r_target)

    # Compute relative velocity magnitude
    v_rel = v_final - v_target
    v_rel_mag = jnp.linalg.norm(v_rel)

    if match_velocity:
        velocity_magnitude_error = jnp.abs(v_rel_mag - desired_flyby_velocity)
    else:
        velocity_magnitude_error = None

    from optimistix import RESULTS

    result_dict = {
        'success': result.result == RESULTS.successful,
        'vx_optimal_kms': float(vx_optimal_kms),
        'vx_optimal_canon': float(vx_optimal_canon),
        'r0_optimal': r0_optimal,
        'v0_optimal': v0_optimal,
        'y0_optimal': float(y0_optimal),
        'z0_optimal': float(z0_optimal),
        'final_position': r_final,
        'final_velocity': v_final,
        'target_position': r_target,
        'target_velocity': v_target,
        'position_error': float(position_error),
        'relative_velocity_magnitude': float(v_rel_mag),
        'trajectory': final_solution,
        'target_body_id': body_id,
        'target_body_name': target_body.name,
        'tf': tf,
        'time_units': time_units,
    }

    if optimize_t0:
        result_dict['t0_optimal'] = float(t0_optimal)
        result_dict['t0_optimal_tu'] = float(t0_optimal_tu)

    if match_velocity:
        result_dict['velocity_magnitude_error'] = float(velocity_magnitude_error)
        result_dict['desired_flyby_velocity'] = float(desired_flyby_velocity)
        result_dict['tf_optimal'] = float(tf_optimal)
        result_dict['tf_optimal_tu'] = float(tf_optimal_tu)

    return result_dict


if __name__ == "__main__":
    # Example 1: Basic position matching (no t0 optimization)
    print("=" * 70)
    print("Example 1: Solving initial arc to body 3 (position matching only)...")
    print("=" * 70)

    result = solve_first_arc(
        body_id=3,
        tf=1.0,  # 1 canonical time unit
        time_units='TU'
    )

    print(f"\nOptimization {'succeeded' if result['success'] else 'failed'}")
    print(f"Target body: {result['target_body_name']} (ID: {result['target_body_id']})")
    print(f"\nOptimal initial conditions:")
    print(f"  Position: {result['r0_optimal']} DU")
    print(f"  Velocity: [{result['vx_optimal_canon']:.6f}, 0.0, 0.0] DU/TU")
    print(f"            ({result['vx_optimal_kms']:.6f} km/s in x-direction)")
    print(f"\nFinal state:")
    print(f"  Position: {result['final_position']} DU")
    print(f"  Velocity: {result['final_velocity']} DU/TU")
    print(f"\nTarget state:")
    print(f"  Position: {result['target_position']} DU")
    print(f"  Velocity: {result['target_velocity']} DU/TU")
    print(f"\nErrors:")
    print(f"  Position error: {result['position_error']:.6e} DU")

    # Example 2: With desired relative velocity magnitude
    print("\n" + "=" * 70)
    print("Example 2: Solving with desired relative velocity magnitude...")
    print("=" * 70)

    # Compute current relative velocity magnitude from Example 1
    v_rel_current = jnp.linalg.norm(result['final_velocity'] - result['target_velocity'])
    print(f"Current relative velocity magnitude: {v_rel_current:.6f} DU/TU")

    # Set a desired relative velocity magnitude (e.g., 50% of current)
    desired_v_rel_mag = 0.5 * float(v_rel_current)
    tf_guess = 1.0

    result2 = solve_first_arc(
        body_id=3,
        tf=tf_guess,
        time_units='TU',
        desired_flyby_velocity=desired_v_rel_mag
    )

    print(f"\nOptimization {'succeeded' if result2['success'] else 'failed'}")
    print(f"\nOptimal flight time: {result2['tf_optimal']:.6f} TU (initial guess was {tf_guess:.6f} TU)")
    print(f"\nDesired relative velocity magnitude: {desired_v_rel_mag:.6f} DU/TU")
    print(f"Actual relative velocity magnitude:  {result2['relative_velocity_magnitude']:.6f} DU/TU")
    print(f"\nTarget velocity:     {result2['target_velocity']} DU/TU")
    print(f"Spacecraft velocity: {result2['final_velocity']} DU/TU")
    print(f"\nErrors:")
    print(f"  Position error:             {result2['position_error']:.6e} DU")
    print(f"  Velocity magnitude error:   {result2['velocity_magnitude_error']:.6e} DU/TU")
