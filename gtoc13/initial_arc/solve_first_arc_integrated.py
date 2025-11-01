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
    initial_vx_guess: float = None
) -> dict:
    """
    Solve for the initial ballistic arc from x=-200 AU to a target body.

    The spacecraft starts at position [-200, 0, 0] AU with velocity [vx, 0, 0].
    We optimize vx such that after propagating for time tf, the spacecraft
    arrives at the target body's position.

    Args:
        body_id: ID of the target body
        tf: Final time in time_units
        time_units: Time units ('s', 'year', 'TU')
        initial_vx_guess: Initial guess for vx in km/s (if None, uses a heuristic)

    Returns:
        dict with keys:
            - 'success': bool, whether optimization converged
            - 'vx_optimal': optimal initial x velocity in km/s
            - 'final_position': final spacecraft position [x, y, z] in AU
            - 'target_position': target body position [x, y, z] in AU
            - 'position_error': distance between final and target positions in AU
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

    # Get target body position at tf in canonical units (DU)
    target_state = target_body.get_state(tf, time_units=time_units, distance_units='DU')
    r_target = target_state.r

    # Canonical gravitational parameter (mu = 1 in canonical units)
    mu_canon = 1.0

    # Define the objective function to minimize
    def objective(design_vars, args):
        """
        Objective function: minimize distance to target body.

        Args:
            design_vars: array [vx, y0, z0] where:
                - vx: initial x-velocity in DU/TU
                - y0: initial y-position in DU
                - z0: initial z-position in DU
            args: unused, required by optimistix interface

        Returns:
            scalar error (distance squared to target)
        """
        # Extract design variables
        vx, y0, z0 = design_vars[0], design_vars[1], design_vars[2]

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
            t0=0.0,
            t1=tf_tu,
            dt0=0.01 * tf_tu,
            y0=y0_state,
            stepsize_controller=stepsize_controller,
            saveat=SaveAt(t1=True),  # Only save final state
            max_steps=10000  # Should be plenty in canonical units
        )

        # Extract final position in DU
        r_final = solution.ys[0][:3]

        # Compute error (distance to target)
        error_vec = r_final - r_target
        error = jnp.sum(error_vec**2)  # Squared distance for smooth gradients

        return error

    # Initial guess for design variables [vx, y0, z0]
    if initial_vx_guess is None:
        # Heuristic: approximate velocity needed to cover ~200 DU in time tf_tu TU
        # For circular orbit at 1 DU: v = sqrt(mu/r) = sqrt(1/1) = 1 DU/TU
        # For larger radius, velocity is smaller
        initial_vx_guess = jnp.sqrt(mu_canon / 200.0)  # Roughly circular orbit velocity at 200 DU
    else:
        # Convert provided guess from km/s to DU/TU
        initial_vx_guess = initial_vx_guess * SPTU / KMPDU

    # Initial guess: [vx, y0, z0]
    # Start with y0=0, z0=0 as initial guess
    design_vars_0 = jnp.array([initial_vx_guess, 0.0, 0.0])

    # Optimize using BFGS
    result = minimise(
        fn=objective,
        solver=BFGS(rtol=1e-9, atol=1e-9),
        y0=design_vars_0,
        args=None,  # No additional args needed
        max_steps=100
    )

    # Extract optimal design variables
    vx_optimal_canon = result.value[0]
    y0_optimal = result.value[1]
    z0_optimal = result.value[2]

    # Convert to km/s for output
    vx_optimal_kms = vx_optimal_canon * KMPDU / SPTU

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
        t0=0.0,
        t1=tf_tu,
        dt0=0.01 * tf_tu,
        y0=y0_state_optimal,
        stepsize_controller=stepsize_controller,
        saveat=SaveAt(ts=jnp.linspace(0, tf_tu, 100)),  # Save full trajectory
        max_steps=10000
    )

    r_final = final_solution.ys[-1, :3]  # In DU
    position_error = jnp.linalg.norm(r_final - r_target)

    from optimistix import RESULTS

    return {
        'success': result.result == RESULTS.successful,
        'vx_optimal_kms': float(vx_optimal_kms),
        'vx_optimal_canon': float(vx_optimal_canon),
        'r0_optimal': r0_optimal,
        'y0_optimal': float(y0_optimal),
        'z0_optimal': float(z0_optimal),
        'final_position': r_final,
        'target_position': r_target,
        'position_error': float(position_error),
        'trajectory': final_solution,
        'target_body_id': body_id,
        'target_body_name': target_body.name,
        'tf': tf,
        'time_units': time_units
    }


if __name__ == "__main__":
    # Example: solve for transfer to body 3 (a planet)
    print("Solving initial arc to body 3...")

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
    print(f"\nFinal position: {result['final_position']} DU")
    print(f"Target position: {result['target_position']} DU")
    print(f"Position error: {result['position_error']:.6e} DU")
