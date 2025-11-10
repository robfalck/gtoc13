"""
Command-line interface for the GTOC13 Dymos solver.

Usage:
    # Using a mission plan file
    python -m gtoc13.dymos_solver --plan mission0.pln

    # Using command-line arguments
    python -m gtoc13.dymos_solver --bodies 10 9 --flyby-times 20.0 40.0 --t0 0.0

    # With additional options
    python -m gtoc13.dymos_solver --plan mission0.pln --num-nodes 30
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from gtoc13.constants import YEAR
from gtoc13.dymos_solver.dymos_solver2 import (
    get_dymos_serial_solver_problem,
    set_initial_guesses,
    create_solution
)


def main():
    """Main entry point for the dymos solver CLI."""
    parser = argparse.ArgumentParser(
        description="GTOC13 Dymos Solver - Trajectory optimization using Dymos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Solve using a mission plan file
  python -m gtoc13.dymos_solver --plan mission0.pln

  # Solve with command-line arguments
  python -m gtoc13.dymos_solver --bodies 10 9 --flyby-times 20.0 40.0

  # Specify initial time and number of nodes
  python -m gtoc13.dymos_solver --bodies 10 --flyby-times 20.0 --t0 5.0 --num-nodes 30

  # Specify control scheme for each arc
  python -m gtoc13.dymos_solver --bodies 10 9 --flyby-times 20.0 40.0 --control uopt pfix
"""
    )

    # Mission plan file or command-line arguments
    input_group = parser.add_mutually_exclusive_group(required=True)

    input_group.add_argument(
        '--bodies', '-b',
        type=int,
        nargs='+',
        metavar='BODY_ID',
        help='Body IDs to visit (space-separated integers)'
    )

    # Additional arguments for command-line mode
    parser.add_argument(
        '--flyby-times', '-f',
        type=float,
        nargs='+',
        metavar='TIME',
        help='Flyby times in years (space-separated floats, required with --bodies)'
    )
    parser.add_argument(
        '--t0',
        type=float,
        default=0.0,
        help='Initial time in years (default: 0.0)'
    )
    parser.add_argument(
        '--control', '-c',
        type=int,
        nargs='+',
        metavar='CONTROL_FLAG',
        help='Control flag for each arc: 0, 1 (space-separated, must match number of bodies)'
    )

    # Solver options
    parser.add_argument(
        '--num-nodes', '-n',
        type=int,
        nargs='+',
        default=20,
        help='Number of collocation nodes per arc (default: 20)'
    )

    # Solver options
    parser.add_argument(
        '--no-opt',
        action='store_true',
        help='If given, just run through the model once without optimization.'
    )

    args = parser.parse_args()

    N = len(args.bodies)

    # Create from command-line arguments
    if not args.flyby_times:
        print("Error: --flyby-times is required when using --bodies", file=sys.stderr)
        sys.exit(1)

    if len(args.bodies) != len(args.flyby_times):
        print(f"Error: Number of bodies ({len(args.bodies)}) must match "
                f"number of flyby times ({len(args.flyby_times)})", file=sys.stderr)
        sys.exit(1)

    # Validate control argument if provided
    if args.control is not None:
        if len(args.control) != len(args.bodies) and len(args.control) != 1:
            print(f"Error: Number of control flags ({len(args.control)}) must match "
                    f"number of bodies ({len(args.bodies)}) if multiple are given", file=sys.stderr)
            sys.exit(1)

        # Validate each control value
        valid_controls = {0, 1}
        for i, ctrl in enumerate(args.control):
            if ctrl not in valid_controls:
                print(f"Error: Invalid control scheme '{ctrl}' at position {i}. "
                        f"Must be one of: {', '.join(sorted(valid_controls))}", file=sys.stderr)
                sys.exit(1)

    if not (isinstance(args.num_nodes, int) or len(args.bodies)):
        print(f"Error: Number of nodes in each arc must be a scalar or must match the number of flyby bodies. ({len(args.bodies)})",
              file=sys.stderr)
        sys.exit(1)

    if isinstance(args.num_nodes, int):
        num_nodes = N * [args.num_nodes]
    else:
        num_nodes = args.num_nodes

    if args.control is None:
        controls = N * [0]
    elif len(args.control) == 1:
        controls = N * [args.controls]
    else:
        controls = args.controls
    
    t0 = np.array(args.t0).reshape((1,))
    dt = np.diff(np.concatenate((t0, args.flyby_times)))

    prob = get_dymos_serial_solver_problem(bodies=args.bodies,
                                           num_nodes=num_nodes,
                                           controls=controls,
                                           warm_start=False,
                                           default_opt_prob=True)
    prob.setup()
    
    set_initial_guesses(prob, bodies=args.bodies, flyby_times=args.flyby_times,
                        t0=args.t0, controls=controls)
    
    # prob.run_model()
    # prob.model.list_vars(print_arrays=True, units=True)
    result = prob.run_driver()

    if result.success:
        sol, sol_file = create_solution(prob, args.bodies)

    # # Display mission plan
    # print(f"\nMission Plan:")
    # print(f"  Bodies: {plan.bodies}")
    # print(f"  Flyby times: {plan.flyby_times} years")
    # print(f"  Initial time: {plan.t0} years")
    # print(f"  Number of nodes: {args.num_nodes}")
    # print()

    # # Compute dt from flyby_times
    # # dt[0] is time from t0 to first flyby
    # # dt[i] is time from flyby i-1 to flyby i (for i > 0)
    # dt = [plan.flyby_times[0] - plan.t0]
    # for i in range(1, len(plan.flyby_times)):
    #     dt.append(plan.flyby_times[i] - plan.flyby_times[i-1])

    # # Solve the trajectory optimization problem
    # print("Starting trajectory optimization...")
    # try:
    #     prob = plan.solve(
    #         num_nodes=args.num_nodes, run_driver=not args.no_opt
    #     )
    #     print("\nOptimization completed successfully!")

    #     # Extract and display optimized results
    #     t0_opt = prob.get_val('t0', units='gtoc_year')[0]
    #     dt_opt = prob.get_val('dt', units='gtoc_year')

    #     # Compute optimized flyby times
    #     flyby_times_opt = [t0_opt + dt_opt[0]]
    #     for i in range(1, len(dt_opt)):
    #         flyby_times_opt.append(flyby_times_opt[-1] + dt_opt[i])

    #     # Get v_infinity magnitudes at each flyby
    #     # v_inf is the relative velocity between spacecraft and body
    #     from gtoc13 import bodies_data
    #     v_body = prob.get_val('body_vel', units='km/s')  # Body velocities
    #     v_in = prob.get_val('final_states:v', units='km/s')  # Spacecraft velocity before flyby

    #     # Calculate v_inf magnitude for each flyby
    #     v_inf_mags = []
    #     for i in range(len(plan.bodies)):
    #         v_rel = v_in[i] - v_body[i]
    #         v_inf_mag = np.linalg.norm(v_rel)
    #         v_inf_mags.append(v_inf_mag)

    #     # Display optimized sequence
    #     print("\nOptimized Sequence:")
    #     print(f"{'body_id':<10} {'flyby_date (year)':<20} {'v_inf (km/s)':<15}")
    #     print("-" * 45)
    #     for body_id, t_flyby, v_inf in zip(plan.bodies, flyby_times_opt, v_inf_mags):
    #         print(f"{body_id:<10} {t_flyby:<20.6f} {v_inf:<15.6f}")

    #     # Display final energy
    #     E_end = prob.get_val('E_end')[0]
    #     print(f"\nFinal specific orbital energy: {E_end:.6f} km^2/s^2")

    #     # Create solution file and plot
    #     print("\nCreating solution file...")
    #     solution, solution_file = create_solution(prob, plan.bodies)
    #     print("Solution file and plot created successfully.")

    #     # Save the mission plan with the same base name
    #     plan_file = Path(solution_file).with_suffix('.pln')

    #     # Update the plan with optimized flyby times
    #     plan.flyby_times = flyby_times_opt
    #     plan.t0 = t0_opt

    #     # Save the updated plan
    #     plan.save(plan_file)
    #     print(f"Mission plan saved to {plan_file}")

    # except Exception as e:
    #     print(f"\nError during optimization: {e}", file=sys.stderr)
    #     import traceback
    #     traceback.print_exc()
    #     sys.exit(1)


if __name__ == '__main__':
    main()
