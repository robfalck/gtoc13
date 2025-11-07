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
from gtoc13.mission_plan import MissionPlan
from gtoc13.dymos_solver.dymos_solver import solve


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
"""
    )

    # Mission plan file or command-line arguments
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--plan', '-p',
        type=Path,
        help='Path to mission plan file (.pln)'
    )
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

    # Solver options
    parser.add_argument(
        '--num-nodes', '-n',
        type=int,
        default=20,
        help='Number of collocation nodes per arc (default: 20)'
    )

    args = parser.parse_args()

    # Load or create mission plan
    if args.plan:
        # Load from file
        if not args.plan.exists():
            print(f"Error: Mission plan file not found: {args.plan}", file=sys.stderr)
            sys.exit(1)

        print(f"Loading mission plan from {args.plan}")
        try:
            plan = MissionPlan.load(args.plan)
        except Exception as e:
            print(f"Error loading mission plan: {e}", file=sys.stderr)
            sys.exit(1)

    else:
        # Create from command-line arguments
        if not args.flyby_times:
            print("Error: --flyby-times is required when using --bodies", file=sys.stderr)
            sys.exit(1)

        if len(args.bodies) != len(args.flyby_times):
            print(f"Error: Number of bodies ({len(args.bodies)}) must match "
                  f"number of flyby times ({len(args.flyby_times)})", file=sys.stderr)
            sys.exit(1)

        print(f"Creating mission plan from command-line arguments")
        plan = MissionPlan(
            bodies=args.bodies,
            flyby_times=YEAR * np.array(args.flyby_times),
            t0=YEAR * args.t0
        )

    # Display mission plan
    print(f"\nMission Plan:")
    print(f"  Bodies: {plan.bodies}")
    print(f"  Flyby times: {plan.flyby_times} years")
    print(f"  Initial time: {plan.t0} years")
    print(f"  Number of nodes: {args.num_nodes}")
    print()

    # Compute dt from flyby_times
    # dt[0] is time from t0 to first flyby
    # dt[i] is time from flyby i-1 to flyby i (for i > 0)
    dt = [plan.flyby_times[0] - plan.t0]
    for i in range(1, len(plan.flyby_times)):
        dt.append(plan.flyby_times[i] - plan.flyby_times[i-1])

    # Solve the trajectory optimization problem
    print("Starting trajectory optimization...")
    try:
        prob = plan.solve(
            num_nodes=args.num_nodes
        )
        print("\nOptimization completed successfully!")

        # Extract and display optimized results
        t0_opt = prob.get_val('t0', units='gtoc_year')[0]
        dt_opt = prob.get_val('dt', units='gtoc_year')

        # Compute optimized flyby times
        flyby_times_opt = [t0_opt + dt_opt[0]]
        for i in range(1, len(dt_opt)):
            flyby_times_opt.append(flyby_times_opt[-1] + dt_opt[i])

        # Get v_infinity magnitudes at each flyby
        # v_inf is the relative velocity between spacecraft and body
        from gtoc13 import bodies_data
        v_body = prob.get_val('body_vel', units='km/s')  # Body velocities
        v_in = prob.get_val('final_states:v', units='km/s')  # Spacecraft velocity before flyby

        # Calculate v_inf magnitude for each flyby
        v_inf_mags = []
        for i in range(len(plan.bodies)):
            v_rel = v_in[i] - v_body[i]
            v_inf_mag = np.linalg.norm(v_rel)
            v_inf_mags.append(v_inf_mag)

        # Display optimized sequence
        print("\nOptimized Sequence:")
        print(f"{'body_id':<10} {'flyby_date (year)':<20} {'v_inf (km/s)':<15}")
        print("-" * 45)
        for body_id, t_flyby, v_inf in zip(plan.bodies, flyby_times_opt, v_inf_mags):
            print(f"{body_id:<10} {t_flyby:<20.6f} {v_inf:<15.6f}")

        # Display final energy
        E_end = prob.get_val('E_end')[0]
        print(f"\nFinal specific orbital energy: {E_end:.6f} km^2/s^2")

    except Exception as e:
        print(f"\nError during optimization: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
