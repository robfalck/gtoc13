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
from gtoc13.dymos_solver.dymos_solver import (
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
  python -m gtoc13.dymos_solver --bodies 10 9 --flyby-times 20.0 40.0 --control 0 1

  # Load a solution file as an initial guess
  python -m gtoc13.dymos_solver --bodies 9 8 7  --flyby-times 20 40 80  --max-time 150 --controls 0 0 0 --load solutions/dymos_solution_32.txt
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
        '--controls', '-c',
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

    parser.add_argument(
        '--max-time',
        type=float,
        default=199.999,
        help='Maximum allowable final time in years. (default: 199.999)'
    )

    # Solver options
    parser.add_argument(
        '--load', '-l',
        type=str,
        nargs='+',
        default=None,
        metavar='LOAD_FILES',
        help='File(s) from which to load the solution. If multiple, they are concatenated.'
    )

    parser.add_argument(
        '--name',
        type=str,
        default=None,
        help='Root name of solution file to be saved. This will overwrite existing files of the same name!'
    )

    parser.add_argument(
        '--obj',
        type=str,
        default='J',
        help='Objective to be used. Either "J" to maximize GTOC objective or "E" to minimize final energy'
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
    if args.controls is not None:
        if len(args.controls) != len(args.bodies) and len(args.controls) != 1:
            print(f"Error: Number of control flags ({len(args.controls)}) must match "
                    f"number of bodies ({len(args.bodies)}) if multiple are given", file=sys.stderr)
            sys.exit(1)

        # Validate each control value
        valid_controls = {0, 1}
        for i, ctrl in enumerate(args.controls):
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

    if args.controls is None:
        controls = N * [0]
    elif len(args.controls) == 1:
        controls = N * [args.controls[0]]
    else:
        controls = args.controls
    
    t0 = np.array(args.t0).reshape((1,))
    dt = np.diff(np.concatenate((t0, args.flyby_times)))

    prob = get_dymos_serial_solver_problem(bodies=args.bodies,
                                           num_nodes=num_nodes,
                                           controls=controls,
                                           warm_start=False,
                                           default_opt_prob=True,
                                           t_max=args.max_time,
                                           obj=args.obj)
    prob.setup()
    
    set_initial_guesses(prob, bodies=args.bodies, flyby_times=args.flyby_times,
                        t0=args.t0, controls=controls)
    
    # prob.run_model()
    # prob.model.list_vars(print_arrays=True, units=True)
    # prob.list_problem_vars(print_arrays=True)
    result = prob.run_driver()


    # if result.success:
    #     sol, sol_file = create_solution(prob, args.bodies, filename=args.name)


if __name__ == '__main__':
    main()
