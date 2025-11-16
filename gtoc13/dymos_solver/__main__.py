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

from gtoc13.dymos_solver.initial_guesses import set_initial_guesses
from gtoc13.solution import GTOC13Solution
from gtoc13.constants import YEAR
from gtoc13.dymos_solver.solve_all import (
    get_dymos_serial_solver_problem,
    create_solution,
    solve_all_arcs
)
from gtoc13.dymos_solver.add_arc import add_arc


def control_type(value):
    """Parse control argument: accepts 0, 1, or 'r'"""
    if value.lower() == 'r':
        return 'r'
    try:
        int_val = int(value)
        if int_val in [0, 1]:
            return int_val
        else:
            raise ValueError(f"Control must be 0, 1, or 'r', got {value}")
    except ValueError:
        raise ValueError(f"Control must be 0, 1, or 'r', got {value}")


def _setup_solve_all_parser(subparsers):
    """
    Set up the solve_all subcommand parser.

    Parameters
    ----------
    subparsers : argparse._SubParsersAction
        The subparsers object from the main parser

    Returns
    -------
    argparse.ArgumentParser
        The configured solve_arcs parser
    """
    solve_all_parser = subparsers.add_parser(
        'solve_all',
        help='Solve trajectory arcs using Dymos optimization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Solve with command-line arguments
  python -m gtoc13.dymos_solver solve_all --bodies 10 9 --flyby-times 20.0 40.0

  # Specify initial time and number of nodes
  python -m gtoc13.dymos_solver solve_all --bodies 10 --flyby-times 20.0 --t0 5.0 --num-nodes 30

  # Specify control scheme for each arc
  python -m gtoc13.dymos_solver solve_all --bodies 10 9 --flyby-times 20.0 40.0 --control 0 1

  # Load a solution file as an initial guess
  python -m gtoc13.dymos_solver solve_all --bodies 9 8 7 --flyby-times 20 40 80 --max-time 150 --controls 0 0 0 --load solutions/dymos_solution_32.txt
"""
    )

    # Mission plan file or command-line arguments
    # input_group = solve_all_parser.add_mutually_exclusive_group(required=True)

    solve_all_parser.add_argument(
        '--bodies', '-b',
        type=int,
        nargs='+',
        metavar='BODY_ID',
        help='Body IDs to visit (space-separated integers)'
    )

    # Additional arguments for command-line mode
    solve_all_parser.add_argument(
        '--flyby-times', '-f',
        type=float,
        nargs='+',
        metavar='TIME',
        help='Flyby times in years (space-separated floats, required with --bodies)'
    )
    solve_all_parser.add_argument(
        '--t0',
        type=float,
        default=0.0,
        help='Initial time in years (default: 0.0)'
    )

    solve_all_parser.add_argument(
        '--controls', '-c',
        type=control_type,
        nargs='+',
        metavar='CONTROL_FLAG',
        help="Control flag for each arc: 0, 1, or 'r' for radial (space-separated, must match number of bodies)"
    )

    # Solver options
    solve_all_parser.add_argument(
        '--num-nodes', '-n',
        type=int,
        nargs='+',
        default=20,
        help='Number of collocation nodes per arc (default: 20)'
    )

    solve_all_parser.add_argument(
        '--no-opt',
        action='store_true',
        help='If given, just run through the model once without optimization.'
    )

    solve_all_parser.add_argument(
        '--max-time',
        type=float,
        default=199.999,
        help='Maximum allowable final time in years. (default: 199.999)'
    )

    solve_all_parser.add_argument(
        '--load', '-l',
        type=str,
        nargs='+',
        default=None,
        metavar='LOAD_FILES',
        help='File(s) from which to load the solution. If multiple, they are concatenated.'
    )

    solve_all_parser.add_argument(
        '--name',
        type=str,
        default=None,
        help='Root name of solution file to be saved. This will overwrite existing files of the same name!'
    )

    solve_all_parser.add_argument(
        '--obj',
        type=str,
        default='J',
        help='Objective to be used. Either "J" to maximize GTOC objective or "E" to minimize final energy'
    )

    solve_all_parser.add_argument(
        '--mode', '-m',
        type=str,
        default='opt',
        help='Mode of operation: "opt", "feas", or "run"'
    )

    return solve_all_parser


def _setup_add_arc_parser(subparsers):
    """
    Set up the add_arc subcommand parser.

    Parameters
    ----------
    subparsers : argparse._SubParsersAction
        The subparsers object from the main parser

    Returns
    -------
    argparse.ArgumentParser
        The configured add_arc parser
    """
    add_arc_parser = subparsers.add_parser(
        'add_arc',
        help='Add a new arc to an existing solution',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Add a new arc to an existing solution
  python -m gtoc13.dymos_solver add_arc solutions/my_solution.txt --max-time 200

  # Add arc and optimize existing arcs too
  python -m gtoc13.dymos_solver add_arc solutions/my_solution.txt --opt-existing

  # Add arc with custom objective
  python -m gtoc13.dymos_solver add_arc solutions/my_solution.txt --obj E --name extended_solution
"""
    )

    # Positional argument: solution file
    add_arc_parser.add_argument(
        'solution_file',
        type=str,
        help='Path to the existing solution file to extend'
    )

    add_arc_parser.add_argument(
        '--body',
        type=int,
        help='Next flyby body to be added.'
    )

    add_arc_parser.add_argument(
        '--flyby-dt',
        type=float,
        help='Flight time to get to next body (yr)'
    )

    add_arc_parser.add_argument(
        '--fix-dt',
        action='store_true',
        help='If True, do not treat flight time as a design variable.'
    )

    add_arc_parser.add_argument(
        '--fix-vin1',
        action='store_true',
        help='If True, do not treat the first flyby incoming velocity as a design variable.'
    )

    add_arc_parser.add_argument(
        '--control', '-c',
        type=control_type,
        default=0,
        help="Control flag for each arc: 0, 1, or 'r' for radial (space-separated, must match number of bodies)"
    )

    add_arc_parser.add_argument(
        '--guess',
        type=str,
        help='Trajectory guess generation. Either "propagate" to propagate the final' \
        ' velocity of the previous solution, or "lambert" to guess via a lambert solution.'
    )

    # Solver options
    add_arc_parser.add_argument(
        '--num-nodes', '-n',
        type=int,
        default=20,
        help='Number of collocation nodes per arc (default: 20)'
    )

    # Optional arguments
    add_arc_parser.add_argument(
        '--max-time',
        type=float,
        default=199.999,
        help='Maximum allowable final time in years. (default: 199.999)'
    )

    add_arc_parser.add_argument(
        '--mode', '-m',
        type=str,
        default='opt',
        help='Mode of operation: "opt", "feas", or "run"'
    )

    add_arc_parser.add_argument(
        '--obj',
        type=str,
        default='J',
        help='Objective to be used. Either "J" to maximize GTOC objective or "E" to minimize final energy'
    )

    return add_arc_parser


def main():
    """Main entry point for the dymos solver CLI."""
    parser = argparse.ArgumentParser(
        description="GTOC13 Dymos Solver - Trajectory optimization using Dymos",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    subparsers.required = True

    # Set up subcommand parsers
    _setup_solve_all_parser(subparsers)
    _setup_add_arc_parser(subparsers)

    # Parse arguments
    args = parser.parse_args()

    # Route to appropriate command handler
    if args.command == 'add_arc':
        add_arc(args)

    elif args.command == 'solve_all':
        solve_all_arcs(args)


if __name__ == '__main__':
    main()
