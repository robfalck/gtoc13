import sys

import numpy as np

import openmdao.api as om

from gtoc13.constants import YEAR
from gtoc13.solution import GTOC13Solution, PropagatedArc, ConicArc

from gtoc13.dymos_solver.solve_arcs import get_dymos_serial_solver_problem, set_initial_guesses, create_solution


def add_arc(args):
    """
    Load the solution as given by the command line arguments.

    Add a single arc following that solution to a specified body
    with a guess of the flyby
    """

    # Create from command-line arguments
    if not args.flyby_dt:
        print("Error: --flyby-dt is required for add_arc", file=sys.stderr)
        sys.exit(1)

    # Validate control argument if provided
    if args.control is not None:

        # Validate each control value
        valid_controls = {0, 1, 'r'}
        if args.control not in valid_controls and not args.control.lower().startswith('p'):
            print(f"Error: Invalid control scheme '{args.control}'. "
                    f"Must be one of: {', '.join(sorted(valid_controls))}", file=sys.stderr)
            sys.exit(1)

    if not isinstance(args.num_nodes, int):
        print("Error: Number of nodes in the new arc must be an integer.",
              file=sys.stderr)
        sys.exit(1)

    guess_sol = GTOC13Solution.load(args.solution_file)

    # Get the last state and time from the solution
    r0 = guess_sol.arcs[-1].position
    v0_in = guess_sol.arcs[-1].velocity_in
    v0_out = guess_sol.arcs[-1].velocity_out
    t0 = guess_sol.arcs[-1].epoch * YEAR
    bodies = [guess_sol.arcs[-1].body_id, args.body]

    prob = get_dymos_serial_solver_problem(bodies=bodies,
                                           num_nodes=args.num_nodes,
                                           controls=args.control,
                                           warm_start=False,
                                           default_opt_prob=True,
                                           t_max=args.max_time,
                                           opt_initial=False,
                                           obj=args.obj)

    prob.setup()

    set_initial_guesses(prob, bodies=bodies, flyby_times=[t0 + args.flyby_dt],
                        t0=t0,
                        controls=args.control, guess_solution=guess_sol)

    save = True
    if args.mode == 'run':
        prob.run_model()
        # prob.check_partials(method='fd', compact_print=True, form='central', includes='*miss_distance_comp*')
    elif args.mode == 'opt':
        result = prob.run_driver()
        save = result.success
    elif args.mode.startswith('feas'):
        prob.find_feasible(iprint=2, method='trf')

    #
    print(f'OpenMDAO output directory: {prob.get_outputs_dir()}')

    # Create solution with control information
    if save:
        sol, sol_file = create_solution(prob, args.bodies, controls=controls, filename=args.name)



if __name__ == '__main__':
    import os
    os.system("python -m gtoc13.dymos_solver add_arc 10_6_4_5_4_minE.txt --body 1000 --flyby-dt 10 --obj E")