import sys

from gtoc13.solution import GTOC13Solution, PropagatedArc, ConicArc, FlybyArc


def add_arc(args):
    """
    Load the solution as given by the command line arguments.
    
    Add a single arc following that solution to a specified body
    with a guess of the flyby
    """

    # Create from command-line arguments
    if not args.flyby_time:
        print("Error: --flyby-times is required when using --bodies", file=sys.stderr)
        sys.exit(1)

    # Validate control argument if provided
    if args.control is not None:

        # Validate each control value
        valid_controls = {0, 1, 'r'}
        if args.control not in valid_controls:
            print(f"Error: Invalid control scheme '{args.control}'. "
                    f"Must be one of: {', '.join(sorted(valid_controls))}", file=sys.stderr)
            sys.exit(1)

    if not isinstance(args.num_nodes, int):
        print(f"Error: Number of nodes in the new arc must be an integer.",
              file=sys.stderr)
        sys.exit(1)
    
    t0 = np.array(args.t0).reshape((1,))
    dt = np.diff(np.concatenate((t0, args.flyby_times)))

    guess_sol = GTOC13Solution.load(args.solution_file)

    # Need to get the existing bodies, the existing flyby times, the existing controls,
    # Then add ours to those. before setting up the problem.
    bodies = []
    num_nodes = []
    controls = []
    fixed_arcs = []
    for i, arc in enumerate(guess_sol.arcs):
        if isinstance(arc, (PropagatedArc, ConicArc)):
            if i == 0 and arc.bodies[0] != -1:
                bodies.append(arc.bodies[0])
            bodies.append(arc.bodies[1])
        if isinstance(arc, PropagatedArc):
            num_nodes.append(len(arc.state_points))
            controls.append(arc.control_type)
        else:
            num_nodes.append(20)
            controls.append[0]
        fixed_arcs.append(True)
        
    bodies.append(args.body)
    num_nodes.append(args.num_nodes)
    controls.append(args.control)
    fixed_arcs.append(False)

    prob = get_dymos_serial_solver_problem(bodies=args.bodies,
                                           num_nodes=num_nodes,
                                           controls=controls,
                                           warm_start=False,
                                           default_opt_prob=True,
                                           t_max=args.max_time,
                                           obj=args.obj)
    prob.setup()

    guess_sol = GTOC13Solution.load(args.solution_file)

    set_initial_guesses(prob, bodies=args.bodies, flyby_times=args.flyby_times,
                        t0=args.t0, controls=controls, guess_solution=guess_sol)
    
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

