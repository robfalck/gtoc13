from collections import deque
from copy import deepcopy
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
import openmdao.api as om
import dymos as dm


from gtoc13 import GTOC13Solution, PropagatedArc, FlybyArc

from gtoc13.dymos_solver.ephem_comp import EphemComp
from gtoc13.dymos_solver.flyby_comp import FlybyDefectComp
from gtoc13.dymos_solver.energy_comp import EnergyComp
from gtoc13.dymos_solver.initial_guesses import set_initial_guesses
from gtoc13.dymos_solver.phases import get_phase
from gtoc13.dymos_solver.v_in_out_comp import VInOutComp
from gtoc13.dymos_solver.score_comp import ScoreComp
from gtoc13.dymos_solver.miss_distance_comp import MissDisanceComp



def get_dymos_serial_solver_problem(bodies: Sequence[int],
                                    controls: Sequence[int] = None,
                                    num_nodes=20,
                                    warm_start=False,
                                    default_opt_prob=True,
                                    opt_initial=True,
                                    t_max=199.999,
                                    obj='J',
                                    prob_name='multi_arc_prob'):
    N = len(bodies)

    if isinstance(num_nodes, int):
        _num_nodes = N * [num_nodes]
    else:
        _num_nodes = num_nodes

    if isinstance(controls, int):
        _control = N * [controls]
    else:
        _control = controls

    prob = om.Problem(name=prob_name)

    prob.model.add_subsystem('ephem', EphemComp(bodies=bodies), promotes=['*'])

    traj = dm.Trajectory()
    prob.model.add_subsystem('traj', traj)

    for i in range(N):
        phase = get_phase(num_nodes=_num_nodes[i], control=_control[i])
        traj.add_phase(f'arc_{i}', phase)

        # Previously we fixed the position at the ends to these, but this
        # is a little more ODE-agnostic
        # If you want to undo this change, uncomment these lines and remove
        # the constraint on r_error below
        # prob.model.connect('event_pos', f'traj.arc_{i}.initial_states:r', src_indices=om.slicer[i, ...])
        # prob.model.connect('event_pos', f'traj.arc_{i}.final_states:r', src_indices=om.slicer[i+1, ...])
        # phase.state_options['r']['fix_initial'] = True
        # phase.state_options['r']['fix_final'] = True

        phase.set_simulate_options(times_per_seg=50, atol=1.0E-12, rtol=1.0E-12)

        prob.model.connect('times', f'traj.arc_{i}.t_initial', src_indices=[i])
        prob.model.connect('dt_out', f'traj.arc_{i}.t_duration', src_indices=[i])

    prob.model.add_subsystem('miss_distance_comp',
                             MissDisanceComp(N=N),
                             promotes_inputs=['event_pos'],
                             promotes_outputs=['r_error'])
    prob.model.add_subsystem('v_in_out_comp', VInOutComp(N=N), promotes_inputs=['v_end'])

    for i in range(N):
        if i > 0:
            prob.model.connect(f'traj.arc_{i}.timeseries.v',
                            f'v_in_out_comp.arc_{i}_v_initial',
                            src_indices=om.slicer[0, ...])

        prob.model.connect(f'traj.arc_{i}.timeseries.v',
                           f'v_in_out_comp.arc_{i}_v_final',
                           src_indices=om.slicer[-1, ...])

        prob.model.connect(f'traj.arc_{i}.timeseries.r',
                           f'miss_distance_comp.arc_{i}_r_initial',
                           src_indices=om.slicer[0, ...])

        prob.model.connect(f'traj.arc_{i}.timeseries.r',
                           f'miss_distance_comp.arc_{i}_r_final',
                           src_indices=om.slicer[-1, ...])

    prob.model.add_subsystem('flyby_comp', FlybyDefectComp(bodies=bodies))

    prob.model.connect('v_in_out_comp.flyby_v_in',
                       'flyby_comp.v_in')

    prob.model.connect('v_in_out_comp.flyby_v_out',
                       'flyby_comp.v_out')

    prob.model.connect('body_vel', 'flyby_comp.v_body')

    prob.model.add_subsystem('energy_comp', EnergyComp(),
                             promotes_inputs=['v_end', 'r_end'],
                             promotes_outputs=['E_end'])
    prob.model.connect('event_pos', 'r_end', src_indices=om.slicer[-1, ...])

    prob.model.add_subsystem('score_comp',
                             ScoreComp(bodies=bodies),
                             promotes_outputs=['J'])

    prob.model.connect('event_pos', 'score_comp.body_pos',
                       src_indices=om.slicer[1:, ...])

    prob.model.connect('flyby_comp.v_inf_in', 'score_comp.v_inf')

    # #
    # # DESIGN VARIABLES
    # #

    if opt_initial:
        # # Start time
        prob.model.add_design_var('t0', lower=0.0, units='gtoc_year')
        # # Start plane position
        prob.model.add_design_var('y0', units='DU')
        prob.model.add_design_var('z0', units='DU')
        # # A constraint on in y and z components of the initial velocity vector
        prob.model.traj.phases.arc_0.add_boundary_constraint('v', loc='initial', indices=[1, 2], equals=0.0)


    # # Times between flyby events
    prob.model.add_design_var('dt', lower=0.001, upper=200, ref=10.0, units='gtoc_year')

    # # Outgoing inertial velocity after last flybyo
    prob.model.add_design_var('v_end', units='DU/TU')

    # #
    # # CONSTRAINTS
    # #

    # Trajectory ends must match event positions
    prob.model.add_constraint('r_error', equals=0, units='DU', ref=0.01)

    # # V-infinity magnitude difference before/after each flyby
    prob.model.add_constraint('flyby_comp.v_inf_mag_defect', equals=0.0, units='km/s')

    # # Periapsis Altitude Constraint for Each flyby
    # # Note that this is a quadratic equation that is negative between the
    # # allowable flyby normalized altitude values, so it just has to be negative.
    # ONLY ADD HPDEFECT TO THOSE ROWS THAT ARE PLANET FLYBYS
    planet_flyby_idxs = np.where(np.asarray(bodies, dtype=int) <= 10)[0]
    if len(planet_flyby_idxs) > 0:
        prob.model.add_constraint('flyby_comp.h_p_norm',
                                indices=planet_flyby_idxs,
                                lower=0.1, upper=100.0)
        # prob.model.add_constraint('flyby_comp.h_p_defect',
        #                           indices=planet_flyby_idxs,
        #                           upper=0.0, ref=1000.0)

    # # Make sure the final time is in the allowable span.
    if obj.lower() != 't':
        prob.model.add_constraint('times', indices=[-1], upper=t_max, units='gtoc_year')

    # prob.model.add_constraint('hz_end', lower=2.0, units='DU**2/TU')

    # # TODO: Add a path constraint for perihelion distance.

    # #
    # # OBJECTIVE
    # #

    # # Minimize specific orbital energy after the last flyby
    if obj.lower() == 'e':
        prob.model.add_objective('E_end', ref=1.0, units='DU**2/TU**2')
    elif obj.lower() == 't':
        prob.model.add_objective('times', ref=10, index=-1, units='gtoc_year')
    else:
        prob.model.add_objective('J', ref=-1.0, units='unitless')


    prob.driver = om.pyOptSparseDriver(optimizer='IPOPT')
    prob.driver.declare_coloring()  # Take advantage of sparsity.
    prob.driver.opt_settings['print_level'] = 5
    prob.driver.opt_settings['tol'] = 1.0E-6

    # Gradient-based autoscaling
    prob.driver.opt_settings['nlp_scaling_method'] = 'gradient-based'

    # Step-size selection
    prob.driver.opt_settings['alpha_for_y'] = 'safer-min-dual-infeas'

    # This following block allows IPOPT to finish if it has a feasible but
    # not optimal solution for several iterations in a row.
    # Acceptable (feasible but suboptimal) tolerance
    prob.driver.opt_settings['acceptable_tol'] = 1.0  # Large value means we don't care about optimality
    prob.driver.opt_settings['acceptable_constr_viol_tol'] = 1.0E-6  # Must satisfy constraints
    prob.driver.opt_settings['acceptable_dual_inf_tol'] = 1.0E10  # Don't care about dual infeasibility
    prob.driver.opt_settings['acceptable_compl_inf_tol'] = 1.0E10  # Don't care about complementarity
    # Number of iterations at acceptable level before terminating
    prob.driver.opt_settings['acceptable_iter'] = 5  # Accept after 5 consecutive "acceptable" iterations

    # How to initialize the constraint bounds of the interior point method
    prob.driver.opt_settings['bound_mult_init_method'] = 'mu-based'

    # How IPOPT changes its barrier parameter (mu) over time.
    # This problem seems to work much better with the default 'adaptive'
    # prob.driver.opt_settings['mu_strategy'] = 'monotone'

    return prob

def create_solution(prob, bodies, controls=None, filename=None, single_arc=False, save_sol=True):
    N = len(bodies)

    num_arcs = 1 if single_arc else N

    # Default controls to all 0 if not provided
    if controls is None:
        controls = [0] * N

    flyby_comp = prob.model.flyby_comp

    flyby_v_in = flyby_comp.get_val('v_in', units='km/s')
    flyby_v_out = flyby_comp.get_val('v_out', units='km/s')
    flyby_v_inf_in = flyby_comp.get_val('v_inf_in', units='km/s')
    flyby_v_inf_out = flyby_comp.get_val('v_inf_out', units='km/s')

    arcs = []
    arc_bodies = deque(maxlen=2)
    arc_bodies.append(-1)
    for i in range(num_arcs):
        t = prob.get_val(f'traj.arc_{i}.timeseries.time', units='s')
        r = prob.get_val(f'traj.arc_{i}.timeseries.r', units='km')
        v = prob.get_val(f'traj.arc_{i}.timeseries.v', units='km/s')
        try:
            u_n = prob.get_val(f'traj.arc_{i}.timeseries.u_n', units='unitless')
        except KeyError as e:
            u_n = prob.get_val(f'traj.arc_{i}.parameters:u_n', units='unitless')
            u_n = np.broadcast_to(u_n, shape=r.shape)

        u_n_norm = np.linalg.norm(u_n, axis=-1)

        # Determine control type for this arc
        control_value = controls[i]
        if control_value == 'r':
            control_type = 'radial'
        elif control_value == 1:
            control_type = 'optimal'
        else:  # control_value == 0
            control_type = 'N/A'

        # Add the i-th propagated arc
        arc_bodies.append(bodies[i])
        arcs.append(PropagatedArc.create(epochs=t, positions=r,
                                         velocities=v, controls=u_n,
                                         control_type=control_type,
                                         bodies=arc_bodies))
        arc_bodies.pop

        # Add the i-th flyby arc
        t_flyby_i = prob.get_val('times', units='s')[i + 1]

        # For now assume we never repeat a body more than 12 times, so
        # each flyby is for science.
        arcs.append(FlybyArc.create(body_id=bodies[i],
                                    epoch=t_flyby_i,
                                    position=r[-1, ...],
                                    velocity_in=flyby_v_in[i],
                                    velocity_out=flyby_v_out[i],
                                    v_inf_in=flyby_v_inf_in[i],
                                    v_inf_out=flyby_v_inf_out[i],
                                    is_science=True))

    # Extract objective values
    E_end = prob.get_val('E_end')[0]

    # Get the objective value (if it exists)
    try:
        J = prob.get_val('J')[0]
    except KeyError:
        J = None

    solution = GTOC13Solution(arcs=arcs,
                              comments=[],
                              objective_J=J)

    if not save_sol:
        return solution, None

    # Find the next available solution filename
    solutions_dir = Path(__file__).parent.parent.parent / 'solutions'
    solutions_dir.mkdir(exist_ok=True)

    if filename is None:
        index = 1
        while (solutions_dir / f'dymos_solution_{index}.txt').exists():
            index += 1

        solution_file = solutions_dir / f'dymos_solution_{index}.txt'
        plot_file = solutions_dir / f'dymos_solution_{index}.png'
    else:
        solution_file = solutions_dir / f'{filename}.txt'
        plot_file = solutions_dir / f'{filename}.png'

    solution.write_to_file(solution_file, precision=11)
    print(f"Solution written to {solution_file}")

    # Create plot and save it
    solution.plot(show_bodies=True, save_path=plot_file,
                  E_end=E_end, J=J)

    return solution, solution_file


def solve_all_arcs(args):

    # Handle solve_arcs command
    if args.bodies is None:
        args.bodies = []
    if args.flyby_times is None:
        args.flyby_times = []

    if args.load:
        guess_sol = GTOC13Solution.load(args.load[0])
    else:
        guess_sol = None

    # Create from command-line arguments
    if len(args.bodies) > 0 and not args.flyby_times:
        print("Error: --flyby-times is required when using --bodies", file=sys.stderr)
        sys.exit(1)

    if len(args.bodies) != len(args.flyby_times):
        print(f"Error: Number of bodies ({len(args.bodies)}) must match "
                f"number of flyby times ({len(args.flyby_times)})", file=sys.stderr)
        sys.exit(1)

    bodies = deepcopy(args.bodies)
    if not bodies:
        if guess_sol is not None:
            bodies = guess_sol.get_bodies()
        else:
            print('Error: If no guess is loaded, bodies must be specified.')
            sys.exit(1)

    flyby_times = deepcopy(args.flyby_times)
    if not flyby_times:
        if guess_sol is not None:
            flyby_times = guess_sol.get_flyby_times()
        else:
            print('Error: If no guess is loaded, flyby-times must be specified.')
            sys.exit(1)

    if guess_sol is not None:
        t0 = guess_sol.get_t0()
    else:
        t0 = args.t0

    N = len(bodies)

    # Validate control argument if provided
    if args.controls is not None:
        if len(args.controls) != len(args.bodies) and len(args.controls) != 1:
            print(f"Error: Number of control flags ({len(args.controls)}) must match "
                    f"number of bodies ({len(args.bodies)}) if multiple are given", file=sys.stderr)
            sys.exit(1)

        # Validate each control value
        valid_controls = {0, 1, 'r'}
        for i, ctrl in enumerate(args.controls):
            if ctrl not in valid_controls:
                print(f"Error: Invalid control scheme '{ctrl}' at position {i}. "
                        f"Must be one of: {', '.join(sorted(valid_controls))}", file=sys.stderr)
                sys.exit(1)

    if isinstance(args.num_nodes, int):
        num_nodes = N * [args.num_nodes]
    else:
        if len(args.num_nodes) == 1:
            num_nodes = N * [args.num_nodes[0]]
        else:
            num_nodes = args.num_nodes

    if len(num_nodes) != len(bodies):
        print(f"Error: Number of nodes in each arc must be a scalar or must match the number of flyby bodies. ({len(args.bodies)})",
              file=sys.stderr)
        sys.exit(1)

    if args.controls is None:
        controls = N * [0]
    elif len(args.controls) == 1:
        controls = N * [args.controls[0]]
    else:
        controls = args.controls

    prob = get_dymos_serial_solver_problem(bodies=bodies,
                                           num_nodes=num_nodes,
                                           controls=controls,
                                           warm_start=False,
                                           default_opt_prob=True,
                                           t_max=args.max_time,
                                           obj=args.obj,
                                           prob_name='solve_all_arcs')
    prob.setup()

    set_initial_guesses(prob, bodies=bodies, flyby_times=flyby_times,
                        t0=args.t0, controls=controls, guess_solution=guess_sol)

    save = True
    if args.mode == 'run':
        prob.run_model()
        # prob.check_partials(method='fd', compact_print=True, form='central', includes='*miss_distance_comp*')
    elif args.mode == 'opt':
        result = prob.run_driver()
        # save = result.success
    elif args.mode.startswith('feas'):
        prob.find_feasible(iprint=2, method='trf')

    #
    print(f'OpenMDAO output directory: {prob.get_outputs_dir()}')

    # Create solution with control information
    if save:
        sol, sol_file = create_solution(prob, bodies, controls=controls, filename=args.name)
