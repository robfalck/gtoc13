from collections.abc import Sequence
from pathlib import Path
import sys

import numpy as np

import numpy as np
import openmdao.api as om
import dymos as dm


from gtoc13.constants import YEAR
from gtoc13 import GTOC13Solution, PropagatedArc, FlybyArc

from gtoc13.dymos_solver.ephem_comp import EphemCompNoStartPlane
from gtoc13.dymos_solver.flyby_comp import FlybyDefectComp
from gtoc13.dymos_solver.energy_comp import EnergyComp
from gtoc13.dymos_solver.initial_guesses import set_initial_guesses
from gtoc13.dymos_solver.v_in_out_comp import SingleArcVInOutComp
from gtoc13.dymos_solver.score_comp import ScoreComp
from gtoc13.dymos_solver.miss_distance_comp import MissDisanceComp

from gtoc13.dymos_solver.solve_all import get_dymos_serial_solver_problem, set_initial_guesses, create_solution, get_phase


def get_dymos_single_arc_problem(bodies: Sequence[int],
                                 control: Sequence[int] = None,
                                 num_nodes=20,
                                 t_max=199.999,
                                 obj='J'):
    N = len(bodies) - 1

    prob = om.Problem()

    prob.model.add_subsystem('ephem', EphemCompNoStartPlane(bodies=bodies), promotes=['*'])

    traj = dm.Trajectory()
    prob.model.add_subsystem('traj', traj)

    phase = get_phase(num_nodes=num_nodes, control=control)
    traj.add_phase('arc_0', phase)

    phase.set_simulate_options(times_per_seg=50, atol=1.0E-12, rtol=1.0E-12)

    prob.model.connect('times', 'traj.arc_0.t_initial', src_indices=[0])
    prob.model.connect('dt_out', 'traj.arc_0.t_duration', src_indices=[0])

    prob.model.add_subsystem('miss_distance_comp',
                             MissDisanceComp(N=N),
                             promotes_inputs=['event_pos'],
                             promotes_outputs=['r_error'])

    prob.model.add_subsystem('v_in_out_comp', SingleArcVInOutComp(), promotes_inputs=['v_in_prev_flyby', 'v_end'])

    # No for loop here unlike the multiphase problem
    prob.model.connect('traj.arc_0.timeseries.v',
                        'v_in_out_comp.arc_0_v_initial',
                        src_indices=om.slicer[0, ...])

    prob.model.connect('traj.arc_0.timeseries.v',
                        'v_in_out_comp.arc_0_v_final',
                        src_indices=om.slicer[-1, ...])

    prob.model.connect('traj.arc_0.timeseries.r',
                        'miss_distance_comp.arc_0_r_initial',
                        src_indices=om.slicer[0, ...])

    prob.model.connect('traj.arc_0.timeseries.r',
                        'miss_distance_comp.arc_0_r_final',
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
    prob.model.connect('body_pos', 'r_end', src_indices=om.slicer[-1, ...])

    prob.model.add_subsystem('score_comp',
                             ScoreComp(bodies=bodies),
                             promotes_outputs=['J'])

    prob.model.connect('body_pos', 'score_comp.body_pos')

    prob.model.connect('flyby_comp.v_inf_in', 'score_comp.v_inf')

    # #
    # # DESIGN VARIABLES
    # #

    # # Times between flyby events
    prob.model.add_design_var('dt', lower=0.0, upper=200, ref=10.0, units='gtoc_year')

    # # Outgoing inertial velocity after last flyby
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
    t0 = guess_sol.arcs[-1].epoch / YEAR
    bodies = [guess_sol.arcs[-1].body_id, args.body]

    prob = get_dymos_single_arc_problem(bodies=bodies,
                                        num_nodes=args.num_nodes,
                                        control=args.control,
                                        t_max=args.max_time,
                                        obj=args.obj)

    prob.setup()

    controls = guess_sol.get_control_flags()
    prev_flyby_time = guess_sol.get_flyby_times()[-1]
    flyby_times = [prev_flyby_time, prev_flyby_time + args.flyby_dt]
    
    set_initial_guesses(prob, bodies=bodies,
                        flyby_times=flyby_times,
                        t0=t0,
                        controls=[args.control],
                        guess_solution=guess_sol,
                        single_arc=True)

    save = True
    if args.mode == 'run':
        prob.run_model()
    elif args.mode == 'opt':
        result = prob.run_driver()
        save = result.success
    elif args.mode.startswith('feas'):
        prob.find_feasible(iprint=2, method='trf')

    prob.list_problem_vars(print_arrays=True)
    #
    print(f'OpenMDAO output directory: {prob.get_outputs_dir()}')

    # Create solution with control information
    if save:
        load_path = Path(args.solution_file)
        new_filename = f"{load_path.stem}_{args.body}"
        sol, sol_file = create_solution(prob, bodies, controls=controls, filename=new_filename, single_arc=True)

if __name__ == '__main__':
    import os
    os.system("python -m gtoc13.dymos_solver add_arc 10_6_4_5_4_minE.txt --body 1000 --flyby-dt 10 --obj E")