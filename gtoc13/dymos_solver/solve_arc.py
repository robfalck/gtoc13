""" Script to solve a single arc from one cartesian position and velocity to another. """
from collections.abc import Sequence

from numba.core.cgutils import false_byte
import numpy as np
import openmdao.api as om
import dymos as dm

from gtoc13.constants import DAY, YEAR
from gtoc13.bodies import bodies_data
from gtoc13.dymos_solver.energy_comp import EnergyComp
from gtoc13.dymos_solver.solve_all import get_phase
from gtoc13.dymos_solver.ephem_comp import EphemCompNoStartPlane
from gtoc13.dymos_solver.flyby_comp import FlybyDefectComp
from gtoc13.dymos_solver.miss_distance_comp import MissDisanceComp
from gtoc13.dymos_solver.v_in_out_comp import SingleArcVInOutComp
from gtoc13.dymos_solver.score_comp import ScoreComp
from gtoc13.dymos_solver.initial_guesses import _guess_propagation, _guess_lambert
from gtoc13.dymos_solver.solve_all import create_solution


def get_single_arc_problem(bodies: Sequence[int],
                           control: Sequence[int] = None,
                           num_nodes=20,
                           t_max=199.999,
                           obj='J',
                           opt_v_in_prev_flyby=True,
                           prob_name='single_arc_problem'):
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
                             promotes_inputs=[('event_pos', 'body_pos')],
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
    # prob.model.add_design_var('dt', lower=0.0, upper=200, ref=10.0, units='gtoc_year')

    # # Incoming inertial velocity before first flyby
    if opt_v_in_prev_flyby:
        prob.model.add_design_var('v_in_prev_flyby', units='DU/TU')

    # # Outgoing inertial velocity after last flyby
    prob.model.add_design_var('v_end', units='DU/TU')

    # #
    # # CONSTRAINTS
    # #

    # Trajectory ends must match event positions
    prob.model.add_constraint('r_error', equals=0, units='DU', ref=0.01)

    # # V-infinity magnitude difference before/after each flyby
    # Don't need for only small bodies
    # prob.model.add_constraint('flyby_comp.v_inf_mag_defect', equals=0.0, units='km/s')

    # # Periapsis Altitude Constraint for Each flyby
    # # Note that this is a quadratic equation that is negative between the
    # # allowable flyby normalized altitude values, so it just has to be negative.

    # ONLY ADD HPDEFECT TO THOSE ROWS THAT ARE PLANET FLYBYS
    planet_flyby_idxs = np.where(np.asarray(bodies, dtype=int) <= 10)[0]

    if len(planet_flyby_idxs) > 0:
        prob.model.add_constraint('flyby_comp.h_p_norm',
                                indices=planet_flyby_idxs,
                                lower=0.1, upper=100.0)

        # Massless flyby DVs are constrained by the flyby defect comp
        prob.model.add_constraint('flyby_comp.v_inf_mag_defect',
                                  indices=planet_flyby_idxs,
                                  equals=0.0, units='km/s')


    # # Make sure the final time is in the allowable span.
    # if obj.lower() != 't':
    #     prob.model.add_constraint('times', indices=[-1], upper=t_max, units='gtoc_year')

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

    return prob, phase


def solve_arc(from_body: int,
              to_body: int,
              t1: float,
              t2: float,
              v_in_1: Sequence,
              v_out_1: Sequence,
              opt_v_in_1: bool=True,
              opt_dt: bool=False,
              control: int=0,
              mode: str='feasible',
              prob_name='solve_arc_prob',
              obj='J',
              num_nodes=20,
              guess='propagate'):

    bodies=[from_body, to_body]

    times_s = np.array([t1, t2]) * YEAR
    dt_s = np.diff(times_s)
    dt_year = dt_s / YEAR

    prob, phase = get_single_arc_problem(bodies=bodies,
                                         control=control,
                                         obj=obj,
                                         num_nodes=num_nodes,
                                         opt_v_in_prev_flyby=opt_v_in_1,
                                         prob_name=prob_name)

    phase.set_time_options(fix_initial=True, fix_duration=True)
    phase.set_simulate_options(times_per_seg=200)

    if opt_dt:
        prob.model.add_design_var('dt',
                                  lower=0.9 * dt_year,
                                  upper=1.1 * dt_year,
                                  ref=1.0, units='gtoc_year')

    prob.setup()

    prob.set_val('t0', times_s[0], units='s')
    prob.set_val('dt', np.diff(times_s), units='s')

    if guess.lower() == 'propagate':
        guess = _guess_propagation(phase,
                                from_body=bodies[0], to_body=bodies[1],
                                t1=times_s[0], t2=times_s[1],
                                v1=v_out_1,
                                control=control)
    elif guess.lower() == 'lambert':
        guess = _guess_lambert(phase, from_body=bodies[0], to_body=bodies[1],
                               t1=times_s[0], t2=times_s[1],
                               control=control)
    else:
        print('Error: Unrecognized guess {guess}. Must be "lambert" or "propagate"')

    phase.set_state_val('r', vals=guess['r'], time_vals=guess['times_s'], units='km')
    phase.set_state_val('v', vals=guess['v'], time_vals=guess['times_s'], units='km/s')

    if control == 1:
        phase.set_control_val('u_n', vals=guess['u'], time_vals=guess['times_s'], units='unitless')
    elif control == 0:
        phase.set_parameter_options('u_n', [0., 0., 0.])

    v_in_prev_flyby = np.array(v_in_1)

    prob.set_val('v_in_prev_flyby', v_in_prev_flyby, units='km/s')
    prob.set_val('v_end', guess['v'][-1, ...], units='km/s')

    if mode.lower().startswith('feas'):
        prob.find_feasible(iprint=2)
        run_opt = False
    elif mode.lower().startswith('opt'):
        run_opt = True
    else:
        run_opt = False

    failed = dm.run_problem(prob, run_driver=run_opt, simulate=True, make_plots=True)

    sol, _ = create_solution(prob, bodies, controls=[control], save_sol=False, single_arc=True)

    return not failed, sol, prob

if __name__ == '__main__':

    success, sol = solve_arc(control=1)
    print(success)
    print(sol)

