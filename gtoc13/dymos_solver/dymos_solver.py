from collections import deque
from pathlib import Path
from typing import Sequence

import pydantic

import numpy as np
import openmdao.api as om
import openmdao.utils.units as om_units
import dymos as dm

from gtoc13 import bodies_data, GTOC13Solution, PropagatedArc, FlybyArc, ConicArc
from gtoc13.constants import MU_ALTAIRA, KMPDU, SPTU, R0

from gtoc13.dymos_solver.ephem_comp import EphemComp
from gtoc13.dymos_solver.v_concat_comp import VConcatComp
from gtoc13.dymos_solver.flyby_comp import FlybyDefectComp
from gtoc13.dymos_solver.energy_comp import EnergyComp
from gtoc13.dymos_solver.v_in_out_comp import VInOutComp
from gtoc13.dymos_solver.score_comp import ScoreComp

from gtoc13.dymos_solver.ode_comp import SolarSailRadialControlODEComp, SolarSailODEComp


def get_phase(num_nodes, control):
    tx = dm.Birkhoff(num_nodes=num_nodes, grid_type='lgl')
    # tx = dm.PicardShooting(num_segments=1, nodes_per_seg=num_nodes, solve_segments='forward')

    ode_cls = SolarSailRadialControlODEComp if control == 'r' else SolarSailODEComp

    phase = dm.Phase(ode_class=ode_cls,
                     transcription=tx)

    phase.add_state('r', rate_source='drdt', units='DU',
                    shape=(3,), fix_initial=True, fix_final=True,
                    targets=['r'])

    phase.add_state('v', rate_source='dvdt', units='DU/TU',
                    shape=(3,), fix_initial=False, fix_final=False,
                    targets=['v'], lower=-100, upper=100,
                    ref=1.0, defect_ref=1.0E-2)

    # We're just going to construct this phase and return it without
    # a control, so that the calling function can handle whether
    # u_n should be a control or a parameter.

    # Control: sail normal unit vector (ballistic = zero for Keplerian orbit)
    if control == 1:
        phase.add_control('u_n', units='unitless', shape=(3,), opt=True,
                        val=np.ones((3,)), targets=['u_n'])
        phase.add_path_constraint('u_n_norm', equals=1.0)
        phase.add_path_constraint('cos_alpha', lower=0.0)
    elif control == 0:
        phase.add_parameter('u_n', units='unitless', shape=(3,),
                            val=np.zeros((3,)), opt=False)
   

    # Set time options
    # The fix_initial here is really a bit of a misnomer.
    # They're not design variables, and we can therefore connect
    # t_initial and t_duration to upstream outputs.
    phase.set_time_options(fix_initial=True,
                           fix_duration=True,
                           units='TU', )
    
    phase.add_timeseries_output('a_grav', units='km/s**2')
    phase.add_timeseries_output('a_sail', units='km/s**2')
    phase.add_timeseries_output('u_n', units='unitless')
    phase.add_timeseries_output('u_n_norm', units='unitless')

    return phase

def get_dymos_serial_solver_problem(bodies: Sequence[int],
                                    controls: Sequence[int] = None,
                                    num_nodes=20,
                                    warm_start=False,
                                    default_opt_prob=True,
                                    t_max=199.999,
                                    obj='J'):
    N = len(bodies)

    if isinstance(num_nodes, int):
        _num_nodes = N * [num_nodes]
    else:
        _num_nodes = num_nodes

    if isinstance(controls, int):
        _control = N * [controls]
    else:
        _control = controls

    prob = om.Problem()

    prob.model.add_subsystem('ephem', EphemComp(bodies=bodies), promotes=['*'])

    traj = dm.Trajectory()
    prob.model.add_subsystem('traj', traj)

    for i in range(N):
        phase = get_phase(num_nodes=_num_nodes[i], control=_control[i])
        traj.add_phase(f'arc_{i}', phase)

        prob.model.connect('event_pos', f'traj.arc_{i}.initial_states:r', src_indices=om.slicer[i, ...])
        prob.model.connect('event_pos', f'traj.arc_{i}.final_states:r', src_indices=om.slicer[i+1, ...])

        phase.set_simulate_options(times_per_seg=50, atol=1.0E-12, rtol=1.0E-12)

        prob.model.connect('times', f'traj.arc_{i}.t_initial', src_indices=[i])
        prob.model.connect('dt_out', f'traj.arc_{i}.t_duration', src_indices=[i])

    prob.model.add_subsystem('v_in_out_comp', VInOutComp(N=N), promotes_inputs=['v_end'])

    for i in range(N):
        if i > 0:
            prob.model.connect(f'traj.arc_{i}.timeseries.v', 
                            f'v_in_out_comp.arc_{i}_v_initial',
                            src_indices=om.slicer[0, ...])
        
        prob.model.connect(f'traj.arc_{i}.timeseries.v', 
                           f'v_in_out_comp.arc_{i}_v_final',
                           src_indices=om.slicer[-1, ...])

    prob.model.add_subsystem('flyby_comp', FlybyDefectComp(bodies=bodies))

    prob.model.connect('v_in_out_comp.flyby_v_in',
                       'flyby_comp.v_in')
    
    prob.model.connect('v_in_out_comp.flyby_v_out',
                       'flyby_comp.v_out')
    
    prob.model.connect('body_vel', 'flyby_comp.v_body')

    prob.model.add_subsystem('energy_comp', EnergyComp(),
                             promotes_inputs=['v_end', 'r_end'],
                             promotes_outputs=['E_end', 'hz_end'])
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

    # # Start time
    prob.model.add_design_var('t0', lower=0.0, units='gtoc_year')

    # # Times between flyby events
    prob.model.add_design_var('dt', lower=0.0, upper=200, units='gtoc_year') 

    # # Start plane position
    prob.model.add_design_var('y0', units='DU')
    prob.model.add_design_var('z0', units='DU')

    # # Outgoing inertial velocity after last flyby
    prob.model.add_design_var('v_end', units='DU/TU')

    # #
    # # CONSTRAINTS
    # #

    # # V-infinity magnitude difference before/after each flyby
    prob.model.add_constraint('flyby_comp.v_inf_mag_defect', equals=0.0, units='km/s')
    
    # # Periapsis Altitude Constraint for Each flyby
    # # Note that this is a quadratic equation that is negative between the
    # # allowable flyby normalized altitude values, so it just has to be negative.
    # ONLY ADD HPDEFECT TO THOSE ROWS THAT ARE PLANET FLYBYS
    planet_flyby_idxs = np.where(np.asarray(bodies, dtype=int) <= 10)[0]
    if len(planet_flyby_idxs) > 0:
        prob.model.add_constraint('flyby_comp.h_p_defect', 
                                indices=planet_flyby_idxs,
                                upper=0.0, ref=1000.0)

    # # Make sure the final time is in the allowable span.
    prob.model.add_constraint('times', indices=[-1], upper=t_max, units='gtoc_year')

    # prob.model.add_constraint('hz_end', lower=2.0, units='DU**2/TU')

    # # A constraint on in y and z components of the initial velocity vector
    prob.model.traj.phases.arc_0.add_boundary_constraint('v', loc='initial', indices=[1, 2], equals=0.0)

    # # TODO: Add a path constraint for perihelion distance.

    # #
    # # OBJECTIVE
    # #

    # # Minimize specific orbital energy after the last flyby
    # # TODO: Convert to problem objective.

    # prob.model.add_objective('E_end')
    if obj == 'E':
        prob.model.add_objective('E_end', ref=1.0, units='DU**2/TU**2')
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


def set_initial_guesses(prob, bodies, flyby_times, t0, controls,
                        guess_solution=None):
    from gtoc13.constants import YEAR

    # Set initial guess values
    N = len(bodies)

    _t0 = np.array(t0).reshape((1,))
    all_times_yr = np.concatenate((_t0, flyby_times))
    dt_yr = np.diff(all_times_yr)
    dt_s = dt_yr * YEAR

    # Set t0 and dt
    prob.set_val('t0', t0, units='gtoc_year')
    prob.set_val('dt', dt_yr, units='gtoc_year')
    prob.set_val('y0', 0.0, units='km')
    prob.set_val('z0', 0.0, units='km')

    # Get body positions and velocities at flyby times
    # Convert flyby times to seconds for get_state
    from gtoc13.constants import KMPDU, YEAR
    flyby_times_s = [t * YEAR for t in flyby_times]

    if guess_solution is not None:
        non_flyby_arcs = [arc for arc in guess_solution.arcs
                        if isinstance(arc, (PropagatedArc, ConicArc))]
    else:
        non_flyby_arcs = []

    # Track the last loaded arc's final time to ensure continuity
    last_loaded_time_s = None

    # Set initial guess for positions and velocities and controls for each arc
    for i in range(N):
        phase = prob.model.traj.phases._get_subsystem(f'arc_{i}')

        try:
            guess_arc = non_flyby_arcs[i]
        except IndexError:
            guess_arc = None

        if guess_arc is None:
            # If we don't have a guess for this arc,
            # guess smoothly interpolated positions from the initial
            # to the final times, and constant velocities based on
            # those times and distances.

            if i == 0:
                t_initial_s = t0 * YEAR
                t_final_s = flyby_times_s[0]
                r1 = np.array([-200.0, 0.0, 0.0]) * KMPDU
                r2 = bodies_data[bodies[0]].get_state(t_final_s).r
            else:
                # If we have a last loaded time, use it to ensure continuity
                if last_loaded_time_s is not None:
                    t_initial_s = last_loaded_time_s
                else:
                    t_initial_s = flyby_times_s[i-1]

                t_final_s = flyby_times_s[i]
                r1 = bodies_data[bodies[i-1]].get_state(t_initial_s).r
                r2 = bodies_data[bodies[i]].get_state(t_final_s).r

            print(f"Arc {i} (no guess): t_initial={t_initial_s/YEAR:.2f} yr, t_final={t_final_s/YEAR:.2f} yr")

            # Update dt for this arc to match actual time span
            dt_arc_s = t_final_s - t_initial_s
            prob.set_val('dt', dt_arc_s / YEAR, indices=[i], units='gtoc_year')

            v_guess = (r2 - r1) / dt_arc_s
        
            # time is connected to an output, no guess necessary
            phase.set_state_val('r', vals=[r1, r2], units='km')
            phase.set_state_val('v', vals=[v_guess, v_guess], units='km/s')
            
            if controls[i] == 0:
                phase.set_parameter_val('u_n', [0., 0., 0.], units='unitless')
            elif controls[i] == 1:
                u_n = np.array([1., 0., 0.])
                phase.set_control_val('u_n', [u_n, u_n], units='unitless')
        
        else:
            # Load the solution from the arc in the existing solution as a guess
            u = np.zeros(3)
            if isinstance(guess_arc, PropagatedArc):
                times_s = np.array([state.epoch for state in guess_arc.state_points])
                r_km = np.array([state.position for state in guess_arc.state_points])
                v_kms = np.array([state.velocity for state in guess_arc.state_points])
                u = np.array([state.control for state in guess_arc.state_points])
            elif isinstance(guess_arc, ConicArc):
                times_s = np.array([guess_arc.epoch_start, guess_arc.epoch_end])
                r_km = np.array([guess_arc.position_start, guess_arc.position_end])
                v_kms = np.array([guess_arc.velocity_start, guess_arc.velocity_end])
                u = np.zeros((2, 3))

            if i == 0:
                prob.set_val('t0', times_s[0], units='s')
                prob.set_val('y0', r_km[0, 1], units='km')
                prob.set_val('z0', r_km[0, 2], units='km')

            # Convert absolute times to phase-relative times
            # Phases in Dymos expect times relative to phase start (tau in [0, duration])
            t_initial_s = times_s[0]
            t_final_s = times_s[-1]

            # Track the final time of this loaded arc for continuity
            last_loaded_time_s = t_final_s

            print(f"Arc {i} (loaded): t_initial={t_initial_s/YEAR:.2f} yr, t_final={t_final_s/YEAR:.2f} yr")

            dt_arc = t_final_s - t_initial_s
            times_relative = times_s - t_initial_s  # Now in range [0, dt_arc]

            prob.set_val('dt', dt_arc, indices=[i], units='s')
            phase.set_state_val('r', vals=r_km, time_vals=times_relative, units='km')
            phase.set_state_val('v', vals=v_kms, time_vals=times_relative, units='km/s')
            if controls is not None:
                if 'u_n' in phase.control_options:
                    u_broadcast = np.broadcast_to(u, (len(times_relative), 3))
                    phase.set_control_val('u_n', vals=u_broadcast, time_vals=times_relative, units='unitless')
                elif 'u_n' in phase.parameter_options:
                    phase.set_parameter_val('u_n', [0, 0, 0], units='unitless')

    # Set the final velocity to a slightly perturbed version of the final arc
    # velocity. Setting them equal results in an infinite flyby radius.
    prob.set_val('v_end',
                 0.9 * phase.get_val('final_states:v', units='km/s'),
                 units='km/s')


def create_solution(prob, bodies, controls=None, filename=None):
    N = len(bodies)

    # Default controls to all 0 if not provided
    if controls is None:
        controls = [0] * N

    flyby_comp = prob.model.flyby_comp

    flyby_v_in = flyby_comp.get_val('v_in', units='km/s')
    flyby_v_out = flyby_comp.get_val('v_out', units='km/s')
    flyby_v_inf_in = flyby_comp.get_val('v_inf_in', units='km/s')
    flyby_v_inf_out = flyby_comp.get_val('v_inf_out', units='km/s')

    # event_pos = prob.get_val('event_pos', units='km')

    arcs = []
    arc_bodies = deque(maxlen=2)
    arc_bodies.append(-1)
    for i in range(N):
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



if __name__ == '__main__':
    # solve(bodies=[10], dt=[20.0], t0=0.0, num_nodes=20)
    bodies = [10, 9]
    dt = [20.,]

    prob = get_dymos_serial_solver_problem(bodies=bodies, num_nodes=20, warm_start=False, default_opt_prob=True)
    prob.setup()
    prob.set_val('dt', dt, units='gtoc_year')
    prob.run_driver()

    # prob.list_problem_vars(print_arrays=True)