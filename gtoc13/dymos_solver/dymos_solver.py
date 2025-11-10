from pathlib import Path
from typing import Sequence

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

from gtoc13.dymos_solver.ode_comp import SolarSailVectorizedODEComp, SolarSailODEComp


def create_solution(prob, bodies):
    N = len(bodies)
    
    flyby_comp = prob.model.flyby_comp
    traj = prob.model.traj
    phase = traj.phases.all_arcs
    sim_prob = traj.sim_prob

    # sol, controls = sim_prob.model.traj.phases.all_arcs.integrator.solve_ivp(method='RK45', dense_output=False, atol=1.0E-12, rtol=1.0E-12)

    # r_sim = traj.sim_prob.get_val('traj.all_arcs.timeseries.r', units='km')
    # v_sim = traj.sim_prob.get_val('traj.all_arcs.timeseries.v', units='km/s')
    # u_n_sim = traj.sim_prob.get_val('traj.all_arcs.timeseries.u_n', units='unitless')
    # u_n_norm_sim = traj.sim_prob.get_val('traj.all_arcs.timeseries.u_n_norm', units='unitless')
    # dt_dtau_s_sim = traj.sim_prob.get_val('traj.all_arcs.parameter_vals:dt_dtau', units='s')
    # tau_sim = traj.sim_prob.get_val('traj.all_arcs.timeseries.tau', units='unitless')

    # r_sim = sol.y.T[:, :3] * KMPDU
    # v_sim = sol.y.T[:, -3:] * KMPDU / SPTU

    # print(controls)
    # exit(0)

    t0_s = prob.get_val('t0', units='s')[0] 
    times = prob.get_val('times', units='s')

    r = prob.get_val('traj.all_arcs.timeseries.r', units='km')
    v = prob.get_val('traj.all_arcs.timeseries.v', units='km/s')
    u_n = prob.get_val('traj.all_arcs.timeseries.u_n', units='unitless')
    u_n_norm = prob.get_val('traj.all_arcs.timeseries.u_n_norm', units='unitless')
    dt_dtau_s = prob.get_val('traj.all_arcs.parameter_vals:dt_dtau', units='s')
    tau = phase.get_val('timeseries.tau', units='unitless')

    # print('Simulation position error at tf')
    # print(r[-1, :, ...])
    # print(r_sim[-1, :, ...])
    # print(r_sim[-1, :, ...] - r[-1, :, ...])

    # print('Simulation velocity error at tf')
    # print(v[-1, :, ...])
    # print(v_sim[-1, :, ...])
    # print(v_sim[-1, :, ...] - v[-1, :, ...])

    # # Get times in seconds using GTOC13's gtoc_year unit definition
    # times = prob.get_val('times', units='s')
    flyby_v_in = flyby_comp.get_val('v_in', units='km/s')
    flyby_v_out = flyby_comp.get_val('v_out', units='km/s')
    flyby_v_inf_in = flyby_comp.get_val('v_inf_in', units='km/s')
    flyby_v_inf_out = flyby_comp.get_val('v_inf_out', units='km/s')

    # event_pos = prob.get_val('event_pos', units='km')

    arcs = []
    for i in range(N):
        # Use the simulation outputs to write the arc
        # dt_dtau_s has shape (1, N) so we need to extract the i-th element correctly
        t_i = times[i] + dt_dtau_s[0, i] * (tau + 1.0).flatten()
        r_i = r[:, i, :]
        v_i = v[:, i, :]
        u_n_i = u_n[:, i, :]

        print('t_i')
        print(t_i)
        print('r_i')
        print(r_i)
        print('v_i')
        print(v_i)
        print('u_n_i')

        # Add the i-th propagated arc 
        arcs.append(PropagatedArc.create(epochs=t_i, positions=r_i,
                                         velocities=v_i, controls=u_n_i))
        
        # Add the i-th flyby arc
        t_flyby_i = prob.get_val('times', units='s')[i + 1]        

        # For now assume we never repeat a body more than 12 times, so 
        # each flyby is for science.
        arcs.append(FlybyArc.create(body_id=bodies[i],
                                    epoch=t_flyby_i,
                                    position=r_i[-1, ...],
                                    velocity_in=flyby_v_in[i],
                                    velocity_out=flyby_v_out[i],
                                    v_inf_in=flyby_v_inf_in[i],
                                    v_inf_out=flyby_v_inf_out[i],
                                    is_science=True))
    
    solution = GTOC13Solution(arcs=arcs,
                              comments=[])

    # Find the next available solution filename
    solutions_dir = Path(__file__).parent.parent.parent / 'solutions'
    solutions_dir.mkdir(exist_ok=True)

    index = 1
    while (solutions_dir / f'dymos_solution_{index}.txt').exists():
        index += 1

    solution_file = solutions_dir / f'dymos_solution_{index}.txt'
    solution.write_to_file(solution_file, precision=11)
    print(f"Solution written to {solution_file}")

    # Extract objective values for plot
    E_end = prob.get_val('E_end')[0]

    # Get the objective value (if it exists)
    try:
        obj_value = prob.get_val('obj')[0]
    except:
        obj_value = E_end  # Fallback to E_end if obj doesn't exist

    # Create plot and save it
    plot_file = solutions_dir / f'dymos_solution_{index}.png'
    solution.plot(show_bodies=True, save_path=plot_file,
                  E_end=E_end, obj_value=obj_value)

    return solution, solution_file

def get_phase(num_nodes):
    tx = dm.Birkhoff(num_nodes=num_nodes, grid_type='lgl')
    # tx = dm.PicardShooting(num_segments=1, nodes_per_seg=num_nodes, solve_segments='forward')

    phase = dm.Phase(ode_class=SolarSailODEComp,
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

    # # Control: sail normal unit vector (ballistic = zero for Keplerian orbit)
    # phase.add_control('u_n', units='unitless', shape=(3,), opt=False,
    #                     val=np.ones((N, 3)), targets=['u_n'])
    # if phase.control_options['u_n']['opt']:
    #     phase.add_path_constraint('u_n_norm', equals=1.0)
    #     phase.add_path_constraint('cos_alpha', lower=0.0)

    # Set time options
    # The fix_initial here is really a bit of a misnomer.
    # They're not design variables, and we can therefore connect
    # t_initial and t_duration to upstream outputs.
    phase.set_time_options(fix_initial=True,
                           fix_duration=True,
                           units='TU', )
    
    phase.add_timeseries_output('a_grav', units='km/s**2')
    phase.add_timeseries_output('a_sail', units='km/s**2')
    phase.add_timeseries_output('u_n_norm', units='unitless')

    return phase

def get_dymos_serial_solver_problem(bodies: Sequence[int], num_nodes=20, warm_start=False,
                                    default_opt_prob=True):
    N = len(bodies)

    if isinstance(num_nodes, int):
        _num_nodes = N * [num_nodes]
    else:
        _num_nodes = num_nodes

    prob = om.Problem()

    prob.model.add_subsystem('ephem', EphemComp(bodies=bodies), promotes=['*'])

    traj = dm.Trajectory()
    prob.model.add_subsystem('traj', traj)

    for i in range(N):
        phase = get_phase(num_nodes=_num_nodes[i])
        traj.add_phase(f'arc_{i}', phase)

        prob.model.connect('event_pos', f'traj.arc_{i}.initial_states:r', src_indices=om.slicer[2*i, ...])
        prob.model.connect('event_pos', f'traj.arc_{i}.final_states:r', src_indices=om.slicer[2*i+1, ...])

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
                             promotes_outputs=['E_end'])
    prob.model.connect('event_pos', 'r_end', src_indices=om.slicer[-1, ...])

    prob.model.add_subsystem('score_comp',
                             ScoreComp(bodies=bodies),
                             promotes_outputs=['J'])

    prob.model.connect('event_pos', 'score_comp.body_pos',
                       src_indices=om.slicer[1:, ...])




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
    prob.model.add_constraint('flyby_comp.v_inf_mag_defect', equals=0.0, units='DU/TU')
    
    # # Periapsis Altitude Constraint for Each flyby
    # # Note that this is a quadratic equation that is negative between the
    # # allowable flyby normalized altitude values, so it just has to be negative.
    # # TODO: Test this for massless body flybys
    prob.model.add_constraint('flyby_comp.h_p_defect', upper=0.0, ref=1000.0)

    # # Make sure the final time is in the allowable span.
    prob.model.add_constraint('times', indices=[-1], upper=199.999, units='gtoc_year')

    # # A constraint on in y and z components of the initial velocity vector
    prob.model.traj.phases.arc_0.add_boundary_constraint('v', loc='initial', indices=[1, 2], equals=0.0)

    # # TODO: Add a path constraint for perihelion distance.

    # #
    # # OBJECTIVE
    # #

    # # Minimize specific orbital energy after the last flyby
    # # TODO: Convert to problem objective.

    # prob.model.add_objective('E_end')
    prob.model.add_objective('J', ref=-1.0)

    prob.driver = om.pyOptSparseDriver(optimizer='IPOPT')
    prob.driver.declare_coloring()  # Take advantage of sparsity.
    prob.driver.opt_settings['print_level'] = 5
    prob.driver.opt_settings['tol'] = 1.0E-6

    # Gradient-based autoscaling
    # prob.driver.opt_settings['nlp_scaling_method'] = 'gradient-based'

    # Step-size selection
    # prob.driver.opt_settings['alpha_for_y'] = 'safer-min-dual-infeas'

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
    # prob.driver.opt_settings['bound_mult_init_method'] = 'mu-based'

    # How IPOPT changes its barrier parameter (mu) over time.
    # This problem seems to work much better with the default 'adaptive'
    # prob.driver.opt_settings['mu_strategy'] = 'monotone'

    return prob


def get_dymos_vectorized_solver_problem(bodies: Sequence[int], num_nodes=20, warm_start=False) -> om.Problem:
    """
    Parameters
    ----------
    bodies : Sequence[int]
        The bodies that make up the solution, in order of visit.
    num_nodes : int
        The number of nodes to be used in each trajectory arc.
    warm_start : bool
        If True, enable IPOPT warm-start settings for faster convergence when
        starting from a good initial guess.

    Returns
    -------
    solution : Solution
        The GTOC solution instance for the posed problem.

    Raises
    ------
    ValueError
        If dymos is unable to find a solution.
    """
    N = len(bodies)

    prob = om.Problem()

    prob.model.add_subsystem('ephem', EphemComp(bodies=bodies), promotes=['*'])

    tx = dm.Birkhoff(num_nodes=num_nodes, grid_type='lgl')
    # tx = dm.PicardShooting(num_segments=1, nodes_per_seg=num_nodes, solve_segments='forward')

    phase = dm.Phase(ode_class=SolarSailVectorizedODEComp,
                     ode_init_kwargs={'N': N, 'r0': R0},
                     transcription=tx)

    traj = dm.Trajectory()
    traj.add_phase('all_arcs', phase, promotes_inputs=['parameters:dt_dtau', 'initial_states:*', 'final_states:*'])
    prob.model.add_subsystem('traj', traj, promotes_inputs=[('parameters:dt_dtau', 'dt_dtau'), 'initial_states:*', 'final_states:*'])

    prob.model.add_subsystem('v_out_comp', VConcatComp(N=N), promotes_inputs=['v_end', 'initial_states:v'], promotes_outputs=['flyby_v_out'])
    
    prob.model.set_input_defaults('initial_states:v', units='km/s', val=np.ones((N, 3))) 

    prob.model.connect('event_pos', 'initial_states:r', src_indices=om.slicer[:-1, ...])
    prob.model.connect('event_pos', 'final_states:r', src_indices=om.slicer[1:, ...])

    prob.model.add_subsystem('flyby_comp', FlybyDefectComp(bodies=bodies),
                             promotes_inputs=[('v_in', 'final_states:v')])
    prob.model.connect('flyby_v_out', 'flyby_comp.v_out')
    prob.model.set_input_defaults('final_states:v', units='km/s', val=np.ones((N, 3)))
    prob.model.connect('body_vel', 'flyby_comp.v_body')

    prob.model.add_subsystem('energy_comp', EnergyComp(),
                             promotes_inputs=['v_end', 'r_end'],
                             promotes_outputs=['E_end', 'obj'])
    prob.model.connect('event_pos', 'r_end', src_indices=om.slicer[-1, ...])

    phase.add_state('r', rate_source='drdt', units='DU',
                    shape=(N, 3), fix_initial=True, fix_final=True,
                    targets=['r'])

    phase.add_state('v', rate_source='dvdt', units='DU/TU',
                    shape=(N, 3), fix_initial=False, fix_final=False,
                    targets=['v'], lower=-100, upper=100)

    # Control: sail normal unit vector (ballistic = zero for Keplerian orbit)
    phase.add_control('u_n', units='unitless', shape=(N, 3), opt=False,
                        val=np.ones((N, 3)), targets=['u_n'])
    if phase.control_options['u_n']['opt']:
        phase.add_path_constraint('u_n_norm', equals=1.0)
        phase.add_path_constraint('cos_alpha', lower=0.0)

    # Time conversion factor
    phase.add_parameter('dt_dtau', units='gtoc_year', val=30/2.0, opt=False,
                        targets=['dt_dtau'], static_target=False, shape=(N,))

    # Set time bounds
    phase.set_time_options(fix_initial=True, fix_duration=True,
                           duration_val=2, units='unitless', name='tau')
    
    phase.add_timeseries_output('a_grav', units='km/s**2')
    phase.add_timeseries_output('a_sail', units='km/s**2')
    phase.add_timeseries_output('u_n_norm', units='unitless')

    # #
    # # DESIGN VARIABLES
    # # 

    # # Start time
    # prob.model.add_design_var('t0', lower=0.0, units='gtoc_year')

    # # Times between flyby events
    # prob.model.add_design_var('dt', lower=0.0, upper=200, units='gtoc_year') 

    # # Start plane position
    # prob.model.add_design_var('y0', units='DU')
    # prob.model.add_design_var('z0', units='DU')

    # # Outgoing inertial velocity after last flyby
    # prob.model.add_design_var('v_end', units='DU/TU')

    # #
    # # CONSTRAINTS
    # #

    # # V-infinity magnitude difference before/after each flyby
    # prob.model.add_constraint('flyby_comp.v_inf_mag_defect', equals=0.0, units='DU/TU')
    
    # # Periapsis Altitude Constraint for Each flyby
    # # Note that this is a quadratic equation that is negative between the
    # # allowable flyby normalized altitude values, so it just has to be negative.
    # # TODO: Test this for massless body flybys
    # prob.model.add_constraint('flyby_comp.h_p_defect', upper=0.0, ref=1000.0)

    # # Make sure the final time is in the allowable span.
    # prob.model.add_constraint('times', indices=[-1], upper=199.999, units='gtoc_year')

    # # A constraint on in y and z components of the initial velocity vector
    # phase.add_boundary_constraint('v', loc='initial', indices=[1, 2], equals=0.0)

    # # TODO: Add a path constraint for perihelion distance.

    # #
    # # OBJECTIVE
    # #

    # # Minimize specific orbital energy after the last flyby
    # # TODO: Convert to problem objective.

    # prob.model.add_objective('E_end')

    prob.driver = om.pyOptSparseDriver(optimizer='IPOPT')
    prob.driver.declare_coloring()  # Take advantage of sparsity.
    prob.driver.opt_settings['print_level'] = 5
    prob.driver.opt_settings['tol'] = 1.0E-6

    # Gradient-based autoscaling
    # prob.driver.opt_settings['nlp_scaling_method'] = 'gradient-based'

    # Step-size selection
    # prob.driver.opt_settings['alpha_for_y'] = 'safer-min-dual-infeas'

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
    # prob.driver.opt_settings['bound_mult_init_method'] = 'mu-based'

    # How IPOPT changes its barrier parameter (mu) over time.
    # This problem seems to work much better with the default 'adaptive'
    # prob.driver.opt_settings['mu_strategy'] = 'monotone'

    phase.set_simulate_options(times_per_seg=50, atol=1.0E-12, rtol=1.0E-12)

    return prob, phase



def solve(bodies: Sequence[int], dt: Sequence[float], t0=0.0, num_nodes=20) -> GTOC13Solution:
    """
    Parameters
    ----------
    bodies : Sequence[int]
        The bodies that make up the solution, in order of visit.
    dt : Sequence[float]
        The time duration (years) of between each body encounter,
        plus the initial arc (the first element).
    t0 : float
        The initial time at the starting plane (years).
    num_nodes : int
        The number of nodes to be used in each trajectory arc.

    Returns
    -------
    solution : Solution
        The GTOC solution instance for the posed problem.

    Raises
    ------
    ValueError
        If dymos is unable to find a solution.
    """
    N = len(bodies)

    prob = om.Problem()

    prob.model.add_subsystem('ephem', EphemComp(bodies=bodies), promotes=['*'])

    tx = dm.Birkhoff(num_nodes=num_nodes, grid_type='lgl')
    # tx = dm.PicardShooting(num_segments=1, nodes_per_seg=num_nodes, solve_segments='forward')

    phase = dm.Phase(ode_class=SolarSailVectorizedODEComp,
                     ode_init_kwargs={'N': N, 'r0': R0},
                     transcription=tx)

    traj = dm.Trajectory()
    traj.add_phase('all_arcs', phase, promotes_inputs=['parameters:dt_dtau', 'initial_states:*', 'final_states:*'])
    prob.model.add_subsystem('traj', traj, promotes_inputs=[('parameters:dt_dtau', 'dt_dtau'), 'initial_states:*', 'final_states:*'])

    prob.model.add_subsystem('v_out_comp', VConcatComp(N=N), promotes_inputs=['v_end', 'initial_states:v'], promotes_outputs=['flyby_v_out'])
    
    prob.model.set_input_defaults('initial_states:v', units='km/s', val=np.ones((N, 3))) 

    prob.model.connect('event_pos', 'initial_states:r', src_indices=om.slicer[:-1, ...])
    prob.model.connect('event_pos', 'final_states:r', src_indices=om.slicer[1:, ...])

    prob.model.add_subsystem('flyby_comp', FlybyDefectComp(bodies=bodies),
                             promotes_inputs=[('v_in', 'final_states:v')])
    prob.model.connect('flyby_v_out', 'flyby_comp.v_out')
    prob.model.set_input_defaults('final_states:v', units='km/s', val=np.ones((N, 3)))
    prob.model.connect('body_vel', 'flyby_comp.v_body')

    prob.model.add_subsystem('energy_comp', EnergyComp(),
                             promotes_inputs=['v_end', 'r_end'],
                             promotes_outputs=['E_end'])
    prob.model.connect('event_pos', 'r_end', src_indices=om.slicer[-1, ...])


    phase.add_state('r', rate_source='drdt', units='DU',
                    shape=(N, 3), fix_initial=True, fix_final=True,
                    targets=['r'])

    phase.add_state('v', rate_source='dvdt', units='DU/TU',
                    shape=(N, 3), fix_initial=False, fix_final=False,
                    targets=['v'], lower=-100, upper=100)

    # Control: sail normal unit vector (ballistic = zero for Keplerian orbit)
    phase.add_control('u_n', units='unitless', shape=(N, 3), opt=False,
                        val=np.ones((N, 3)), targets=['u_n'])
    if phase.control_options['u_n']['opt']:
        phase.add_path_constraint('u_n_norm', equals=1.0)
        phase.add_path_constraint('cos_alpha', lower=0.0)

    # Time conversion factor
    phase.add_parameter('dt_dtau', units='gtoc_year', val=30/2.0, opt=False,
                        targets=['dt_dtau'], static_target=False, shape=(N,))

    # Set time bounds
    phase.set_time_options(fix_initial=True, fix_duration=True,
                           duration_val=2, units='unitless', name='tau')
    
    phase.add_timeseries_output('a_grav', units='km/s**2')
    phase.add_timeseries_output('a_sail', units='km/s**2')
    phase.add_timeseries_output('u_n_norm', units='unitless')

    #
    # DESIGN VARIABLES
    # 

    # Start time
    prob.model.add_design_var('t0', lower=0.0, units='gtoc_year')

    # Times between flyby events
    prob.model.add_design_var('dt', lower=0.0, upper=200, units='gtoc_year') 

    # Start plane position
    prob.model.add_design_var('y0', units='DU')
    prob.model.add_design_var('z0', units='DU')

    # Outgoing inertial velocity after last flyby
    prob.model.add_design_var('v_end', units='DU/TU')

    #
    # CONSTRAINTS
    #

    # V-infinity magnitude difference before/after each flyby
    prob.model.add_constraint('flyby_comp.v_inf_mag_defect', equals=0.0, units='DU/TU')
    
    # Periapsis Altitude Constraint for Each flyby
    # Note that this is a quadratic equation that is negative between the
    # allowable flyby normalized altitude values, so it just has to be negative.
    # TODO: Test this for massless body flybys
    prob.model.add_constraint('flyby_comp.h_p_defect', upper=0.0, ref=1000.0)

    # Make sure the final time is in the allowable span.
    prob.model.add_constraint('times', indices=[-1], upper=199.999, units='gtoc_year')

    # A constraint on in y and z components of the initial velocity vector
    phase.add_boundary_constraint('v', loc='initial', indices=[1, 2], equals=0.0)

    # TODO: Add a path constraint for perihelion distance.

    #
    # OBJECTIVE
    #

    # Minimize specific orbital energy after the last flyby
    # TODO: Convert to problem objective.

    prob.model.add_objective('E_end')

    prob.driver = om.pyOptSparseDriver(optimizer='IPOPT')
    prob.driver.declare_coloring()  # Take advantage of sparsity.
    prob.driver.opt_settings['print_level'] = 5
    prob.driver.opt_settings['tol'] = 1.0E-6

    # Warm-start settings for faster convergence when starting from good initial guess
    if warm_start:
        prob.driver.opt_settings['warm_start_init_point'] = 'yes'
        prob.driver.opt_settings['warm_start_bound_push'] = 1e-9
        prob.driver.opt_settings['warm_start_bound_frac'] = 1e-9
        prob.driver.opt_settings['warm_start_slack_bound_push'] = 1e-9
        prob.driver.opt_settings['warm_start_slack_bound_frac'] = 1e-9
        prob.driver.opt_settings['warm_start_mult_bound_push'] = 1e-9
        prob.driver.opt_settings['mu_init'] = 1e-4

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

    phase.set_simulate_options(times_per_seg=50, atol=1.0E-12, rtol=1.0E-12)

    prob.setup()

    prob.set_val('t0', t0, units='gtoc_year')
    prob.set_val('dt', dt, units='gtoc_year')
    prob.set_val('v_end', [6.43e+01, -7.38e-03, 1.678e-03], units='km/s')

    # Set initial guess - linearly interpolate between initial and final states
    r0 = np.array([[-29919574016.0, 7479893504.0, 0.000000000000]])
    v0 = np.array([[64.269462585449 -15.136305809021 -7.331562519073]])

    rf = np.array([[-19776389120.0, 5091000832.0, -1156782976.0]])
    vf = np.array([[64.303253173828 -15.144878387451 -7.330575942993]])


    phase.set_state_val('r', vals=([r0, r0]), units='km')
    phase.set_state_val('v', vals=([v0, v0]), units='km/s')
    phase.set_time_val(initial=-1.0, duration=2.0, units='unitless')
    u_n = np.zeros((N, 3))
    if phase.control_options['u_n']['opt']:
        u_n[:, 0] = 1.0  # Keep u_n as zeros for ballistic trajectory
    phase.set_control_val('u_n', [u_n, u_n])
    phase.set_parameter_val('dt_dtau', np.asarray(dt) / 2., units='gtoc_year')

    # # Run the problem
    # prob.find_feasible()
    dm.run_problem(prob, run_driver=True, simulate=False)


    times = prob.model.get_val('times', units='s')
    r = prob.model.get_val('traj.all_arcs.timeseries.r', units='km')
    planet_pos = bodies_data[bodies[0]].get_state(times[-1]).r

    # print('After solving the propagation ends at')
    # print(r)
    # print(times)
    # print(f'Body {bodies[0]} is at')
    # print(planet_pos)
    # print(r[-1, ...] - planet_pos)

    solution = create_solution(prob, bodies)
    return solution


if __name__ == '__main__':
    # solve(bodies=[10], dt=[20.0], t0=0.0, num_nodes=20)
    prob = get_dymos_serial_solver_problem(bodies=[10], num_nodes=20, warm_start=False, default_opt_prob=True)
    prob.setup()
    prob.set_val('dt', [20.], units='gtoc_year')
    prob.run_driver()

    prob.list_problem_vars(print_arrays=True)