from typing import Sequence

import numpy as np
import openmdao.api as om
import openmdao.utils.units as om_units
import dymos as dm

from gtoc13.solution import GTOC13Solution
from gtoc13.dymos_solver.ephem_comp import EphemComp
from gtoc13.dymos_solver.v_concat_comp import VConcatComp
from gtoc13.dymos_solver.flyby_comp import FlybyDefectComp
from gtoc13.dymos_solver.energy_comp import EnergyComp

from gtoc13.dymos_solver.ode_comp import SolarSailODEComp
from gtoc13.dymos_solver.ode_comp import SolarSailODEComp
from gtoc13.constants import MU_ALTAIRA, KMPAU, R0

# Add our specific DU and TU to OpenMDAO's recognized units.
om_units.add_unit('DU', f'{KMPAU}*1000*m')
period = 2 * np.pi * np.sqrt(KMPAU**3 / MU_ALTAIRA)
om_units.add_unit('TU', f'{period}*s')

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

    phase = dm.Phase(ode_class=SolarSailODEComp,
                     ode_init_kwargs={'N': N, 'r0': 1.0},
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
    phase.add_control('u_n', units='unitless', shape=(N, 3), opt=True,
                        val=np.ones((N, 3)), targets=['u_n'])
    if phase.control_options['u_n']['opt']:
        phase.add_path_constraint('u_n_norm', equals=1.0)
        phase.add_path_constraint('cos_alpha', lower=0.0)

    # Time conversion factor
    phase.add_parameter('dt_dtau', units='year', val=30/2.0, opt=False,
                        targets=['dt_dtau'], static_target=False, shape=(N,))

    # Set time bounds
    phase.set_time_options(fix_initial=True, fix_duration=True,
                           duration_val=2, units='unitless', name='tau')
    
    phase.add_timeseries_output('a_grav', units='km/s**2')
    phase.add_timeseries_output('a_sail', units='km/s**2')


    prob.model.add_design_var('dt', lower=0.0, upper=200) 
    prob.model.add_design_var('y0', units='DU')
    prob.model.add_design_var('z0', units='DU')
    prob.model.add_design_var('v_end', units='DU/TU')
    prob.model.add_constraint('flyby_comp.v_inf_mag_defect', equals=0.0, units='DU/TU')
    prob.model.add_constraint('flyby_comp.h_p_defect', upper=0.0, ref=1000.0)
    # prob.model.add_design_var('vx0', lower=0.0, units='DU/TU')
    # prob.model.add_design_var('v_final', units='DU/TU')
    phase.add_boundary_constraint('v', loc='initial', indices=[1, 2], equals=0.0)
    # phase.add_objective('tau', loc='final')
    prob.model.add_objective('E_end')

    prob.driver = om.pyOptSparseDriver(optimizer='IPOPT')
    prob.driver.opt_settings['print_level'] = 5
    prob.driver.opt_settings['tol'] = 1.0E-5

    phase.set_simulate_options(times_per_seg=num_nodes)

    prob.setup()

    prob.set_val('t0', t0, units='year')
    prob.set_val('dt', dt, units='year')
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
    u_n[:, 0] = 1.0  # Keep u_n as zeros for ballistic trajectory
    phase.set_control_val('u_n', [u_n, u_n])
    phase.set_parameter_val('dt_dtau', np.asarray(dt) / 2., units='year')

    # # Run the problem
    dm.run_problem(prob, run_driver=True, simulate=False)

    # # prob.run_model()
    
    t0_s = prob.get_val('t0', units='s')
    tau = phase.get_val('timeseries.tau', units='unitless')

    body_id = 0
    flag = 1

    # prob = traj.sim_prob

    r = prob.get_val('traj.all_arcs.timeseries.r', units='km')
    v = prob.get_val('traj.all_arcs.timeseries.v', units='km/s')
    u_n = prob.get_val('traj.all_arcs.timeseries.u_n', units='unitless')
    dt_dtau_s = prob.get_val('traj.all_arcs.parameter_vals:dt_dtau', units='s')

    for arc_i in range(N):
        for node_j in range(num_nodes):
            r_ij = r[node_j, arc_i, :]
            v_ij = v[node_j, arc_i, :]
            u_n_ij = u_n[node_j, arc_i, :]
            t_s_j = t0_s + dt_dtau_s[arc_i] * (tau[node_j] + 1.0)

            print(f'{body_id} {flag} {t_s_j[0]:.15e} {r_ij[0]:.15e} {r_ij[1]:.15e} {r_ij[2]:.15e} {v_ij[0]:.15e} {v_ij[1]:.15e} {v_ij[2]:.15e} {u_n_ij[0]:.15e} {u_n_ij[1]:.15e} {u_n_ij[2]:.15e}')

    # prob.model.list_vars(print_arrays=True)

if __name__ == '__main__':
    solve(bodies=[10], dt=[5.0], t0=0.0, num_nodes=10)