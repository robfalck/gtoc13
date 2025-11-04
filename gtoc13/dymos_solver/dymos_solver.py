from typing import Sequence

import numpy as np
import openmdao.api as om
import openmdao.utils.units as om_units
import dymos as dm

from gtoc13.solution import GTOC13Solution
from gtoc13.dymos_solver.ephem_comp import EphemComp

from gtoc13.dymos_solver.ode_comp import SolarSailODEComp
from gtoc13.dymos_solver.ode_comp import SolarSailODEComp
from gtoc13.constants import MU_ALTAIRA, KMPAU

# Add our specific DU and TU to OpenMDAO's recognized units.
om_units.add_unit('DU', f'{KMPAU}*1000*m')
period = 2 * np.pi * np.sqrt(KMPAU**3 / MU_ALTAIRA)
om_units.add_unit('TU', f'{period}*s')

def solve(bodies: Sequence[int], times: Sequence[float]) -> GTOC13Solution:
    """
    Parameters
    ----------
    bodies : Sequence[int]
        The bodies that make up the solution, in order of visit.
    times : Sequence[float]
        The approximate encounter year of each event. times has 
        one more element than bodies because the initial time
        of the trajectory at the starting plane is included.

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
    dt_year = np.diff(times)
    num_nodes=20

    prob = om.Problem()

    prob.model.add_subsystem('ephem', EphemComp(bodies=bodies))

    tx = dm.Birkhoff(num_nodes=num_nodes, grid_type='lgl')
    # tx = dm.PicardShooting(num_segments=1, nodes_per_seg=num_nodes, solve_segments='forward')

    phase = dm.Phase(ode_class=SolarSailODEComp,
                     ode_init_kwargs={'N': N, 'r0': 1.0},
                     transcription=tx)

    phase.add_state('r', rate_source='drdt', units='DU',
                    shape=(N, 3), fix_initial=True, fix_final=True,
                    targets=['r'])

    phase.add_state('v', rate_source='dvdt', units='DU/year',
                    shape=(N, 3), fix_initial=False, fix_final=False,
                    targets=['v'])

    # Control: sail normal unit vector (ballistic = zero for Keplerian orbit)
    phase.add_control('u_n', units=None, shape=(N, 3), opt=False,
                        val=np.zeros((N, 3)), targets=['u_n'])

    # Time conversion factor
    phase.add_parameter('dt_dtau', units='year', val=30/2.0, opt=False,
                        targets=['dt_dtau'], static_target=False, shape=(N,))

    # Set time bounds
    phase.set_time_options(fix_initial=True, fix_duration=True,
                           duration_val=2, units='unitless', name='tau')
    
    phase.add_timeseries_output('a_grav', units='km/s**2')
    phase.add_timeseries_output('a_sail', units='km/s**2')


    traj = dm.Trajectory()
    traj.add_phase('all_arcs', phase)
    prob.model.add_subsystem('traj', traj)

    prob.model.connect('ephem.event_pos', 'traj.all_arcs.initial_states:r', src_indices=om.slicer[:-1, ...])
    prob.model.connect('ephem.event_pos', 'traj.all_arcs.final_states:r', src_indices=om.slicer[1:, ...])
    prob.model.connect('ephem.dt_dtau', 'traj.all_arcs.parameters:dt_dtau')

    prob.add_design_var('dt', lower=0.0, upper=200)
    phase.add_objective('tau', loc='final')

    prob.setup()

    prob.set_val('ephem.times', times, units='year')

    # Set initial guess - linearly interpolate between initial and final states
    # phase.set_state_val('r', vals=([r0, rf]), units='DU')
    # phase.set_state_val('v', vals=([vf, vf]), units='DU/year')
    phase.set_time_val(initial=-1.0, duration=2.0, units='unitless')
    u_n = np.zeros((N, 3))
    # # u_n[:, 0] = 1.0  # Keep u_n as zeros for ballistic trajectory
    phase.set_control_val('u_n', [u_n, u_n])
    phase.set_parameter_val('dt_dtau', dt_year / 2., units='year')

    # # Run the problem
    # dm.run_problem(prob, run_driver=False, simulate=True)

    prob.run_model()


if __name__ == '__main__':
    solve(bodies=[10], times=[0.0, 30.0])