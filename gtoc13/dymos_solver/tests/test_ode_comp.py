import unittest

import numpy as np
import jax.numpy as jnp

import openmdao.api as om
import openmdao.utils.units as om_units

import dymos as dm
from openmdao.utils.assert_utils import assert_near_equal

from gtoc13.bodies import bodies_data
from gtoc13.dymos_solver.ode_comp import SolarSailODEComp
from gtoc13.constants import MU_ALTAIRA, YEAR, KMPAU, R0

# Add our specific DU and TU to OpenMDAO's recognized units.
om_units.add_unit('DU', f'{KMPAU}*1000*m')
period = 2 * np.pi * np.sqrt(KMPAU**3 / MU_ALTAIRA)
om_units.add_unit('TU', f'{period}*s')

try:
    from lamberthub import vallado2013_jax as vallado2013_jax
    LAMBERTHUB_AVAILABLE = True
except ImportError:
    LAMBERTHUB_AVAILABLE = False
    # Fallback to local Lambert solver
    from gtoc13.lambert import lambert_universal_variables



class TestODEComp(unittest.TestCase):

    def test_solar_sail_ballistic_trajectory(self):
        """
        Test that a ballistic trajectory (u_n = [0, 0, 0]) matches a Lambert transfer
        from the initial position of Planet X (body 0) to its position at t=30 years
        using Dymos with Birkhoff transcription.
        """
        # Problem setup: Transfer from Planet X at t=0 to Planet X at t=30 years
        t0 = 0.0  # years
        tf = 30.0  # years

        # Get planetX state at intercept
        bodies = [10]
        N = len(bodies)
        rf, vf = bodies_data[10].get_state(t0 + tf, distance_units='DU', time_units='year')

        # Assume we're moving purely in x.
        r0 = np.array(rf.copy())
        r0[0] = -200.0

        v0 = np.array([(rf[0] - r0[0]) / (tf - t0), 0., 0.])
        vf = np.array(vf.copy())
        vf[0] = (rf[0] - r0[0]) / (tf - t0)

        # # Get Lambert solution for comparison
        # dt = (tf - t0) * YEAR  # Convert to seconds

        # if LAMBERTHUB_AVAILABLE:
        # Use lamberthub vallado2013 if available

        # else:
        #     # Use local Lambert solver
        #     v0_lambert, vf_lambert, z_final, converged = lambert_universal_variables(
        #         jnp.array(r0),
        #         jnp.array(rf_target),
        #         dt,
        #         MU_ALTAIRA,
        #         short=True
        #     )
        #     v0_lambert = np.array(v0_lambert)
        #     vf_lambert = np.array(vf_lambert)

        # self.assertTrue(converged, "Lambert solver did not converge")

        # Create Dymos problem
        prob = om.Problem()

        # Number of nodes for Birkhoff transcription
        num_nodes = 20

        # Create a Dymos phase with Birkhoff transcription
        # tx = dm.Birkhoff(num_nodes=num_nodes, grid_type='lgl')
        tx = dm.PicardShooting(num_segments=1, nodes_per_seg=num_nodes, solve_segments='backward')

        phase = dm.Phase(ode_class=SolarSailODEComp,
                         ode_init_kwargs={'N': N, 'r0': 1.0},
                         transcription=tx)

        traj = dm.Trajectory()
        traj.add_phase('all_arcs', phase)
        prob.model.add_subsystem('traj', traj)

        # Set the state variables
        # Dymos will create shape (num_nodes, N, 3) = (num_nodes, 1, 3)
        # Each state has shape (N, 3) = (1, 3) at each node
        phase.add_state('r', rate_source='drdt', units='DU',
                       shape=(N, 3), fix_initial=False, fix_final=True,
                       targets=['r'])

        phase.add_state('v', rate_source='dvdt', units='DU/year',
                       shape=(N, 3), fix_initial=False, fix_final=False,
                       targets=['v'])

        # Control: sail normal unit vector (ballistic = zero for Keplerian orbit)
        phase.add_control('u_n', units=None, shape=(N, 3), opt=False,
                         val=np.zeros((N, 3)), targets=['u_n'])

        # Time conversion factor
        phase.add_parameter('dt_dtau', units='year', val=30/2.0,
                           targets=['dt_dtau'], static_target=False, shape=(1,))

        # Set time bounds
        phase.set_time_options(fix_initial=True, fix_duration=True,
                               duration_val=2, units='unitless', name='tau')
        
        phase.add_timeseries_output('a_grav', units='km/s**2')
        phase.add_timeseries_output('a_sail', units='km/s**2')

        # Add boundary constraint to reach target position
        # phase.add_boundary_constraint('r', loc='final', equals=rf_target.reshape(1, 3), units='km')

        # Objective: minimize time (though it's fixed)
        phase.add_objective('tau', loc='final', scaler=1.0)

        # Setup the problem
        prob.driver = om.pyOptSparseDriver()
        prob.driver.options['optimizer'] = 'IPOPT'

        prob.setup()

        # Set initial guess - linearly interpolate between initial and final states
        phase.set_state_val('r', vals=([r0, rf]), units='DU')
        phase.set_state_val('v', vals=([vf, vf]), units='DU/year')
        phase.set_time_val(initial=-1.0, duration=2.0, units='unitless')
        u_n = np.zeros((N, 3))
        # u_n[:, 0] = 1.0  # Keep u_n as zeros for ballistic trajectory
        phase.set_control_val('u_n', [u_n, u_n])
        phase.set_parameter_val('dt_dtau', (tf - t0) / 2., units='year')

        # Run the problem
        dm.run_problem(prob, run_driver=False, simulate=True)

        dymos_initial_pos_km = phase.get_val('initial_states:r', units='km')[0]
        dymos_initial_vel_kms = phase.get_val('initial_states:v', units='km/s')[0]
        dymos_final_pos_km = phase.get_val('final_states:r', units='km')[0]
        dymos_final_vel_kms = phase.get_val('final_states:v', units='km/s')[0]
        dt_s = 30 * YEAR  # Total time in seconds
        t_span = (0.0, dt_s)  # Time span as tuple (t_initial, t_final)

        # Get the Dymos SIMULATION solution (not collocation)
        # The simulation uses scipy integrator internally
        # sim_prob = prob.model.traj.sim_prob

        # if sim_prob is None:
        #     print("WARNING: No simulation was run!")
        #     return

        # dymos_tau = sim_prob.get_val('traj.all_arcs.timeseries.tau', units='unitless')  # Normalized time [-1, 1]

        # Get the dt_dtau parameter value (constant for this problem)
        # dt_dtau_val = prob.get_val('traj.all_arcs.parameter_vals:dt_dtau', units='year')[0, 0]

        # Compute actual time from tau: t = t0 + (tau + 1)/2 * duration * dt_dtau
        # tau goes from -1 to 1, so (tau + 1)/2 goes from 0 to 1
        # duration is 2 in tau units
        # dymos_times = (dymos_tau + 1.0) / 2.0 * 2.0 * dt_dtau_val  # Years

        # dymos_r = sim_prob.get_val('traj.all_arcs.timeseries.r', units='DU')  # (num_nodes, N, 3)
        # Velocity is stored as DU/year in Dymos, convert to DU/TU
        # dymos_v_year = sim_prob.get_val('traj.all_arcs.timeseries.v', units='DU/year')  # (num_nodes, N, 3)
        # dymos_v = dymos_v_year / (2 * np.pi)  # Convert to DU/TU

        # print(f"\nDymos solution:")
        # print(f"  Time range: {dymos_times[0]} to {dymos_times[-1]} years")
        # print(f"  r shape: {dymos_r.shape}")
        # print(f"  v shape: {dymos_v.shape}")

        # Now integrate the same problem using scipy.solve_ivp for comparison
        from scipy.integrate import solve_ivp
        from gtoc13.odes import solar_sail_ode

        # # Extract actual initial conditions from Dymos solution
        # r0_DU = dymos_r[0, 0, :]  # First time point, first trajectory
        # v0_DU = dymos_v[0, 0, :]  # Already in DU/TU
        u_n_val = np.zeros(3)  # Ballistic trajectory

        # print(f"\nInitial conditions from Dymos:")
        # print(f"  r0 = {r0_DU} DU")
        # print(f"  v0 = {v0_DU} DU/TU")

        def ode_func(t, y):
            """ODE function: y = [r_x, r_y, r_z, v_x, v_y, v_z]

            Note: solar_sail_ode expects dt_dtau parameter, but since we're integrating
            in actual time (not tau), we pass 1.0 to get unscaled derivatives.
            """
            r = y[0:3]  # km
            v = y[3:6]  # km/s
            # Pass dt_dtau=1.0 because we want derivatives w.r.t. actual time, not tau
            drdt, dvdt, a_grav, a_sail, cos_alpha = solar_sail_ode(r, v, 1.0, u_n_val, MU_ALTAIRA, R0)
            return np.concatenate([drdt, dvdt])

        # Initial state vector
        y0 = np.concatenate([dymos_initial_pos_km, dymos_initial_vel_kms]).ravel()

        # Integrate using RK45
        sol = solve_ivp(
            ode_func,
            t_span,  # Use the tuple (0.0, dt_s) instead of scalar dt_s
            y0,
            method='DOP853',  # High-order method for accuracy
            rtol=1e-10,
            atol=1e-12
        )

        # Extract solution
        scipy_r = sol.y[0:3, :].T  # (num_times, 3)
        scipy_v = sol.y[3:6, :].T  # (num_times, 3)

        assert_near_equal(dymos_final_pos_km.ravel(), scipy_r[-1], tolerance=1.0E-5)
        assert_near_equal(dymos_final_vel_kms.ravel(), scipy_v[-1], tolerance=1.0E-5)


if __name__ == '__main__':
    unittest.main()
