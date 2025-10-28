import jax
import jax.numpy as jnp
import openmdao.api as om
import dymos as dm

from gtoc13 import gtoc13_ballistic_ode, gtoc13_ode, bodies_data, YEAR


def ballistic_ode(t, r, v):
    rv = jnp.hstack((r, v))
    args = tuple()
    return gtoc13_ballistic_ode(t, rv, args)


ballistic_ode_vmap = jax.vmap(ballistic_ode, in_axes=(0, 0, 0), out_axes=0)


class EphemComp(om.JaxExplicitComponent):
    """
    Ephemeris component that computes body position and velocity.

    Inputs:
        time: Time in seconds
        body_id: Body ID (discrete input)

    Outputs:
        r: Body position vector in km (3,)
        v: Body velocity vector in km/s (3,)
    """

    def setup(self):
        self.add_input('time', val=0.0, units='s', desc='Time')
        self.add_discrete_input('body_id', val=-1, desc='Target body ID')
        self.add_output('r', shape=(3,), units='km', desc='Body position')
        self.add_output('v', shape=(3,), units='km/s', desc='Body velocity')

        # Set check_partials options for finite differencing
        # Use relative step size because ephemeris data has limited resolution
        # A 1% relative step ensures adequate resolution for derivative checks
        self.set_check_partial_options(wrt='time', method='fd', step_calc='rel', step=0.01)

    def compute_primal(self, time, body_id):
        """Compute body ephemeris at given time."""
        # body_id is a discrete input, so it's always concrete (not a JAX tracer)
        # This means we can safely do the dictionary lookup
        body_id_int = int(body_id)
        body = bodies_data[body_id_int]

        # Body.get_state() is JAX-compatible and can be traced/differentiated
        state = body.get_state(time / YEAR)

        return state.r, state.v


class FlybyComp(om.JaxExplicitComponent):

    def setup(self):
        # Continuous input: time (scalar)
        self.add_input('r', shape=(3,), units='km', desc='cartesian position of the spacecraft')
        self.add_input('v', shape=(3,), units='km/s', desc='cartesian velocity of the spacecraft')
        self.add_input('r_body', shape=(3,), units='km', desc='cartesian position of the planet')
        self.add_input('v_body', shape=(3,), units='km/s', desc='cartesian velocity of the planet')

        self.add_output('r_err', shape=(3,), units='km', desc='cartesian difference in position')
        self.add_output('v_inf', shape=(3,), units='km/s', desc='cartesian difference in velocity')
        self.add_output('v_inf_mag', shape=(1,), units='km/s', desc='norm of the v_infinity vector')

        self.set_check_partial_options(wrt='*', form='forward', step_calc='rel_element')

    def compute_primal(self, r, v, r_body, v_body):
        r_err = r - r_body
        v_inf = v - v_body
        v_inf_mag = jnp.array([jnp.linalg.norm(v_inf)])  # Shape (1,) to match declaration
        return r_err, v_inf, v_inf_mag


class BallisticODEComp(om.JaxExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes')

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('t', shape=(nn,), units='s', tags=['dymos_state_target:'])
        self.add_input('r', shape=(nn, 3), units='km')
        self.add_input('v', shape=(nn, 3), units='km/s')
        self.add_output('rdot', shape=(nn, 3), units='km/s', tags=['dymos.state_rate_source:r'])
        self.add_output('vdot', shape=(nn, 3), units='km/s**2', tags=['dymos.state_rate_source:v'])

    def compute_primal(self, t, r, v):
        rv_dot = ballistic_ode_vmap(t, r, v)
        rdot = rv_dot[:, :3]
        vdot = rv_dot[:, -3:]
        return rdot, vdot


def solve_first_arc(target_id, time, use_sail=False):
    """
    Solve for the first arc to a target body.

    TODO: Make the instantiated OpenMDAO model persist so that we can just repeatly query it.
    """
    prob = om.Problem()

    traj = dm.Trajectory()
    phase = dm.Phase(ode_class=BallisticODEComp,
                     transcription=dm.PicardShooting(num_segments=1,
                                                     nodes_per_seg=21,
                                                     solve_segments='backward'))
    
    phase.set_time_options(fix_initial=True, fix_duration=True, units='s', targets=['t'])
    phase.set_state_options('r', fix_initial=False, fix_final=True, units='km')
    phase.set_state_options('v', fix_initial=False, fix_final=False, units='km/s')
    
    
    traj.add_phase('phase', phase, promotes_inputs=['t_duration'])
    
    prob.model.add_subsystem('target_body_ephem', EphemComp(),
                             promotes_inputs=['time', 'body_id'],
                             promotes_outputs=[('v', 'v_body'), ('r', 'r_body')]) 
    prob.model.add_subsystem('traj', traj,
                             promotes_inputs=[('t_duration', 'time')])
    prob.model.add_subsystem('flyby', FlybyComp())
    
    # Remove any ambiguities from promoted time values from different components
    # This sets the units and default value of time in the independent variables.
    prob.model.set_input_defaults('time', val=200.0, units='year')

    # Make connections that we need.
    prob.model.connect('r_body', ('flyby.r_body', 'traj.phase.final_states:r'))
    prob.model.connect('v_body', 'flyby.v_body')
    prob.model.connect('traj.phase.states:r', 'flyby.r', src_indices=om.slicer[-1, :])
    prob.model.connect('traj.phase.states:v', 'flyby.v', src_indices=om.slicer[-1, :])

    # We want to find the final velocity such that our initial position is [-200 AU, free, free]
    # and the initial velocity is [free, 0 km/s, 0 km/s ]
    # If we also target a flyby v_infinity and free up the time of flight, this results
    # in 4 free variables and 4 residuals.
    
    resids_comp = om.ExecComp(['resid_x = r_x - 200',
                               'resid_vx0 = vx0 - 0.0',
                               'resid_vy0 = vy0 - 0.0',
                               'resid_vinf = v_inf_mag - v_inf_target'],
                               resid_x={'units': 'AU'},
                               resid_vx0={'units': 'km/s'},
                               resid_vy0={'units': 'km/s'},
                               resid_vinf={'units': 'km/s'},
                               v_inf_target={'units': 'km/s'})

    implicit_var_comp = om.InputResidsComp()
    implicit_var_comp.add_input('resid_x', shape_by_conn=True, units='AU')
    implicit_var_comp.add_input('resid_vx0', shape_by_conn=True, units='km/s')
    implicit_var_comp.add_input('resid_vy0', shape_by_conn=True, units='km/s')
    implicit_var_comp.add_input('resid_vinf', shape_by_conn=True, units='km/s')

    implicit_var_comp.add_output('vf', shape=(3,), units='km/s')
    implicit_var_comp.add_output('time', units='year')

    prob.model.add_subsystem('resids_comp', resids_comp, promotes_outputs=['*'], promotes_inputs=['v_inf_target'])
    prob.model.add_subsystem('implicit_var_comp', implicit_var_comp, promotes_inputs=['*'])

    prob.model.connect('implicit_var_comp.time', 'time')
    prob.model.connect('implicit_var_comp.vf', 'traj.phase.final_states:v')
    # Connect initial state values to residual component
    prob.model.connect('traj.phase.initial_states:r', 'resids_comp.r_x', src_indices=om.slicer[:, 0])  # x-component
    prob.model.connect('traj.phase.initial_states:v', 'resids_comp.vx0', src_indices=om.slicer[:, 0])  # vx component
    prob.model.connect('traj.phase.initial_states:v', 'resids_comp.vy0', src_indices=om.slicer[:, 1])  # vy component
    prob.model.connect('flyby.v_inf_mag', 'resids_comp.v_inf_mag')

    prob.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
    prob.model.linear_solver = om.DirectSolver()

    prob.setup(force_alloc_complex=True)

    phase.set_time_val(initial=0.0, units='year')
    phase.set_state_val('r',
                        vals=[[-2.9919574e+10, 7.4798935e+09, 0.0000000e+00],
                              [-1.9776317e+10, 5.0909875e+09, -1.1567910e+09]],
                        units='km')
    phase.set_state_val('v',
                        vals=[[64.26946, -15.136306, -7.3315625],
                              [64.303246, -15.144867,  -7.330577]], units='km/s' )
    
    # Set the discrete input value of the target body id
    prob.model.set_val('body_id', 10)
    prob.model.set_val('time', 5.0, units='year')
    # prob.model.set_val('implicit_var_comp.vf', [64.303246, -15.144867,  -7.330577], units='km/s')
    prob.model.set_val('target_body_ephem.body_id', 10)
    prob.model.set_val('v_inf_target', 10, units='km/s')
    
    dm.run_problem(prob, simulate=True, make_plots=True)

    prob.check_partials(compact_print=True, method='cs')

    return

    # Extract solution values
    initial_time = float(prob.model.get_val('traj.phase.t_initial', units='year')[0])
    duration = float(prob.model.get_val('time', units='year')[0])
    final_time = initial_time + duration
    initial_pos = prob.model.get_val('traj.phase.initial_states:r', units='km').flatten()
    initial_vel = prob.model.get_val('traj.phase.initial_states:v', units='km/s').flatten()
    final_pos = prob.model.get_val('traj.phase.final_states:r', units='km').flatten()
    final_vel = prob.model.get_val('traj.phase.final_states:v', units='km/s').flatten()
    v_inf = prob.model.get_val('flyby.v_inf', units='km/s').flatten()
    v_inf_mag = float(prob.model.get_val('flyby.v_inf_mag', units='km/s')[0])
    r_err = prob.model.get_val('flyby.r_err', units='km').flatten()
    r_err_mag = float(jnp.linalg.norm(r_err))

    # Print summary table
    print("\n" + "="*80)
    print("TRAJECTORY SOLUTION SUMMARY")
    print("="*80)
    print(f"\nInitial Time:  {initial_time:12.6f} years")
    print(f"Final Time:    {final_time:12.6f} years")
    print(f"Duration:      {final_time - initial_time:12.6f} years")
    print("\n" + "-"*80)
    print("INITIAL STATE")
    print("-"*80)
    print(f"Position (km):   [{initial_pos[0]:14.6e}, {initial_pos[1]:14.6e}, {initial_pos[2]:14.6e}]")
    print(f"Velocity (km/s): [{initial_vel[0]:13.6f}, {initial_vel[1]:13.6f}, {initial_vel[2]:13.6f}]")
    print("\n" + "-"*80)
    print("FINAL STATE")
    print("-"*80)
    print(f"Position (km):   [{final_pos[0]:14.6e}, {final_pos[1]:14.6e}, {final_pos[2]:14.6e}]")
    print(f"Velocity (km/s): [{final_vel[0]:13.6f}, {final_vel[1]:13.6f}, {final_vel[2]:13.6f}]")
    print("\n" + "-"*80)
    print("FLYBY INFORMATION")
    print("-"*80)
    print(f"V_infinity (km/s): [{v_inf[0]:13.6f}, {v_inf[1]:13.6f}, {v_inf[2]:13.6f}]")
    print(f"V_infinity mag:     {v_inf_mag:13.6f} km/s")
    print(f"\nPosition error (km): [{r_err[0]:14.6e}, {r_err[1]:14.6e}, {r_err[2]:14.6e}]")
    print(f"Position error mag:  {r_err_mag:14.6e} km ({r_err_mag/1.496e8:14.6e} AU)")
    print("="*80 + "\n")


if __name__ == '__main__':

    # ======================================================================
    # Propagating initial conditions with diffrax (Keplerian dynamics)
    # ======================================================================
    # Propagating from t=0.0 s to t=157788000.0 s (5.0 years)
    # Initial state: r = [-2.9919574e+10  7.4798935e+09  0.0000000e+00] km
    #             v = [ 64.26946   -15.136306   -7.3315625] km/s

    # Propagated final state:
    # Position: [-1.9776317e+10  5.0909875e+09 -1.1567910e+09] km
    # Velocity: [ 64.303246 -15.144867  -7.330577] km/s

    # Lambert solution final state:
    # Position: [-1.977639e+10  5.091001e+09 -1.156783e+09] km
    # Velocity: [ 64.30325  -15.144878  -7.330576] km/s

    # Errors (Propagated - Lambert):
    # Position error: [ 71680. -13312.  -8064.] km
    # Position error magnitude: 7.335026e+04 km (4.903162e-04 AU)
    # Velocity error: [-7.6293945e-06  1.1444092e-05 -9.5367432e-07] km/s
    # Velocity error magnitude: 1.378711e-05 km/s

    solve_first_arc(target_id=10, time=5, use_sail=False)
    
