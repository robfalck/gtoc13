from gtoc13.dymos_solver.ode_comp import SolarSailODEComp, SolarSailRadialControlODEComp
from gtoc13.dymos_solver.ballistic_prop_comp import BallisticPropagationComp

import openmdao.api as om
import dymos as dm
import numpy as np


def get_phase(num_nodes, control):
    tx = dm.Birkhoff(num_nodes=num_nodes, grid_type='lgl')

    ode_cls = SolarSailRadialControlODEComp if control == 'r' else SolarSailODEComp

    phase = dm.Phase(ode_class=ode_cls,
                     transcription=tx)

    phase.add_state('r', rate_source='drdt', units='DU',
                    shape=(3,), fix_initial=False, fix_final=False,
                    targets=['r'], ref=10.0, defect_ref=10.0)

    phase.add_state('v', rate_source='dvdt', units='DU/TU',
                    shape=(3,), fix_initial=False, fix_final=False,
                    targets=['v'],
                    ref=1.0, defect_ref=1.0)

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


def get_ballistic_phase(num_nodes):

    phase = dm.AnalyticPhase(ode_class=BallisticPropagationComp,
                             num_nodes=num_nodes,
                             ode_init_kwargs={'use_jit': False})

    phase.add_state('r', units='km', shape=(3,))
    phase.add_state('v', units='km/s', shape=(3,))
    phase.add_parameter('r0', units='km')
    phase.add_parameter('v0', units='km/s')
    phase.set_time_options(units='s')

    return phase


if __name__ == '__main__':

    p = om.Problem()
    traj = p.model.add_subsystem('traj', dm.Trajectory())
    phase = get_ballistic_phase(num_nodes=10)
    traj.add_phase('ballistic_arc', phase)
    

    p.setup()

    phase.set_time_val(0.0, 25, units='gtoc_year')
    phase.set_parameter_val('r0', [-1.34378052e+08, -4.33550636e+08,  3.72593516e+07], units='km')
    phase.set_parameter_val('v0', [-5.54081412, -21.83816121,  -0.43231826], units='km/s')

    p.run_model()

    print(phase.get_val('timeseries.r', units='km'))
    print(phase.get_val('timeseries.v', units='km/s'))

# === Lambert Solution Test ===
# Transfer from body 4 (Hoth) to body 1005 (Asteroid_1005)
# Transfer time: 25.00 years
# Initial position: [-1.34378052e+08 -4.33550636e+08  3.72593516e+07]
# Final position: [-8.58202793e+07 -6.03180105e+08 -1.38351112e+08]
# Lambert initial velocity: [ -5.54081412 -21.83816121  -0.43231826]
# Lambert final velocity: [ 3.44349658 17.99927189  2.46876791]

# Initial state verification:
#   Position error: 0.00e+00 km (should be ~0)
#   Velocity error: 0.00e+00 km/s (should be ~0)

# Final state verification:
#   Propagated final position: [-8.58202793e+07 -6.03180105e+08 -1.38351112e+08]
#   Target final position: [-8.58202793e+07 -6.03180105e+08 -1.38351112e+08]
#   Position error: 7.71e-04 km
#   Propagated final velocity: [ 3.44349658 17.99927189  2.46876791]
#   Lambert final velocity: [ 3.44349658 17.99927189  2.46876791]
#   Velocity error: 1.42e-11 km/s