import jax.numpy as jnp

import openmdao.api as om


class VConcatComp(om.JaxExplicitComponent):

    def initialize(self):
        self.options.declare('N', types=(int,), desc='Number of bodies in the solution')

    def setup(self):
        # The final inertial velocity after the last flyby
        self.add_input('v_final', shape=(1, 3), units='km/s')

        # The initial velocity of all but the first arc. These are the inertial
        # outgoing velocities for each flyby calc, except the final one.
        self.add_input('initial_states:v', primal_name='arc_initial_vel', shape=(self.options['N'], 3), units='km/s')
        self.add_output('flyby_v_out', shape=(self.options['N'], 3), units='km/s')
    
    def compute_primal(self, v_final, arc_initial_vel):
        flyby_v_out = jnp.vstack((arc_initial_vel[1:], v_final))
        return flyby_v_out