import jax.numpy as jnp

import openmdao.api as om

from gtoc13.constants import MU_ALTAIRA, KMPDU, SPTU


class EnergyComp(om.JaxExplicitComponent):

    def initialize(self):
        self.options.declare('N', types=(int,), desc='Number of bodies in the solution')

    def setup(self):
        # The final inertial velocity after the last flyby
        self.add_input('r_end', shape=(1, 3), units='km')
        self.add_input('v_end', shape=(1, 3), units='km/s')
        self.add_output('E_end', shape=(1,), units='km**2/s**2')
        self.add_output('hz_end', shape=(1,),units='km**2/s')

    def compute_primal(self, r_end, v_end):
        # Flatten to 1D vectors for dot product
        r_vec = r_end.ravel()
        v_vec = v_end.ravel()

        # Specific orbital energy: E = v²/2 - μ/r
        E_end = jnp.array([jnp.dot(v_vec, v_vec) / 2.0 - MU_ALTAIRA / jnp.linalg.norm(r_vec)])
        hz_end = jnp.cross(r_end, v_end)[:, 2]
        return E_end, hz_end