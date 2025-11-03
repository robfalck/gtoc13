from collections.abc import Sequence

import jax
import jax.numpy as jnp

import openmdao.api as om

from gtoc13.bodies import bodies_data
from gtoc13.constants import MU_ALTAIRA, KMPDU
from gtoc13.astrodynamics import elements_to_pos_vel


class EphemComp(om.JaxExplicitComponent):
    """
    Ephemeris component that computes body position and velocity.

    Note that n is the number of bodies to be visited.

    Options:
        units: One of 'km/s' or 'DU/TU'
        bodies: A sequence of ints that are the bodies to be visited.

    Inputs:
        time: Time in seconds (n + 1,)
        body_id: Body ID (discrete input)

    Outputs:
        r: Body position vector in km (n, 3)
        v: Body velocity vector in km/s (n,)
    """
    def initialize(self):
        self.options.declare('units', values=('km/s', 'DU/TU'), 
                             desc='km/s for metric or DU/TU for canonical units')
        self.options.declare('bodies', types=Sequence,
                             desc='The bodies to be visited, in sequence')

    def setup(self):
        n = len(self.options['bodies'])
        self.add_input('times', shape=(n + 1,), units='year', desc='Time of each encounter, including the start time in element 0.')
        self.add_output('body_pos', shape=(n, 3), units='AU', desc='Body position')
        self.add_output('body_vel', shape=(n, 3), units='AU/year', desc='Body velocity')

        self._ELEMENTS = jnp.zeros((n, 6))

        for i, body_id in enumerate(self.options['bodies']):
            self._ELEMENTS = self._ELEMENTS.at[i, :].set(bodies_data[body_id].elements.to_array())

        if self.options['units'] == 'DU/TU':
            self._MU = 1.0
            self._ELEMENTS = self._ELEMENTS.at[:, 0].set(self._ELEMENTS[:, 0] / KMPDU)
        else:
            self._MU = MU_ALTAIRA
        
        # Set check_partials options for finite differencing
        # Use relative step size because ephemeris data has limited resolution
        # A 1% relative step ensures adequate resolution for derivative checks
        self.set_check_partial_options(wrt='time', method='fd', step_calc='rel', step=0.01)

    def get_self_statics(self):
        """
        self.ELEMENTS is effectively a static input to the compute_primal
        method. By declaring that here, jax will successfully handle ange
        changes that might happen to it, redoing just-in-time compilation
        as necessary.
        """
        return (self._ELEMENTS, self._MU)

    def compute_primal(self, times):
        """Compute body ephemeris at given time."""

        body_pos, body_vel = elements_to_pos_vel(self._ELEMENTS, times[1:], self._MU)

        return body_pos, body_vel
