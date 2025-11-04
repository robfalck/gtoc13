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
        self.options.declare('bodies', types=Sequence,
                             desc='The bodies to be visited, in sequence')

    def setup(self):
        N = len(self.options['bodies'])
        self.add_input('t0', shape=(1,), val=0.0, units='year', desc='Initial time')
        self.add_input('dt', shape=(N,), units='year', desc='Transfer time for each event.')
        self.add_input('y0', shape=(1,), units='AU', desc='Initial y-position')
        self.add_input('z0', shape=(1,), units='AU', desc='Initial z-position')
        self.add_input('vx0', shape=(1,), units='AU/year', desc='Initial x-velocity')
        self.add_output('event_pos', shape=(N + 1, 3), units='AU', desc='Positions at the starting time and each body intercept.')
        self.add_output('event_vel', shape=(N + 1, 3), units='AU/year', desc='Velocities at the starting time and each body intercept.')
        self.add_output('dt_dtau', shape=(N,), units='year', desc='Time span vs tau for each arc.')
        self.add_output('times', shape=(N + 1,), units='year', desc='Times of events')

        self._ELEMENTS = jnp.zeros((N, 6))

        for i, body_id in enumerate(self.options['bodies']):
            self._ELEMENTS = self._ELEMENTS.at[i, :].set(bodies_data[body_id].elements.to_array())

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

    def compute_primal(self, t0, dt, y0, z0, vx0):
        """Compute body ephemeris at given time and append to initial state."""
        times = jnp.concatenate((t0, jnp.cumsum(dt)))

        body_pos, body_vel = elements_to_pos_vel(self._ELEMENTS, times[1:], self._MU)

        initial_pos = jnp.array([[-200, y0[0], z0[0]]])
        initial_vel = jnp.array([[vx0[0], 0.0, 0.0]])

        event_pos = jnp.vstack((initial_pos, body_pos))
        event_vel = jnp.vstack((initial_vel, body_vel))

        dt_dtau = dt / 2.0

        return event_pos, event_vel, dt_dtau, times
