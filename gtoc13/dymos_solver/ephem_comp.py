from collections.abc import Sequence

import jax
import jax.numpy as jnp

import openmdao.api as om

from gtoc13.bodies import bodies_data
from gtoc13.constants import MU_ALTAIRA, KMPDU, YEAR
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
        self.add_input('t0', shape=(1,), val=0.0, units='gtoc_year', desc='Initial time')
        self.add_input('dt', shape=(N,), units='gtoc_year', desc='Transfer time for each event.')
        self.add_input('y0', shape=(1,), units='DU', desc='Initial y-position')
        self.add_input('z0', shape=(1,), units='DU', desc='Initial z-position')
        # self.add_input('vx0', shape=(1,), units='DU/gtoc_year', desc='Initial x-velocity')
        self.add_output('event_pos', shape=(N + 1, 3), units='km', desc='Positions at the starting time and each body intercept.')
        self.add_output('body_vel', shape=(N, 3), units='km/s', desc='Intertial velocity of each body at intercept (km/s)')
        self.add_output('dt_dtau', shape=(N,), units='gtoc_year', desc='Time span vs tau for each arc.')
        self.add_output('times', shape=(N + 1,), units='gtoc_year', desc='Times of events')
        self.add_output('dt_out', shape=(N,), units='gtoc_year', desc='dt echoed as an output to be connected downstream.')

        self._ELEMENTS = jnp.zeros((N, 6))

        for i, body_id in enumerate(self.options['bodies']):
            self._ELEMENTS = self._ELEMENTS.at[i, :].set(bodies_data[body_id].elements.to_array())
        self._ELEMENTS = tuple(tuple(row) for row in self._ELEMENTS.tolist())

        self._MU = MU_ALTAIRA

    def get_self_statics(self):
        """
        self._ELEMENTS is effectively a static input to the compute_primal
        method. By declaring that here, jax will successfully handle any
        changes that might happen to it, redoing just-in-time compilation
        as necessary.

        Note: We convert arrays to tuples to make them hashable for OpenMDAO.
        """
        # Convert JAX array to nested tuples so it's hashable
        return (self._ELEMENTS, self._MU)

    def compute_primal(self, t0, dt, y0, z0):
        """Compute body ephemeris at given time and append to initial state.

        Parameters
        ----------
        t0 : array
            Initial time
        dt : array
            Time deltas for each arc
        y0 : array
            Initial y position
        z0 : array
            Initial z position
        ELEMENTS_tuple : tuple of tuples
            Orbital elements for each body (from get_self_statics, as nested tuples)
        MU : float
            Gravitational parameter (from get_self_statics)
        """
        # Convert hashable tuple back to JAX array for computation
        ELEMENTS = jnp.array(self._ELEMENTS)

        # Compute event times: t0, t0+dt[0], t0+dt[0]+dt[1], ...
        # Note: times are in gtoc_year units (as specified in the input declaration)
        times = jnp.concatenate((t0, t0 + jnp.cumsum(dt)))

        # Convert times from gtoc_year to seconds for elements_to_pos_vel
        # Since gtoc_year is defined as YEAR * s, and our inputs have units='gtoc_year',
        # the values in compute_primal are the numeric values in gtoc_year units.
        # But elements_to_pos_vel expects seconds, so we multiply by YEAR.
        times_seconds = YEAR * times[1:]

        # Using MU in km**3/s**2, make sure to pass times in seconds, not years.
        body_pos, body_vel = elements_to_pos_vel(ELEMENTS, times_seconds, self._MU)

        # Convert initial conditions from DU to km
        # Note: inputs are in DU, outputs must be in km
        # 1 DU = KMPAU km (defined in constants.py)
        initial_pos_km = jnp.array([[-200 * KMPDU, y0[0] * KMPDU, z0[0] * KMPDU]])

        event_pos = jnp.vstack((initial_pos_km, body_pos))

        dt_dtau = dt / 2.0

        dt_out = dt

        return event_pos, body_vel, dt_dtau, times, dt_out


class EphemCompNoStartPlane(om.JaxExplicitComponent):
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
        self.add_input('t0', shape=(1,), val=0.0, units='gtoc_year', desc='Initial time')
        self.add_input('dt', shape=(1,), units='gtoc_year', desc='Transfer time for each event.')

        self.add_output('body_pos', shape=(N, 3), units='km', desc='Positions at the starting time and each body intercept.')
        self.add_output('body_vel', shape=(N, 3), units='km/s', desc='Intertial velocity of each body at intercept (km/s)')
        self.add_output('times', shape=(N,), units='gtoc_year', desc='Times of events')
        self.add_output('dt_out', shape=(1,), units='gtoc_year', desc='dt echoed as an output to be connected downstream.')

        self._ELEMENTS = jnp.zeros((N, 6))

        for i, body_id in enumerate(self.options['bodies']):
            self._ELEMENTS = self._ELEMENTS.at[i, :].set(bodies_data[body_id].elements.to_array())
        self._ELEMENTS = tuple(tuple(row) for row in self._ELEMENTS.tolist())

        self._MU = MU_ALTAIRA

    def get_self_statics(self):
        """
        self._ELEMENTS is effectively a static input to the compute_primal
        method. By declaring that here, jax will successfully handle any
        changes that might happen to it, redoing just-in-time compilation
        as necessary.

        Note: We convert arrays to tuples to make them hashable for OpenMDAO.
        """
        # Convert JAX array to nested tuples so it's hashable
        return (self._ELEMENTS, self._MU)

    def compute_primal(self, t0, dt):
        """Compute body ephemeris at given time and append to initial state.

        Parameters
        ----------
        t0 : array
            Initial time
        dt : array
            Time deltas for each arc
        y0 : array
            Initial y position
        z0 : array
            Initial z position
        ELEMENTS_tuple : tuple of tuples
            Orbital elements for each body (from get_self_statics, as nested tuples)
        MU : float
            Gravitational parameter (from get_self_statics)
        """
        # Convert hashable tuple back to JAX array for computation
        ELEMENTS = jnp.array(self._ELEMENTS)

        # Compute event times: t0, t0+dt[0], t0+dt[0]+dt[1], ...
        # Note: times are in gtoc_year units (as specified in the input declaration)
        times = jnp.concatenate((t0, t0 + dt))

        # Convert times from gtoc_year to seconds for elements_to_pos_vel
        # Since gtoc_year is defined as YEAR * s, and our inputs have units='gtoc_year',
        # the values in compute_primal are the numeric values in gtoc_year units.
        # But elements_to_pos_vel expects seconds, so we multiply by YEAR.
        times_s = YEAR * times

        # Using MU in km**3/s**2, make sure to pass times in seconds, not years.
        body_pos, body_vel = elements_to_pos_vel(ELEMENTS, times_s, self._MU)

        dt_out = dt

        return body_pos, body_vel, times, dt_out