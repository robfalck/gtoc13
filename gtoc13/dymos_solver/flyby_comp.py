"""
OpenMDAO component for computing flyby defects using JAX.
"""
from collections.abc import Sequence

import jax.numpy as jnp
import openmdao.api as om

from gtoc13.astrodynamics import flyby_defects_in_out
from gtoc13.bodies import bodies_data


class FlybyDefectComp(om.JaxExplicitComponent):
    """
    Compute flyby defects for trajectory optimization.

    This component wraps the flyby_defects_in_out function to compute constraint
    defects that must be satisfied for valid gravity assist flybys.

    For N flybys, this component computes:
    - v_inf_mag_defect: Difference in incoming and outgoing v-infinity magnitudes
    - h_p_defect: Parabolic altitude constraint defect

    Both defects should be constrained to zero (or negative for h_p_defect) for
    a feasible trajectory.

    Options
    -------
    bodies : Sequence[int]
        Sequence of body IDs for each flyby (length N)

    Inputs
    ------
    v_in : ndarray (N, 3)
        Incoming inertial velocities at each flyby (km/s)
    v_out : ndarray (N, 3)
        Outgoing inertial velocities at each flyby (km/s)
    v_body : ndarray (N, 3)
        Body inertial velocities at each flyby (km/s)

    Outputs
    -------
    v_inf_mag_defect : ndarray (N,)
        V-infinity magnitude defects (should be zero)
    h_p_defect : ndarray (N,)
        Altitude constraint defects (should be negative for valid flyby)
    """

    def initialize(self):
        """Declare options."""
        self.options.declare('bodies', types=Sequence,
                             desc='Body IDs for each flyby')

    def setup(self):
        """Set up inputs and outputs."""
        N = len(self.options['bodies'])

        # Inputs: velocities at each flyby
        self.add_input('v_in', shape=(N, 3), units='km/s',
                       desc='Incoming inertial velocities')
        self.add_input('v_out', shape=(N, 3), units='km/s',
                       desc='Outgoing inertial velocities')
        self.add_input('v_body', shape=(N, 3), units='km/s',
                       desc='Body inertial velocities')

        # Outputs: defects for each flyby
        self.add_output('v_inf_in', shape=(N, 3), units='km/s',
                        desc='Incoming V-infinity vector')
        self.add_output('v_inf_out', shape=(N, 3), units='km/s',
                        desc='Outgoing V-infinity vector')
        self.add_output('v_inf_mag_defect', shape=(N,), units='km/s',
                        desc='V-infinity magnitude defects')
        self.add_output('h_p_defect', shape=(N,), units='unitless',
                        desc='Altitude constraint defects (parabolic)')

        # Store body parameters as static data
        self._MU_BODY = jnp.zeros(N)
        self._R_BODY = jnp.zeros(N)

        for i, body_id in enumerate(self.options['bodies']):
            body = bodies_data[body_id]
            self._MU_BODY = self._MU_BODY.at[i].set(body.mu)
            self._R_BODY = self._R_BODY.at[i].set(body.radius)

        # Convert to tuples for hashability
        self._MU_BODY = tuple(self._MU_BODY.tolist())
        self._R_BODY = tuple(self._R_BODY.tolist())

    def get_self_statics(self):
        """
        Return static body parameters for JIT compilation.

        These are not component inputs but static values that define
        the body properties for each flyby.
        """
        return (self._MU_BODY, self._R_BODY)

    def compute_primal(self, v_in, v_out, v_body):
        """
        Compute flyby defects.

        Parameters
        ----------
        v_in : array (N, 3)
            Incoming velocities
        v_out : array (N, 3)
            Outgoing velocities
        v_body : array (N, 3)
            Body velocities

        Returns
        -------
        v_inf_mag_defect : array (N,)
            V-infinity magnitude defects
        h_p_defect : array (N,)
            Altitude constraint defects
        """
        # Convert tuples back to JAX arrays
        mu_body = jnp.array(self._MU_BODY)
        r_body = jnp.array(self._R_BODY)

        N = len(mu_body)

        # Initialize output arrays
        v_inf_in = jnp.zeros((N, 3))
        v_inf_out = jnp.zeros((N, 3))
        v_inf_mag_defect = jnp.zeros(N)
        h_p_defect = jnp.zeros(N)

        # Compute defects for each flyby
        for i in range(N):
            v_inf_in_i, v_inf_out_i, v_inf_defect, h_p_def = flyby_defects_in_out(
                v_in[i, :],
                v_out[i, :],
                v_body[i, :],
                mu_body[i],
                r_body[i]
            )
            v_inf_in = v_inf_in.at[i].set(v_inf_in_i)
            v_inf_out = v_inf_out.at[i].set(v_inf_out_i)
            v_inf_mag_defect = v_inf_mag_defect.at[i].set(v_inf_defect)
            h_p_defect = h_p_defect.at[i].set(h_p_def)

        return v_inf_in, v_inf_out, v_inf_mag_defect, h_p_defect
