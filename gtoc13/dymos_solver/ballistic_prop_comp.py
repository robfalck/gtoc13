"""
OpenMDAO JAX component for ballistic (Keplerian) trajectory propagation.

This component uses the analytic f-g propagation method to compute positions
and velocities at specified times given initial conditions.
"""
import jax.numpy as jnp
import openmdao.api as om

from gtoc13.analytic import propagate_ballistic
from gtoc13.constants import MU_ALTAIRA


class BallisticPropagationComp(om.JaxExplicitComponent):
    """
    Propagates a ballistic trajectory from initial conditions to specified times.

    This component uses the analytic Lagrange f-g method to compute the trajectory,
    which is exact for two-body Keplerian motion and much faster than numerical
    integration.

    Inputs
    ------
    r0 : array (3,)
        Initial position vector [x, y, z] in km
    v0 : array (3,)
        Initial velocity vector [vx, vy, vz] in km/s
    times : array (num_nodes,)
        Times at which to evaluate the trajectory in seconds.
        times[0] is treated as t0 (the time corresponding to r0, v0)

    Outputs
    -------
    r : array (num_nodes, 3)
        Position vectors at each time in km
    v : array (num_nodes, 3)
        Velocity vectors at each time in km/s
    """

    def initialize(self):
        """Declare options."""
        self.options.declare('num_nodes', types=int, desc='Number of time nodes')

    def setup(self):
        """Set up inputs and outputs."""
        nn = self.options['num_nodes']

        # Inputs
        # For Dymos AnalyticPhase, parameters are scalars or vectors per node
        # r0 and v0 should be shape (1, 3) to work with Dymos
        self.add_input('r0', shape=(1, 3), units='km',
                       desc='Initial position vector',
                       tags=['dymos.static_target'])
        self.add_input('v0', shape=(1, 3), units='km/s',
                       desc='Initial velocity vector',
                       tags=['dymos.static_target'])
        self.add_input('time', shape=(nn,), units='s',
                       desc='Times at which to evaluate the trajectory')

        # Outputs
        self.add_output('r', shape=(nn, 3), units='km',
                        desc='Position vectors at each time')
        self.add_output('v', shape=(nn, 3), units='km/s',
                        desc='Velocity vectors at each time')

    def compute_primal(self, r0, v0, time):
        """
        Compute the ballistic trajectory.

        JAX will automatically compute derivatives of this function.

        Parameters
        ----------
        r0 : jnp.ndarray, shape (1, 3)
            Initial position vector (reshaped to (3,) internally)
        v0 : jnp.ndarray, shape (1, 3)
            Initial velocity vector (reshaped to (3,) internally)
        times : jnp.ndarray, shape (num_nodes,)
            Times at which to evaluate

        Returns
        -------
        r : jnp.ndarray, shape (num_nodes, 3)
            Position vectors
        v : jnp.ndarray, shape (num_nodes, 3)
            Velocity vectors
        """
        print(r0, v0)
        print(time)
        # Reshape r0 and v0 from (1, 3) to (3,) for propagate_ballistic
        r0_vec = r0.ravel()
        v0_vec = v0.ravel()

        # Propagate the trajectory using the analytic method
        # Uses MU_ALTAIRA by default
        r, v = propagate_ballistic(r0_vec, v0_vec, time)

        return r, v
