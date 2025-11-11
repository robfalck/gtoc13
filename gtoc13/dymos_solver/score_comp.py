from collections.abc import Sequence

import jax
from jax import jit
import jax.numpy as jnp

import openmdao.api as om

from gtoc13 import bodies_data



class ScoreComp(om.JaxExplicitComponent):
    """
    Ephemeris component that computes body position and velocity.

    Note that n is the number of bodies to be visited.

    Options:
        bodies: A sequence of ints that are the bodies to be visited.

    Inputs:
        body_pos: Body position vectors at each flyby in km (N, 3)
        v_inf: V_infinity of each flyby (N, 3)

    Outputs:
        J : GTOC objective to be maximized (1,)
    """
    def initialize(self):
        self.options.declare('bodies', types=Sequence,
                             desc='The bodies to be visited, in sequence')

    def setup(self):
        N = len(self.options['bodies'])
        self.add_input('body_pos', shape=(N, 3), units='km', desc='Positions of bodies at times of flyby.')
        self.add_input('v_inf', shape=(N, 3), units='km/s', desc='Flyby v-inf of each flyby.')
        self.add_output('J', shape=(1,), units='unitless', desc='GTOC13 objective to be maximized.')

        # Store as tuples for hashability (will convert to arrays in compute_primal)
        self._body_ids = tuple(self.options['bodies'])
        self._body_weights = tuple([bodies_data[id].weight for id in self.options['bodies']])

    def get_self_statics(self):
        """
        self._ELEMENTS is effectively a static input to the compute_primal
        method. By declaring that here, jax will successfully handle any
        changes that might happen to it, redoing just-in-time compilation
        as necessary.

        Note: We convert arrays to tuples to make them hashable for OpenMDAO.
        """
        # Return tuples so they're hashable
        return (self._body_ids, self._body_weights)

    def compute_primal(self,
                       body_pos: jnp.ndarray,
                       v_inf: jnp.ndarray
    ) -> float:
        """
        Compute the objective function J from section 3 of the GTOC13 problem statement.

        This implements:
        J = b * c * Sum_k w_k Sum_i (S(r_hat,k,i) x F(v_inf,k,i))

        where b=1 and c=1 (grand tour and time bonuses not considered here).

        Inputs:
            body_pos: heliocentric position of each body at each flyby (shape: (N, 3))
                    where N is the total number of flybys
            v_inf: hyperbolic excess velocity vectors at each flyby (shape: (N, 3))
            body_ids: body ID for each flyby (shape: (N,))
            body_weights: scientific weight for each flyby (shape: (N,))
                        can be obtained by mapping body_ids to weights from bodies_data

        Outputs:
            J: total score (scalar)

        Note:
            This function is JAX-differentiable with respect to body_pos and v_inf.
        """
        # Convert tuples back to JAX arrays for computation
        body_ids = jnp.array(self._body_ids)
        body_weights = jnp.array(self._body_weights)

        # Normalize body positions to get unit vectors
        # Shape: (N, 3)
        r_norm = jnp.linalg.norm(body_pos, axis=1, keepdims=True)
        r_hat = body_pos / r_norm

        # Compute seasonal penalty for all flybys
        # Shape: (N,)
        #S = seasonal_penalty_vectorized(r_hat, body_ids)
        S = 1.0

        # Compute velocity magnitude for each flyby
        # Shape: (N,)
        v_inf_mag = jnp.linalg.norm(v_inf, axis=1)

        # Compute flyby velocity penalty for all flybys
        # Shape: (N,)
        F = jax.vmap(flyby_velocity_penalty)(v_inf_mag)

        # Compute contribution from each flyby: w_k * S * F
        # Shape: (N,)
        contributions = body_weights * S * F

        # Replace any NaN or Inf contributions with 0
        contributions = jnp.where(jnp.isfinite(contributions), contributions, 0.0)

        # Sum all contributions
        # Return as shape (1,) array to match declared output shape
        J = jnp.sum(contributions)
        J = jnp.reshape(J, (1,))

        # Final safety check
        J = jnp.where(jnp.isfinite(J), J, 0.0)

        return J


def flyby_velocity_penalty(v_infinity: float) -> float:
    """
    Compute flyby velocity penalty term F.

    Args:
        v_infinity: hyperbolic excess velocity magnitude (km/s)

    Returns:
        F: velocity penalty factor
    """
    F = 0.2 + jnp.exp(-v_infinity / 13.0) / (1.0 + jnp.exp(-5.0 * (v_infinity - 1.5)))
    return F


def seasonal_penalty_vectorized(r_hat: jnp.ndarray, body_ids: jnp.ndarray) -> jnp.ndarray:
    """
    Compute seasonal penalty term S for all flybys across all bodies.

    This implements the formula from section 3.3:
    S(r_hat_k,i) = 0.1 + 0.9 / (1 + 10 * �_{j=1}^{i-1} exp(-(acosd(r_hat_k,i dot r_hat_k,j))^2 / 50))

    where the sum is over previous flybys of the SAME body k, and S(r_hat_k,1) = 1 for the first flyby of each body.

    Args:
        r_hat: unit heliocentric position vectors at all flybys (shape: (N, 3))
               where N is the total number of flybys across all bodies
        body_ids: body ID for each flyby (shape: (N,))
                  flybys must be in chronological order

    Returns:
        S: seasonal penalty factors for each flyby (shape: (N,))
    """
    N = r_hat.shape[0]

    # Compute pairwise dot products: result[i, j] = r_hat[i] · r_hat[j]
    # Shape: (N, N)
    dot_products = jnp.dot(r_hat, r_hat.T)
    # Clip to valid range for arccos to avoid NaN
    dot_products = jnp.clip(dot_products, -1.0, 1.0)

    # Convert to angles in degrees
    # Shape: (N, N)
    angles_deg = jnp.arccos(dot_products) * 180.0 / jnp.pi

    # Replace NaN values with 0 (can occur with identical vectors)
    angles_deg = jnp.where(jnp.isnan(angles_deg), 0.0, angles_deg)

    # Compute exponential terms
    # Shape: (N, N)
    exp_terms = jnp.exp(-angles_deg**2 / 50.0)

    # Create mask for same-body previous flybys
    # mask[i, j] = 1 if j < i (previous flyby) AND body_ids[i] == body_ids[j] (same body)
    # Shape: (N, N)

    # Lower triangular mask (j < i)
    lower_tri_mask = jnp.tril(jnp.ones((N, N)), k=-1)

    # Same body mask: body_ids[i] == body_ids[j]
    # Broadcast body_ids to compare all pairs
    same_body_mask = body_ids[:, None] == body_ids[None, :]

    # Combined mask: previous flyby of same body
    mask = lower_tri_mask * same_body_mask

    # Apply mask to get only contributions from previous flybys of same body
    masked_exp = exp_terms * mask

    # Sum over previous flybys of same body (sum along columns for each row)
    # Shape: (N,)
    exp_sums = jnp.sum(masked_exp, axis=1)

    # Compute S for each flyby
    # S(r_hat_k,i) = 0.1 + 0.9 / (1 + 10 * exp_sum_i)
    # Shape: (N,)
    S = 0.1 + 0.9 / (1.0 + 10.0 * exp_sums)

    # Safety check: replace any NaN or Inf with 1.0 (no penalty)
    S = jnp.where(jnp.isfinite(S), S, 1.0)

    return S
