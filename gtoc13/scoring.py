"""
GTOC13 Scoring Functions.

This module contains functions for computing the mission score including
seasonal penalties, flyby velocity penalties, time bonuses, and the overall
score calculation.
"""
import jax
import jax.numpy as jnp
from jax import jit


def seasonal_penalty(r_hat_current: jnp.ndarray, r_hat_previous: jnp.ndarray) -> float:
    """
    Compute seasonal penalty term S for a single previous flyby.

    Args:
        r_hat_current: unit heliocentric position vector at current flyby
        r_hat_previous: array of unit heliocentric position vectors at previous flybys (shape: (n_prev, 3))

    Returns:
        S: seasonal penalty factor
    """
    # Handle first flyby case
    def first_flyby():
        return 1.0

    def subsequent_flyby():
        # Compute angles with all previous flybys
        dot_products = jnp.dot(r_hat_previous, r_hat_current)
        dot_products = jnp.clip(dot_products, -1.0, 1.0)
        angles_deg = jnp.arccos(dot_products) * 180.0 / jnp.pi

        # Sum of exponential terms
        exp_sum = jnp.sum(jnp.exp(-angles_deg**2 / 50.0))

        S = 0.1 + 0.9 / (1.0 + 10.0 * exp_sum)
        return S

    n_prev = r_hat_previous.shape[0]
    return jax.lax.cond(n_prev == 0, first_flyby, subsequent_flyby)


@jit
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


@jit
def time_bonus(t_submission_days: float) -> float:
    """
    Compute time bonus term c based on submission time.

    Args:
        t_submission_days: days elapsed from competition start

    Returns:
        c: time bonus factor
    """
    return jax.lax.cond(
        t_submission_days <= 7.0,
        lambda: 1.13,
        lambda: -0.005 * t_submission_days + 1.165
    )


def compute_score(
    flybys: list,
    body_weights: dict,
    grand_tour_achieved: bool,
    submission_time_days: float
) -> float:
    """
    Compute total mission score.

    Args:
        flybys: list of flyby data, each containing:
                {'body_id', 'r_hat', 'v_infinity', 'is_scientific', 'r_hat_previous'}
        body_weights: dict mapping body_id to scientific weight
        grand_tour_achieved: whether grand tour bonus applies
        submission_time_days: days from competition start

    Returns:
        J: total score
    """
    b = 1.2 if grand_tour_achieved else 1.0
    c = time_bonus(submission_time_days)

    # Group flybys by body
    body_flybys = {}
    for fb in flybys:
        if not fb['is_scientific']:
            continue

        body_id = fb['body_id']
        if body_id not in body_flybys:
            body_flybys[body_id] = []
        body_flybys[body_id].append(fb)

    total_score = 0.0

    for body_id, fb_list in body_flybys.items():
        w_k = body_weights.get(body_id, 0.0)

        for i, fb in enumerate(fb_list[:13]):  # Max 13 scientific flybys per body
            # Get previous flyby positions for this body
            r_hat_prev = jnp.array([fb_list[j]['r_hat'] for j in range(i)])

            S = seasonal_penalty(fb['r_hat'], r_hat_prev)
            F = flyby_velocity_penalty(fb['v_infinity'])

            total_score += w_k * S * F

    J = b * c * total_score

    return J
