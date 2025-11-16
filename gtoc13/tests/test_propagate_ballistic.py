"""
Tests for the analytic ballistic propagation function.

This module tests the propagate_ballistic function by comparing its results
against the body ephemeris data from get_state().
"""
import pytest
import jax.numpy as jnp
import numpy as np
from gtoc13.analytic import propagate_ballistic
from gtoc13.bodies import bodies_data
from gtoc13.constants import YEAR, MU_ALTAIRA
from lamberthub import izzo2015 as lambert


def test_propagate_ballistic_against_ephemeris():
    """
    Test propagate_ballistic against body ephemeris for a planet.

    This test propagates a body's orbit using the analytic f-g method and
    compares the result to the body's ephemeris at multiple points along the orbit.
    """
    # Use a planet with a shorter, well-defined orbit (Bespin)
    body_id = 10
    body = bodies_data[body_id]

    # Get initial state at t=0
    t0 = 0.0
    state0 = body.get_state(t0, time_units='s', distance_units='km')
    r0 = np.array(state0.r)
    v0 = np.array(state0.v)

    # Get the body's orbital period
    period_years = body.get_period(units='year')
    period_s = period_years * YEAR

    # Test at multiple points over half an orbit (to avoid numerical issues with long periods)
    # Use 20 points distributed over half the orbit
    n_points = 20
    times = np.linspace(0, period_s / 2, n_points)

    # Propagate using our analytic function
    positions, velocities = propagate_ballistic(
        jnp.array(r0),
        jnp.array(v0),
        jnp.array(times),
        mu=MU_ALTAIRA
    )

    # Convert to numpy for comparison
    positions = np.array(positions)
    velocities = np.array(velocities)

    # Compare against ephemeris at each time point
    max_pos_error = 0.0
    max_vel_error = 0.0

    for i, t in enumerate(times):
        # Get reference state from ephemeris
        state_ref = body.get_state(t, time_units='s', distance_units='km')
        r_ref = np.array(state_ref.r)
        v_ref = np.array(state_ref.v)

        # Compute errors
        pos_error = np.linalg.norm(positions[i] - r_ref)
        vel_error = np.linalg.norm(velocities[i] - v_ref)

        max_pos_error = max(max_pos_error, pos_error)
        max_vel_error = max(max_vel_error, vel_error)

        # Position error should be very small (< 1 km for one orbit)
        # Velocity error should be very small (< 0.001 km/s for one orbit)
        assert pos_error < 1.0, \
            f"Position error too large at t={t/YEAR:.2f} years: {pos_error:.2e} km"
        assert vel_error < 0.001, \
            f"Velocity error too large at t={t/YEAR:.2f} years: {vel_error:.2e} km/s"

    print(f"\nTest passed for body {body_id} ({body.name})")
    print(f"  Orbital period: {period_years:.2f} years")
    print(f"  Max position error: {max_pos_error:.2e} km")
    print(f"  Max velocity error: {max_vel_error:.2e} km/s")


def test_propagate_ballistic_multiple_bodies():
    """
    Test propagate_ballistic against ephemeris for multiple bodies.

    This ensures the function works correctly for different orbital types
    (planets, asteroids, comets).
    """
    # Test with a few different bodies
    test_bodies = [
        10,   # Bespin (planet)
        1001, # An asteroid
        2006, # A comet
    ]

    for body_id in test_bodies:
        body = bodies_data[body_id]

        # Get initial state
        t0 = 0.0
        state0 = body.get_state(t0, time_units='s', distance_units='km')
        r0 = np.array(state0.r)
        v0 = np.array(state0.v)

        # Test at 5 points over half an orbit
        period_years = body.get_period(units='year')
        period_s = period_years * YEAR
        times = np.linspace(0, period_s / 2, 5)

        # Propagate
        positions, velocities = propagate_ballistic(
            jnp.array(r0),
            jnp.array(v0),
            jnp.array(times),
            mu=MU_ALTAIRA
        )

        positions = np.array(positions)
        velocities = np.array(velocities)

        # Check each time point
        for i, t in enumerate(times):
            state_ref = body.get_state(t, time_units='s', distance_units='km')
            r_ref = np.array(state_ref.r)
            v_ref = np.array(state_ref.v)

            pos_error = np.linalg.norm(positions[i] - r_ref)
            vel_error = np.linalg.norm(velocities[i] - v_ref)

            # Relaxed tolerances for diverse body types
            assert pos_error < 10.0, \
                f"Body {body_id}: Position error at t={t/YEAR:.2f} yr: {pos_error:.2e} km"
            assert vel_error < 0.01, \
                f"Body {body_id}: Velocity error at t={t/YEAR:.2f} yr: {vel_error:.2e} km/s"


def test_propagate_ballistic_initial_condition():
    """
    Test that propagate_ballistic returns the initial state at t=t0.
    """
    # Use a simple test case
    body_id = 10  # Bespin
    body = bodies_data[body_id]

    t0 = 0.0
    state0 = body.get_state(t0, time_units='s', distance_units='km')
    r0 = jnp.array(state0.r)
    v0 = jnp.array(state0.v)

    # Propagate at just the initial time
    times = jnp.array([t0])
    positions, velocities = propagate_ballistic(r0, v0, times)

    # Should return exactly the initial state
    np.testing.assert_allclose(positions[0], r0, rtol=1e-12)
    np.testing.assert_allclose(velocities[0], v0, rtol=1e-12)


def test_propagate_ballistic_energy_conservation():
    """
    Test that orbital energy is conserved during propagation.
    """
    # Use a comet
    body_id = 2006
    body = bodies_data[body_id]

    t0 = 0.0
    state0 = body.get_state(t0, time_units='s', distance_units='km')
    r0 = np.array(state0.r)
    v0 = np.array(state0.v)

    # Compute initial orbital energy
    r0_mag = np.linalg.norm(r0)
    v0_mag = np.linalg.norm(v0)
    E0 = 0.5 * v0_mag**2 - MU_ALTAIRA / r0_mag

    # Propagate over one orbit
    period_years = body.get_period(units='year')
    period_s = period_years * YEAR
    times = np.linspace(0, period_s, 50)

    positions, velocities = propagate_ballistic(
        jnp.array(r0),
        jnp.array(v0),
        jnp.array(times),
        mu=MU_ALTAIRA
    )

    positions = np.array(positions)
    velocities = np.array(velocities)

    # Check energy at each point
    for i in range(len(times)):
        r_mag = np.linalg.norm(positions[i])
        v_mag = np.linalg.norm(velocities[i])
        E = 0.5 * v_mag**2 - MU_ALTAIRA / r_mag

        # Energy should be conserved to high precision
        energy_error = abs((E - E0) / E0)
        assert energy_error < 1e-10, \
            f"Energy not conserved at t={times[i]/YEAR:.2f} yr: relative error = {energy_error:.2e}"


def test_propagate_ballistic_angular_momentum_conservation():
    """
    Test that angular momentum is conserved during propagation.
    """
    # Use an asteroid
    body_id = 1001
    body = bodies_data[body_id]

    t0 = 0.0
    state0 = body.get_state(t0, time_units='s', distance_units='km')
    r0 = np.array(state0.r)
    v0 = np.array(state0.v)

    # Compute initial angular momentum
    h0 = np.cross(r0, v0)
    h0_mag = np.linalg.norm(h0)

    # Propagate
    period_years = body.get_period(units='year')
    period_s = period_years * YEAR
    times = np.linspace(0, period_s / 2, 20)

    positions, velocities = propagate_ballistic(
        jnp.array(r0),
        jnp.array(v0),
        jnp.array(times),
        mu=MU_ALTAIRA
    )

    positions = np.array(positions)
    velocities = np.array(velocities)

    # Check angular momentum at each point
    for i in range(len(times)):
        h = np.cross(positions[i], velocities[i])
        h_mag = np.linalg.norm(h)

        # Angular momentum should be conserved to high precision
        h_error = abs((h_mag - h0_mag) / h0_mag)
        assert h_error < 1e-10, \
            f"Angular momentum not conserved at t={times[i]/YEAR:.2f} yr: relative error = {h_error:.2e}"


def test_propagate_ballistic_vs_lambert():
    """
    Test propagate_ballistic against a Lambert solution from lamberthub.izzo2015.

    This test sets up a transfer from body ID 4 to an asteroid, solves the Lambert
    problem to get the initial velocity, then propagates that trajectory and verifies
    that the initial and final states match.
    """
    # Use body 4 as the departure body
    from_body_id = 4
    from_body = bodies_data[from_body_id]

    # Use an asteroid as the target
    to_body_id = 1005
    to_body = bodies_data[to_body_id]

    # Set up the transfer times
    t_departure = 10.0 * YEAR  # Start at 10 years
    transfer_time = 25.0 * YEAR  # 25 year transfer time in seconds
    t_arrival = t_departure + transfer_time

    # Get the initial and final positions
    state_departure = from_body.get_state(t_departure, time_units='s', distance_units='km')
    state_arrival = to_body.get_state(t_arrival, time_units='s', distance_units='km')

    r1 = np.array(state_departure.r, dtype=float, copy=True)
    r2 = np.array(state_arrival.r, dtype=float, copy=True)

    # Solve the Lambert problem to get the required initial velocity
    # lambert returns (v1, v2, tof, theta, n_iter, err_msg)
    lambert_result = lambert(MU_ALTAIRA, r1, r2, transfer_time)
    v1_lambert = np.array(lambert_result[0])
    v2_lambert = np.array(lambert_result[1])

    print(f"\n=== Lambert Solution Test ===")
    print(f"Transfer from body {from_body_id} ({from_body.name}) to body {to_body_id} ({to_body.name})")
    print(f"Transfer time: {transfer_time / YEAR:.2f} years")
    print(f"Initial position: {r1}")
    print(f"Final position: {r2}")
    print(f"Lambert initial velocity: {v1_lambert}")
    print(f"Lambert final velocity: {v2_lambert}")

    # Now propagate the trajectory using the Lambert initial velocity
    times = np.array([t_departure, t_arrival])
    positions, velocities = propagate_ballistic(
        jnp.array(r1),
        jnp.array(v1_lambert),
        jnp.array(times),
        mu=MU_ALTAIRA
    )

    positions = np.array(positions)
    velocities = np.array(velocities)

    # Check that the initial state matches
    r1_prop = positions[0]
    v1_prop = velocities[0]

    pos_error_initial = np.linalg.norm(r1_prop - r1)
    vel_error_initial = np.linalg.norm(v1_prop - v1_lambert)

    print(f"\nInitial state verification:")
    print(f"  Position error: {pos_error_initial:.2e} km (should be ~0)")
    print(f"  Velocity error: {vel_error_initial:.2e} km/s (should be ~0)")

    # The initial state should match exactly (within numerical precision)
    assert pos_error_initial < 1e-6, \
        f"Initial position mismatch: {pos_error_initial:.2e} km"
    assert vel_error_initial < 1e-9, \
        f"Initial velocity mismatch: {vel_error_initial:.2e} km/s"

    # Check that the final state matches the target
    r2_prop = positions[1]
    v2_prop = velocities[1]

    pos_error_final = np.linalg.norm(r2_prop - r2)
    vel_error_final = np.linalg.norm(v2_prop - v2_lambert)

    print(f"\nFinal state verification:")
    print(f"  Propagated final position: {r2_prop}")
    print(f"  Target final position: {r2}")
    print(f"  Position error: {pos_error_final:.2e} km")
    print(f"  Propagated final velocity: {v2_prop}")
    print(f"  Lambert final velocity: {v2_lambert}")
    print(f"  Velocity error: {vel_error_final:.2e} km/s")

    # The final position should match the target position very closely
    # (within a reasonable tolerance for numerical integration vs Lambert solution)
    assert pos_error_final < 0.001, \
        f"Final position error too large: {pos_error_final:.2e} km"

    # The final velocity from propagation should match the Lambert final velocity
    assert vel_error_final < 0.001, \
        f"Final velocity error too large: {vel_error_final:.2e} km/s"

    # Additional check: verify the Lambert solution itself by checking that
    # the initial velocity magnitude is reasonable for an interplanetary transfer
    v1_mag = np.linalg.norm(v1_lambert)
    print(f"\nLambert solution sanity checks:")
    print(f"  Initial velocity magnitude: {v1_mag:.6f} km/s")

    # For an interplanetary transfer, velocity should be reasonable
    # (not zero, not absurdly large)
    assert 0.1 < v1_mag < 100.0, \
        f"Lambert velocity seems unreasonable: {v1_mag:.2e} km/s"

    print(f"\n✓ Lambert solution test passed!")


if __name__ == '__main__':
    # Run tests with verbose output
    pytest.main([__file__, '-v', '-s'])
