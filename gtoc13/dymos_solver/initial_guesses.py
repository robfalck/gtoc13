from lamberthub import izzo2015 as lambert
from gtoc13 import ConicArc, FlybyArc, PropagatedArc, bodies_data


import numpy as np

from gtoc13.analytic import propagate_ballistic
from gtoc13.constants import KMPDU, MU_ALTAIRA


def _guess_from_solution(guess_arc):
    """
    Extract initial guess data from a solution arc.

    This function processes an arc from a previously computed solution and extracts
    the trajectory data in a format suitable for initializing a Dymos optimization
    problem. It handles three types of arcs: PropagatedArc (thrust arcs), ConicArc
    (coast arcs), and FlybyArc (planetary flybys).

    Parameters
    ----------
    guess_arc : PropagatedArc, ConicArc, or FlybyArc
        An arc from a GTOC13Solution that will be used as an initial guess.

    Returns
    -------
    dict
        A dictionary containing the following keys:
        - 'dt' : float
            Arc duration in seconds. For FlybyArc, this is 0.0.
        - 'r' : ndarray, shape (n, 3)
            Position trajectory in km. For PropagatedArc, contains all state points.
            For ConicArc, contains start and end positions. For FlybyArc, contains
            the flyby position.
        - 'v' : ndarray, shape (n, 3) or None
            Velocity trajectory in km/s. For PropagatedArc and ConicArc, contains
            velocities at state points. For FlybyArc, this is None (use v_in/v_out instead).
        - 'u' : ndarray, shape (n, 3) or None
            Control vector (thrust direction) trajectory. For PropagatedArc, contains
            all control points. For ConicArc, contains zeros. For FlybyArc, this is None.

    Notes
    -----
    - For PropagatedArc: Extracts all state points (time, position, velocity, control)
      from the arc's trajectory data.
    - For ConicArc: Extracts only the start and end states, with zero control vectors.
    - For FlybyArc: Extracts the flyby epoch, position, and incoming/outgoing velocities.
      Duration is computed as 0 since flyby is instantaneous.

    This function is primarily used by set_initial_guesses() to warm-start the
    optimization with data from a previous solution.

    Examples
    --------
    >>> # Extract guess from a propagated arc
    >>> guess = _guess_from_solution(solution.arcs[0])
    >>> phase.set_state_val('r', vals=guess['r'], units='km')
    >>> phase.set_state_val('v', vals=guess['v'], units='km/s')
    """
    if isinstance(guess_arc, PropagatedArc):
        times_s = np.array([state.epoch for state in guess_arc.state_points])
        r_km = np.array([state.position for state in guess_arc.state_points])
        v_kms = np.array([state.velocity for state in guess_arc.state_points])
        u = np.array([state.control for state in guess_arc.state_points])
    elif isinstance(guess_arc, ConicArc):
        times_s = np.array([guess_arc.epoch_start, guess_arc.epoch_end])
        r_km = np.array([guess_arc.position_start, guess_arc.position_end])
        v_kms = np.array([guess_arc.velocity_start, guess_arc.velocity_end])
        u = np.zeros((2, 3))
    else:
        raise RuntimeError('_guess_from_solution only works for PropagatedArc and ConicArc')

    t_initial_s = times_s[0]
    t_final_s = times_s[-1]
    dt_arc_s = t_final_s - t_initial_s

    return {'t_initial': t_initial_s,
            'times_s': times_s,
            'dt': dt_arc_s,
            'r': r_km,
            'v': v_kms,
            'u': u}


def _guess_linear(phase, from_body, to_body, t1, t2, control):
    """
    Generate a linear initial guess for an arc trajectory.

    This function creates a simple initial guess for an arc by linearly interpolating
    positions between the start and end points, and using a constant average velocity.
    For thrust arcs with radial control, it also generates anti-radial unit vectors
    as the control guess.

    Parameters
    ----------
    phase : dymos.Phase
        The Dymos phase object for which to generate the guess. Used to interpolate
        values onto the collocation nodes.
    from_body : int
        The body id of the starting body, or -1 for the starting plane.
    to_body : int
        The body id of the target body.
    t1 : float
        Initial time in seconds.
    t2 : float
        Final time in seconds.
    control : int
        Control type for the arc:
        - 0: Coast arc (no thrust, zero control)
        - 1 or 'r': Thrust arc with radial control (anti-radial unit vector)

    Returns
    -------
    dict
        A dictionary containing the initial guess data with the following keys:
        - 'dt' : float
            Arc duration in seconds (t2 - t1).
        - 'r' : ndarray, shape (n, 3)
            Position trajectory in km, linearly interpolated between r1 and r2
            at all collocation nodes.
        - 'v' : ndarray, shape (n, 3)
            Velocity trajectory in km/s. Constant average velocity computed as
            (r2 - r1) / dt at all nodes.
        - 'u' : ndarray, shape (n, 3)
            Control vector trajectory. For coast arcs (control=0), this is zeros.
            For thrust arcs, this is the anti-radial unit vector at each node.

    Notes
    -----
    - This function provides a very simple guess and is mainly used as a fallback
      when Lambert solver fails or for the initial arc from the problem start point.
    - The velocity guess is not physically accurate for orbital mechanics, but serves
      as a reasonable starting point for the optimizer.
    - For thrust arcs with radial control, the control vector is computed as the
      anti-radial direction: u = -r / ||r||, which points toward the sun.
    - The phase.interp() method is used to interpolate values onto the phase's
      collocation nodes based on the transcription scheme.

    See Also
    --------
    _guess_from_solution : Extract guess from a previously computed solution
    set_initial_guesses : Main function that sets all initial guesses

    Examples
    --------
    >>> # Generate linear guess for an arc
    >>> guess = _guess_linear(phase, r_start, r_end, t_start, t_end, control=1)
    >>> phase.set_state_val('r', vals=guess['r'], units='km')
    >>> phase.set_state_val('v', vals=guess['v'], units='km/s')
    >>> phase.set_control_val('u_n', vals=guess['u'], units='unitless')
    """
    r2 = bodies_data[to_body].get_state(t2).r
    if from_body == -1:
        r1 = np.array([-200 * KMPDU, r2[1], r2[2]])
    else:
        r1 = bodies_data[to_body].get_state(t1).r

    dt_arc_s = t2 - t1
    r_km = phase.interp('r', [r1, r2])
    v_avg = (r2 - r1) / dt_arc_s
    v_kms = phase.interp('v', [v_avg, v_avg])

    times_s = phase.interp('t', [t1, t2])

    if control == 0:
        u = np.zeros((1, 3))
    else:
        r_mag = np.linalg.norm(r_km, axis=-1, keepdims=True)
        u = -r_km / r_mag

    return {'t_initial': t1,
            'times_s': times_s,
            'dt': dt_arc_s,
            'r': r_km,
            'v': v_kms,
            'u': u}


def _guess_lambert(phase, from_body, to_body, t1, t2, control):
    """
    Generate a Lambert initial guess for an arc trajectory.

    This function creates a simple initial guess for an arc by solving Lamberts
    problem between the two bodies at the given times, and then propagating
    that trajectory using an analytic 2-body approach.
    For thrust arcs with radial control, it also generates anti-radial unit vectors
    as the control guess.

    Parameters
    ----------
    phase : dymos.Phase
        The Dymos phase object for which to generate the guess. Used to interpolate
        values onto the collocation nodes.
    from_body : int
        The body id of the starting body, or -1 for the starting plane.
    to_body : int
        The body id of the target body.
    t1 : float
        Initial time in seconds.
    t2 : float
        Final time in seconds.
    control : int
        Control type for the arc:
        - 0: Coast arc (no thrust, zero control)
        - 1 or 'r': Thrust arc with radial control (anti-radial unit vector)

    Returns
    -------
    dict
        A dictionary containing the initial guess data with the following keys:
        - 'dt' : float
            Arc duration in seconds (t2 - t1).
        - 'r' : ndarray, shape (n, 3)
            Position trajectory in km, linearly interpolated between r1 and r2
            at all collocation nodes.
        - 'v' : ndarray, shape (n, 3)
            Velocity trajectory in km/s. Constant average velocity computed as
            (r2 - r1) / dt at all nodes.
        - 'u' : ndarray, shape (n, 3)
            Control vector trajectory. For coast arcs (control=0), this is zeros.
            For thrust arcs, this is the anti-radial unit vector at each node.

    """
    r2 = bodies_data[to_body].get_state(t2).r
    if from_body == -1:
        r1 = np.array([-200 * KMPDU, r2[1], r2[2]])
    else:
        r1 = bodies_data[to_body].get_state(t1).r

    dt_arc_s = t2 - t1

    # Convert to standard numpy arrays to avoid numba compilation issues with read-only buffers
    r1 = np.array(r1, dtype=float, copy=True)
    r2 = np.array(r2, dtype=float, copy=True)

    lambert_sol = lambert(MU_ALTAIRA, r1, r2, dt_arc_s)
    v1 = lambert_sol[0]
    # v1, _, resid = vallado2013_jax(MU_ALTAIRA, r1, r2, dt_arc_s)
    nodes_tau = phase.options['transcription'].grid_data.node_ptau
    node_times = t1 + 0.5 * (nodes_tau + 1) * dt_arc_s

    if np.any(np.isnan(v1)):
        print(f'{phase.name}: Lambert solve failed - falling back to linear guess' )
        # LAMBERT FAILED - FALLBACK TO LINEAR
        guess = _guess_linear(phase, from_body, to_body, t1, t2, control)
        r_km = guess['r']
        v_kms = guess['v']
    else:
        r_km, v_kms = propagate_ballistic(r1, v1, node_times)

    if control == 0:
        u = np.zeros((1, 3))
    else:
        r_mag = np.linalg.norm(r_km, axis=-1, keepdims=True)
        u = -r_km / r_mag

    return {'t_initial': t1,
            'times_s': node_times,
            'dt': dt_arc_s,
            'r': r_km,
            'v': v_kms,
            'u': u}

def set_initial_guesses(prob, bodies, flyby_times, t0, controls,
                        guess_solution=None, single_arc=False):
    from gtoc13.constants import YEAR

    # Set initial guess values
    N = len(bodies)

    if single_arc:
        dt_yr = np.diff(flyby_times)
    else:
        _t0 = np.array(t0).reshape((1,))
        all_times_yr = np.concatenate((_t0, flyby_times))
        dt_yr = np.diff(all_times_yr)

    # Set t0 and dt
    prob.set_val('t0', t0, units='gtoc_year')
    prob.set_val('dt', dt_yr, units='gtoc_year')
    if single_arc:
        prob.set_val('v_in_prev_flyby', guess_solution.arcs[-1].velocity_in, units='km/s')
    else:
        prob.set_val('y0', 0.0, units='km')
        prob.set_val('z0', 0.0, units='km')

    # Get body positions and velocities at flyby times
    # Convert flyby times to seconds for get_state
    flyby_times_s = [t * YEAR for t in flyby_times]

    last_guess_flyby_arc = None
    if guess_solution is not None:
        non_flyby_arcs = [arc for arc in guess_solution.arcs
                        if isinstance(arc, (PropagatedArc, ConicArc))]
        last_guess_flyby_arc = guess_solution.arcs[-1]
        if not isinstance(last_guess_flyby_arc, FlybyArc):
            last_guess_flyby_arc=None
    else:
        non_flyby_arcs = []

    # Set initial guess for positions and velocities and controls for each arc
    if single_arc:
        _bodies = bodies
        _times = flyby_times_s
    else:
        _bodies = [-1] + bodies
        _times = [t0 * YEAR] + flyby_times_s

    num_arcs = (N-1 if single_arc else N)

    for i in range(num_arcs):
        phase = prob.model.traj.phases._get_subsystem(f'arc_{i}')
        from_body = _bodies[i]
        to_body = _bodies[i+1]
        t_initial_s = _times[i]
        t_final_s = _times[i+1]

        try:
            guess_arc = non_flyby_arcs[i]
        except IndexError:
            guess_arc = None

        if guess_arc is None or single_arc:
            # If we don't have a guess for this arc,
            # try lambert, which will fall back to linear if it fails to converge.
            print(f'arc {i} attempting guess from lambert')
            guess = _guess_lambert(phase, from_body, to_body, t_initial_s, t_final_s, controls[i])
        else:
            print(f'arc {i} attempting guess from solution')
            guess = _guess_from_solution(guess_arc)

        # Set the top level problem variables
        if i == 0 and not single_arc:
            prob.set_val('t0', guess['t_initial'], units='s')
            prob.set_val('y0', guess['r'][0, 1], units='km')
            prob.set_val('z0', guess['r'][0, 2], units='km')
        prob.set_val('dt', guess['dt'] / YEAR, indices=[i], units='gtoc_year')

        # Set the phase states
        phase.set_state_val('r', guess['r'], time_vals=guess['times_s'], units='km')
        phase.set_state_val('v', guess['v'], time_vals=guess['times_s'], units='km/s')

        # Set the phase controls
        if controls[i] == 0:
            phase.set_parameter_val('u_n', [0., 0., 0.], units='unitless')
        elif controls[i] == 1:
            # If we've changed any controls to optimal but they
            # were 0 before, we need to set the guess so that the optimizer
            # starts with something reasonable.
            if np.all(np.abs(guess['u']) < 1.0E-2):
                r_mag = np.linalg.norm(guess['r'], axis=-1, keepdims=True)
                r_hat = guess['r'] / r_mag
                guess['u'] = -r_hat

            phase.set_control_val('u_n', guess['u'], units='unitless')
    else:
        # If the last arc is in the guess, use that flyby v_out as v_end.
        # Otherwise, the final velocity to a slightly perturbed version
        # of the final arc velocity. Setting them equal results in an infinite flyby radius.
        if guess_arc is not None:
            prob.set_val('v_end',
                         guess_solution.arcs[-1].velocity_out,
                         units='km/s')
        else:
            prob.set_val('v_end',
                        0.9 * guess['v'][-1, ...],
                        units='km/s')