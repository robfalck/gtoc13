"""
Animated 3D visualization of GTOC13 orbits (planets, asteroids, comets)
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import jax.numpy as jnp

from gtoc13 import (
    Body,
    load_bodies_data,
    OrbitalElements,
    elements_to_cartesian,
    AU,
    MU_ALTAIRA,
    YEAR,
    DAY,
    ballistic_ode
)
from scipy.integrate import solve_ivp


def compute_orbit_points(elements: OrbitalElements, n_points: int = 100) -> np.ndarray:
    """
    Compute points along an orbit for visualization using uniformly spaced true anomaly.
    Directly computes positions from orbital elements to avoid numerical issues.
    Returns array of shape (n_points, 3) with [x, y, z] positions in AU.
    """
    a, e, i, Omega, omega = elements.a, elements.e, elements.i, elements.Omega, elements.omega

    # Uniformly spaced true anomalies (endpoint=False to avoid duplicate at 0/2π)
    true_anomalies = np.linspace(0, 2 * np.pi, n_points, endpoint=False)

    # Precompute rotation matrix elements
    cos_Omega = np.cos(Omega)
    sin_Omega = np.sin(Omega)
    cos_i = np.cos(i)
    sin_i = np.sin(i)
    cos_omega = np.cos(omega)
    sin_omega = np.sin(omega)

    positions = []
    for nu in true_anomalies:
        # Compute radius from true anomaly
        r_mag = a * (1.0 - e**2) / (1.0 + e * np.cos(nu))

        # Position in orbital plane (perifocal frame)
        x_orb = r_mag * np.cos(nu)
        y_orb = r_mag * np.sin(nu)

        # Rotate from orbital plane to inertial frame
        # R = R3(-Omega) * R1(-i) * R3(-omega)
        x = (cos_Omega * cos_omega - sin_Omega * sin_omega * cos_i) * x_orb + \
            (-cos_Omega * sin_omega - sin_Omega * cos_omega * cos_i) * y_orb

        y = (sin_Omega * cos_omega + cos_Omega * sin_omega * cos_i) * x_orb + \
            (-sin_Omega * sin_omega + cos_Omega * cos_omega * cos_i) * y_orb

        z = (sin_omega * sin_i) * x_orb + (cos_omega * sin_i) * y_orb

        positions.append(np.array([x, y, z]))

    # Convert to numpy array and scale to AU
    return np.array(positions) / AU


def propagate_final_state(final_state: dict, duration_years: float = 50.0, n_points: int = 1000) -> np.ndarray:
    """
    Propagate the final state of a trajectory using ballistic (Keplerian) motion.

    Args:
        final_state: Dictionary containing 'position' and 'velocity' arrays in km and km/s
        duration_years: Duration to propagate in years (default: 50.0)
        n_points: Number of points in the propagated trajectory (default: 1000)

    Returns:
        Array of shape (n_points, 3) with propagated positions in km
    """
    # Initial state [x, y, z, vx, vy, vz]
    y0 = np.concatenate([final_state['position'], final_state['velocity']])

    # Time span for propagation (in seconds)
    t_span = (0.0, duration_years * YEAR)
    t_eval = np.linspace(0.0, duration_years * YEAR, n_points)

    # Integrate using ballistic ODE (Keplerian motion)
    # Note: ballistic_ode expects args=(mu,)
    def ode_func(t, y):
        return np.asarray(ballistic_ode(t, y, (MU_ALTAIRA,)))

    sol = solve_ivp(
        ode_func,
        t_span,
        y0,
        method='DOP853',  # High-order Runge-Kutta method
        t_eval=t_eval,
        rtol=1e-12,
        atol=1e-12
    )

    # Extract positions (first 3 components)
    positions = sol.y[:3, :].T  # Shape: (n_points, 3)

    return positions, t_eval


def load_solution_trajectory(filepath: str) -> dict:
    """
    Load a solution trajectory from a GTOC13 solution file.
    Returns dict with 'state_points' list, 'max_time', and 'propagated_trajectory'.
    """
    state_points = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith('#') or line.startswith('!'):
                continue

            # Parse data row: body_id flag epoch x y z vx vy vz cx cy cz
            parts = line.replace(',', ' ').split()
            if len(parts) != 12:
                continue

            state_points.append({
                'body_id': int(float(parts[0])),
                'flag': int(float(parts[1])),
                'epoch': float(parts[2]),
                'position': np.array([float(parts[3]), float(parts[4]), float(parts[5])]),
                'velocity': np.array([float(parts[6]), float(parts[7]), float(parts[8])]),
                'control': np.array([float(parts[9]), float(parts[10]), float(parts[11])])
            })

    max_time = max(pt['epoch'] for pt in state_points) if state_points else 0.0

    # Propagate final state for 100 years after last flyby
    propagated_positions = None
    propagated_times = None
    if state_points:
        final_state = state_points[-1]
        print(f"Propagating final state for 100 years from t={max_time/YEAR:.2f} years...")
        propagated_positions, propagated_times = propagate_final_state(final_state, duration_years=100.0)
        # Adjust times to be relative to mission start
        propagated_times = propagated_times + max_time
        print(f"Propagation complete. Final propagated time: t={propagated_times[-1]/YEAR:.2f} years")

    return {
        'state_points': state_points,
        'max_time': max_time,
        'propagated_positions': propagated_positions,
        'propagated_times': propagated_times
    }


def create_animation(
    duration_years: float = 50.0,
    fps: int = 30,
    n_orbit_points: int = 100,
    solution_file: str = None
):
    """
    Create an animated 3D visualization of the solar system.

    Args:
        duration_years: Duration of animation in years
        fps: Frames per second
        n_orbit_points: Number of points to plot for each orbit
        solution_file: Optional path to solution trajectory file
    """
    print("Loading orbital data...")
    bodies = load_bodies_data()

    # Separate bodies by type
    planets = [b for b in bodies.values() if b.is_planet()]
    small_bodies = [b for b in bodies.values() if b.is_small_body()]

    # Further separate small bodies into asteroids and comets
    # Asteroids: IDs 1001-1100, Comets: IDs 2001-2200
    asteroids = [b for b in small_bodies if 1001 <= b.id <= 1100]
    comets = [b for b in small_bodies if 2001 <= b.id <= 2200]

    print(f"Loaded {len(planets)} planets, {len(asteroids)} asteroids, {len(comets)} comets")

    # Load solution trajectory if provided
    solution_data = None
    if solution_file:
        solution_data = load_solution_trajectory(solution_file)
        print(f"Loaded solution with {len(solution_data['state_points'])} state points")
        print(f"Solution trajectory ends at t={solution_data['max_time'] / YEAR:.2f} years")

    # Compute orbit paths (static)
    print("Computing orbit paths...")
    planet_orbits = [compute_orbit_points(p.elements, n_orbit_points) for p in planets]
    asteroid_orbits = [compute_orbit_points(a.elements, n_orbit_points) for a in asteroids]
    comet_orbits = [compute_orbit_points(c.elements, n_orbit_points * 10) for c in comets]

    # Setup figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Set up the plot limits
    max_dist = max([np.max(np.abs(orbit)) for orbit in planet_orbits]) * 1.25

    # If solution is loaded, set initial zoom to show trajectory better (exclude starting point)
    if solution_data:
        # Get positions excluding the first few points (likely the far starting position)
        positions = np.array([pt['position'] / AU for pt in solution_data['state_points'][5:]])
        if len(positions) > 0:
            traj_max = np.max(np.abs(positions)) * 1.5
            max_dist = max(traj_max, max_dist * 0.3)  # At least 30% of planet scale

    initial_max_dist = max_dist
    ax.set_xlim([-max_dist, max_dist])
    ax.set_ylim([-max_dist, max_dist])
    ax.set_zlim([-max_dist, max_dist])

    # Zoom state
    zoom_state = {'scale': 1.0}

    # Selected body state
    selected_body_state = {'body': None, 'body_type': None}

    def on_scroll(event):
        """Handle mouse wheel scroll for zooming"""
        # Get the current scaling factor
        scale = zoom_state['scale']

        # Determine zoom direction
        if event.button == 'up':
            # Zoom in
            scale *= 0.9
        elif event.button == 'down':
            # Zoom out
            scale *= 1.1

        # Limit zoom range (allow much closer zoom)
        scale = max(0.001, min(scale, 100.0))
        zoom_state['scale'] = scale

        # Update axis limits
        new_max_dist = initial_max_dist * scale
        ax.set_xlim([-new_max_dist, new_max_dist])
        ax.set_ylim([-new_max_dist, new_max_dist])
        ax.set_zlim([-new_max_dist, new_max_dist])

        fig.canvas.draw_idle()

    # Connect scroll event
    fig.canvas.mpl_connect('scroll_event', on_scroll)

    def on_click(event):
        """Handle mouse click to select a body"""
        if event.inaxes != ax:
            return

        # Only process left mouse button clicks
        if event.button != 1:
            return

        # Get current positions of all bodies at current time
        t = time_rate_state['current_time']

        # Get all body positions
        all_bodies = []
        all_positions = []

        # Add planets
        for p in planets:
            state = elements_to_cartesian(p.elements, float(t))
            pos = np.array(state.r) / AU
            all_bodies.append(p)
            all_positions.append(pos)

        # Add asteroids
        for a in asteroids:
            state = elements_to_cartesian(a.elements, float(t))
            pos = np.array(state.r) / AU
            all_bodies.append(a)
            all_positions.append(pos)

        # Add comets
        for c in comets:
            state = elements_to_cartesian(c.elements, float(t))
            pos = np.array(state.r) / AU
            all_bodies.append(c)
            all_positions.append(pos)

        if not all_positions:
            return

        all_positions = np.array(all_positions)

        # Convert click coordinates to data coordinates
        # For 3D plots, we need to project to 2D screen space
        try:
            from mpl_toolkits.mplot3d import proj3d

            # Get mouse position in display coordinates
            if event.xdata is None or event.ydata is None:
                return

            # Project all 3D positions to 2D display coordinates
            x2d_disp, y2d_disp = [], []
            for pos in all_positions:
                # Project 3D point to 2D
                x_proj, y_proj, _ = proj3d.proj_transform(pos[0], pos[1], pos[2], ax.get_proj())

                # Transform from axes coordinates to display coordinates
                x_disp, y_disp = ax.transData.transform((x_proj, y_proj))
                x2d_disp.append(x_disp)
                y2d_disp.append(y_disp)

            x2d_disp = np.array(x2d_disp)
            y2d_disp = np.array(y2d_disp)

            # Get click position in display coordinates
            click_x_disp, click_y_disp = event.x, event.y

            # Find closest body in 2D display space (pixels)
            distances = np.sqrt((x2d_disp - click_x_disp)**2 + (y2d_disp - click_y_disp)**2)
            closest_idx = np.argmin(distances)

            # Debug print
            print(f"Click at ({click_x_disp:.1f}, {click_y_disp:.1f}), closest body at ({x2d_disp[closest_idx]:.1f}, {y2d_disp[closest_idx]:.1f}), distance={distances[closest_idx]:.1f}px")

            # Only select if click is reasonably close (within threshold in pixels)
            if distances[closest_idx] < 30:  # 30 pixels threshold
                selected_body = all_bodies[closest_idx]
                selected_body_state['body'] = selected_body

                # Update info text
                period_years = selected_body.get_period(units='year')
                info_str = (
                    f"Selected Body:\n"
                    f"  Name:   {selected_body.name}\n"
                    f"  ID:     {selected_body.id}\n"
                    f"  Radius: {selected_body.radius:.2f} km\n"
                    f"  Weight: {selected_body.weight:.4f}\n"
                    f"  μ:      {selected_body.mu:.6e} km³/s²\n"
                    f"  Period: {period_years:.2f} years"
                )
                info_text.set_text(info_str)
                fig.canvas.draw_idle()
                print(f"Selected: {selected_body.name}")
            else:
                # Click too far from any body, clear selection
                selected_body_state['body'] = None
                info_text.set_text('')
                fig.canvas.draw_idle()
                print(f"No body within threshold (closest was {distances[closest_idx]:.1f}px away)")

        except Exception as e:
            print(f"Error selecting body: {e}")
            import traceback
            traceback.print_exc()

    # Connect click event
    fig.canvas.mpl_connect('button_press_event', on_click)

    ax.set_xlabel('X (AU)')
    ax.set_ylabel('Y (AU)')
    ax.set_zlabel('Z (AU)')
    ax.set_title('GTOC13 Solar System')

    # Plot sun at origin
    ax.scatter([0], [0], [0], c='yellow', s=200, marker='*', label='Altaira')

    # Plot orbit paths (static) - store references for toggling visibility
    print("Plotting orbit paths...")
    planet_orbit_lines = []
    for orbit in planet_orbits:
        line, = ax.plot(orbit[:, 0], orbit[:, 1], orbit[:, 2], 'b-', alpha=0.6, linewidth=1)
        planet_orbit_lines.append(line)

    asteroid_orbit_lines = []
    for orbit in asteroid_orbits:
        line, = ax.plot(orbit[:, 0], orbit[:, 1], orbit[:, 2], 'g-', alpha=0.6, linewidth=0.5)
        asteroid_orbit_lines.append(line)

    comet_orbit_lines = []
    for orbit in comet_orbits:
        line, = ax.plot(orbit[:, 0], orbit[:, 1], orbit[:, 2], 'r-', alpha=0.6, linewidth=0.5)
        comet_orbit_lines.append(line)

    # Plot solution trajectory if provided
    solution_line = None
    propagated_line = None
    if solution_data:
        # Plot the full trajectory path (static)
        positions = np.array([pt['position'] / AU for pt in solution_data['state_points']])
        solution_line, = ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
                'c-', alpha=0.8, linewidth=2, label='Solution Trajectory')

        # Plot the propagated trajectory (100 years after last flyby)
        if solution_data['propagated_positions'] is not None:
            prop_pos_au = solution_data['propagated_positions'] / AU
            propagated_line, = ax.plot(prop_pos_au[:, 0], prop_pos_au[:, 1], prop_pos_au[:, 2],
                    'm--', alpha=0.6, linewidth=1.5, label='Propagated (100 years)')

    # Create scatter plots for moving bodies
    planet_scatter = ax.scatter([], [], [], c='blue', s=50, marker='o', label='Planets')
    asteroid_scatter = ax.scatter([], [], [], c='green', s=10, marker='.', label='Asteroids', alpha=1.0)
    comet_scatter = ax.scatter([], [], [], c='red', s=20, marker='^', label='Comets', alpha=1.0)

    # Create scatter plot for spacecraft if solution is loaded
    spacecraft_scatter = None
    if solution_data:
        spacecraft_scatter = ax.scatter([], [], [], c='cyan', s=100, marker='*', label='Spacecraft')

    # Create text labels for planets
    planet_labels = []
    for p in planets:
        label = ax.text(0, 0, 0, p.name, fontsize=10, color='blue',
                       ha='left', va='bottom', rotation=0)
        planet_labels.append(label)

    ax.legend(loc='upper right')

    # Add text for displaying time
    time_text = fig.text(0.02, 0.95, '', fontsize=12, family='monospace',
                         verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Add text for displaying selected body info
    info_text = fig.text(0.02, 0.05, '', fontsize=10, family='monospace',
                         verticalalignment='bottom',
                         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    # Animation parameters
    total_time = duration_years * YEAR  # seconds
    n_frames = int(duration_years * fps)
    times = np.linspace(0, total_time, n_frames)

    # Time rate control - track cumulative time
    time_rate_state = {
        'rate': 1.0,
        'current_time': 0.0,
        'last_frame': 0,
        'paused': False
    }

    # Visibility state for body types and orbits
    visibility_state = {
        'planets': True,
        'asteroids': True,
        'comets': True,
        'orbits': True
    }

    def on_key_press(event):
        """Handle keyboard input for time rate control and visibility toggles"""
        if event.key == ' ':
            # Toggle pause
            time_rate_state['paused'] = not time_rate_state['paused']
            if time_rate_state['paused']:
                print("Animation PAUSED")
            else:
                print("Animation RESUMED")
        elif event.key == '+' or event.key == '=':
            # Speed up (double the rate)
            time_rate_state['rate'] *= 2.0
            time_rate_state['rate'] = min(time_rate_state['rate'], 64.0)  # Max 64x
            print(f"Time rate: {time_rate_state['rate']:.1f}x")
        elif event.key == '-' or event.key == '_':
            # Slow down (halve the rate)
            time_rate_state['rate'] /= 2.0
            time_rate_state['rate'] = max(time_rate_state['rate'], 0.00125)  # Min 0.00125x (100x slower)
            print(f"Time rate: {time_rate_state['rate']:.4f}x")
        elif event.key == '0':
            # Reset to normal speed
            time_rate_state['rate'] = 1.0
            print(f"Time rate: {time_rate_state['rate']:.1f}x (reset)")
        elif event.key == 'p':
            # Toggle planets visibility
            visibility_state['planets'] = not visibility_state['planets']
            planet_scatter.set_visible(visibility_state['planets'])
            for label in planet_labels:
                label.set_visible(visibility_state['planets'])
            if visibility_state['orbits']:
                for line in planet_orbit_lines:
                    line.set_visible(visibility_state['planets'])
            print(f"Planets: {'ON' if visibility_state['planets'] else 'OFF'}")
            fig.canvas.draw_idle()
        elif event.key == 'a':
            # Toggle asteroids visibility
            visibility_state['asteroids'] = not visibility_state['asteroids']
            asteroid_scatter.set_visible(visibility_state['asteroids'])
            if visibility_state['orbits']:
                for line in asteroid_orbit_lines:
                    line.set_visible(visibility_state['asteroids'])
            print(f"Asteroids: {'ON' if visibility_state['asteroids'] else 'OFF'}")
            fig.canvas.draw_idle()
        elif event.key == 'c':
            # Toggle comets visibility
            visibility_state['comets'] = not visibility_state['comets']
            comet_scatter.set_visible(visibility_state['comets'])
            if visibility_state['orbits']:
                for line in comet_orbit_lines:
                    line.set_visible(visibility_state['comets'])
            print(f"Comets: {'ON' if visibility_state['comets'] else 'OFF'}")
            fig.canvas.draw_idle()
        elif event.key == 'o':
            # Toggle orbit tracks visibility
            visibility_state['orbits'] = not visibility_state['orbits']
            if visibility_state['planets']:
                for line in planet_orbit_lines:
                    line.set_visible(visibility_state['orbits'])
            if visibility_state['asteroids']:
                for line in asteroid_orbit_lines:
                    line.set_visible(visibility_state['orbits'])
            if visibility_state['comets']:
                for line in comet_orbit_lines:
                    line.set_visible(visibility_state['orbits'])
            print(f"Orbit tracks: {'ON' if visibility_state['orbits'] else 'OFF'}")
            fig.canvas.draw_idle()

    # Connect keyboard event
    fig.canvas.mpl_connect('key_press_event', on_key_press)

    def init():
        """Initialize animation"""
        planet_scatter._offsets3d = ([], [], [])
        asteroid_scatter._offsets3d = ([], [], [])
        comet_scatter._offsets3d = ([], [], [])
        if spacecraft_scatter:
            spacecraft_scatter._offsets3d = ([], [], [])
        time_text.set_text('')
        info_text.set_text('')
        time_rate_state['current_time'] = 0.0
        time_rate_state['last_frame'] = 0
        for label in planet_labels:
            label.set_position((0, 0))
            label.set_3d_properties(0)
        artists = [planet_scatter, asteroid_scatter, comet_scatter, time_text, info_text] + planet_labels
        if spacecraft_scatter:
            artists.append(spacecraft_scatter)
        return artists

    def update(frame):
        """Update animation frame"""
        # Calculate time increment based on frame advance and current rate
        frame_delta = frame - time_rate_state['last_frame']
        time_rate_state['last_frame'] = frame

        # Only advance time if not paused
        if not time_rate_state['paused']:
            # Advance time by the delta scaled by rate
            base_time_step = total_time / n_frames
            time_rate_state['current_time'] += frame_delta * base_time_step * time_rate_state['rate']

            # Wrap time to stay within bounds
            time_rate_state['current_time'] = time_rate_state['current_time'] % total_time

        t = time_rate_state['current_time']

        # Compute planet positions
        planet_pos = []
        for p in planets:
            state = elements_to_cartesian(p.elements, float(t))
            planet_pos.append(np.array(state.r) / AU)
        planet_pos = np.array(planet_pos)

        # Compute asteroid positions
        asteroid_pos = []
        for a in asteroids:
            state = elements_to_cartesian(a.elements, float(t))
            asteroid_pos.append(np.array(state.r) / AU)
        asteroid_pos = np.array(asteroid_pos)

        # Compute comet positions
        comet_pos = []
        for c in comets:
            state = elements_to_cartesian(c.elements, float(t))
            comet_pos.append(np.array(state.r) / AU)
        comet_pos = np.array(comet_pos)

        # Update scatter plots
        planet_scatter._offsets3d = (planet_pos[:, 0], planet_pos[:, 1], planet_pos[:, 2])
        asteroid_scatter._offsets3d = (asteroid_pos[:, 0], asteroid_pos[:, 1], asteroid_pos[:, 2])
        comet_scatter._offsets3d = (comet_pos[:, 0], comet_pos[:, 1], comet_pos[:, 2])

        # Update planet labels to follow planets
        for i, label in enumerate(planet_labels):
            x, y, z = planet_pos[i]
            label.set_position((x, y))
            label.set_3d_properties(z, zdir='z')

        # Update spacecraft position if solution is loaded
        if spacecraft_scatter and solution_data:
            # Find spacecraft position at current time
            state_points = solution_data['state_points']
            max_solution_time = solution_data['max_time']
            propagated_positions = solution_data['propagated_positions']
            propagated_times = solution_data['propagated_times']

            if t <= max_solution_time:
                # Interpolate position between state points
                # Find the two state points that bracket current time
                for i in range(len(state_points) - 1):
                    if state_points[i]['epoch'] <= t <= state_points[i + 1]['epoch']:
                        # Linear interpolation
                        t0 = state_points[i]['epoch']
                        t1 = state_points[i + 1]['epoch']
                        p0 = state_points[i]['position']
                        p1 = state_points[i + 1]['position']

                        if t1 > t0:
                            alpha = (t - t0) / (t1 - t0)
                            sc_pos = p0 + alpha * (p1 - p0)
                        else:
                            sc_pos = p0

                        sc_pos_au = sc_pos / AU
                        spacecraft_scatter._offsets3d = ([sc_pos_au[0]], [sc_pos_au[1]], [sc_pos_au[2]])
                        break
                else:
                    # If t is before first state point or at exact match
                    if t <= state_points[0]['epoch']:
                        sc_pos_au = state_points[0]['position'] / AU
                        spacecraft_scatter._offsets3d = ([sc_pos_au[0]], [sc_pos_au[1]], [sc_pos_au[2]])
            elif propagated_positions is not None and t <= propagated_times[-1]:
                # Interpolate position in the propagated trajectory
                # Find the two propagated points that bracket current time
                idx = np.searchsorted(propagated_times, t)
                if idx == 0:
                    sc_pos = propagated_positions[0]
                elif idx >= len(propagated_times):
                    sc_pos = propagated_positions[-1]
                else:
                    # Linear interpolation between propagated points
                    t0 = propagated_times[idx - 1]
                    t1 = propagated_times[idx]
                    p0 = propagated_positions[idx - 1]
                    p1 = propagated_positions[idx]

                    if t1 > t0:
                        alpha = (t - t0) / (t1 - t0)
                        sc_pos = p0 + alpha * (p1 - p0)
                    else:
                        sc_pos = p0

                sc_pos_au = sc_pos / AU
                spacecraft_scatter._offsets3d = ([sc_pos_au[0]], [sc_pos_au[1]], [sc_pos_au[2]])
            else:
                # Keep spacecraft at final position after propagated trajectory ends
                if propagated_positions is not None:
                    final_pos = propagated_positions[-1] / AU
                else:
                    final_pos = state_points[-1]['position'] / AU
                spacecraft_scatter._offsets3d = ([final_pos[0]], [final_pos[1]], [final_pos[2]])

        # Update title with current time
        ax.set_title(f'GTOC13 Solar System')

        # Update time display
        time_years = t / YEAR
        time_days = t / DAY
        rate = time_rate_state['rate']
        paused = time_rate_state['paused']
        pause_str = " [PAUSED]" if paused else ""
        time_text.set_text(
            f'Epoch: {t:.2f} s\n'
            f'Time:  {time_years:.2f} years\n'
            f'       {time_days:.1f} days\n'
            f'Rate:  {rate:.4f}x{pause_str}'
        )

        artists = [planet_scatter, asteroid_scatter, comet_scatter, time_text, info_text] + planet_labels
        if spacecraft_scatter:
            artists.append(spacecraft_scatter)
        return artists

    print(f"Creating animation with {n_frames} frames...")
    anim = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=n_frames,
        interval=1000/fps,
        blit=False
    )

    return fig, anim


def main():
    """Main function to create and display animation"""
    import sys

    # Check for solution file argument
    solution_file = None
    if len(sys.argv) > 1:
        solution_file = sys.argv[1]
        print(f"Loading solution from: {solution_file}")

    # Create animation
    fig, anim = create_animation(
        duration_years=200.0,  # Animate 200 years
        fps=30,
        n_orbit_points=100,
        solution_file=solution_file
    )

    # Save or show animation
    print("Displaying animation...")
    print("\nControls:")
    print("  Mouse click: Select body (shows info in lower left)")
    print("  Mouse wheel: Zoom in/out")
    print("  Mouse drag:  Rotate view")
    print("  Spacebar:    Pause/Resume")
    print("  + or =:      Speed up time (2x)")
    print("  - or _:      Slow down time (0.5x)")
    print("  0:           Reset time rate to 1x")
    print("  p:           Toggle planets visibility")
    print("  a:           Toggle asteroids visibility")
    print("  c:           Toggle comets visibility")
    print("  o:           Toggle orbit tracks visibility")
    print("\nClose the window to exit.")
    plt.show()

    # Optionally save to file
    # print("Saving animation to file...")
    # anim.save('gtoc13_orbits.mp4', writer='ffmpeg', fps=30, dpi=100)
    # print("Animation saved!")


if __name__ == '__main__':
    main()
