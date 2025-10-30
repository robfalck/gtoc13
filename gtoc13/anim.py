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
)
from gtoc13.constants import MU_ALTAIRA, YEAR, DAY


def compute_orbit_points(elements: OrbitalElements, n_points: int = 100) -> np.ndarray:
    """
    Compute points along an orbit for visualization.
    Returns array of shape (n_points, 3) with [x, y, z] positions in AU.
    """
    # Compute one full orbital period
    period = 2 * np.pi * np.sqrt(elements.a**3 / MU_ALTAIRA)
    times = np.linspace(0, period, n_points)

    positions = []
    for t in times:
        state = elements_to_cartesian(elements, float(t))
        positions.append(np.array(state.r))

    # Convert to numpy array and scale to AU
    return np.array(positions) / AU


def load_solution_trajectory(filepath: str) -> dict:
    """
    Load a solution trajectory from a GTOC13 solution file.
    Returns dict with 'state_points' list and 'max_time'.
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

    return {
        'state_points': state_points,
        'max_time': max_time
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
    initial_max_dist = max_dist
    ax.set_xlim([-max_dist, max_dist])
    ax.set_ylim([-max_dist, max_dist])
    ax.set_zlim([-max_dist, max_dist])

    # Zoom state
    zoom_state = {'scale': 1.0}

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

        # Limit zoom range
        scale = max(0.1, min(scale, 10.0))
        zoom_state['scale'] = scale

        # Update axis limits
        new_max_dist = initial_max_dist * scale
        ax.set_xlim([-new_max_dist, new_max_dist])
        ax.set_ylim([-new_max_dist, new_max_dist])
        ax.set_zlim([-new_max_dist, new_max_dist])

        fig.canvas.draw_idle()

    # Connect scroll event
    fig.canvas.mpl_connect('scroll_event', on_scroll)

    ax.set_xlabel('X (AU)')
    ax.set_ylabel('Y (AU)')
    ax.set_zlabel('Z (AU)')
    ax.set_title('GTOC13 Solar System')

    # Plot sun at origin
    ax.scatter([0], [0], [0], c='yellow', s=200, marker='*', label='Altaira')

    # Plot orbit paths (static)
    print("Plotting orbit paths...")
    for orbit in planet_orbits:
        ax.plot(orbit[:, 0], orbit[:, 1], orbit[:, 2], 'b-', alpha=0.6, linewidth=1)

    for orbit in asteroid_orbits:
        ax.plot(orbit[:, 0], orbit[:, 1], orbit[:, 2], 'g-', alpha=0.6, linewidth=0.5)

    for orbit in comet_orbits:
        ax.plot(orbit[:, 0], orbit[:, 1], orbit[:, 2], 'r-', alpha=0.6, linewidth=0.5)

    # Plot solution trajectory if provided
    solution_line = None
    if solution_data:
        # Plot the full trajectory path (static)
        positions = np.array([pt['position'] / AU for pt in solution_data['state_points']])
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
                'c-', alpha=0.8, linewidth=2, label='Solution Trajectory')

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

    # Animation parameters
    total_time = duration_years * YEAR  # seconds
    n_frames = int(duration_years * fps)
    times = np.linspace(0, total_time, n_frames)

    # Time rate control - track cumulative time
    time_rate_state = {
        'rate': 1.0,
        'current_time': 0.0,
        'last_frame': 0
    }

    def on_key_press(event):
        """Handle keyboard input for time rate control"""
        if event.key == '+' or event.key == '=':
            # Speed up (double the rate)
            time_rate_state['rate'] *= 2.0
            time_rate_state['rate'] = min(time_rate_state['rate'], 64.0)  # Max 64x
            print(f"Time rate: {time_rate_state['rate']:.1f}x")
        elif event.key == '-' or event.key == '_':
            # Slow down (halve the rate)
            time_rate_state['rate'] /= 2.0
            time_rate_state['rate'] = max(time_rate_state['rate'], 0.125)  # Min 0.125x
            print(f"Time rate: {time_rate_state['rate']:.1f}x")
        elif event.key == '0':
            # Reset to normal speed
            time_rate_state['rate'] = 1.0
            print(f"Time rate: {time_rate_state['rate']:.1f}x (reset)")

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
        time_rate_state['current_time'] = 0.0
        time_rate_state['last_frame'] = 0
        for label in planet_labels:
            label.set_position((0, 0))
            label.set_3d_properties(0)
        artists = [planet_scatter, asteroid_scatter, comet_scatter, time_text] + planet_labels
        if spacecraft_scatter:
            artists.append(spacecraft_scatter)
        return artists

    def update(frame):
        """Update animation frame"""
        # Calculate time increment based on frame advance and current rate
        frame_delta = frame - time_rate_state['last_frame']
        time_rate_state['last_frame'] = frame

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
            else:
                # Keep spacecraft at final position after trajectory ends
                final_pos = state_points[-1]['position'] / AU
                spacecraft_scatter._offsets3d = ([final_pos[0]], [final_pos[1]], [final_pos[2]])

        # Update title with current time
        ax.set_title(f'GTOC13 Solar System')

        # Update time display
        time_years = t / YEAR
        time_days = t / DAY
        rate = time_rate_state['rate']
        time_text.set_text(
            f'Epoch: {t:.2f} s\n'
            f'Time:  {time_years:.2f} years\n'
            f'       {time_days:.1f} days\n'
            f'Rate:  {rate:.2f}x'
        )

        artists = [planet_scatter, asteroid_scatter, comet_scatter, time_text] + planet_labels
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
    print("  Mouse wheel: Zoom in/out")
    print("  Mouse drag:  Rotate view")
    print("  + or =:      Speed up time (2x)")
    print("  - or _:      Slow down time (0.5x)")
    print("  0:           Reset time rate to 1x")
    print("\nClose the window to exit.")
    plt.show()

    # Optionally save to file
    # print("Saving animation to file...")
    # anim.save('gtoc13_orbits.mp4', writer='ffmpeg', fps=30, dpi=100)
    # print("Animation saved!")


if __name__ == '__main__':
    main()
