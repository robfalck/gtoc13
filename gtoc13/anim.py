"""
Animated 3D visualization of GTOC13 orbits (planets, asteroids, comets)
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import csv
from pathlib import Path
import jax.numpy as jnp

from gtoc13 import OrbitalElements, elements_to_cartesian, AU, MU_ALTAIRA


def load_planets(filepath: str) -> list[OrbitalElements]:
    """Load planet data from CSV file"""
    planets = []
    with open(filepath, 'r', encoding='latin-1') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Handle the # prefix in the first column name
            planet_id_key = '#Planet ID' if '#Planet ID' in row else 'Planet ID'

            # Convert degrees to radians
            elements = OrbitalElements(
                a=float(row['Semi-Major Axis (km)']),
                e=float(row['Eccentricity ()']),
                i=np.deg2rad(float(row['Inclination (deg)'])),
                Omega=np.deg2rad(float(row['Longitude of the Ascending Node (deg)'])),
                omega=np.deg2rad(float(row['Argument of Periapsis (deg)'])),
                M0=np.deg2rad(float(row['Mean Anomaly at t=0 (deg)'])),
                mu_body=float(row['GM (km3/s2)']),
                radius=float(row['Radius (km)']),
                weight=float(row['Weight ()'])
            )
            planets.append({
                'id': row[planet_id_key],
                'name': row['Name'],
                'elements': elements
            })
    return planets


def load_small_bodies(filepath: str, id_prefix: str) -> list[OrbitalElements]:
    """Load asteroid or comet data from CSV file"""
    bodies = []
    with open(filepath, 'r', encoding='latin-1') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Handle the # prefix in the first column name (with or without space)
            id_key_options = [f'#{id_prefix} ID', f'# {id_prefix} ID', f'{id_prefix} ID']
            id_key = next((key for key in id_key_options if key in row), None)

            if id_key is None:
                raise KeyError(f"Could not find ID column. Available columns: {list(row.keys())}")

            # Convert degrees to radians
            # Handle different column names for Mean Anomaly
            m0_key = 'Mean Anomaly at t=0 (deg)' if 'Mean Anomaly at t=0 (deg)' in row else 'Mean Anomaly at t=0'

            elements = OrbitalElements(
                a=float(row['Semi-Major Axis (km)']),
                e=float(row['Eccentricity ()']),
                i=np.deg2rad(float(row['Inclination (deg)'])),
                Omega=np.deg2rad(float(row['Longitude of the Ascending Node (deg)'])),
                omega=np.deg2rad(float(row['Argument of Periapsis (deg)'])),
                M0=np.deg2rad(float(row[m0_key])),
                mu_body=0.0,  # Small bodies have negligible mass
                radius=0.0,
                weight=float(row['Weight ()'])
            )
            body_id = row[id_key]
            bodies.append({
                'id': body_id,
                'elements': elements
            })
    return bodies


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


def create_animation(
    planets_file: str,
    asteroids_file: str,
    comets_file: str,
    duration_years: float = 50.0,
    fps: int = 30,
    n_orbit_points: int = 200
):
    """
    Create an animated 3D visualization of the solar system.

    Args:
        planets_file: Path to planets CSV
        asteroids_file: Path to asteroids CSV
        comets_file: Path to comets CSV
        duration_years: Duration of animation in years
        fps: Frames per second
        n_orbit_points: Number of points to plot for each orbit
    """
    print("Loading orbital data...")
    planets = load_planets(planets_file)
    asteroids = load_small_bodies(asteroids_file, 'Asteroid')
    comets = load_small_bodies(comets_file, 'Comet')

    print(f"Loaded {len(planets)} planets, {len(asteroids)} asteroids, {len(comets)} comets")

    # Compute orbit paths (static)
    print("Computing orbit paths...")
    planet_orbits = [compute_orbit_points(p['elements'], n_orbit_points) for p in planets]
    asteroid_orbits = [compute_orbit_points(a['elements'], n_orbit_points) for a in asteroids]
    comet_orbits = [compute_orbit_points(c['elements'], n_orbit_points) for c in comets]

    # Setup figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Set up the plot limits
    max_dist = max([np.max(np.abs(orbit)) for orbit in planet_orbits]) * 1.1
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
        ax.plot(orbit[:, 0], orbit[:, 1], orbit[:, 2], 'b-', alpha=0.3, linewidth=1)

    for orbit in asteroid_orbits:
        ax.plot(orbit[:, 0], orbit[:, 1], orbit[:, 2], 'g-', alpha=0.1, linewidth=0.5)

    for orbit in comet_orbits:
        ax.plot(orbit[:, 0], orbit[:, 1], orbit[:, 2], 'r-', alpha=0.1, linewidth=0.5)

    # Create scatter plots for moving bodies
    planet_scatter = ax.scatter([], [], [], c='blue', s=50, marker='o', label='Planets')
    asteroid_scatter = ax.scatter([], [], [], c='green', s=10, marker='.', label='Asteroids', alpha=0.6)
    comet_scatter = ax.scatter([], [], [], c='red', s=20, marker='^', label='Comets', alpha=0.6)

    ax.legend(loc='upper right')

    # Add text for displaying time
    time_text = fig.text(0.02, 0.95, '', fontsize=12, family='monospace',
                         verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Animation parameters
    from gtoc13 import YEAR, DAY
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
        time_text.set_text('')
        time_rate_state['current_time'] = 0.0
        time_rate_state['last_frame'] = 0
        return planet_scatter, asteroid_scatter, comet_scatter, time_text

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
            state = elements_to_cartesian(p['elements'], float(t))
            planet_pos.append(np.array(state.r) / AU)
        planet_pos = np.array(planet_pos)

        # Compute asteroid positions
        asteroid_pos = []
        for a in asteroids:
            state = elements_to_cartesian(a['elements'], float(t))
            asteroid_pos.append(np.array(state.r) / AU)
        asteroid_pos = np.array(asteroid_pos)

        # Compute comet positions
        comet_pos = []
        for c in comets:
            state = elements_to_cartesian(c['elements'], float(t))
            comet_pos.append(np.array(state.r) / AU)
        comet_pos = np.array(comet_pos)

        # Update scatter plots
        planet_scatter._offsets3d = (planet_pos[:, 0], planet_pos[:, 1], planet_pos[:, 2])
        asteroid_scatter._offsets3d = (asteroid_pos[:, 0], asteroid_pos[:, 1], asteroid_pos[:, 2])
        comet_scatter._offsets3d = (comet_pos[:, 0], comet_pos[:, 1], comet_pos[:, 2])

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

        return planet_scatter, asteroid_scatter, comet_scatter, time_text

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
    # Get data directory
    data_dir = Path(__file__).parent / 'data'

    planets_file = data_dir / 'gtoc13_planets.csv'
    asteroids_file = data_dir / 'gtoc13_asteroids.csv'
    comets_file = data_dir / 'gtoc13_comets.csv'

    # Create animation
    fig, anim = create_animation(
        str(planets_file),
        str(asteroids_file),
        str(comets_file),
        duration_years=50.0,  # Animate 50 years
        fps=30,
        n_orbit_points=200
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
