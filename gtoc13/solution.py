"""
GTOC13 Solution representation using Pydantic models.
Based on the GTOC13 Solution File Format Specification.
"""
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Literal, Tuple, TextIO
import sys
import numpy as np
from pathlib import Path

from scipy.interpolate import BarycentricInterpolator
from scipy.integrate import solve_ivp

from gtoc13.constants import MU_ALTAIRA, R0, KMPDU
from gtoc13.odes import solar_sail_ode, solar_sail_acceleration


class StatePoint(BaseModel):
    """
    A single state point in the trajectory.
    Each row in the solution file represents one state point.
    """
    body_id: int = Field(
        ...,
        description="Body identifier: 0 for heliocentric, >0 for flyby body ID"
    )
    flag: int = Field(
        ...,
        description="Arc type flag: 0 for conic/non-science, 1 for propagated/science"
    )
    epoch: float = Field(
        ...,
        description="Time in seconds past t=0"
    )
    position: Tuple[float, float, float] = Field(
        ...,
        description="Heliocentric position vector [x, y, z] in km"
    )
    velocity: Tuple[float, float, float] = Field(
        ...,
        description="Heliocentric velocity vector [vx, vy, vz] in km/s"
    )
    control: Tuple[float, float, float] = Field(
        ...,
        description="Control vector: unit normal for propagated arcs, v_inf for flybys, [0,0,0] for conic"
    )

    @field_validator('body_id')
    @classmethod
    def validate_body_id(cls, v):
        if v < 0:
            raise ValueError("body_id must be non-negative")
        return v

    @field_validator('flag')
    @classmethod
    def validate_flag(cls, v):
        if v not in [0, 1]:
            raise ValueError("flag must be 0 or 1")
        return v

    def to_row(self) -> str:
        """Convert state point to a solution file row"""
        return (
            f"{self.body_id:d} {self.flag:d} {self.epoch:.6f} "
            f"{self.position[0]:.6f} {self.position[1]:.6f} {self.position[2]:.6f} "
            f"{self.velocity[0]:.6f} {self.velocity[1]:.6f} {self.velocity[2]:.6f} "
            f"{self.control[0]:.6f} {self.control[1]:.6f} {self.control[2]:.6f}"
        )


class FlybyArc(BaseModel):
    """
    Represents a flyby arc (gravity assist maneuver).
    Consists of exactly 2 consecutive state points with body_id > 0.
    """
    body_id: int = Field(..., gt=0, description="Body identifier for flyby")
    is_science: bool = Field(..., description="Whether this flyby counts for science scoring")
    epoch: float = Field(..., description="Flyby epoch in seconds")
    position: Tuple[float, float, float] = Field(..., description="Spacecraft position at flyby (km)")
    velocity_in: Tuple[float, float, float] = Field(..., description="Incoming heliocentric velocity (km/s)")
    velocity_out: Tuple[float, float, float] = Field(..., description="Outgoing heliocentric velocity (km/s)")
    v_inf_in: Tuple[float, float, float] = Field(..., description="Incoming v_infinity vector (km/s)")
    v_inf_out: Tuple[float, float, float] = Field(..., description="Outgoing v_infinity vector (km/s)")

    def to_state_points(self) -> List[StatePoint]:
        """Convert flyby arc to two state points"""
        flag = 1 if self.is_science else 0
        return [
            StatePoint(
                body_id=self.body_id,
                flag=flag,
                epoch=self.epoch,
                position=self.position,
                velocity=self.velocity_in,
                control=self.v_inf_in
            ),
            StatePoint(
                body_id=self.body_id,
                flag=flag,
                epoch=self.epoch,
                position=self.position,
                velocity=self.velocity_out,
                control=self.v_inf_out
            )
        ]

    @staticmethod
    def create(
        body_id: int,
        epoch: float,
        position: Tuple[float, float, float],
        velocity_in: Tuple[float, float, float],
        velocity_out: Tuple[float, float, float],
        v_inf_in: Tuple[float, float, float],
        v_inf_out: Tuple[float, float, float],
        is_science: bool = True
    ) -> 'FlybyArc':
        """
        Create a flyby arc.

        Args:
            body_id: Body identifier (must be > 0)
            epoch: Time of flyby (seconds)
            position: Spacecraft position at flyby (km)
            velocity_in: Incoming heliocentric velocity (km/s)
            velocity_out: Outgoing heliocentric velocity (km/s)
            v_inf_in: Incoming v_infinity vector (km/s)
            v_inf_out: Outgoing v_infinity vector (km/s)
            is_science: Whether this flyby counts for science scoring (default: True)

        Returns:
            FlybyArc object
        """
        return FlybyArc(
            body_id=body_id,
            is_science=is_science,
            epoch=epoch,
            position=position,
            velocity_in=velocity_in,
            velocity_out=velocity_out,
            v_inf_in=v_inf_in,
            v_inf_out=v_inf_out
        )


class ConicArc(BaseModel):
    """
    Represents a conic arc (ballistic coast, Keplerian motion).
    Consists of exactly 2 consecutive state points with body_id=0, flag=0.
    """
    epoch_start: float = Field(..., description="Start epoch in seconds")
    epoch_end: float = Field(..., description="End epoch in seconds")
    position_start: Tuple[float, float, float] = Field(..., description="Start position (km)")
    position_end: Tuple[float, float, float] = Field(..., description="End position (km)")
    velocity_start: Tuple[float, float, float] = Field(..., description="Start velocity (km/s)")
    velocity_end: Tuple[float, float, float] = Field(..., description="End velocity (km/s)")

    @model_validator(mode='after')
    def validate_time_order(self):
        if self.epoch_end <= self.epoch_start:
            raise ValueError("epoch_end must be greater than epoch_start")
        return self

    def to_state_points(self) -> List[StatePoint]:
        """Convert conic arc to two state points"""
        return [
            StatePoint(
                body_id=0,
                flag=0,
                epoch=self.epoch_start,
                position=self.position_start,
                velocity=self.velocity_start,
                control=(0.0, 0.0, 0.0)
            ),
            StatePoint(
                body_id=0,
                flag=0,
                epoch=self.epoch_end,
                position=self.position_end,
                velocity=self.velocity_end,
                control=(0.0, 0.0, 0.0)
            )
        ]

    @staticmethod
    def create(
        epoch_start: float,
        epoch_end: float,
        position_start: Tuple[float, float, float],
        position_end: Tuple[float, float, float],
        velocity_start: Tuple[float, float, float],
        velocity_end: Tuple[float, float, float]
    ) -> 'ConicArc':
        """
        Create a conic arc (ballistic coast).

        Args:
            epoch_start: Start time (seconds)
            epoch_end: End time (seconds)
            position_start: Start position (km)
            position_end: End position (km)
            velocity_start: Start velocity (km/s)
            velocity_end: End velocity (km/s)

        Returns:
            ConicArc object
        """
        return ConicArc(
            epoch_start=epoch_start,
            epoch_end=epoch_end,
            position_start=position_start,
            position_end=position_end,
            velocity_start=velocity_start,
            velocity_end=velocity_end
        )


class PropagatedArc(BaseModel):
    """
    Represents a propagated arc (solar sail propulsion or numerical integration).
    Consists of 2 or more state points with body_id=0, flag=1.
    """
    state_points: List[StatePoint] = Field(
        ...,
        min_length=2,
        description="List of state points (minimum 2)"
    )

    @model_validator(mode='after')
    def validate_propagated_arc(self):
        # Check all points have body_id=0 and flag=1
        for i, pt in enumerate(self.state_points):
            if pt.body_id != 0:
                raise ValueError(f"State point {i}: body_id must be 0 for propagated arc")
            if pt.flag != 1:
                raise ValueError(f"State point {i}: flag must be 1 for propagated arc")

        # Check time is non-decreasing and minimum step is 60s
        for i in range(1, len(self.state_points)):
            dt = self.state_points[i].epoch - self.state_points[i-1].epoch
            if dt < 0:
                raise ValueError(f"Time must be non-decreasing at point {i}")
            # Allow dt=0 for control discontinuities, otherwise require >= 60s
            if dt > 0 and dt < 60.0:
                raise ValueError(f"Minimum time step is 60s, got {dt}s at point {i}")

        return self

    def to_state_points(self) -> List[StatePoint]:
        """Return the list of state points"""
        return self.state_points

    @staticmethod
    def create(
        epochs: List[float],
        positions: List[Tuple[float, float, float]],
        velocities: List[Tuple[float, float, float]],
        controls: List[Tuple[float, float, float]]
    ) -> 'PropagatedArc':
        """
        Create a propagated arc from lists of epochs, positions, velocities, and controls.

        To provide an accurate simulated solution, provide each of these items at points that
        correspond to LGL or CGL nodes in a polynomial.

        Args:
            epochs: List of time points (seconds)
            positions: List of position vectors (km)
            velocities: List of velocity vectors (km/s)
            controls: List of control vectors (sail normal unit vectors)

        Returns:
            PropagatedArc object

        Raises:
            ValueError: If lists have different lengths
        """
        if not (len(epochs) == len(positions) == len(velocities) == len(controls)):
            raise ValueError("All lists must have the same length")

        state_points = [
            StatePoint(
                body_id=0,
                flag=1,
                epoch=epoch,
                position=pos,
                velocity=vel,
                control=ctrl
            )
            for epoch, pos, vel, ctrl in zip(epochs, positions, velocities, controls)
        ]

        # t, r, v, u = PropagatedArc.simulate(epochs, positions, velocities, controls)

        # state_points = [
        #     StatePoint(
        #         body_id=0,
        #         flag=1,
        #         epoch=epoch,
        #         position=pos,
        #         velocity=vel,
        #         control=ctrl
        #     )
        #     for epoch, pos, vel, ctrl in zip(t, r, v, u)
        # ]

        return PropagatedArc(state_points=state_points)

    @staticmethod
    def simulate(epochs, positions, velocities, controls):

        # Convert to numpy arrays
        epochs = np.array(epochs)
        positions = np.array(positions)
        velocities = np.array(velocities)
        controls = np.array(controls)

        # Map epochs onto [-1, 1]
        t0 = epochs[0]
        tf = epochs[-1]
        tau = 2.0 * (epochs - t0) / (tf - t0) - 1.0

        # Create separate interpolators for each control component
        # BarycentricInterpolator works on 1D data, so we need one per component
        # Ensure we're using flattened 1D arrays
        interp_u0 = BarycentricInterpolator(epochs.flatten(), controls[:, 0].flatten())
        interp_u1 = BarycentricInterpolator(epochs.flatten(), controls[:, 1].flatten())
        interp_u2 = BarycentricInterpolator(epochs.flatten(), controls[:, 2].flatten())
  
        def _sim_ode(t, y):
            r = y[:3]
            v = y[-3:]
            u_n = np.hstack([interp_u0(t), interp_u1(t), interp_u2(t)])

            r_mag = np.linalg.norm(r)
            
            a_grav = -MU_ALTAIRA * r / r_mag**3    
            a_sail, cos_alpha = solar_sail_acceleration(r, u_n, 1.0)
            a_total = a_grav #+ a_sail

            ydot = np.concatenate([v, a_total])
            return ydot

        y0 = np.concatenate((positions[0, :], velocities[0, :]))

        sol = solve_ivp(_sim_ode, t_span=(t0, tf), y0=y0, method='RK45', t_eval=None,
                        dense_output=False, first_step=86400.0, atol=1.0E-12, rtol=1.0E-12)

        t = sol.t
        r = sol.y.T[:, :3]
        v = sol.y.T[:, 3:]

        # Evaluate control at all times
        u = np.array([interp_u0(t), interp_u1(t), interp_u2(t)]).T
        u = np.zeros_like(v)

        return t, r, v, u



class GTOC13Solution(BaseModel):
    """
    Complete GTOC13 trajectory solution.
    Contains a sequence of arcs (flyby, conic, and propagated).
    """
    arcs: List[FlybyArc | ConicArc | PropagatedArc] = Field(
        ...,
        description="Sequence of trajectory arcs"
    )
    comments: List[str] = Field(
        default_factory=list,
        description="Optional comments to include in the file header"
    )

    @model_validator(mode='after')
    def validate_solution(self):
        if len(self.arcs) == 0:
            raise ValueError("Solution must contain at least one arc")

        # First arc must be heliocentric
        first_states = self.arcs[0].to_state_points()
        if first_states[0].body_id != 0:
            raise ValueError("First arc must be heliocentric (body_id=0)")

        return self

    def to_state_points(self) -> List[StatePoint]:
        """Convert all arcs to a flat list of state points"""
        all_points = []
        for arc in self.arcs:
            all_points.extend(arc.to_state_points())
        return all_points

    def write(self, stream: TextIO = sys.stdout, precision: int = 15) -> None:
        """
        Write the solution to a stream in GTOC13 submission format.

        Args:
            stream: Output stream (default: sys.stdout)
            precision: Number of decimal places for floating point numbers (default: 15)
        """
        from datetime import datetime
        from gtoc13.bodies import bodies_data
        from gtoc13.constants import YEAR

        # Write header comments
        stream.write("# GTOC13 Solution File\n")
        stream.write(f"# Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        stream.write("# Units: epoch(s), position(km), velocity(km/s)\n")

        # Write custom comments
        for comment in self.comments:
            if not comment.startswith('#'):
                comment = f"# {comment}"
            stream.write(f"{comment}\n")

        stream.write("#\n")

        # Write arcs with headers
        for arc in self.arcs:
            if isinstance(arc, PropagatedArc):
                # Propagated arc header
                t_start = arc.state_points[0].epoch / YEAR
                t_end = arc.state_points[-1].epoch / YEAR
                stream.write(f"# Propagated Arc: Body 0 (heliocentric) from t={t_start:.6f} years to t={t_end:.6f} years\n")
                stream.write(f"# {'body_id':>10} {'flag':>6} {'epoch (s)':>{16+8}} "
                           f"{'x (km)':>{precision+8}} {'y (km)':>{precision+8}} {'z (km)':>{precision+8}} "
                           f"{'vx (km/s)':>{precision+8}} {'vy (km/s)':>{precision+8}} {'vz (km/s)':>{precision+8}} "
                           f"{'cx':>{precision+8}} {'cy':>{precision+8}} {'cz':>{precision+8}}\n")
            elif isinstance(arc, ConicArc):
                # Conic arc header
                t_start = arc.epoch_start / YEAR
                t_end = arc.epoch_end / YEAR
                stream.write(f"# Conic Arc: Body 0 (heliocentric) from t={t_start:.6f} years to t={t_end:.6f} years\n")
                stream.write(f"# {'body_id':>10} {'flag':>6} {'epoch (s)':>{16+8}} "
                           f"{'x (km)':>{precision+8}} {'y (km)':>{precision+8}} {'z (km)':>{precision+8}} "
                           f"{'vx (km/s)':>{precision+8}} {'vy (km/s)':>{precision+8}} {'vz (km/s)':>{precision+8}} "
                           f"{'cx':>{precision+8}} {'cy':>{precision+8}} {'cz':>{precision+8}}\n")
            elif isinstance(arc, FlybyArc):
                # Flyby arc header
                t_flyby = arc.epoch / YEAR
                body_name = bodies_data.get(arc.body_id, None)
                if body_name and hasattr(body_name, 'name'):
                    name_str = f" ({body_name.name})"
                else:
                    name_str = ""
                # Calculate v_inf magnitude from the incoming v_inf vector
                v_inf_mag = np.sqrt(arc.v_inf_in[0]**2 + arc.v_inf_in[1]**2 + arc.v_inf_in[2]**2)
                stream.write(f"# Flyby of Body {arc.body_id}{name_str} at t={t_flyby:.6f} years, v_inf={v_inf_mag:.6f} km/s\n")
                stream.write(f"# {'body_id':>10} {'flag':>6} {'epoch (s)':>{16+8}} "
                           f"{'x (km)':>{precision+8}} {'y (km)':>{precision+8}} {'z (km)':>{precision+8}} "
                           f"{'vx (km/s)':>{precision+8}} {'vy (km/s)':>{precision+8}} {'vz (km/s)':>{precision+8}} "
                           f"{'v_inf_x':>{precision+8}} {'v_inf_y':>{precision+8}} {'v_inf_z':>{precision+8}}\n")

            # Write state points for this arc
            for point in arc.to_state_points():
                # Format with specified precision (scientific notation)
                line = (
                    f"  {point.body_id:>10d} {point.flag:>6d} {point.epoch:>{16+8}.{16}e} "
                    f"{point.position[0]:>{precision+8}.{precision}e} {point.position[1]:>{precision+8}.{precision}e} {point.position[2]:>{precision+8}.{precision}e} "
                    f"{point.velocity[0]:>{precision+8}.{precision}e} {point.velocity[1]:>{precision+8}.{precision}e} {point.velocity[2]:>{precision+8}.{precision}e} "
                    f"{point.control[0]:>{precision+8}.{precision}e} {point.control[1]:>{precision+8}.{precision}e} {point.control[2]:>{precision+8}.{precision}e}\n"
                )
                stream.write(line)

    def write_to_file(self, filepath: str | Path, precision: int = 15) -> None:
        """
        Write the solution to a file in GTOC13 submission format.

        Args:
            filepath: Path to output file
            precision: Number of decimal places for floating point numbers (default: 15)
        """
        filepath = Path(filepath)
        with open(filepath, 'w') as f:
            self.write(stream=f, precision=precision)

    def plot(self, show_bodies: bool = True, figsize: tuple = (12, 10), save_path: str | Path | None = None,
             E_end: float | None = None, int_cos_alpha_end: float | None = None, obj_value: float | None = None):
        """
        Plot the heliocentric trajectory arcs and body orbits in the x-y plane.

        Args:
            show_bodies: If True, plot the orbits of bodies involved in flybys
            figsize: Figure size (width, height) in inches
            save_path: Optional path to save the figure. If None, displays the plot.
            E_end: Final specific orbital energy (km^2/s^2)
            int_cos_alpha_end: Integral of cos(alpha) at the end
            obj_value: Final objective value

        Returns:
            matplotlib figure and axis objects
        """
        import matplotlib.pyplot as plt
        from gtoc13.bodies import bodies_data
        from gtoc13.constants import YEAR

        fig, ax = plt.subplots(figsize=figsize)

        # Plot the sun at origin
        ax.plot(0, 0, 'yo', markersize=15, label='Altaira (Sun)', zorder=10)

        # Collect all body IDs involved in flybys
        body_ids = set()
        for arc in self.arcs:
            if isinstance(arc, FlybyArc):
                body_ids.add(arc.body_id)

        # Plot body orbits if requested
        if show_bodies and body_ids:
            import numpy as np

            # Generate full orbits for each body
            for body_id in sorted(body_ids):
                body = bodies_data.get(body_id)
                if body is None:
                    continue

                # Get body name
                name = body.name if hasattr(body, 'name') else f'Body {body_id}'

                # Generate orbit points (one full period)
                period = body.get_period('s')  # Period in seconds
                times = np.linspace(0, period, 200)

                orbit_x = []
                orbit_y = []
                for t in times:
                    state = body.get_state(t, time_units='s')
                    orbit_x.append(state.r[0])
                    orbit_y.append(state.r[1])

                # Plot orbit as dashed line
                ax.plot(orbit_x, orbit_y, '--', alpha=0.5, linewidth=1, label=f'{name} orbit')

        # Plot trajectory arcs
        for i, arc in enumerate(self.arcs):
            if isinstance(arc, PropagatedArc):
                # Extract x, y positions from propagated arc
                x = [pt.position[0] for pt in arc.state_points]
                y = [pt.position[1] for pt in arc.state_points]

                # Plot propagated arc
                ax.plot(x, y, 'b-', linewidth=2, alpha=0.7,
                       label='Trajectory' if i == 0 else None)

                # Mark start and end points
                ax.plot(x[0], y[0], 'go', markersize=8, zorder=5)
                ax.plot(x[-1], y[-1], 'ro', markersize=8, zorder=5)

                # Plot sail normal vectors for points with non-zero control
                import numpy as np
                for j, pt in enumerate(arc.state_points):
                    control_mag = np.linalg.norm(pt.control)
                    # Only plot every 5th point to avoid clutter, and only if control is significant
                    if j % 5 == 0 and control_mag > 1e-6:
                        # Scale the arrow by a reasonable amount for visibility
                        # Use 5% of the typical distance scale
                        scale = 5e8  # Scale factor for arrow length
                        dx = pt.control[0] * scale
                        dy = pt.control[1] * scale

                        ax.arrow(pt.position[0], pt.position[1], dx, dy,
                                head_width=3e8, head_length=5e8,
                                fc='red', ec='red', alpha=0.6, linewidth=1.5,
                                label='Sail Normal' if i == 0 and j == 0 else None,
                                zorder=4)

            elif isinstance(arc, FlybyArc):
                # Mark flyby location
                x_flyby = arc.position[0] if hasattr(arc, 'position') else None
                y_flyby = arc.position[1] if hasattr(arc, 'position') else None

                if x_flyby is not None and y_flyby is not None:
                    body = bodies_data.get(arc.body_id)
                    name = body.name if body and hasattr(body, 'name') else f'Body {arc.body_id}'

                    ax.plot(x_flyby, y_flyby, 'r*', markersize=15,
                           label=f'Flyby: {name}' if i < 5 else None, zorder=6)

                    # Add annotation for flyby
                    t_flyby = arc.epoch / YEAR if hasattr(arc, 'epoch') else None
                    if t_flyby is not None:
                        ax.annotate(f'{name}\nt={t_flyby:.1f} yr',
                                  xy=(x_flyby, y_flyby),
                                  xytext=(10, 10), textcoords='offset points',
                                  fontsize=8, alpha=0.7)

        # Formatting
        ax.set_xlabel('X Position (km)', fontsize=12)
        ax.set_ylabel('Y Position (km)', fontsize=12)
        ax.set_title('GTOC13 Trajectory Solution (X-Y Plane)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        ax.legend(loc='best', fontsize=10)

        # Add mission statistics as text
        num_flybys = sum(1 for arc in self.arcs if isinstance(arc, FlybyArc))
        num_prop_arcs = sum(1 for arc in self.arcs if isinstance(arc, PropagatedArc))

        # Calculate total mission time
        last_epoch = 0
        for arc in self.arcs:
            if isinstance(arc, PropagatedArc) and arc.state_points:
                last_epoch = max(last_epoch, arc.state_points[-1].epoch)
            elif isinstance(arc, FlybyArc) and hasattr(arc, 'epoch'):
                last_epoch = max(last_epoch, arc.epoch)

        mission_time = last_epoch / YEAR if last_epoch > 0 else 0

        # Build info text with mission statistics
        info_lines = [
            f'Flybys: {num_flybys}',
            f'Prop. Arcs: {num_prop_arcs}',
            f'Mission Time: {mission_time:.2f} years'
        ]

        # Add objective values if provided
        if obj_value is not None:
            info_lines.append(f'Objective: {obj_value:.6f}')
        if E_end is not None:
            info_lines.append(f'E_end: {E_end:.6f} km²/s²')
        if int_cos_alpha_end is not None:
            info_lines.append(f'∫cos(α): {int_cos_alpha_end:.6f}')

        info_text = '\n'.join(info_lines)
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
               verticalalignment='top', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()

        return fig, ax

    @classmethod
    def from_file(cls, filepath: str | Path) -> 'GTOC13Solution':
        """
        Read a solution from a GTOC13 submission file.

        Args:
            filepath: Path to input file

        Returns:
            GTOC13Solution object
        """
        filepath = Path(filepath)
        state_points = []
        comments = []

        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()

                # Skip empty lines
                if not line:
                    continue

                # Handle comments
                if line.startswith('#') or line.startswith('!'):
                    comments.append(line)
                    continue

                # Parse data row
                parts = line.replace(',', ' ').split()
                if len(parts) != 12:
                    raise ValueError(f"Expected 12 fields, got {len(parts)}: {line}")

                state_points.append(StatePoint(
                    body_id=int(float(parts[0])),
                    flag=int(float(parts[1])),
                    epoch=float(parts[2]),
                    position=(float(parts[3]), float(parts[4]), float(parts[5])),
                    velocity=(float(parts[6]), float(parts[7]), float(parts[8])),
                    control=(float(parts[9]), float(parts[10]), float(parts[11]))
                ))

        # Parse state points into arcs
        arcs = cls._parse_arcs_from_state_points(state_points)

        return cls(arcs=arcs, comments=comments)

    @staticmethod
    def _parse_arcs_from_state_points(state_points: List[StatePoint]) -> List[FlybyArc | ConicArc | PropagatedArc]:
        """
        Parse a list of state points into structured arcs.
        This is a simplified parser - could be enhanced with more validation.
        """
        arcs = []
        i = 0

        while i < len(state_points):
            pt = state_points[i]

            # Flyby arc (body_id > 0)
            if pt.body_id > 0:
                if i + 1 >= len(state_points):
                    # Final single-row flyby
                    arcs.append(FlybyArc(
                        body_id=pt.body_id,
                        is_science=(pt.flag == 1),
                        epoch=pt.epoch,
                        position=pt.position,
                        velocity_in=pt.velocity,
                        velocity_out=pt.velocity,
                        v_inf_in=pt.control,
                        v_inf_out=pt.control
                    ))
                    i += 1
                else:
                    pt_next = state_points[i + 1]
                    arcs.append(FlybyArc(
                        body_id=pt.body_id,
                        is_science=(pt.flag == 1),
                        epoch=pt.epoch,
                        position=pt.position,
                        velocity_in=pt.velocity,
                        velocity_out=pt_next.velocity,
                        v_inf_in=pt.control,
                        v_inf_out=pt_next.control
                    ))
                    i += 2

            # Conic arc (body_id=0, flag=0)
            elif pt.flag == 0:
                if i + 1 >= len(state_points):
                    raise ValueError("Conic arc must have at least 2 points")
                pt_next = state_points[i + 1]
                arcs.append(ConicArc(
                    epoch_start=pt.epoch,
                    epoch_end=pt_next.epoch,
                    position_start=pt.position,
                    position_end=pt_next.position,
                    velocity_start=pt.velocity,
                    velocity_end=pt_next.velocity
                ))
                i += 2

            # Propagated arc (body_id=0, flag=1)
            else:
                prop_points = [pt]
                i += 1
                # Collect all consecutive propagated points
                while i < len(state_points) and state_points[i].body_id == 0 and state_points[i].flag == 1:
                    prop_points.append(state_points[i])
                    i += 1
                arcs.append(PropagatedArc(state_points=prop_points))

        return arcs
