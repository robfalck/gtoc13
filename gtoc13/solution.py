"""
GTOC13 Solution representation using Pydantic models.
Based on the GTOC13 Solution File Format Specification.
"""
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Literal, Tuple
import numpy as np
from pathlib import Path


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

    def write_solution_file(self, filepath: str | Path, precision: int = 6) -> None:
        """
        Write the solution to a file in GTOC13 submission format.

        Args:
            filepath: Path to output file
            precision: Number of decimal places for floating point numbers
        """
        filepath = Path(filepath)

        with open(filepath, 'w') as f:
            # Write header comments
            f.write("# GTOC13 Solution File\n")
            f.write("# Format: body_id flag epoch x y z vx vy vz cx cy cz\n")
            f.write("# Units: epoch(s), position(km), velocity(km/s)\n")

            # Write custom comments
            for comment in self.comments:
                if not comment.startswith('#'):
                    comment = f"# {comment}"
                f.write(f"{comment}\n")

            f.write("#\n")

            # Write state points
            state_points = self.to_state_points()
            for point in state_points:
                # Format with specified precision
                line = (
                    f"{point.body_id:d} {point.flag:d} {point.epoch:.{precision}f} "
                    f"{point.position[0]:.{precision}f} {point.position[1]:.{precision}f} {point.position[2]:.{precision}f} "
                    f"{point.velocity[0]:.{precision}f} {point.velocity[1]:.{precision}f} {point.velocity[2]:.{precision}f} "
                    f"{point.control[0]:.{precision}f} {point.control[1]:.{precision}f} {point.control[2]:.{precision}f}\n"
                )
                f.write(line)

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


# Convenience functions for creating arcs
def create_flyby(
    body_id: int,
    epoch: float,
    position: Tuple[float, float, float],
    velocity_in: Tuple[float, float, float],
    velocity_out: Tuple[float, float, float],
    v_inf_in: Tuple[float, float, float],
    v_inf_out: Tuple[float, float, float],
    is_science: bool = True
) -> FlybyArc:
    """Convenience function to create a flyby arc"""
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


def create_conic(
    epoch_start: float,
    epoch_end: float,
    position_start: Tuple[float, float, float],
    position_end: Tuple[float, float, float],
    velocity_start: Tuple[float, float, float],
    velocity_end: Tuple[float, float, float]
) -> ConicArc:
    """Convenience function to create a conic arc"""
    return ConicArc(
        epoch_start=epoch_start,
        epoch_end=epoch_end,
        position_start=position_start,
        position_end=position_end,
        velocity_start=velocity_start,
        velocity_end=velocity_end
    )


def create_propagated(
    epochs: List[float],
    positions: List[Tuple[float, float, float]],
    velocities: List[Tuple[float, float, float]],
    controls: List[Tuple[float, float, float]]
) -> PropagatedArc:
    """Convenience function to create a propagated arc from lists"""
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

    return PropagatedArc(state_points=state_points)
