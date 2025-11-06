import numpy as np
import time
from typing import List, Optional, Dict, Iterable
from dataclasses import dataclass, field
import functools


class Timer:
    def __enter__(self):
        self._enter_time = time.time()

    def __exit__(self, *exc_args):
        self._exit_time = time.time()
        print(f"{self._exit_time - self._enter_time:.2f} seconds elapsed\n")


def timer(func):
    """Print the runtime of the decorated function"""

    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()  # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()  # 2
        run_time = end_time - start_time  # 3
        print(f"{func.__name__!r} elapsed {run_time:.4f} secs")
        return value

    return wrapper_timer


def lin_dots_penalty(r_i: np.array, r_j: np.array) -> np.float32:
    dots = np.clip(np.dot(r_i, r_j) / (np.linalg.norm(r_i) * np.linalg.norm(r_j)), -1.0, 1.0)
    if dots < 0.875:
        return 0.0
    else:
        return np.clip(  # polynomial fit of penalty
            3e-07 * dots**6
            - 2e-05 * dots**5
            + 0.0006 * dots**4
            - 0.0072 * dots**3
            + 0.0426 * dots**2
            - 0.108 * dots
            + 1.0822,
            0,
            1,
        )


@dataclass
class IndexParams:
    bodies_ID: list[int]
    n_timesteps: int
    seq_length: int


@dataclass
class DisBody:
    weight: float
    r_du: dict = field(
        default_factory=lambda: {
            1: np.ndarray,
        }
    )
    v_dtu: dict = field(
        default_factory=lambda: {
            1: np.ndarray,
        }
    )
    t_tu: np.ndarray = field(default_factory=lambda: np.zeros(3))


@dataclass
class DiscreteDict:
    bodies: dict = field(
        default_factory=lambda: {
            1: DisBody,
        }
    )
    dv_table: dict = field(
        default_factory=lambda: {
            tuple[int, int, int, int]: {"tof": float, "dv1": float, "dv2": float}
        }
    )  # {kimj: {tof: float, dv1: float, dv2: float}
