import numpy as np
import time
from typing import Optional
from dataclasses import dataclass, field
import functools
from gtoc13 import YEAR, SPTU, KMPDU
from tqdm import tqdm

np.set_printoptions(legacy="1.25")


@dataclass
class IndexParams:
    bodies_ID: list[int]
    n_timesteps: int
    seq_length: int
    flyby_limit: int
    gt_planets: int
    first_arcs: Optional[
        list[int | tuple[int, list[int, int]]]
    ]  # integer of body, or body with bounds on timesteps


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


@dataclass
class SolverParams:
    solver_name: str
    toconsole: bool = True
    write_nl: bool = False
    write_log: bool = False
    solv_iter: int = 1


@dataclass
class SequenceTarget:
    order: int
    body_state: tuple[str, tuple[int, int]]
    year: float


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
        print(f"{func.__name__!r} elapsed {run_time:.4f} secs\n")
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


@timer
def create_discrete_dataset(
    Yo: float, Yf: float, bodies_data: dict, perYear: int = 2
) -> DiscreteDict:
    To = Yo * YEAR
    Tf = Yf * YEAR  # years in seconds
    num = int(np.ceil((Tf - To) / (YEAR / perYear)))

    ## Generate position tables for just the bodies
    print("...discretizing body data...")
    with Timer():
        dis_ephm = DiscreteDict()
        k_body = []
        for b_idx, body in tqdm(bodies_data.items()):
            if body.is_planet() or body.name == "Yandi":
                k_body.append(b_idx)
                timestep = np.linspace(To, Tf, num) / SPTU
                dis_ephm.bodies[b_idx] = DisBody(
                    weight=body.weight,
                    r_du=np.array([body.get_state(timestep[idx]).r / KMPDU for idx in range(num)]),
                    v_dtu=np.array(
                        [body.get_state(timestep[idx]).v * SPTU / KMPDU for idx in range(num)]
                    ),
                    t_tu=timestep,
                )
    return dis_ephm, k_body, num


# print("...calculating lambert delta-vs...")
# with Timer():
#     dv_limit *= SPTU.tolist() / KMPDU  # km/s
#     dv_1 = {}
#     dv_2 = {}
#     for kimj in tqdm(
#         [
#             (k, i, m, j)
#             for (k, m) in list(product(k_body, repeat=2))
#             for i, __ in enumerate(discrete_data[k]["t_tu"])
#             for j, __ in enumerate(discrete_data[m]["t_tu"])
#         ]
#     ):
#         k, i, m, j = kimj
#         tu_i = discrete_data[k]["t_tu"][i]
#         tu_j = discrete_data[m]["t_tu"][j]
#         vk_dtu_i = discrete_data[k]["v_dtu"][i]
#         vm_dtu_j = discrete_data[m]["v_dtu"][j]
#         tof = (tu_j - tu_i).tolist()
#         # if tof > 0:
#         #     ki_to_mj = lambert_problem(
#         #         r1=discrete_data[k]["r_du"][i].tolist(),
#         #         r2=discrete_data[m]["r_du"][j].tolist(),
#         #         tof=(tu_j - tu_i).tolist(),
#         #     )
#         #     dv_1[(k, i + 1, m, j + 1)] = np.linalg.norm(
#         #         np.array(ki_to_mj.get_v1()[0]) - vk_dtu_i
#         #     )
#         #     dv_2[(k, i + 1, m, j + 1)] = np.linalg.norm(
#         #         np.array(ki_to_mj.get_v2()[0]) - vm_dtu_j
#         #     )

#         # else:
#         dv_1[(k, i + 1, m, j + 1)], dv_2[(k, i + 1, m, j + 1)] = np.random.rand(2) * (
#             dv_limit + 50.0
#         )
# [(v * KMPDU / SPTU).tolist() for v in test_3.get_v1()[0]]
# dvi_check = [val <= dv_limit for key, val in dv_1.items()]
# dvf_check = [val <= dv_limit for key, val in dv_2.items()]
# print("dvi % :", sum(dvi_check) * 100 / len(dvi_check))
# print("dvf % :", sum(dvf_check) * 100 / len(dvf_check))
# print("\n")
