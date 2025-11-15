"""
Might have DU and TU unit errors?

"""

from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
import time
import functools
from tqdm import tqdm

import numpy as np
from numpy.linalg import norm

from lamberthub import izzo2015

import plotly
import plotly.graph_objects as go

from gtoc13 import YPTU, bodies_data, Body, SPTU, KMPDU, DAY


np.set_printoptions(legacy="1.25")


@dataclass
class IndexParams:
    """
    Parameters that dictate the dimensionality of the BINLP problem.

    bodies_ID : list of body indices for the model
    n_timesteps : number of discretizations for time and position per body, does not indicate equal delta-ts!
    seq_length : target number of bodies
    flyby_limit : max number of scientific flybys that will count towards scoring
    gt_planets : number of unique planets (large) needed for the grand tour bonus
    dv_limit: delta-v limit in km/s
    first_arcs : list of initial planets and timestep index bounds to pre-constraint the model

    """

    bodies_ID: list[int]
    n_timesteps: int
    seq_length: int
    flyby_limit: int
    gt_planets: int
    gt_smalls: int
    dv_limit: float = 200
    dE_tol: float = 10  # km**2/s**2
    dot_tol: float = 0.95
    first_arcs: Optional[None | list[int | tuple[int, tuple[int, int]]]] = (
        None  # integer of body, or body with bounds on timesteps
    )
    disallowed: Optional[None | list[tuple[int, str]]] = None  # str = 'end' or 'all'


@dataclass
class DisBody:
    name: str
    weight: float
    tp_tu: np.float32
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
class ArcTable:
    tofs: dict = field(
        default_factory=lambda: {tuple[int, int, int, int]: float}
    )  # {kimj: {tof: float, dv: float}
    # dv_tot: dict = field(default_factory=lambda: {tuple[int, int, int, int]: float})
    energy: dict = field(default_factory=lambda: {tuple[int, int, int, int]: float})
    vinf_a: dict = field(default_factory=lambda: {tuple[int, int, int, int]: np.ndarray})
    vinf_d: dict = field(default_factory=lambda: {tuple[int, int, int, int]: np.ndarray})
    dotprod_l: dict = field(default_factory=lambda: {tuple[int, int, int, int]: float})
    dotprod_u: dict = field(default_factory=lambda: {tuple[int, int, int, int]: float})


@dataclass
class SolverParams:
    """
    Parameters that dictate solver behavior and solution generation.

    solver_name : str of AMPL solver to use
    toconsole : display the solver's default output logs to console
    write_nl : create a .nl file with the model for debugging purposes
    write_log : write the output log to file
    solv_iter : number of solver iterations for the problem

    """

    solver_name: str
    soln_gap: float = 0.0
    toconsole: bool = True
    write_nl: bool = False
    write_log: bool = False
    solv_iter: int = 1


@dataclass
class SequenceTarget:
    place: int
    name: str
    body_id: int
    ts_idx: int
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


def vinf_penalty(vinf_in: np.float32) -> np.float32:
    # converts from DU/TU to km/s
    return 0.2 + (np.exp(-vinf_in * KMPDU / (SPTU * 13))) / (
        1 + np.exp(-5 * (vinf_in * KMPDU / SPTU - 1.5))
    )


def dot_angle(vinf: np.float32, b_id: int, mode: str) -> np.float32:
    body = bodies_data[b_id]
    f = 101 if mode == "min" else 1.1
    dot_prod = (
        (
            np.arcsin(
                (body.mu * (SPTU / KMPDU) ** 2 / (f * body.radius))
                / (vinf**2 + (body.mu * (SPTU / KMPDU) ** 2 / (f * body.radius)))
            )
            * 2
        )
        if b_id <= 10
        else 0
    )

    return dot_prod


@timer
def create_discrete_dataset(
    Yo: float, Yf: float, bodies_data: dict[int:Body], perYear: int = 2, include_small: bool = False
) -> tuple[dict, list[int], int, np.ndarray]:
    To = float(Yo / YPTU)
    Tf = float(Yf / YPTU)  # years per TU
    num = int((Yf - Yo) * perYear)
    timesteps = np.linspace(To, Tf, num)
    ## Generate position tables for just the bodies
    dis_ephm = dict()
    k_body = []
    for b_idx, body in tqdm(bodies_data.items()):
        if not include_small:
            if not (body.is_planet() or body.name == "Yandi"):
                continue
        k_body.append(b_idx)
        dis_ephm[b_idx] = DisBody(
            name=body.name,
            weight=body.weight,
            r_du=np.array(
                [
                    body.get_state(timesteps[idx], time_units="TU", distance_units="DU").r
                    for idx in range(num)
                ]
            ),
            t_tu=timesteps,
            tp_tu=np.float32(body.get_period(units="TU")),
        )
    return dis_ephm, k_body, num, timesteps


def lambert_arc(
    k: int, ti: float, m: int, tof: float, debug: bool = False, prograde: bool = True
) -> dict:
    tj = tof + ti
    state_ki = bodies_data[k].get_state(ti, time_units="TU", distance_units="DU")
    state_mj = bodies_data[m].get_state(tj, time_units="TU", distance_units="DU")
    r_ki, v_ki = state_ki
    r_mj, v_mj = state_mj
    if debug:
        print(tof, state_ki)

    pro = izzo2015(
        1,
        np.array(r_ki, dtype=np.float64),
        np.array(r_mj, dtype=np.float64),
        np.float64(tof),
        prograde=prograde,
        low_path=True,
    )
    energy = (norm(pro[0]) ** 2) / 2 - 1 / norm(r_ki)
    vinf_a = pro[1] - np.array(v_mj)
    vinf_d = pro[0] - np.array(v_ki)
    dotprod_l = dot_angle(norm(vinf_a), m, "max")
    dotprod_u = dot_angle(norm(vinf_a), m, "min")

    return (
        energy,
        vinf_a,
        vinf_d,
        dotprod_l,
        dotprod_u,
    )


@timer
def build_arc_table(body_list: list[int], timesteps: np.ndarray) -> ArcTable:
    tofs = {}
    energy = {}
    vinf_a = {}
    vinf_d = {}
    dotprod_l = {}
    dotprod_u = {}

    for kimj in tqdm(
        [
            (k, i, m, j)
            for k in body_list
            for i in range(len(timesteps))
            for m in body_list
            for j in range(len(timesteps))
            if (i != j)
        ]
    ):
        k, i, m, j = kimj
        tof = timesteps[j] - timesteps[i]
        dt_tol = DAY / SPTU if k != m else bodies_data[k].get_period(units="TU")
        if tof >= dt_tol:
            arc_soln = lambert_arc(k, timesteps[i], m, tof)
            if norm(arc_soln[1]) > 0.01638258 * 2:  # 1km/s
                (
                    energy[(k, i + 1, m, j + 1)],
                    vinf_a[(k, i + 1, m, j + 1)],
                    vinf_d[(k, i + 1, m, j + 1)],
                    dotprod_l[(k, i + 1, m, j + 1)],
                    dotprod_u[(k, i + 1, m, j + 1)],
                ) = arc_soln
                tofs[(k, i + 1, m, j + 1)] = tof

    arc_table = ArcTable(
        tofs=tofs,
        energy=energy,
        vinf_a=vinf_a,
        vinf_d=vinf_d,
        dotprod_l=dotprod_l,
        dotprod_u=dotprod_u,
    )
    return arc_table


def plot_porkchop(
    body1: int,
    body2: int,
    t_dep: np.ndarray,
    tof: np.ndarray,
    dvs: np.ndarray,
    dv_limit: float,
    show: bool = True,
    years: bool = True,
):
    k_name = bodies_data[body1].name
    m_name = bodies_data[body2].name
    x_options = dict(title=f"depart from {k_name} at tu_i", tickmode="auto")
    y_options = dict(title=f"arrive at {m_name} in delta-tu", tickmode="auto")
    main_title = f"from {k_name} to {m_name}"

    layout_options = dict(
        autosize=False,
        width=700,
        height=700,
        title=main_title,
        xaxis=x_options,
        yaxis=y_options,
    )
    max_val = np.min((dv_limit, np.max(dvs)))
    contour_dict = dict(
        start=0.0,
        end=max_val,
        size=max_val / 50,
    )

    if years:
        t_dep *= YPTU
        tof *= YPTU
    fig = go.Figure(
        layout=layout_options,
        data=go.Contour(
            x=t_dep,
            y=tof,
            z=dvs,
            contours=contour_dict,
            colorscale="dense",
        ),
    )
    if show:
        fig.show()
    else:
        output_path = Path.cwd() / "outputs"
        output_path.mkdir(exist_ok=True)
        plotly.offline.plot(
            fig, filename=(output_path / (main_title + ".html")).as_posix(), auto_open=False
        )
