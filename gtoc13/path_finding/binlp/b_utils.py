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
import pykep as pk

import plotly
import plotly.graph_objects as go

from gtoc13 import YPTU, bodies_data, Body


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
    dv_limit: Optional[float]
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
class DVTable:
    tofs: dict = field(
        default_factory=lambda: {tuple[int, int, int, int]: float}
    )  # {kimj: {tof: float, dv: float}
    dvs: dict = field(default_factory=lambda: {tuple[int, int, int, int]: float})


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


def min_dv_lam(
    k: int, ti: float, m: int, tof: float, debug: bool = False
) -> tuple[pk.lambert_problem, pk.lambert_problem]:
    tj = tof + ti
    state_ki = bodies_data[k].get_state(ti, time_units="TU", distance_units="DU")
    state_mj = bodies_data[m].get_state(tj, time_units="TU", distance_units="DU")
    r_ki, v_ki = state_ki
    r_mj, v_mj = state_mj
    if debug:
        print(tof, state_ki)

    cw = pk.lambert_problem(
        r1=np.array(r_ki, dtype=np.float64),
        r2=np.array(r_mj, dtype=np.float64),
        tof=np.float64(tof),
        cw=True,
        max_revs=0,
    )
    ccw = pk.lambert_problem(
        r1=np.array(r_ki, dtype=np.float64),
        r2=np.array(r_mj, dtype=np.float64),
        tof=np.float64(tof),
        cw=False,
        max_revs=0,
    )
    dvs_cw = norm(cw.get_v1()[0] - np.array(v_ki)) + norm(cw.get_v2()[0] - np.array(v_mj))
    dvs_ccw = norm(ccw.get_v1()[0] - np.array(v_ki)) + norm(ccw.get_v2()[0] - np.array(v_mj))
    min_dv = min(dvs_cw, dvs_ccw)
    return min_dv


@timer
def build_dv_table(body_list: list[int], timesteps: np.ndarray):
    tof_dict = {}
    dv_dict = {}
    for kimj in tqdm(
        [
            (k, i, m, j)
            for k in body_list
            for i in range(len(timesteps))
            for m in body_list
            for j in range(len(timesteps))
            if (k != m and i != j)
        ]
    ):
        k, i, m, j = kimj
        tof = timesteps[j] - timesteps[i]
        if tof >= 0:
            dv_dict[(k, i + 1, m, j + 1)] = min_dv_lam(k, timesteps[i], m, tof)
            tof_dict[(k, i + 1, m, j + 1)] = tof
    dv_table = DVTable(tofs=tof_dict, dvs=dv_dict)
    return dv_table


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
