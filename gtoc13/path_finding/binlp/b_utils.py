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

from gtoc13 import YPTU, bodies_data, Body, MU_ALTAIRA


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
    dE_tol: float  # km**2/s**2
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
    pro_energy: dict = field(default_factory=lambda: {tuple[int, int, int, int]: float})
    pro_vinf_a: dict = field(default_factory=lambda: {tuple[int, int, int, int]: np.ndarray})
    pro_vinf_d: dict = field(default_factory=lambda: {tuple[int, int, int, int]: np.ndarray})
    ret_energy: dict = field(default_factory=lambda: {tuple[int, int, int, int]: float})
    ret_vinf_a: dict = field(default_factory=lambda: {tuple[int, int, int, int]: np.ndarray})
    ret_vinf_d: dict = field(default_factory=lambda: {tuple[int, int, int, int]: np.ndarray})


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
    # for DU/TU units
    if vinf_in < 2.35:
        return 13.469 * vinf_in - 0.0142
    else:
        return (
            -0.0192 * vinf_in**5
            + 0.2069 * vinf_in**4
            - 0.8811 * vinf_in**3
            + 1.8804 * vinf_in**2
            - 2.0656 * vinf_in
            + 1.1715
        )


def dotangle_min(vinf: np.float32, b_id: int) -> np.float32:
    match b_id:
        case 1:
            dot_prod = (
                -0.0072 * vinf**5
                + 0.1039 * vinf**4
                - 0.5393 * vinf**3
                + 1.071 * vinf**2
                + 0.0043 * vinf
                - 1.0081
            )
        case 2:
            dot_prod = (
                0.0128 * vinf**5
                - 0.1978 * vinf**4
                + 1.1691 * vinf**3
                - 3.2878 * vinf**2
                + 4.3814 * vinf
                - 1.2201
            )

        case 3:
            if vinf < 2.42:
                dot_prod = (
                    -0.2266 * vinf**6
                    + 2.2211 * vinf**5
                    - 8.7353 * vinf**4
                    + 17.544 * vinf**3
                    - 18.865 * vinf**2
                    + 10.249 * vinf
                    - 1.2005
                )
            else:
                dot_prod = 0.9997
        case 4:
            if vinf < 1.116:
                dot_prod = (
                    -5.2052 * vinf**4 + 17.381 * vinf**3 - 21.218 * vinf**2 + 11.231 * vinf - 1.1924
                )
            else:
                dot_prod = 0.9999

        case 5:
            if vinf < 1.416:
                dot_prod = (
                    1.7535 * vinf**4 - 5.7851 * vinf**3 + 5.4358 * vinf**2 + 0.2489 * vinf - 1.0146
                )
            else:
                dot_prod = 0.011 * vinf**3 - 0.1237 * vinf**2 + 0.4598 * vinf + 0.4298

        case 6:
            dot_prod = (
                0.0063 * vinf**6
                - 0.1031 * vinf**5
                + 0.6479 * vinf**4
                - 1.8878 * vinf**3
                + 2.2357 * vinf**2
                + 0.2692 * vinf
                - 1.0311
            )

        case 7:
            if vinf < 1.516:
                dot_prod = (
                    4.9725 * vinf**6
                    - 26.501 * vinf**5
                    + 54.764 * vinf**4
                    - 53.667 * vinf**3
                    + 22.6 * vinf**2
                    - 0.258 * vinf
                    - 1.0004
                )
            else:
                dot_prod = -0.0028 * vinf**2 + 0.0218 * vinf + 0.9582

        case 8:
            if vinf < 1.016:
                dot_prod = (
                    5.5151 * vinf**4 - 13.68 * vinf**3 + 9.8893 * vinf**2 + 0.0724 * vinf - 1.0057
                )
            elif vinf < 3.715:
                dot_prod = (
                    -0.0206 * vinf**4 + 0.2278 * vinf**3 - 0.9334 * vinf**2 + 1.6946 * vinf - 0.168
                )
            else:
                dot_prod = 0.9999

        case 9:
            if vinf < 1.51589:
                dot_prod = (
                    -1.8327 * vinf**5
                    + 8.6614 * vinf**4
                    - 14.843 * vinf**3
                    + 9.963 * vinf**2
                    - 0.2283 * vinf
                    - 0.9981
                )
            else:
                dot_prod = 0.0066 * vinf**3 - 0.0747 * vinf**2 + 0.281 * vinf + 0.6471

        case 10:
            if vinf < 0.81612:
                dot_prod = (
                    15.469 * vinf**4 - 29.675 * vinf**3 + 16.231 * vinf**2 + 0.3922 * vinf - 1.0117
                )
            else:
                dot_prod = (
                    -0.0016 * vinf**6
                    + 0.0298 * vinf**5
                    - 0.2311 * vinf**4
                    + 0.9311 * vinf**3
                    - 2.0559 * vinf**2
                    + 2.3702 * vinf
                    - 0.1293
                )

    return dot_prod


def dotangle_max(vinf: np.float32, b_id: int) -> np.float32:
    match b_id:
        case 1:
            if vinf < 0.715:
                dot_prod = 9.1669 * vinf**3 - 17.04 * vinf**2 + 10.514 * vinf - 1.1815
            else:
                dot_prod = 0.0015 * vinf + 0.9946
        case 2:
            if vinf < 0.117:
                dot_prod = 16.483 * vinf - 1.0121
            else:
                dot_prod = (
                    -0.0011 * vinf**6
                    + 0.0181 * vinf**5
                    - 0.1159 * vinf**4
                    + 0.3643 * vinf**3
                    - 0.5786 * vinf**2
                    + 0.4252 * vinf
                    + 0.8932
                )

        case 3:
            if vinf < 0.117:
                dot_prod = 10.441 * vinf - 0.2197
            else:
                dot_prod = 0.0001 * vinf + 0.9996

        case 4:
            if vinf < 0.11635:
                dot_prod = 9.0037 * vinf - 0.0506
            else:
                dot_prod = 1

        case 5:
            if vinf < 0.11635:
                dot_prod = 15.894 * vinf - 1.1365
            elif vinf < 1.416:
                dot_prod = (
                    -5.9664 * vinf**6
                    + 30.566 * vinf**5
                    - 62.319 * vinf**4
                    + 64.273 * vinf**3
                    - 35.094 * vinf**2
                    + 9.5522 * vinf
                    - 0.0104
                )
            else:
                dot_prod = 0.99998

        case 6:
            if vinf < 1.116:
                dot_prod = (
                    21.075 * vinf**5
                    - 73.869 * vinf**4
                    + 99.059 * vinf**3
                    - 63.253 * vinf**2
                    + 19.214 * vinf
                    - 1.242
                )
            else:
                dot_prod = 0.999999

        case 7:
            if vinf < 0.11635:
                dot_prod = 16.089 * vinf - 0.9315
            elif vinf < 0.6162:
                dot_prod = (
                    -9.1064 * vinf**4 + 15.656 * vinf**3 - 9.762 * vinf**2 + 2.614 * vinf + 0.7456
                )
            else:
                dot_prod = 0.99999

        case 8:
            if vinf < 0.11635:
                dot_prod = 16.551 * vinf - 1.0791
            elif vinf < 0.2165:
                dot_prod = 1.3347 * vinf + 0.6914
            elif vinf < 1.016:
                dot_prod = (
                    -2.7264 * vinf**6
                    + 11.294 * vinf**5
                    - 19.11 * vinf**4
                    + 16.912 * vinf**3
                    - 8.2716 * vinf**2
                    + 2.1299 * vinf
                    + 0.772
                )
            else:
                dot_prod = 3e-6 * vinf**3 - 3e-5 * vinf**2 + 0.0001 * vinf + 0.9999

        case 9:
            if vinf < 0.5162:
                dot_prod = (
                    -268.28 * vinf**4 + 357 * vinf**3 - 168.7 * vinf**2 + 33.387 * vinf - 1.3423
                )
            elif vinf < 1.216:
                dot_prod = (
                    -0.0159 * vinf**4 + 0.0633 * vinf**3 - 0.0943 * vinf**2 + 0.0628 * vinf + 0.984
                )
            else:
                dot_prod = (
                    -5e-7 * vinf**6
                    + 1e-5 * vinf**5
                    - 9e-5 * vinf**4
                    + 0.0004 * vinf**3
                    - 0.0009 * vinf**2
                    + 0.0012 * vinf
                    + 0.9994
                )

        case 10:
            if vinf < 0.11635:
                dot_prod = 16.066 * vinf - 0.9277
            elif vinf < 0.616186:
                dot_prod = (
                    31.241 * vinf**5
                    - 66.149 * vinf**4
                    + 54.758 * vinf**3
                    - 22.165 * vinf**2
                    + 4.3994 * vinf
                    + 0.655
                )
            elif vinf < 1.51589:
                dot_prod = (
                    -0.0006 * vinf**4 + 0.0032 * vinf**3 - 0.0058 * vinf**2 + 0.0047 * vinf + 0.9985
                )
            else:
                dot_prod = (
                    -4e-8 * vinf**6
                    + 8e-7 * vinf**5
                    - 7e-6 * vinf**4
                    + 3e-5 * vinf**3
                    - 9e-5 * vinf**2
                    + 0.0001 * vinf
                    + 0.9999
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


def lambert_arc(k: int, ti: float, m: int, tof: float, debug: bool = False) -> dict:
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
        prograde=True,
        low_path=True,
    )
    ret = izzo2015(
        1,
        np.array(r_ki, dtype=np.float64),
        np.array(r_mj, dtype=np.float64),
        np.float64(tof),
        prograde=False,
        low_path=True,
    )
    pro_energy = (norm(pro[0]) ** 2) / 2 - 1 / norm(r_ki)
    pro_vinf_a = pro[1] - np.array(v_mj)
    pro_vinf_d = pro[0] - np.array(v_ki)

    ret_energy = (norm(ret[0]) ** 2) / 2 - 1 / norm(r_ki)
    ret_vinf_a = ret[1] - np.array(v_mj)
    ret_vinf_d = ret[0] - np.array(v_ki)

    return pro_energy, pro_vinf_a, pro_vinf_d, ret_energy, ret_vinf_a, ret_vinf_d


@timer
def build_arc_table(body_list: list[int], timesteps: np.ndarray) -> ArcTable:
    tofs = {}
    p_se = {}
    p_vi_a = {}
    p_vi_d = {}
    r_se = {}
    r_vi_a = {}
    r_vi_d = {}
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
            tofs[(k, i + 1, m, j + 1)] = tof
            (
                p_se[(k, i + 1, m, j + 1)],
                p_vi_a[(k, i + 1, m, j + 1)],
                p_vi_d[(k, i + 1, m, j + 1)],
                r_se[(k, i + 1, m, j + 1)],
                r_vi_a[(k, i + 1, m, j + 1)],
                r_vi_d[(k, i + 1, m, j + 1)],
            ) = lambert_arc(k, timesteps[i], m, tof)

    arc_table = ArcTable(
        tofs=tofs,
        pro_energy=p_se,
        pro_vinf_a=p_vi_a,
        pro_vinf_d=p_vi_d,
        ret_energy=r_se,
        ret_vinf_a=r_vi_a,
        ret_vinf_d=r_vi_d,
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
