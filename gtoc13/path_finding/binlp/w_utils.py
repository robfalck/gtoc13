import pykep as pk
import numpy as np
from numpy.linalg import norm
from gtoc13 import YPTU, bodies_data
import plotly
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

np.set_printoptions(legacy="1.25")


def min_dv_lam(
    k: int, ti: float, m: int, tof: float, debug: bool = False
) -> tuple[pk.lambert_problem, pk.lambert_problem]:
    tj = ti + tof
    state_ki = bodies_data[k].get_state(ti, time_units="TU", distance_units="DU")
    state_mj = bodies_data[m].get_state(tj, time_units="TU", distance_units="DU")
    r_ki, v_ki = state_ki
    r_mj, v_mj = state_mj

    cw = pk.lambert_problem(
        r1=np.array(r_ki, dtype=np.float64),
        r2=np.array(r_mj, dtype=np.float64),
        tof=np.float64(tof),
        cw=True,
        max_revs=1,
    )
    ccw = pk.lambert_problem(
        r1=np.array(r_ki, dtype=np.float64),
        r2=np.array(r_mj, dtype=np.float64),
        tof=np.float64(tof),
        cw=False,
        max_revs=1,
    )
    dvs_cw = norm(cw.get_v1()[0] - np.array(v_ki)) + norm(cw.get_v2()[0] - np.array(v_mj))
    dvs_ccw = norm(ccw.get_v1()[0] - np.array(v_ki)) + norm(ccw.get_v2()[0] - np.array(v_mj))
    min_dv = min(dvs_cw, dvs_ccw)
    return min_dv


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
    y_options = dict(title=f"arrive at {m_name} in delta-t", tickmode="auto")
    main_title = f"from {k_name} to {m_name}"

    layout_options = dict(
        autosize=False,
        width=700,
        height=700,
        title=main_title,
        xaxis=x_options,
        yaxis=y_options,
    )

    contour_dict = dict(
        start=np.min(dvs),
        end=np.min((dv_limit, np.max(dvs))),
        size=np.min((dv_limit, np.max(dvs))) / 100,
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
