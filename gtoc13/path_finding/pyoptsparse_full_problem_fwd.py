#!/usr/bin/env python3
"""
GTOC13 multiple-shooting, solved with pyOptSparse + IPOPT.
This is a self-contained replacement for the pygmo-based driver.

What it does:
- reads a beam-search JSON to get a sequence of bodies and epochs
- builds the tour problem (times, vinf, sail controls)
- defines objective + eq + ineq just like the pygmo UDP
- calls IPOPT through pyOptSparse
"""

from __future__ import annotations
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Union

import numpy as np
import jax
import jax.numpy as jnp
import diffrax as dfx
from pyoptsparse import Optimization, IPOPT
from gtoc13.constants import (
    KMPAU,
    DAY,
    YEAR,
    MU_ALTAIRA,
    C_FLUX,
    SAIL_AREA,
    SPACECRAFT_MASS,
)

# -------------------------------------------------------------------
# Constants (mirroring gtoc13.constants so this stays drop-in)
# -------------------------------------------------------------------
AU_KM = KMPAU
YEAR_S = YEAR
DEFAULT_A0_1AU = 2.0 * C_FLUX * SAIL_AREA / SPACECRAFT_MASS / 1000.0  # km/s^2

jax.config.update("jax_enable_x64", True)

# scaling used in the pygmo version
L_REF = 1.0e6
T_REF = 100.0 * YEAR_S
V_REF = 1.0e2

INTERSTELLAR_BODY_ID = -1  # keep placeholder


# -------------------------------------------------------------------
# Minimal bodies catalog – in your real code, import from gtoc13.bodies
# here we assume the JSON gives ephemeris times, so we only need the star mu
# and a way to say "mu=0 for asteroids".
# Replace this with your actual catalog if you have it.
# -------------------------------------------------------------------
@dataclass
class BodyEphem:
    body_id: int
    mu: float
    radius: float


class Catalog:
    def __init__(self, mu_star_km3_s2: float):
        self.mu_star = mu_star_km3_s2
        # real code: load bodies_data; here just fallback
        self._default_radius = 3000.0

    def body_radius(self, body_id: int) -> float:
        return self._default_radius

    def body_mu(self, body_id: int) -> float:
        # planets vs massless bodies — you can plug your real logic here
        if body_id < 10:
            return 4.0e5  # fake planet mu
        return 0.0

    def state(self, body_id: int, t: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # placeholder ephemeris: circular orbit in xy
        # real code should call your actual ephemeris
        w = 2.0 * jnp.pi / YEAR_S
        radius = 1.0e8 + 1.0e6 * body_id
        ct = jnp.cos(w * t)
        st = jnp.sin(w * t)
        r = jnp.array([ct, st, 0.0]) * radius
        v = jnp.array([-st, ct, 0.0]) * radius * w
        return r, v

    def orbital_period(self, body_id: int) -> float:
        return YEAR_S


# -------------------------------------------------------------------
# Sail dynamics
# -------------------------------------------------------------------
@dataclass
class SailParams:
    a0_1au_km_s2: float


def sail_accel(r: jnp.ndarray, v: jnp.ndarray, alpha: float, sigma: float, a0_1au: float) -> jnp.ndarray:
    rnorm = jnp.linalg.norm(r)
    rnorm_safe = jnp.where(rnorm > 0.0, rnorm, 1.0)
    rhat = r / rnorm_safe
    h = jnp.cross(r, v)
    hnorm = jnp.linalg.norm(h)
    def handle_degenerate():
        that = jnp.array([0.0, 1.0, 0.0])
        nhat = jnp.array([0.0, 0.0, 1.0])
        return that, nhat
    def handle_regular():
        hhat = h / jnp.where(hnorm > 0.0, hnorm, 1.0)
        that = jnp.cross(hhat, rhat)
        that = that / jnp.where(jnp.linalg.norm(that) > 0.0, jnp.linalg.norm(that), 1.0)
        nhat = hhat
        return that, nhat
    that, nhat = jax.lax.cond(hnorm < 1e-12, lambda: handle_degenerate(), lambda: handle_regular())
    cos_alpha = jnp.cos(alpha)
    sin_alpha = jnp.sin(alpha)
    n_hat = cos_alpha * rhat + sin_alpha * (jnp.cos(sigma) * that + jnp.sin(sigma) * nhat)
    scale = a0_1au * (AU_KM / jnp.where(rnorm > 0.0, rnorm, 1.0)) ** 2 * (cos_alpha ** 2)
    return scale * n_hat


def safe_norm(vec: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    return jnp.sqrt(jnp.dot(vec, vec) + eps)


@dataclass
class IntOptions:
    solver: str = "tsit5"
    rtol: float = 1e-6
    atol: float = 1e-9
    max_steps: int = 4096
    dt0_fraction: float = 0.1


class SailPropagator:
    _SOLVERS = {
        "tsit5": dfx.Tsit5,
        "dopri5": dfx.Dopri5,
        "heun": dfx.Heun,
    }

    def __init__(self, mu_star: float, sail: SailParams, accel_smoothing: float = 0.0, opts: IntOptions = IntOptions()):
        self.mu = mu_star
        self.sail = sail
        self.opts = opts
        self.accel_smoothing = accel_smoothing
        solver_cls = self._SOLVERS.get(opts.solver.lower(), dfx.Tsit5)
        self._solver = solver_cls()
        self._ode_term = dfx.ODETerm(self._rhs)
        self._saveat = dfx.SaveAt(t1=True)

    def _rhs(self, t, y, args):
        mu_star, a0, alpha, sigma, smoothing = args
        r = y[:3]
        v = y[3:]
        rnorm = jnp.linalg.norm(r)
        acc_grav = -mu_star * r / ((rnorm ** 3) + 1e-30)
        acc_sail = sail_accel(r, v, alpha, sigma, a0)
        if smoothing > 0.0:
            mag = jnp.linalg.norm(acc_sail)
            floor = smoothing
            acc_sail = jax.lax.cond(
                mag < floor,
                lambda _: jnp.where(mag > 1e-12, acc_sail * (floor / mag), floor * r / jnp.where(rnorm > 0.0, rnorm, 1.0)),
                lambda _: acc_sail,
                operand=None,
            )
        return jnp.concatenate([v, acc_grav + acc_sail])

    def propagate_piecewise(self, t0, y0, t1, controls):
        controls = jnp.asarray(controls)
        if controls.size == 0:
            return y0
        total_dt = t1 - t0
        nseg = controls.shape[0]
        dt = total_dt / nseg
        y0 = jnp.asarray(y0, dtype=jnp.float64)

        def integrate_segment(carry, ctrl):
            t_start, y_start = carry
            alpha, sigma = ctrl
            sol = dfx.diffeqsolve(
                self._ode_term,
                self._solver,
                t0=t_start,
                t1=t_start + dt,
                dt0=jnp.maximum(jnp.abs(dt) * self.opts.dt0_fraction, 1e-6),
                y0=y_start,
                args=(self.mu, self.sail.a0_1au_km_s2, alpha, sigma, self.accel_smoothing),
                max_steps=self.opts.max_steps,
                saveat=self._saveat,
            )
            y_end = jnp.reshape(sol.ys, y_start.shape)
            return (t_start + dt, y_end), y_end

        def do_integrate():
            (_, y_last), _ = jax.lax.scan(integrate_segment, (t0, y0), controls)
            return y_last

        return jax.lax.cond(jnp.abs(total_dt) < 1e-12, lambda _: y0, lambda _: do_integrate(), operand=None)


# -------------------------------------------------------------------
# Problem definition (replacement for the pygmo UDP)
# -------------------------------------------------------------------
@dataclass
class LegCtrlSpec:
    nseg_leg: int = 5


@dataclass
class ProblemOptions:
    same_body_gap_factor: float = 1.0 / 3.0
    progress_every_evals: int = 0
    flyby_altitude_bounds: Tuple[float, float] = (0.1, 100.0)
    objective: str = "vinf_rss"
    optimize_t0: bool = False
    constrain_interstellar_direction: bool = False
    accel_smoothing: float = 0.0


class GTOC13TourProblem:
    """
    Standalone version of the tour problem, no pygmo.
    fitness(x) -> [obj, eq..., ineq...]
    """
    def __init__(
        self,
        body_sequence: List[int],
        t_guess_s: List[float],
        catalog: Catalog,
        mu_star_km3_s2: float,
        sail: SailParams,
        leg_ctrl: LegCtrlSpec,
        opts: ProblemOptions,
    ):
        self.seq = body_sequence
        self.t_guess = t_guess_s
        self.cat = catalog
        self.mu = mu_star_km3_s2
        self.sail = sail
        self.leg_ctrl = leg_ctrl
        self.opts = opts
        self.prop = SailPropagator(mu_star_km3_s2, sail, accel_smoothing=opts.accel_smoothing)
        self.M = len(self.seq) - 1
        self._build_indexing()
        self._precompute_leg_columns()
        self._build_autodiff_handles()

    def _build_indexing(self):
        idx = 0
        self.idx_t = list(range(idx, idx + len(self.seq)))
        idx += len(self.seq)
        self.idx_vinf = []
        for _ in range(self.M):
            self.idx_vinf.append((idx, idx + 3))
            idx += 3
        self.idx_ctrl = []
        for _ in range(self.M):
            lo = idx
            hi = idx + 2 * self.leg_ctrl.nseg_leg
            self.idx_ctrl.append((lo, hi))
            idx = hi
        self.nx = idx

    def _precompute_leg_columns(self):
        self._leg_columns = []
        for j in range(self.M):
            cols = {self.idx_t[j], self.idx_t[j + 1]}
            lo, hi = self.idx_vinf[j]
            cols.update(range(lo, hi))
            lo, hi = self.idx_ctrl[j]
            cols.update(range(lo, hi))
            self._leg_columns.append(sorted(cols))

    def _build_autodiff_handles(self):
        def fitness_fn(x):
            return self._fitness_vector_impl(x)

        jac_fn = jax.jacrev(fitness_fn)
        self._fitness_fn = fitness_fn
        self._jacobian_fn = jac_fn

    def _fitness_vector_impl(self, x):
        x = jnp.asarray(x, dtype=jnp.float64)
        idx_t = jnp.array(self.idx_t, dtype=jnp.int32)
        t_scaled = x[idx_t]
        t = t_scaled * T_REF
        vinf_dep = []
        for j in range(self.M):
            lo, hi = self.idx_vinf[j]
            idx = jnp.arange(lo, hi, dtype=jnp.int32)
            vinf_dep.append(x[idx] * V_REF)
        vinf_dep = jnp.stack(vinf_dep) if vinf_dep else jnp.zeros((0, 3), dtype=jnp.float64)

        rB_list = []
        vB_list = []
        for tj, bid in zip(t, self.seq):
            rj, vj = self.cat.state(bid, tj)
            rB_list.append(rj)
            vB_list.append(vj)
        rB = jnp.stack(rB_list)
        vB = jnp.stack(vB_list)

        ceq_parts = []
        cineq_parts = []

        if self.M > 0:
            time_order = (t[:-1] - t[1:]) / T_REF
            cineq_parts.append(time_order)
            dt_leg = t[1:] - t[:-1]
            dt_min = 3.0 * DAY
            min_leg = (dt_min - dt_leg) / T_REF
            cineq_parts.append(min_leg)
        else:
            dt_leg = jnp.zeros(0, dtype=jnp.float64)

        vinf_arr = jnp.zeros((len(self.seq), 3), dtype=jnp.float64)

        for j in range(self.M):
            ctrls = self._extract_ctrls_jax(x, *self.idx_ctrl[j])
            y0 = jnp.concatenate([rB[j], vB[j] + vinf_dep[j]])
            yf = self.prop.propagate_piecewise(t[j], y0, t[j + 1], ctrls)
            ceq_parts.append((yf[:3] - rB[j + 1]) / L_REF)
            vinf_arr = vinf_arr.at[j + 1].set(yf[3:] - vB[j + 1])

        for j in range(self.M):
            if self.seq[j] == self.seq[j + 1]:
                Tbody = self.cat.orbital_period(self.seq[j])
                gap = (self.opts.same_body_gap_factor * Tbody - (t[j + 1] - t[j])) / T_REF
                cineq_parts.append(jnp.array([gap], dtype=jnp.float64))

        h_min_norm, h_max_norm = self.opts.flyby_altitude_bounds
        for b in range(1, self.M):
            body_id = self.seq[b]
            vinf_in = vinf_arr[b]
            vinf_out = vinf_dep[b]
            mu_body = self.cat.body_mu(body_id)
            if mu_body <= 0.0:
                ceq_parts.append((vinf_out - vinf_in) / V_REF)
                continue
            vinf_in_mag = safe_norm(vinf_in)
            vinf_out_mag = safe_norm(vinf_out)
            ceq_parts.append((vinf_out_mag - vinf_in_mag) / V_REF)
            dot = jnp.dot(vinf_in, vinf_out)
            cos_delta = jnp.clip(dot / (vinf_in_mag * vinf_out_mag + 1e-30), -1.0, 1.0)
            delta = jnp.arccos(jnp.clip(cos_delta, -1.0 + 1e-9, 1.0 - 1e-9))
            s = jnp.maximum(jnp.sin(0.5 * delta), 1e-8)
            e = 1.0 / s
            rp = (mu_body / (vinf_in_mag ** 2 + 1e-30)) * (e - 1.0)
            R = self.cat.body_radius(body_id)
            h_p = rp - R
            cineq_parts.append(jnp.array([h_min_norm - h_p / R], dtype=jnp.float64))
            cineq_parts.append(jnp.array([h_p / R - h_max_norm], dtype=jnp.float64))

        final_vinf = vinf_arr[-1]
        if self.opts.objective == "tof":
            f0 = (t[-1] - t[0]) / YEAR_S
        elif self.opts.objective == "vinf":
            f0 = safe_norm(final_vinf)
        elif self.opts.objective == "vinf_rss":
            norms = jnp.sqrt(jnp.sum(vinf_arr[1:] ** 2, axis=1) + 1e-12)
            f0 = jnp.sqrt(jnp.sum(norms ** 2))
        else:
            f0 = safe_norm(final_vinf)

        ceq_vec = (
            jnp.concatenate([c.reshape(-1) for c in ceq_parts])
            if ceq_parts
            else jnp.zeros(0, dtype=jnp.float64)
        )
        cineq_vec = (
            jnp.concatenate([c.reshape(-1) for c in cineq_parts])
            if cineq_parts
            else jnp.zeros(0, dtype=jnp.float64)
        )

        return jnp.concatenate([jnp.array([f0], dtype=jnp.float64), ceq_vec, cineq_vec])

    # ------------------------------------------------------------------
    # counts and bounds
    # ------------------------------------------------------------------
    def get_bounds(self):
        lb = [-1e12] * self.nx
        ub = [1e12] * self.nx
        # time decision vars are stored scaled by T_REF (100 yr), so convert desired 200 yr window
        t_max_scaled = (200.0 * YEAR_S) / T_REF
        for k in self.idx_t:
            lb[k] = 0.0
            ub[k] = t_max_scaled
        if (not self.opts.optimize_t0) and self.idx_t:
            t0_scaled = self.t_guess[0] / T_REF
            lb[self.idx_t[0]] = t0_scaled
            ub[self.idx_t[0]] = t0_scaled
        # vinf box
        vinf_max = 150.0 / V_REF
        for lo, hi in self.idx_vinf:
            for k in range(lo, hi):
                lb[k] = -vinf_max
                ub[k] = vinf_max
        # controls
        for lo, hi in self.idx_ctrl:
            raw_len = hi - lo
            for k in range(0, raw_len, 2):
                lb[lo + k] = 1e-4
                ub[lo + k] = 0.5 * math.pi - 1e-4
                lb[lo + k + 1] = -math.pi
                ub[lo + k + 1] = math.pi
        return lb, ub

    def get_nec(self) -> int:
        nec = 3 * self.M  # position match each leg
        # flyby equality for massive interior bodies: magnitude continuity
        for b in range(1, self.M):
            if self.cat.body_mu(self.seq[b]) > 0.0:
                nec += 1
            else:
                nec += 3
        return nec

    def get_nic(self) -> int:
        nic = 0
        # time ordering
        nic += self.M
        # min leg duration
        nic += self.M
        # same-body gap
        for j in range(self.M):
            if self.seq[j] == self.seq[j + 1]:
                nic += 1
        # flyby altitude
        for b in range(1, self.M):
            if self.cat.body_mu(self.seq[b]) > 0.0:
                nic += 2
        return nic

    def _extract_ctrls_jax(self, x: jnp.ndarray, lo: int, hi: int) -> jnp.ndarray:
        raw = x[lo:hi]
        raw = raw.reshape((self.leg_ctrl.nseg_leg, 2))
        alpha = jnp.clip(raw[:, 0], 1e-4, 0.5 * jnp.pi - 1e-4)
        sigma = raw[:, 1]
        return jnp.stack([alpha, sigma], axis=1)

    def fitness(self, x: Union[List[float], np.ndarray]) -> List[float]:
        x = np.asarray(x, dtype=float)
        f = np.asarray(self._fitness_fn(x))
        expected = 1 + self.get_nec() + self.get_nic()
        if f.size != expected:
            raise RuntimeError(f"fitness length mismatch: expected {expected}, got {f.size}")
        return f.tolist()

    def partials(self, x: Union[List[float], np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = np.asarray(x, dtype=float)
        J = np.asarray(self._jacobian_fn(x))
        n_eq = self.get_nec()
        grad = J[0, :]
        jac_eq = J[1 : 1 + n_eq, :]
        jac_ineq = J[1 + n_eq :, :]
        return grad, jac_eq, jac_ineq

    def ceq_sparsity(self) -> Tuple[np.ndarray, np.ndarray]:
        rows = []
        cols = []
        row = 0
        for j in range(self.M):
            leg_cols = self._leg_columns[j]
            for axis in range(3):
                rows.extend([row + axis] * len(leg_cols))
                cols.extend(leg_cols)
            row += 3
        for b in range(1, self.M):
            mu_body = self.cat.body_mu(self.seq[b])
            cols_union = sorted(set(self._leg_columns[b - 1] + list(range(*self.idx_vinf[b]))))
            n_rows = 3 if mu_body <= 0.0 else 1
            for axis in range(n_rows):
                rows.extend([row + axis] * len(cols_union))
                cols.extend(cols_union)
            row += n_rows
        return np.asarray(rows, dtype=int), np.asarray(cols, dtype=int)

    def cineq_sparsity(self) -> Tuple[np.ndarray, np.ndarray]:
        rows = []
        cols = []
        row = 0
        # time ordering
        for j in range(self.M):
            idx_pair = [self.idx_t[j], self.idx_t[j + 1]]
            rows.extend([row] * len(idx_pair))
            cols.extend(idx_pair)
            row += 1
        # min duration
        for j in range(self.M):
            idx_pair = [self.idx_t[j], self.idx_t[j + 1]]
            rows.extend([row] * len(idx_pair))
            cols.extend(idx_pair)
            row += 1
        # same-body gaps
        for j in range(self.M):
            if self.seq[j] == self.seq[j + 1]:
                idx_pair = [self.idx_t[j], self.idx_t[j + 1]]
                rows.extend([row] * len(idx_pair))
                cols.extend(idx_pair)
                row += 1
        # flyby altitude
        for b in range(1, self.M):
            if self.cat.body_mu(self.seq[b]) <= 0.0:
                continue
            cols_union = sorted(set(self._leg_columns[b - 1] + list(range(*self.idx_vinf[b]))))
            rows.extend([row] * len(cols_union))
            cols.extend(cols_union)
            row += 1
            rows.extend([row] * len(cols_union))
            cols.extend(cols_union)
            row += 1
        return np.asarray(rows, dtype=int), np.asarray(cols, dtype=int)


# -------------------------------------------------------------------
# JSON helper (reimplemented here)
# -------------------------------------------------------------------
def build_from_lambert_json(
    json_path: Union[str, Path],
    solution_rank: int = 1,
    start_index: int = 0,
    end_index: int = -1,
    catalog: Optional[Catalog] = None,
    sail: Optional[SailParams] = None,
    leg_ctrl: Optional[LegCtrlSpec] = None,
    opts: Optional[ProblemOptions] = None,
):
    path = Path(json_path)
    data = json.loads(path.read_text())
    sols = data["solutions"]
    target = next(s for s in sols if int(s["rank"]) == solution_rank)
    encs = target["encounters"]
    if end_index < 0:
        end_index = len(encs) + end_index
    enc_slice = encs[start_index : end_index + 1]

    body_ids = [int(e["body_id"]) for e in enc_slice]
    epochs_s = [float(e["epoch_days"]) * DAY for e in enc_slice]

    if catalog is None:
        catalog = Catalog(mu_star_km3_s2=MU_ALTAIRA)
    if sail is None:
        sail = SailParams(a0_1au_km_s2=DEFAULT_A0_1AU)
    if leg_ctrl is None:
        leg_ctrl = LegCtrlSpec()
    if opts is None:
        opts = ProblemOptions()

    prob = GTOC13TourProblem(
        body_sequence=body_ids,
        t_guess_s=epochs_s,
        catalog=catalog,
        mu_star_km3_s2=MU_ALTAIRA,
        sail=sail,
        leg_ctrl=leg_ctrl,
        opts=opts,
    )

    # init x
    x0 = np.zeros(prob.nx)
    for k, tk in zip(prob.idx_t, epochs_s):
        x0[k] = tk / T_REF
    # zero vinf, default sail controls
    for lo, hi in prob.idx_ctrl:
        raw = x0[lo:hi]
        for jj in range(0, len(raw), 2):
            raw[jj] = 0.5 * math.pi - 0.03  # near edge
            raw[jj + 1] = 0.0
        x0[lo:hi] = raw

    return prob, x0


# -------------------------------------------------------------------
# pyOptSparse glue
# -------------------------------------------------------------------
def build_pyoptsparse_problem(prob: GTOC13TourProblem, x0: np.ndarray):
    n_eq = prob.get_nec()
    n_ineq = prob.get_nic()
    lb, ub = prob.get_bounds()
    def objfunc(xdict):
        x = xdict["x"]
        f = np.asarray(prob.fitness(x), dtype=float)
        return {
            "obj": float(f[0]),
            "ceq": f[1 : 1 + n_eq],
            "cineq": f[1 + n_eq : 1 + n_eq + n_ineq],
        }

    def sensfunc(xdict, funcs):
        x = xdict["x"]
        grad_obj, jac_eq, jac_ineq = prob.partials(x)
        return {
            "obj": {"x": grad_obj},
            "ceq": {"x": jac_eq},
            "cineq": {"x": jac_ineq},
        }

    optProb = Optimization("GTOC13_IPOPT", objfunc)
    optProb.addVarGroup("x", prob.nx, lower=lb, upper=ub, value=x0)
    optProb.addObj("obj")
    if n_eq > 0:
        optProb.addConGroup("ceq", n_eq, lower=0.0, upper=0.0)
    if n_ineq > 0:
        optProb.addConGroup("cineq", n_ineq, lower=-1e20, upper=0.0)

    return optProb, sensfunc


def main():
    # change this to your real file
    json_path = "/Users/home/codes/gtoc13/results/beam/bs_mission-raw_bw100_d20_top100_20251107-234707Z.json"
    prob, x0 = build_from_lambert_json(
        json_path,
        solution_rank=1,
        start_index=0,
        end_index=1,
    )
    optProb, sensfunc = build_pyoptsparse_problem(prob, x0)

    ipopt_opts = {
        "tol": 1e-6,
        "constr_viol_tol": 1e-6,
        "acceptable_tol": 1e-4,
        "max_iter": 800,
        "print_level": 5,
    }
    opt = IPOPT(options=ipopt_opts)
    sol = opt(optProb, sens=sensfunc)

    x_opt = sol.getDVs()["x"]
    f = np.asarray(prob.fitness(x_opt), dtype=float)
    n_eq = prob.get_nec()
    print("Objective:", f[0])
    print("eq norm:", np.linalg.norm(f[1 : 1 + n_eq]))
    print("max ineq:", np.max(f[1 + n_eq :]))


if __name__ == "__main__":
    main()
