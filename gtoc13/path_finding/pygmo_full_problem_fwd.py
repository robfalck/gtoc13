"""
GTOC13 multiple-shooting UDP in pygmo

This module implements a flexible multiple-shooting formulation for
heliocentric 2-body dynamics with an ideal solar sail, patched conic
flybys at planets, and continuity at massless bodies (asteroids/comets).

Core features
-------------
- Arbitrary-length sequence of bodies (planets + small bodies)
- Event-based multiple shooting with mid-point matching per leg
- Piecewise-constant sail controls per half-leg: (cone alpha, clock sigma)
- Planetary flyby modeling via v-infinity rotation and altitude window
- Time ordering, same-body spacing constraint

Notes
-----
* Units: SI-like, using kilometers and seconds throughout.
  - Distances in km, velocities km/s, time in s, angles in radians.
  - The central star gravitational parameter mu_star must be provided.
  - a0_1au is the sail max acceleration at 1 AU in km/s^2.
* Ephemerides: bodies loaded from the provided CSVs in /mnt/data.
  - Planets: provide GM (km^3/s^2) and radius (km), and Keplerian elements
    at contest epoch t = 0.
  - Asteroids/comets: massless for flyby; only Keplerian elements.
* Initial arrival condition from interstellar space is left as a TODO hook
  (see `enforce_initial_arrival_constraints`). The rest of the formulation
  is fully functional.

Author: ChatGPT (GTOC helper)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import json
import math
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import diffrax as dfx

from gtoc13.bodies import bodies_data, INTERSTELLAR_BODY_ID
from gtoc13.constants import MU_ALTAIRA, C_FLUX, SAIL_AREA, SPACECRAFT_MASS, DAY
from gtoc13.ephemerides_jax import keplerian_state

try:
    import pygmo as pg
except Exception as e:  # pragma: no cover
    pg = None  # allow import for static analysis without pygmo


# ---------------------------
# Constants & small utilities
# ---------------------------
AU_KM = 149_597_870.7  # km
DEFAULT_A0_1AU = 2.0 * C_FLUX * SAIL_AREA / SPACECRAFT_MASS / 1000.0  # km/s^2

YEAR_S = 365.25 * 86400.0
# -------------
# Scaling constants for nondimensionalization
L_REF = 1.0e5           # length scale for constraint scaling: 100,000 km
T_REF = 100.0 * YEAR_S  # time scale: 100 years in s
V_REF = 1.0e2           # velocity scale: 100 km/s
TIME_SCALE = DAY        # scale inequality times to days
DEG2RAD = math.pi / 180.0

jax.config.update("jax_enable_x64", True)


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _set_equal_xy_limits(ax, points: np.ndarray, padding: float = 0.2) -> None:
    if points.size == 0:
        return
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    centers = 0.5 * (mins + maxs)
    span = (maxs - mins).max()
    radius = 0.5 * span if span > 0 else 1.0
    radius *= 1.0 + max(0.0, padding)
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)


class Catalog:
    """Thin wrapper around gtoc13.bodies for reusable ephemerides/metadata."""

    def __init__(self, mu_star_km3_s2: float):
        self.mu_star = mu_star_km3_s2
        self.bodies = bodies_data

    def _body(self, body_id: int):
        try:
            return self.bodies[body_id]
        except KeyError as exc:  # pragma: no cover
            raise KeyError(f"Unknown body id {body_id}") from exc

    def is_planet(self, body_id: int) -> bool:
        return self._body(body_id).is_planet()

    def body_radius(self, body_id: int) -> float:
        return float(self._body(body_id).radius)

    def body_mu(self, body_id: int) -> float:
        return float(self._body(body_id).mu)

    def state(self, body_id: int, t: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Heliocentric (r,v) at time t for body id using shared body ephemerides.
        t in seconds since contest epoch. r in km, v in km/s.
        """
        body = self._body(body_id)
        state = body.get_state(t, time_units="s", distance_units="km")
        r = jnp.asarray(state.r, dtype=jnp.float64)
        v = jnp.asarray(state.v, dtype=jnp.float64)
        return r, v


def _elements_vector(body) -> jnp.ndarray:
    elems = body.elements
    return jnp.asarray(
        [
            float(elems.a),
            float(elems.e),
            float(elems.i),
            float(elems.Omega),
            float(elems.omega),
            float(elems.M0),
        ],
        dtype=jnp.float64,
    )
    def orbital_period(self, body_id: int) -> float:
        """Keplerian period [s] of body around the star."""
        body = self._body(body_id)
        return float(body.get_period("s"))


# ---------------------------
# Sail dynamics & integrator
# ---------------------------
@dataclass
class SailParams:
    a0_1au_km_s2: float  # max accel at 1 AU (km/s^2)


def sail_accel(
    r: Union[np.ndarray, jnp.ndarray],
    v: Union[np.ndarray, jnp.ndarray],
    alpha: float,
    sigma: float,
    a0_1au: float,
) -> jnp.ndarray:
    """Ideal sail acceleration in km/s^2 (JAX-compatible)."""
    r = jnp.asarray(r, dtype=jnp.float64)
    v = jnp.asarray(v, dtype=jnp.float64)
    rnorm = jnp.linalg.norm(r)
    rnorm_safe = jnp.where(rnorm > 0.0, rnorm, 1.0)
    rhat = r / rnorm_safe
    h = jnp.cross(r, v)
    hnorm = jnp.linalg.norm(h)

    def _degenerate_frame(_):
        that = jnp.array([0.0, 1.0, 0.0], dtype=jnp.float64)
        nhat = jnp.array([0.0, 0.0, 1.0], dtype=jnp.float64)
        return that, nhat

    def _regular_frame(_):
        hhat = h / jnp.where(hnorm > 0.0, hnorm, 1.0)
        that = jnp.cross(hhat, rhat)
        that_norm = jnp.linalg.norm(that)
        that = that / jnp.where(that_norm > 0.0, that_norm, 1.0)
        nhat = hhat
        return that, nhat

    that, nhat = jax.lax.cond(hnorm < 1e-12, _degenerate_frame, _regular_frame, operand=None)
    cos_alpha = jnp.cos(alpha)
    sin_alpha = jnp.sin(alpha)
    n_hat = cos_alpha * rhat + sin_alpha * (
        jnp.cos(sigma) * that + jnp.sin(sigma) * nhat
    )
    scale = a0_1au * (AU_KM / rnorm_safe) ** 2 * (cos_alpha ** 2)
    return scale * n_hat


@dataclass
class IntOptions:
    solver: str = "tsit5"
    rtol: float = 1e-6
    atol: float = 1e-9
    max_steps: int = 1_000_000  # keep finite for reverse-mode adjoint memory budgeting
    dt0_fraction: float = 0.1




class SailPropagator:
    _SOLVERS = {
        "tsit5": dfx.Tsit5,
        "dopri5": dfx.Dopri5,
        "heun": dfx.Heun,
    }

    def __init__(
        self,
        mu_star: float,
        sail: SailParams,
        *,
        accel_smoothing: float = 0.0,
        opts: IntOptions = IntOptions(),
        use_solar_sail: bool = True,
    ):
        self.mu = mu_star
        self.sail = sail
        self.opts = opts
        self.accel_smoothing = max(0.0, accel_smoothing)
        self.use_solar_sail = use_solar_sail
        solver_cls = self._SOLVERS.get((opts.solver or "tsit5").lower(), dfx.Tsit5)
        self._solver = solver_cls()
        self._ode_term = dfx.ODETerm(self._rhs)
        self._controller = dfx.PIDController(rtol=opts.rtol, atol=opts.atol)
        self._saveat_t1 = dfx.SaveAt(t1=True)

    def _rhs(self, t, y, args):
        mu_star, a0, alpha, sigma, smoothing = args
        r = y[:3]
        v = y[3:]
        rnorm = jnp.linalg.norm(r)
        acc_grav = -mu_star * r / (jnp.power(rnorm, 3) + 1e-30)
        if self.use_solar_sail:
            acc_sail = sail_accel(r, v, alpha, sigma, a0)
        else:
            acc_sail = jnp.zeros_like(r)

        def _smooth(_):
            mag = jnp.linalg.norm(acc_sail)
            floor = smoothing
            scaled = acc_sail * (floor / jnp.where(mag > 1e-12, mag, 1.0))
            return jnp.where(mag < floor, scaled, acc_sail)

        acc_sail = jax.lax.cond(
            smoothing > 0.0,
            _smooth,
            lambda _: acc_sail,
            operand=None,
        )

        return jnp.concatenate([v, acc_grav + acc_sail])

    def _dt0_hint(self, dt: jnp.ndarray) -> jnp.ndarray:
        mag = jnp.maximum(jnp.abs(dt) * self.opts.dt0_fraction, 1e-6)
        sign = jnp.where(dt >= 0.0, 1.0, -1.0)
        return sign * mag

    def propagate_piecewise_jax(
        self,
        t0: float,
        y0: jnp.ndarray,
        t1: float,
        controls: jnp.ndarray,
    ) -> jnp.ndarray:
        controls = jnp.asarray(controls, dtype=jnp.float64)
        if controls.size == 0:
            return y0
        total_dt = t1 - t0
        nseg = controls.shape[0]
        dt = total_dt / nseg

        def integrate_segment(carry, ctrl):
            t_cur, y_cur = carry
            alpha, sigma = ctrl
            dt0 = self._dt0_hint(dt)
            sol = dfx.diffeqsolve(
                self._ode_term,
                self._solver,
                t0=t_cur,
                t1=t_cur + dt,
                dt0=dt0,
                y0=y_cur,
                args=(self.mu, self.sail.a0_1au_km_s2, alpha, sigma, self.accel_smoothing),
                stepsize_controller=self._controller,
                max_steps=self.opts.max_steps,
                saveat=self._saveat_t1,
            )
            y_end = jnp.reshape(sol.ys, y_cur.shape)
            return (t_cur + dt, y_end), y_end

        def do_integrate():
            (_, y_last), _ = jax.lax.scan(integrate_segment, (t0, y0), controls)
            return y_last

        return jax.lax.cond(
            jnp.abs(total_dt) < 1e-12,
            lambda _: y0,
            lambda _: do_integrate(),
            operand=None,
        )

    def _propagate_with_samples(
        self,
        t0: float,
        y0: np.ndarray,
        t1: float,
        controls: np.ndarray,
        samples_per_segment: int,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        samples: List[np.ndarray] = [y0[:3].copy()]
        total_dt = float(t1 - t0)
        if abs(total_dt) < 1e-12:
            return y0.copy(), samples
        nseg = max(1, len(controls))
        dt = total_dt / nseg
        y = y0.copy()
        t = float(t0)
        for alpha, sigma in controls:
            seg_t1 = t + dt
            ts = np.linspace(t, seg_t1, max(2, samples_per_segment + 1))
            saveat = dfx.SaveAt(ts=jnp.asarray(ts[1:], dtype=jnp.float64))
            dt0 = float(math.copysign(max(abs(dt) * self.opts.dt0_fraction, 1e-6), dt if dt != 0 else 1.0))
            sol = dfx.diffeqsolve(
                self._ode_term,
                self._solver,
                t0=t,
                t1=seg_t1,
                dt0=dt0,
                y0=jnp.asarray(y, dtype=jnp.float64),
                args=(self.mu, self.sail.a0_1au_km_s2, alpha, sigma, self.accel_smoothing),
                stepsize_controller=self._controller,
                max_steps=self.opts.max_steps,
                saveat=saveat,
            )
            ys = np.asarray(sol.ys)
            samples.extend(ys[:, :3])
            y = ys[-1]
            t = seg_t1
        return y, samples

    def propagate_piecewise(
        self,
        t0: float,
        y0: np.ndarray,
        t1: float,
        controls: List[Tuple[float, float]],
        collect_samples: bool = False,
        samples_per_segment: int = 200,
    ) -> Tuple[np.ndarray, List[np.ndarray], float]:
        controls_arr = np.asarray(controls, dtype=float).reshape((-1, 2)) if controls else np.zeros((0, 2))
        y0_arr = np.asarray(y0, dtype=float)
        if collect_samples:
            y_end_np, samples = self._propagate_with_samples(t0, y0_arr, t1, controls_arr, samples_per_segment)
        else:
            y_end = self.propagate_piecewise_jax(
                float(t0),
                jnp.asarray(y0_arr, dtype=jnp.float64),
                float(t1),
                jnp.asarray(controls_arr, dtype=jnp.float64),
            )
            y_end_np = np.asarray(y_end, dtype=float)
            samples = []
        r_start = float(np.linalg.norm(y0_arr[:3]))
        r_end = float(np.linalg.norm(y_end_np[:3]))
        if samples:
            norms = [float(np.linalg.norm(pt)) for pt in samples]
            rmin = min(norms) if norms else min(r_start, r_end)
        else:
            rmin = min(r_start, r_end)
        return y_end_np, samples, rmin


# ---------------------------
# Flyby utilities (patched conics)
# ---------------------------
def turn_angle(vin_minus: np.ndarray, vin_plus: np.ndarray) -> float:
    c = np.dot(vin_minus, vin_plus) / (np.linalg.norm(vin_minus) * np.linalg.norm(vin_plus) + 1e-30)
    return math.acos(clamp(c, -1.0, 1.0))


def rp_from_turn(vinf: float, mu_p: float, delta: float) -> float:
    """Pericenter radius from turn angle (hyperbola): delta = 2*asin(1/e), e=1+rp*vinf^2/mu.
    Solve rp = (mu / vinf^2) * (1/(sin(delta/2)) - 1).
    """
    s = math.sin(0.5 * delta)
    s = max(1e-12, s)
    e = 1.0 / s
    rp = (mu_p / (vinf ** 2 + 1e-30)) * (e - 1.0)
    return rp


# ---------------------------
# UDP: decision vector layout
# ---------------------------
@dataclass
class LegCtrlSpec:
    nseg_leg: int = 5  # segments per full leg


@dataclass
class ProblemOptions:
    same_body_gap_factor: float = 1.0 / 3.0
    enforce_initial_arrival: bool = False  # TODO hook (see notes)
    progress_every_evals: int = 0  # print progress every N fitness evals; 0 disables printing
    flyby_altitude_bounds: Tuple[float, float] = (0.1, 100.0)  # normalized by body radius (min, max)
    objective: str = "energy"  # "energy", "vinf", "tof", "vinf_rss", or "feasibility"
    optimize_t0: bool = False  # optimize the start time when True
    constrain_interstellar_direction: bool = False  # enforce vy=vz=0 at body 0 when True
    accel_smoothing: float = 0.0  # min sail accel (km/s^2) applied via smoothing floor
    use_jit: bool = True  # compile fitness/jacobian with jax.jit for speed when True
    objective_scale: Optional[float] = None  # optional manual scaling of the objective value
    use_solar_sail: bool = True  # disable to fly purely under 2-body dynamics


class GTOC13TourUDP:
    """pygmo UDP implementing multiple-shooting for a flyby tour with a sail.

    Decision vector x packs:
    - Event times: t[0..M] (s) for each body in the sequence
    - For each leg j (from body j to body j+1):
        v_inf_depart_j (3) body-centric hyperbolic excess velocity components
        piecewise-constant sail controls over the full leg: (alpha,sigma)*nseg_leg
    Notes:
    * alpha in [0, pi/2], sigma in [0, 2*pi).
    * Angles are stored unbounded and reduced mod bounds in unpacking for robustness.
    """

    def __init__(
        self,
        body_sequence: List[int],
        t_guess_s: List[float],
        catalog: Catalog,
        mu_star_km3_s2: float,
        sail: SailParams,
        leg_ctrl: LegCtrlSpec = LegCtrlSpec(),
        opts: ProblemOptions = ProblemOptions(),
    ):
        assert len(body_sequence) >= 1
        assert len(t_guess_s) == len(body_sequence)
        self.seq = list(body_sequence)
        self.M = len(self.seq) - 1
        self.t_guess = list(t_guess_s)
        self.cat = catalog
        self.mu = mu_star_km3_s2
        self.prop = SailPropagator(
            mu_star_km3_s2,
            sail,
            accel_smoothing=opts.accel_smoothing,
            use_solar_sail=opts.use_solar_sail,
        )
        self.sail = sail
        self.leg_ctrl = leg_ctrl
        self.popts = opts
        self.controls_enabled = bool(opts.use_solar_sail)
        self._ctrl_shape = (self.leg_ctrl.nseg_leg, 2)
        self._zero_ctrls_np = np.zeros(self._ctrl_shape)
        self._zero_ctrls_jax = jnp.zeros(self._ctrl_shape, dtype=jnp.float64)
        self._has_interstellar_dir_constraint = (
            self.M > 0
            and self.seq[0] == INTERSTELLAR_BODY_ID
            and self.popts.constrain_interstellar_direction
        )

        # Progress printing
        self.progress_every = opts.progress_every_evals
        self._eval_ctr = 0

        # Variable indexing map
        self._build_indexing()
        self._precompute_leg_columns()
        self._build_constraint_sparsity()
        self._prepare_ephemerides()
        self._build_autodiff_handles()

    # -----------------------
    # Variable packing helpers
    # -----------------------
    def _build_indexing(self):
        idx = 0
        self.idx_t = list(range(idx, idx + len(self.seq)))
        idx += len(self.seq)

        # Event state blocks
        # Leg departure v-infinity vectors (body-centric)
        self.idx_vinf = []
        for j in range(self.M):
            self.idx_vinf.append((idx, idx + 3))
            idx += 3

        # Controls per leg
        self.idx_ctrl = []
        npar_leg = 2 * self.leg_ctrl.nseg_leg
        for j in range(self.M):
            if self.controls_enabled and npar_leg > 0:
                self.idx_ctrl.append((idx, idx + npar_leg))
                idx += npar_leg
            else:
                self.idx_ctrl.append((None, None))
        self.nx = idx
        self._grad_sparsity = [(0, col) for col in range(self.nx)]

    def _precompute_leg_columns(self) -> None:
        self._leg_columns: List[List[int]] = []
        for j in range(self.M):
            cols = {self.idx_t[j], self.idx_t[j + 1]}
            lo, hi = self.idx_vinf[j]
            cols.update(range(lo, hi))
            ctrl_idx = self.idx_ctrl[j]
            if ctrl_idx[0] is not None:
                lo, hi = ctrl_idx
                cols.update(range(lo, hi))
            self._leg_columns.append(sorted(cols))

    def _build_constraint_sparsity(self) -> None:
        eq_rows: List[Tuple[int, ...]] = []
        if self._has_interstellar_dir_constraint and self.idx_vinf:
            vinf_cols = list(range(*self.idx_vinf[0]))
            if len(vinf_cols) >= 3:
                eq_rows.append((vinf_cols[1],))
                eq_rows.append((vinf_cols[2],))
        for leg_cols in self._leg_columns:
            eq_rows.extend([tuple(leg_cols)] * 3)
        for body_idx in range(1, self.M):
            mu_body = self.cat.body_mu(self.seq[body_idx])
            cols_union = sorted(
                set(self._leg_columns[body_idx - 1] + list(range(*self.idx_vinf[body_idx])))
            )
            if mu_body <= 0.0:
                eq_rows.extend([tuple(cols_union)] * 3)
            else:
                eq_rows.append(tuple(cols_union))
        self._eq_sparsity_by_row = eq_rows
        eq_pairs: List[Tuple[int, int]] = []
        row_idx = 0
        for cols in eq_rows:
            eq_pairs.extend((row_idx, col) for col in cols)
            row_idx += 1
        self._eq_sparsity_pairs = eq_pairs

        ineq_rows: List[Tuple[int, ...]] = []
        for j in range(self.M):
            cols_pair = (self.idx_t[j], self.idx_t[j + 1])
            ineq_rows.append(cols_pair)
        for j in range(self.M):
            cols_pair = (self.idx_t[j], self.idx_t[j + 1])
            ineq_rows.append(cols_pair)
        for j in range(self.M):
            if self.seq[j] == self.seq[j + 1]:
                cols_pair = (self.idx_t[j], self.idx_t[j + 1])
                ineq_rows.append(cols_pair)
        for body_idx in range(1, self.M):
            if self.cat.body_mu(self.seq[body_idx]) <= 0.0:
                continue
            cols_union = sorted(
                set(self._leg_columns[body_idx - 1] + list(range(*self.idx_vinf[body_idx])))
            )
            cols_tuple = tuple(cols_union)
            ineq_rows.append(cols_tuple)
            ineq_rows.append(cols_tuple)
        self._ineq_sparsity_by_row = ineq_rows
        ineq_pairs: List[Tuple[int, int]] = []
        row_idx = 0
        for cols in ineq_rows:
            ineq_pairs.extend((row_idx, col) for col in cols)
            row_idx += 1
        self._ineq_sparsity_pairs = ineq_pairs

    def _prepare_ephemerides(self) -> None:
        radii: List[float] = []
        body_mu: List[float] = []
        periods: List[float] = []
        elem_stack: List[jnp.ndarray] = []
        has_elem: List[bool] = []
        fixed_r: List[jnp.ndarray] = []
        fixed_v: List[jnp.ndarray] = []
        for body_id in self.seq:
            body = self.cat._body(body_id)
            radii.append(float(body.radius))
            body_mu.append(float(body.mu))
            try:
                periods.append(float(body.get_period("s")))
            except ValueError:
                periods.append(float("inf"))
            elems = getattr(body, "elements", None)
            if elems is None:
                r0, v0 = self.cat.state(body_id, 0.0)
                elem_stack.append(jnp.zeros(6, dtype=jnp.float64))
                has_elem.append(False)
                fixed_r.append(jnp.asarray(r0, dtype=jnp.float64))
                fixed_v.append(jnp.asarray(v0, dtype=jnp.float64))
            else:
                elem_stack.append(_elements_vector(body))
                has_elem.append(True)
                fixed_r.append(jnp.zeros(3, dtype=jnp.float64))
                fixed_v.append(jnp.zeros(3, dtype=jnp.float64))
        self._seq_elements = jnp.stack(elem_stack)
        self._seq_has_elements = jnp.asarray(has_elem, dtype=jnp.bool_)
        self._seq_fixed_r = jnp.stack(fixed_r)
        self._seq_fixed_v = jnp.stack(fixed_v)
        self._seq_radii = jnp.asarray(radii, dtype=jnp.float64)
        self._seq_body_mu = jnp.asarray(body_mu, dtype=jnp.float64)
        self._seq_periods = jnp.asarray(periods, dtype=jnp.float64)

    def _build_autodiff_handles(self) -> None:
        def fitness_fn(vec):
            return self._fitness_vector_impl(vec)

        jac_fn = jax.jacrev(fitness_fn)
        if self.popts.use_jit:
            fitness_fn = jax.jit(fitness_fn)
            jac_fn = jax.jit(jac_fn)

        self._fitness_fn = fitness_fn
        self._jacobian_fn = jac_fn

    def get_bounds(self) -> Tuple[List[float], List[float]]:  # pygmo API
        lb = [-1e12] * self.nx
        ub = [1e12] * self.nx
        # Times bounded to mission window [0, 200 years], scaled
        for k in self.idx_t:
            lb[k] = 0.0
            ub[k] = 2.0
        if (not self.popts.optimize_t0) and self.idx_t:
            t0_scaled = self.t_guess[0] / T_REF
            lb[self.idx_t[0]] = t0_scaled
            ub[self.idx_t[0]] = t0_scaled
        # Limit v-infinity components to a reasonable search box (km/s scaled by V_REF)
        vinf_max = 150.0 / V_REF  # store in scaled units
        for lo, hi in self.idx_vinf:
            for k in range(lo, hi):
                lb[k] = -vinf_max
                ub[k] = vinf_max
        # Bound control angles directly to avoid wrapping
        alpha_min = 1e-4
        alpha_max = 0.5 * math.pi - 1e-4
        for lo, hi in self.idx_ctrl:
            if lo is None or hi is None:
                continue
            for k in range(lo, hi, 2):
                lb[k] = alpha_min
                ub[k] = alpha_max
                lb[k + 1] = -math.pi
                ub[k + 1] = math.pi
        return lb, ub

    # -----------------------
    # Constraint counts
    # -----------------------
    def get_nec(self) -> int:  # pygmo API
        nec = 3 * self.M
        if self._has_interstellar_dir_constraint:
            nec += 2
        # Flyby or continuity equalities at interior bodies
        for body_idx in range(1, self.M):
            body_id = self.seq[body_idx]
            if self.cat.body_mu(body_id) > 0.0:
                nec += 1  # magnitude continuity
            else:
                nec += 3  # vector continuity for massless body
        # (Optional) initial arrival constraints — TODO: typically 3 eq
        # if self.popts.enforce_initial_arrival: nec += 3
        return nec

    def get_nic(self) -> int:  # pygmo API
        nic = 0
        # Time ordering: t[j+1]-t[j] >= 0
        nic += self.M
        # Minimum leg duration (e.g., 3 days) to avoid dt->0 pathologies
        nic += self.M
        # Same-body gap: for successive equal bodies: dt - T/3 >= 0
        for j in range(self.M):
            if self.seq[j] == self.seq[j + 1]:
                nic += 1
        # Flyby altitude feasibility at interior massive bodies: two inequalities (lower/upper)
        for body_idx in range(1, self.M):
            body_id = self.seq[body_idx]
            if self.cat.body_mu(body_id) > 0.0:
                nic += 2
        return nic

    def _get_ctrls(self, x: np.ndarray, idx_pair: Tuple[int, int]) -> List[Tuple[float, float]]:
        lo, hi = idx_pair
        if lo is None or hi is None:
            return [(0.0, 0.0)] * self.leg_ctrl.nseg_leg
        raw = x[lo:hi]
        ctrls = []
        for k in range(0, len(raw), 2):
            alpha = clamp(raw[k], 0.0, 0.5 * math.pi)
            sigma = raw[k + 1]
            ctrls.append((alpha, sigma))
        return ctrls

    def _extract_ctrls_jax(self, x: jnp.ndarray, idx_pair: Tuple[int, int]) -> jnp.ndarray:
        lo, hi = idx_pair
        if lo is None or hi is None:
            return self._zero_ctrls_jax
        raw = x[lo:hi].reshape((self.leg_ctrl.nseg_leg, 2))
        alpha = jnp.clip(raw[:, 0], 1e-4, 0.5 * jnp.pi - 1e-4)
        sigma = raw[:, 1]
        return jnp.stack([alpha, sigma], axis=1)

    # -----------------------
    # Fitness
    # -----------------------
    def _fitness_vector_impl(self, x: Union[np.ndarray, jnp.ndarray]) -> jnp.ndarray:
        x = jnp.asarray(x, dtype=jnp.float64)
        idx_t = jnp.asarray(self.idx_t, dtype=jnp.int32)
        t = x[idx_t] * T_REF

        vinf_dep_list = []
        for lo, hi in self.idx_vinf:
            vinf_dep_list.append(x[lo:hi] * V_REF)
        vinf_dep_arr = (
            jnp.stack(vinf_dep_list) if vinf_dep_list else jnp.zeros((0, 3), dtype=jnp.float64)
        )

        def _state_eval(elem, has_elem, r_fix, v_fix, tt):
            return jax.lax.cond(
                has_elem,
                lambda _: keplerian_state(elem, self.mu, tt),
                lambda _: (r_fix, v_fix),
                operand=None,
            )

        rb, vb = jax.vmap(_state_eval)(
            self._seq_elements,
            self._seq_has_elements,
            self._seq_fixed_r,
            self._seq_fixed_v,
            t,
        )

        ceq_parts: List[jnp.ndarray] = []
        cineq_parts: List[jnp.ndarray] = []

        if self.M > 0:
            cineq_parts.append((t[:-1] - t[1:]) / TIME_SCALE)
            dt_leg = t[1:] - t[:-1]
            dt_min = 3.0 * DAY
            cineq_parts.append((dt_min - dt_leg) / TIME_SCALE)
        vinf_arr = jnp.zeros((len(self.seq), 3), dtype=jnp.float64)

        if self._has_interstellar_dir_constraint:
            vinf0 = vinf_dep_arr[0]
            ceq_parts.append(jnp.array([vinf0[1] / V_REF], dtype=jnp.float64))
            ceq_parts.append(jnp.array([vinf0[2] / V_REF], dtype=jnp.float64))

        for j in range(self.M):
            ctrls = self._extract_ctrls_jax(x, self.idx_ctrl[j])
            y0 = jnp.concatenate([rb[j], vb[j] + vinf_dep_arr[j]])
            yf = self.prop.propagate_piecewise_jax(t[j], y0, t[j + 1], ctrls)
            ceq_parts.append((yf[:3] - rb[j + 1]) / L_REF)
            vinf_arr = vinf_arr.at[j + 1].set(yf[3:] - vb[j + 1])

        for j in range(self.M):
            if self.seq[j] == self.seq[j + 1]:
                T_body = self._seq_periods[j]
                gap = (self.popts.same_body_gap_factor * T_body - (t[j + 1] - t[j])) / TIME_SCALE
                cineq_parts.append(jnp.array([gap], dtype=jnp.float64))

        h_min_norm, h_max_norm = self.popts.flyby_altitude_bounds
        for body_idx in range(1, self.M):
            body_id = self.seq[body_idx]
            vinf_in = vinf_arr[body_idx]
            vinf_out = vinf_dep_arr[body_idx]
            mu_body = self._seq_body_mu[body_idx]
            if mu_body <= 0.0:
                ceq_parts.append((vinf_out - vinf_in) / V_REF)
                continue
            vinf_in_mag = jnp.linalg.norm(vinf_in)
            vinf_out_mag = jnp.linalg.norm(vinf_out)
            ceq_parts.append((vinf_out_mag - vinf_in_mag) / V_REF)
            dot = jnp.dot(vinf_in, vinf_out)
            cos_delta = jnp.clip(dot / (vinf_in_mag * vinf_out_mag + 1e-30), -1.0, 1.0)
            delta = jnp.arccos(cos_delta)
            s = jnp.maximum(jnp.sin(0.5 * delta), 1e-8)
            e = 1.0 / s
            rp = (mu_body / (vinf_in_mag ** 2 + 1e-30)) * (e - 1.0)
            R = self._seq_radii[body_idx]
            h_p = rp - R
            cineq_parts.append(jnp.array([h_min_norm - h_p / R], dtype=jnp.float64))
            cineq_parts.append(jnp.array([h_p / R - h_max_norm], dtype=jnp.float64))

        final_vinf = vinf_arr[-1]
        obj_mode = (self.popts.objective or "energy").lower()
        if obj_mode == "feasibility":
            f0 = jnp.array(0.0, dtype=jnp.float64)
        elif obj_mode == "tof":
            f0 = (t[-1] - t[0]) / YEAR_S
        elif obj_mode == "vinf":
            f0 = jnp.linalg.norm(final_vinf)
        elif obj_mode == "vinf_rss":
            vinf_norms = jnp.linalg.norm(vinf_arr[1:], axis=1)
            f0 = jnp.sqrt(jnp.sum(vinf_norms ** 2))
        else:
            r_final = rb[-1]
            v_final = vb[-1] + final_vinf
            f0 = 0.5 * jnp.dot(v_final, v_final) - self.mu / (jnp.linalg.norm(r_final) + 1e-12)

        scale = self.popts.objective_scale
        if scale is None:
            if obj_mode == "energy":
                scale = V_REF ** 2
            elif obj_mode in ("vinf", "vinf_rss"):
                scale = V_REF
            elif obj_mode == "tof":
                scale = 100.0  # normalize to ~centuries
            else:
                scale = 1.0
        f0 = f0 / max(scale, 1e-16)

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

    def fitness(self, x: List[float]) -> List[float]:  # pygmo API
        x_np = np.asarray(x, dtype=float)
        f_vec = np.asarray(self._fitness_fn(x_np))
        expected_len = 1 + self.get_nec() + self.get_nic()
        if f_vec.size != expected_len:
            raise RuntimeError(
                f"Fitness length mismatch: expected {expected_len}, got {f_vec.size} "
                f"(nec+nic={expected_len - 1})."
            )
        self._eval_ctr += 1
        if self.progress_every > 0 and (self._eval_ctr % self.progress_every == 0):
            eq_size = self.get_nec()
            eq_norm = float(np.linalg.norm(f_vec[1 : 1 + eq_size]))
            cineq = f_vec[1 + eq_size :]
            ineq_penalty = float(np.sum(np.minimum(0.0, cineq) ** 2))
            print(
                f"[fit] eval {self._eval_ctr:6d}  f0={f_vec[0]:.4e}  ||eq||={eq_norm:.2e}  ineq_pen={ineq_penalty:.2e}",
                flush=True,
            )
        return f_vec.tolist()

    # -----------------------
    # Problem meta
    # -----------------------
    def get_nobj(self) -> int:  # pygmo API
        return 1

    def gradient(self, x: List[float]) -> List[float]:
        x_np = np.asarray(x, dtype=float)
        J = np.asarray(self._jacobian_fn(x_np))
        return J[0, :].tolist()

    def has_gradient(self) -> bool:
        return True

    def gradient_sparsity(self):
        return list(self._grad_sparsity)

    def has_gradient_sparsity(self) -> bool:
        return True

    def jacobian(self, x: List[float]) -> List[float]:
        x_np = np.asarray(x, dtype=float)
        J = np.asarray(self._jacobian_fn(x_np))
        n_eq = self.get_nec()
        jac_eq = J[1 : 1 + n_eq]
        jac_ineq = J[1 + n_eq :]
        values: List[float] = []
        for row_idx, cols in enumerate(self._eq_sparsity_by_row):
            if len(cols) == 1:
                values.append(float(jac_eq[row_idx, cols[0]]))
            else:
                values.extend(jac_eq[row_idx, list(cols)])
        for row_idx, cols in enumerate(self._ineq_sparsity_by_row):
            if len(cols) == 1:
                values.append(float(jac_ineq[row_idx, cols[0]]))
            else:
                values.extend(jac_ineq[row_idx, list(cols)])
        return np.asarray(values, dtype=float).tolist()

    def has_jacobian(self) -> bool:
        return True

    def jacobian_sparsity(self):
        offset = self.get_nec()
        eq_pairs = list(self._eq_sparsity_pairs)
        ineq_pairs = [(row_idx + offset, col) for row_idx, col in self._ineq_sparsity_pairs]
        return eq_pairs + ineq_pairs

    def has_jacobian_sparsity(self) -> bool:
        return True

    def get_name(self) -> str:  # pygmo API
        return "GTOC13 multiple-shooting tour (sail)"

    def get_extra_info(self) -> str:  # pygmo API
        return (
            f"legs={self.M}, nseg_leg={self.leg_ctrl.nseg_leg}"
        )


# ---------------------------
# Convenience builder / example
# ---------------------------

def make_initial_guess(udp: GTOC13TourUDP, body_sequence: List[int], t_guess_s: List[float]) -> np.ndarray:
    """Construct a coarse initial guess vector x consistent with catalogs.
    - Sets event times from the provided guess (scaled).
    - Sets all leg departure v_infinity vectors to zero (coasting start).
    - Sets all controls to alpha ~ 90° - 3° (small thrust) so sensitivities exist, sigma=0.
    - Times written as scaled (t_j / T_REF).
    """
    x = np.zeros(udp.nx)
    # Times (write scaled)
    for k, tk in zip(udp.idx_t, t_guess_s):
        x[k] = tk / T_REF  # scale to nondimensional
    # Departure v_infinity guesses (all zero)
    for lo, hi in udp.idx_vinf:
        x[lo:hi] = 0.0
    # Controls
    alpha0 = max(0.0, 0.5 * math.pi - (0.01 * DEG2RAD))
    for j in range(udp.M):
        lo, hi = udp.idx_ctrl[j]
        if lo is None or hi is None:
            continue
        raw = np.zeros(hi - lo)
        for k in range(0, len(raw), 2):
            raw[k] = alpha0
            raw[k + 1] = 0.0
        x[lo:hi] = raw
    return x


def _extract_lambert_segment(
    json_path: Union[str, Path],
    solution_rank: int = 1,
    start_index: int = 0,
    end_index: int = -1,
) -> Tuple[List[int], List[float], List[Optional[np.ndarray]]]:
    """Extract a contiguous segment of encounters from a Lambert beam-search JSON."""
    path = Path(json_path)
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    solutions = data.get("solutions", [])
    target = next((sol for sol in solutions if int(sol.get("rank", 0)) == solution_rank), None)
    if target is None:
        raise ValueError(f"Rank {solution_rank} not found in {path}")
    encounters = target.get("encounters", [])
    if not encounters:
        raise ValueError(f"No encounters stored for rank {solution_rank} in {path}")

    n_enc = len(encounters)
    if end_index is None:
        end_index = n_enc - 1
    if end_index < 0:
        end_index = n_enc + end_index
    if not (0 <= start_index < n_enc):
        raise IndexError(f"start_index {start_index} out of range (n={n_enc})")
    if not (0 <= end_index < n_enc):
        raise IndexError(f"end_index {end_index} out of range (n={n_enc})")
    if end_index <= start_index:
        raise ValueError("Need at least two encounters (end_index must be > start_index).")

    enc_slice = encounters[start_index : end_index + 1]
    body_ids = [int(enc["body_id"]) for enc in enc_slice]
    epochs_days = [float(enc["epoch_days"]) for enc in enc_slice]

    vinf_out: List[Optional[np.ndarray]] = []
    for enc in enc_slice[:-1]:  # legs originate at every encounter except the last
        vec = enc.get("vinf_out_vec_km_s")
        if vec is None:
            vinf_out.append(None)
        else:
            vinf_out.append(np.asarray(vec, dtype=float))

    return body_ids, epochs_days, vinf_out


def make_initial_guess_from_lambert_json(
    json_path: Union[str, Path],
    solution_rank: int = 1,
    start_index: int = 0,
    end_index: int = -1,
    catalog: Optional[Catalog] = None,
    sail: Optional[SailParams] = None,
    mu_star_km3_s2: float = MU_ALTAIRA,
    leg_ctrl: Optional[LegCtrlSpec] = None,
    opts: Optional[ProblemOptions] = None,
) -> Tuple[GTOC13TourUDP, np.ndarray]:
    """Build a UDP + initial guess from a Lambert beam-search JSON export.

    Parameters
    ----------
    json_path : path-like
        Path to the JSON file generated by ``bs_lambert.py``.
    solution_rank : int
        Rank (1-indexed) of the stored solution to use.
    start_index, end_index : int
        Encounter slice bounds within that solution. The slice is inclusive and
        must contain at least two encounters. Negative ``end_index`` follows
        Python indexing semantics (e.g., ``-1`` selects the final encounter).
    catalog, sail : optional
        Pre-built :class:`Catalog` and :class:`SailParams` instances. When omitted,
        defaults are constructed using Altaira constants.
    mu_star_km3_s2 : float
        Central body gravitational parameter (defaults to ``MU_ALTAIRA``).
    leg_ctrl, opts : optional
        Control discretization and problem options. Defaults are created when None.

    Returns
    -------
    udp : GTOC13TourUDP
        Problem instance configured with the extracted body sequence and epochs.
    x0 : np.ndarray
        Decision vector initialized from the JSON data (times + departure v∞ guesses).
    """
    body_ids, epochs_days, vinf_out = _extract_lambert_segment(
        json_path,
        solution_rank=solution_rank,
        start_index=start_index,
        end_index=end_index,
    )
    if catalog is None:
        catalog = Catalog(mu_star_km3_s2=mu_star_km3_s2)
    if sail is None:
        sail = SailParams(a0_1au_km_s2=DEFAULT_A0_1AU)
    if leg_ctrl is None:
        leg_ctrl = LegCtrlSpec()
    if opts is None:
        opts = ProblemOptions()

    epochs_s = [d * DAY for d in epochs_days]
    udp = GTOC13TourUDP(body_ids, epochs_s, catalog, mu_star_km3_s2, sail, leg_ctrl, opts=opts)
    x0 = make_initial_guess(udp, body_ids, epochs_s)

    for j, vinf_vec in enumerate(vinf_out):
        if vinf_vec is None:
            continue
        lo, hi = udp.idx_vinf[j]
        x0[lo:hi] = vinf_vec / V_REF

    return udp, x0


def plot_solution_2d(udp: GTOC13TourUDP, x: np.ndarray) -> None:
    """Plot a 2-D (x-y) view of the trajectory implied by decision vector x."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        print("matplotlib not available; skipping trajectory plot.")
        return

    x = np.atleast_1d(np.asarray(x, dtype=float))
    # Unpack times (unscaled, same logic as in fitness)
    t = x[udp.idx_t] * T_REF
    rb_list = []
    vb_list = []
    for tj, body_id in zip(t, udp.seq):
        rb, vb = udp.cat.state(body_id, float(tj))
        rb_list.append(rb)
        vb_list.append(vb)
    vinf_dep = []
    for lo, hi in udp.idx_vinf:
        vinf_dep.append(x[lo:hi] * V_REF)

    # Body positions at event times
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("GTOC13 Tour (heliocentric XY plane)")
    ax.set_xlabel("x [km]")
    ax.set_ylabel("y [km]")
    ax.grid(True, which="both", linestyle=":", linewidth=0.5)

    body_xy = np.array([[rb[0], rb[1]] for rb in rb_list])
    ax.scatter(body_xy[:, 0], body_xy[:, 1], c="tab:orange", marker="o", label="Body position")
    for (bx, by), body_id in zip(body_xy, udp.seq):
        ax.text(bx, by, f" {body_id}", fontsize=9, color="black", ha="left", va="bottom")

    all_xy = [body_xy]
    all_xy.append(np.array([[0.0, 0.0]]))
    ax.scatter([0.0], [0.0], color="#e0c200", s=80, edgecolors="k", label="Star")

    for j in range(udp.M):
        if udp.M > 1:
            t_color = j / (udp.M - 1)
        else:
            t_color = 0.0
        leg_color = (t_color, 1.0 - t_color, 0.0)  # green -> red
        ctrls = udp._get_ctrls(x, udp.idx_ctrl[j])
        y0 = np.hstack([rb_list[j], vb_list[j] + vinf_dep[j]])
        _, samples, _ = udp.prop.propagate_piecewise(
            float(t[j]),
            y0,
            float(t[j + 1]),
            ctrls,
            collect_samples=True,
            samples_per_segment=300,
        )
        if samples:
            leg_arr = np.asarray(samples)
            if leg_arr.size:
                all_xy.append(leg_arr[:, :2])
            ax.plot(leg_arr[:, 0], leg_arr[:, 1], linewidth=1.5, label=f"Leg {j+1}", color=leg_color)
            ax.scatter([leg_arr[-1, 0]], [leg_arr[-1, 1]], c="red", s=25, zorder=5)

    if all_xy:
        stacked = np.vstack(all_xy)
        _set_equal_xy_limits(ax, stacked, padding=0.2)

    ax.legend(loc="best")
    plt.show()


# ---------------------------
# Minimal usage sketch (not executed here)
# ---------------------------
if __name__ == "__main__":  # pragma: no cover
    if pg is None:
        raise SystemExit("pygmo is required to run this module")

    # Problem setup
    MU_STAR = MU_ALTAIRA  # Altaira gravitational parameter (km^3/s^2)
    A0_1AU = DEFAULT_A0_1AU  # Sail acceleration at 1 AU implied by constants (km/s^2)

    catalog = Catalog(mu_star_km3_s2=MU_STAR)
    sail = SailParams(a0_1au_km_s2=A0_1AU)

    leg_ctrl = LegCtrlSpec(nseg_leg=1)
    popts = ProblemOptions(progress_every_evals=0, objective="vinf_rss", 
                           optimize_t0=True, constrain_interstellar_direction=False, accel_smoothing=1e-10,
                           use_solar_sail=False)

    # Build UDP + initial guess directly from a Lambert beam-search JSON
    #json_path = Path("results/beam/bs_medium_bw5000_d30_top5_20251101-202939Z.json")
    json_path = Path("/Users/home/codes/gtoc13/results/beam/bs_mission-raw_bw100_d20_top100_20251107-234707Z.json")
    udp, x0 = make_initial_guess_from_lambert_json(
        json_path=json_path,
        solution_rank=1,
        start_index=0,
        end_index=1,
        catalog=catalog,
        sail=sail,
        mu_star_km3_s2=MU_STAR,
        leg_ctrl=leg_ctrl,
        opts=popts,
    )

    print(udp)
    #plot_solution_2d(udp, x0)

    # Set integrator options on the propagator
    int_opts = IntOptions()
    udp.prop.opts = int_opts

    prob = pg.problem(udp)
    print(prob)
    print("fitness at x0:", udp.fitness(x0))
 
    # Single-stage: IPOPT with adaptive settings
    pop = pg.population(prob, 0)
    pop.push_back(x0)
    if hasattr(pg, "ipopt"):
        ip = pg.ipopt()
        ip.set_string_option("mu_strategy", "adaptive")
        #ip.set_string_option("linear_solver", "mumps")
        ip.set_numeric_option("tol", 1e-6)
        ip.set_numeric_option("acceptable_tol", 1)
        #ip.set_numeric_option("barrier_tol_factor", 0.1)
        #ip.set_numeric_option("constr_viol_tol", 1e-5)
        #ip.set_numeric_option("dual_inf_tol", 1e-4)
        ip.set_numeric_option('acceptable_constr_viol_tol', 1e-6)
        #ip.set_numeric_option('acceptable_dual_inf_tol', 1e10)
        #ip.set_numeric_option('acceptable_compl_inf_tol', 1e10)
        #ip.set_string_option("mu_oracle", "loqo")
        ip.set_integer_option("max_iter", 1000)
        ip.set_integer_option("print_level", 4)
        ip.set_integer_option("acceptable_iter", 5)
        #ip.set_integer_option("mumps_mem_percent", 2000)
        #ip.set_string_option("nlp_scaling_method", "gradient-based")
        #ip.set_numeric_option("obj_scaling_factor", 1.0)  # let Ipopt pick

        algo = pg.algorithm(ip)
        algo.set_verbosity(1)
        print(algo)
        pop = algo.evolve(pop)
    else:
        raise SystemExit("IPOPT not available in this pagmo build")

    xf = pop.get_x()[0]
    print("Final f:", prob.fitness(xf)[0])
    plot_solution_2d(udp, xf)
