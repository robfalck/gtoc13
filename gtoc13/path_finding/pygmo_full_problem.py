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
from typing import Dict, Tuple, List, Optional

import math
import numpy as np
from scipy.integrate import solve_ivp
from pathlib import Path

try:
    import pygmo as pg
except Exception as e:  # pragma: no cover
    pg = None  # allow import for static analysis without pygmo


# ---------------------------
# Constants & small utilities
# ---------------------------
DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi
AU_KM = 149_597_870.7  # km

YEAR_S = 365.25 * 86400.0
# -------------
# Scaling constants for nondimensionalization
L_REF = AU_KM           # length scale: 1 AU in km
T_REF = YEAR_S          # time scale: 1 year in s
V_REF = L_REF / T_REF   # velocity scale: AU/year in km/s (~4.74)


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


# ---------------------------
# Keplerian propagation (2-body)
# ---------------------------
@dataclass
class KeplerianElements:
    a_km: float
    e: float
    i_rad: float
    Omega_rad: float
    omega_rad: float
    M0_rad: float  # mean anomaly at t=0


def kepler_mean_to_eccentric(M: float, e: float, tol: float = 1e-13, itmax: int = 50) -> float:
    """Solve Kepler's equation M = E - e*sin(E) for E.
    Uses Newton-Raphson with safe-guarding.
    """
    M = (M + math.pi) % (2 * math.pi) - math.pi  # wrap to [-pi, pi]
    if e < 1e-12:
        return M
    E = M if e < 0.8 else math.pi
    for _ in range(itmax):
        f = E - e * math.sin(E) - M
        fp = 1 - e * math.cos(E)
        d = -f / fp
        E += d
        if abs(d) < tol:
            break
    return E


def kepler_elements_to_rv(mu: float, kep: KeplerianElements, t: float) -> Tuple[np.ndarray, np.ndarray]:
    """State (r, v) at time t given elements at t=0, under 2-body gravity mu.
    Returns r [km], v [km/s].
    """
    a, e, inc, Om, om, M0 = (
        kep.a_km, kep.e, kep.i_rad, kep.Omega_rad, kep.omega_rad, kep.M0_rad
    )
    n = math.sqrt(mu / abs(a ** 3))  # mean motion (rad/s), a>0 assumed here
    M = M0 + n * t
    E = kepler_mean_to_eccentric(M, e)
    cosE, sinE = math.cos(E), math.sin(E)
    # Perifocal coordinates
    r_pf = np.array([a * (cosE - e), a * math.sqrt(1 - e * e) * sinE, 0.0])
    v_pf = np.array([
        -math.sqrt(mu * a) * sinE / (a * (1 - e * cosE)),
        math.sqrt(mu * a) * math.sqrt(1 - e * e) * cosE / (a * (1 - e * cosE)),
        0.0,
    ])
    # Rotation matrix PQW->IJK
    cO, sO = math.cos(Om), math.sin(Om)
    ci, si = math.cos(inc), math.sin(inc)
    co, so = math.cos(om), math.sin(om)
    R = np.array([
        [cO * co - sO * so * ci, -cO * so - sO * co * ci, sO * si],
        [sO * co + cO * so * ci, -sO * so + cO * co * ci, -cO * si],
        [so * si, co * si, ci],
    ])
    r = R.dot(r_pf)
    v = R.dot(v_pf)
    return r, v


# ---------------------------
# Body catalogs and ephemerides
# ---------------------------
@dataclass
class Planet:
    pid: int
    name: str
    gm_km3_s2: float
    radius_km: float
    kep: KeplerianElements


@dataclass
class SmallBody:
    sid: int
    kep: KeplerianElements


class Catalog:
    """Loads GTOC13 CSVs from /mnt/data and provides states/params.

    Assumes t=0 corresponds to the epoch at which the Kepler elements are given.
    """

    def __init__(self, mu_star_km3_s2: float):
        import pandas as pd  # lazy import

        self.mu_star = mu_star_km3_s2
        data_dir = Path(__file__).resolve().parent.parent / "data"
        # Load planets
        p = pd.read_csv(data_dir / "gtoc13_planets.csv", encoding="latin1")
        self.planets: Dict[int, Planet] = {}
        for _, row in p.iterrows():
            pid = int(row["#Planet ID"])  # 1..10 expected
            kep = KeplerianElements(
                a_km=float(row["Semi-Major Axis (km)"]),
                e=float(row["Eccentricity ()"]),
                i_rad=float(row["Inclination (deg)"]) * DEG2RAD,
                Omega_rad=float(row["Longitude of the Ascending Node (deg)"]) * DEG2RAD,
                omega_rad=float(row["Argument of Periapsis (deg)"]) * DEG2RAD,
                M0_rad=float(row["Mean Anomaly at t=0 (deg)"]) * DEG2RAD,
            )
            self.planets[pid] = Planet(
                pid=pid,
                name=str(row["Name"]).strip(),
                gm_km3_s2=float(row["GM (km3/s2)"]),
                radius_km=float(row["Radius (km)"]),
                kep=kep,
            )

        # Asteroids
        a = pd.read_csv(data_dir / "gtoc13_asteroids.csv", encoding="latin1")
        self.asteroids: Dict[int, SmallBody] = {}
        for _, row in a.iterrows():
            sid = int(row["#Asteroid ID"])  # 1001..
            kep = KeplerianElements(
                a_km=float(row["Semi-Major Axis (km)"]),
                e=float(row["Eccentricity ()"]),
                i_rad=float(row["Inclination (deg)"]) * DEG2RAD,
                Omega_rad=float(row["Longitude of the Ascending Node (deg)"]) * DEG2RAD,
                omega_rad=float(row["Argument of Periapsis (deg)"]) * DEG2RAD,
                M0_rad=float(row["Mean Anomaly at t=0"])
                * (DEG2RAD if "deg" in str(a.columns[-2]).lower() else 1.0),
            )
            self.asteroids[sid] = SmallBody(sid=sid, kep=kep)

        # Comets
        c = pd.read_csv(data_dir / "gtoc13_comets.csv", encoding="latin1")
        self.comets: Dict[int, SmallBody] = {}
        for _, row in c.iterrows():
            sid = int(row["# Comet ID"])  # 2001..
            kep = KeplerianElements(
                a_km=float(row["Semi-Major Axis (km)"]),
                e=float(row["Eccentricity ()"]),
                i_rad=float(row["Inclination (deg)"]) * DEG2RAD,
                Omega_rad=float(row["Longitude of the Ascending Node (deg)"]) * DEG2RAD,
                omega_rad=float(row["Argument of Periapsis (deg)"]) * DEG2RAD,
                M0_rad=float(row["Mean Anomaly at t=0 (deg)"]) * DEG2RAD,
            )
            self.comets[sid] = SmallBody(sid=sid, kep=kep)

    def is_planet(self, body_id: int) -> bool:
        return body_id in self.planets

    def body_radius(self, body_id: int) -> float:
        if self.is_planet(body_id):
            return self.planets[body_id].radius_km
        return 0.0

    def body_mu(self, body_id: int) -> float:
        if self.is_planet(body_id):
            return self.planets[body_id].gm_km3_s2
        return 0.0

    def state(self, body_id: int, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """Heliocentric (r,v) at time t for body id.
        t in seconds since contest epoch. r in km, v in km/s.
        """
        if self.is_planet(body_id):
            kep = self.planets[body_id].kep
        elif body_id in self.asteroids:
            kep = self.asteroids[body_id].kep
        elif body_id in self.comets:
            kep = self.comets[body_id].kep
        else:
            raise KeyError(f"Unknown body id {body_id}")
        return kepler_elements_to_rv(self.mu_star, kep, t)

    def orbital_period(self, body_id: int) -> float:
        """Keplerian period [s] of body around the star."""
        if self.is_planet(body_id):
            a = self.planets[body_id].kep.a_km
        elif body_id in self.asteroids:
            a = self.asteroids[body_id].kep.a_km
        else:
            a = self.comets[body_id].kep.a_km
        return 2 * math.pi * math.sqrt(a ** 3 / self.mu_star)


# ---------------------------
# Sail dynamics & integrator
# ---------------------------
@dataclass
class SailParams:
    a0_1au_km_s2: float  # max accel at 1 AU (km/s^2)


def sail_accel(r: np.ndarray, v: np.ndarray, alpha: float, sigma: float, a0_1au: float) -> np.ndarray:
    """Ideal sail acceleration in km/s^2.
    alpha in [0, pi/2], sigma in [0, 2*pi).
    """
    rnorm = np.linalg.norm(r)
    if rnorm == 0:
        return np.zeros(3)
    rhat = r / rnorm
    h = np.cross(r, v)
    hnorm = np.linalg.norm(h)
    if hnorm < 1e-12:
        # Degenerate: choose arbitrary transverse frame
        # Here, pick t-hat orthogonal to r
        arbitrary = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(arbitrary, rhat)) > 0.9:
            arbitrary = np.array([0.0, 1.0, 0.0])
        that = np.cross(np.cross(rhat, arbitrary), rhat)
        that /= np.linalg.norm(that)
        nhat = np.cross(rhat, that)
    else:
        hhat = h / hnorm
        that = np.cross(hhat, rhat)
        that /= np.linalg.norm(that)
        nhat = hhat
    # Sail normal direction
    n_hat = math.cos(alpha) * rhat + math.sin(alpha) * (math.cos(sigma) * that + math.sin(sigma) * nhat)
    # Magnitude scales as cos^2(alpha) / r^2
    scale = a0_1au * (AU_KM / rnorm) ** 2 * (math.cos(alpha) ** 2)
    return scale * n_hat


@dataclass
class IntOptions:
    method: str = "DOP853"
    rtol: float = 1e-4
    atol: float = 1e-6
    h_init: Optional[float] = None
    h_min: float = 1.0
    h_max: Optional[float] = None




class SailPropagator:
    def __init__(self, mu_star: float, sail: SailParams, opts: IntOptions = IntOptions()):
        self.mu = mu_star
        self.sail = sail
        self.opts = opts

    def deriv(self, t: float, y: np.ndarray, ctl_alpha: float, ctl_sigma: float) -> np.ndarray:
        r = y[:3]
        v = y[3:]
        rnorm = np.linalg.norm(r)
        acc_grav = -self.mu * r / (rnorm ** 3 + 1e-30)
        acc_sail = sail_accel(r, v, ctl_alpha, ctl_sigma, self.sail.a0_1au_km_s2)
        dydt = np.zeros_like(y)
        dydt[:3] = v
        dydt[3:] = acc_grav + acc_sail
        return dydt


    def propagate_piecewise(
        self,
        t0: float,
        y0: np.ndarray,
        t1: float,
        controls: List[Tuple[float, float]],
        collect_samples: bool = False,
        samples_per_segment: int = 200,
    ) -> Tuple[np.ndarray, List[np.ndarray], float]:
        """Propagate from t0 to t1 with piecewise-constant controls.
        controls: list of (alpha, sigma) for equal-duration segments across [t0,t1].
        Returns (y_end, sampled_positions, r_min_along_arc)
        """
        if t1 == t0:
            base_samples: List[np.ndarray] = []
            if collect_samples:
                base_samples.append(y0[:3].copy())
            return y0.copy(), base_samples, np.linalg.norm(y0[:3])

        forward = t1 > t0
        T = abs(t1 - t0)
        nseg = max(1, len(controls))
        dt_seg = T / nseg
        y = y0.copy()
        t = t0
        samples_r: List[np.ndarray] = []
        if collect_samples:
            samples_r.append(y[:3].copy())
        rmin = np.linalg.norm(y[:3])

        for s in range(nseg):
            alpha, sigma = controls[s]
            seg_end = t + (dt_seg if forward else -dt_seg)
            def rhs(tt, yy):
                return self.deriv(tt, yy, alpha, sigma)
            # Let solve_ivp choose adaptive step sizes by default; tighten if collecting samples
            rtol = self.opts.rtol
            atol = self.opts.atol
            if collect_samples:
                rtol = min(rtol, 1e-8)
                atol = min(atol, 1e-10)
            kwargs = dict(method=self.opts.method, rtol=rtol, atol=atol, dense_output=collect_samples)

            step_hint = None
            if self.opts.h_max is not None:
                step_hint = self.opts.h_max
            if collect_samples:
                seg_duration = abs(seg_end - t)
                desired = seg_duration / max(4, samples_per_segment)
                step_hint = desired if step_hint is None else min(step_hint, desired)
            if step_hint is not None:
                kwargs["max_step"] = step_hint

            if self.opts.h_init is not None:
                first_step = self.opts.h_init
                if step_hint is not None:
                    first_step = min(first_step, step_hint)
                kwargs["first_step"] = first_step
            sol = solve_ivp(
                rhs,
                t_span=(t, seg_end),
                y0=y,
                **kwargs,
            )
            # After integration, update t and y to last point
            t = sol.t[-1]
            y = sol.y[:, -1]
            if collect_samples:
                if sol.sol is not None:
                    eval_t = np.linspace(sol.t[0], sol.t[-1], max(2, samples_per_segment + 1))
                    seg_states = sol.sol(eval_t)
                    seg_pts = seg_states[:3, 1:].T  # drop first point (already included)
                else:
                    seg_pts = sol.y[:3, 1:].T
                if seg_pts.size:
                    samples_r.extend(seg_pts)
            # Update minimum radius
            rmin = min(rmin, np.min(np.linalg.norm(sol.y[:3, :], axis=0)))
        return y, (samples_r if collect_samples else []), rmin


# ---------------------------
# Flyby utilities (patched conics)
# ---------------------------

def vinf_vectors(body_v: np.ndarray, v_pre: np.ndarray, v_post: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Return v_inf- and v_inf+, and their magnitudes."""
    vin_minus = v_pre - body_v
    vin_plus = v_post - body_v
    return vin_minus, vin_plus, float(np.linalg.norm(vin_minus)), float(np.linalg.norm(vin_plus))


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
    nseg_half: int = 5  # segments per half leg


@dataclass
class ProblemOptions:
    same_body_gap_factor: float = 1.0 / 3.0
    enforce_initial_arrival: bool = False  # TODO hook (see notes)
    progress_every_evals: int = 0  # print progress every N fitness evals; 0 disables printing


class GTOC13TourUDP:
    """pygmo UDP implementing multiple-shooting for a flyby tour with a sail.

    Decision vector x packs:
    - Event times: t[0..M] (s)
    - For each event j:
        r_j (3) position at event in heliocentric frame (km)
        v_pre_j (3) incoming heliocentric velocity (km/s)  [except j=0]
        v_post_j (3) outgoing heliocentric velocity (km/s) [except j=M]
    - For each leg j (from event j to j+1):
        forward half controls: (alpha,sigma)*nseg_half
        backward half controls: (alpha,sigma)*nseg_half
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
        self.prop = SailPropagator(mu_star_km3_s2, sail)
        self.sail = sail
        self.leg_ctrl = leg_ctrl
        self.popts = opts

        # Progress printing
        self.progress_every = opts.progress_every_evals
        self._eval_ctr = 0

        # Pre-compute which events are planets
        self.is_planet = [self.cat.is_planet(b) for b in self.seq]

        # Variable indexing map
        self._build_indexing()

    # -----------------------
    # Variable packing helpers
    # -----------------------
    def _build_indexing(self):
        idx = 0
        self.idx_t = list(range(idx, idx + len(self.seq)))
        idx += len(self.seq)

        # Event state blocks
        self.idx_r = []
        self.idx_vpre = []
        self.idx_vpost = []
        for j in range(len(self.seq)):
            self.idx_r.append((idx, idx + 3))
            idx += 3
            if j > 0:
                self.idx_vpre.append((idx, idx + 3))
                idx += 3
            else:
                self.idx_vpre.append(None)
            if j < len(self.seq) - 1:
                self.idx_vpost.append((idx, idx + 3))
                idx += 3
            else:
                self.idx_vpost.append(None)

        # Controls per leg
        self.idx_ctrl_fwd = []
        self.idx_ctrl_bwd = []
        npar_half = 2 * self.leg_ctrl.nseg_half
        for j in range(self.M):
            self.idx_ctrl_fwd.append((idx, idx + npar_half))
            idx += npar_half
            self.idx_ctrl_bwd.append((idx, idx + npar_half))
            idx += npar_half

        self.nx = idx

    def get_bounds(self) -> Tuple[List[float], List[float]]:  # pygmo API
        lb = [-1e12] * self.nx
        ub = [1e12] * self.nx
        # Times bounded to mission window [0, 200 years], scaled
        for k in self.idx_t:
            lb[k] = 0.0
            ub[k] = 200.0
        # No further bounds; angles are unbounded but wrapped in unpacking
        return lb, ub

    # -----------------------
    # Constraint counts
    # -----------------------
    def get_nec(self) -> int:  # pygmo API
        nec = 0
        # Event position equalities: r_j - r_body(t_j) = 0  (3 each)
        nec += 3 * len(self.seq)
        # Mid-point matches: 6 per leg
        nec += 6 * self.M
        # Massless events: velocity continuity 3 eq at each non-planet
        for j, is_pl in enumerate(self.is_planet):
            if not is_pl:
                # both sides exist except at endpoints; handle endpoints too (then it's 3 eq only if both sides exist)
                if self.idx_vpre[j] is not None and self.idx_vpost[j] is not None:
                    nec += 3
        # Planetary flyby equalities: |v_inf+| - |v_inf-| = 0 (magnitude only)
        for j, is_pl in enumerate(self.is_planet):
            if is_pl:
                if self.idx_vpre[j] is not None and self.idx_vpost[j] is not None:
                    nec += 1  # v_inf magnitude equality only
        # (Optional) initial arrival constraints — TODO: typically 3 eq
        # if self.popts.enforce_initial_arrival: nec += 3
        return nec

    def get_nic(self) -> int:  # pygmo API
        nic = 0
        # Time ordering: t[j+1]-t[j] >= 0
        nic += self.M
        # Same-body gap: for successive equal bodies: dt - T/3 >= 0
        for j in range(self.M):
            if self.seq[j] == self.seq[j + 1]:
                nic += 1
        # Planet flyby altitude window: rp - (R+0.1R) >=0 and (100R) - rp >=0  => two inequalities per planet event
        for j, is_pl in enumerate(self.is_planet):
            if is_pl and self.idx_vpre[j] is not None and self.idx_vpost[j] is not None:
                nic += 2
        return nic

    # -----------------------
    # Unpacking helpers
    # -----------------------
    def _wrap_angles(self, a: float, s: float) -> Tuple[float, float]:
        # Ensure alpha in [0, pi/2], sigma in [0, 2*pi)
        alpha = abs(a) % (math.pi)
        if alpha > 0.5 * math.pi:
            alpha = math.pi - alpha
        sigma = s % (2 * math.pi)
        return alpha, sigma

    def _get_ctrls(self, x: np.ndarray, idx_pair: Tuple[int, int]) -> List[Tuple[float, float]]:
        lo, hi = idx_pair
        raw = x[lo:hi]
        ctrls = []
        for k in range(0, len(raw), 2):
            alpha, sigma = self._wrap_angles(raw[k], raw[k + 1])
            ctrls.append((alpha, sigma))
        return ctrls

    # -----------------------
    # Fitness
    # -----------------------
    def fitness(self, x: List[float]) -> List[float]:  # pygmo API
        x = np.asarray(x, dtype=float)
        # Unpack times (scaled)
        t_scaled = x[self.idx_t]
        t = t_scaled * T_REF  # unscale to seconds
        # Unpack event states (positions and velocities are scaled)
        r = []
        vpre = []
        vpost = []
        for j in range(len(self.seq)):
            i0, i1 = self.idx_r[j]
            r_scaled = x[i0:i1]
            r_phys = r_scaled * L_REF  # unscale to km
            r.append(r_phys)
            if self.idx_vpre[j] is not None:
                i0, i1 = self.idx_vpre[j]
                vpre_scaled = x[i0:i1]
                vpre_phys = vpre_scaled * V_REF  # unscale to km/s
                vpre.append(vpre_phys)
            else:
                vpre.append(None)
            if self.idx_vpost[j] is not None:
                i0, i1 = self.idx_vpost[j]
                vpost_scaled = x[i0:i1]
                vpost_phys = vpost_scaled * V_REF  # unscale to km/s
                vpost.append(vpost_phys)
            else:
                vpost.append(None)

        # Cache body states at event times
        rb_list = []
        vb_list = []
        for j, body_id in enumerate(self.seq):
            rb_phys, vb_phys = self.cat.state(body_id, float(t[j]))
            rb_list.append(rb_phys)
            vb_list.append(vb_phys)

        # Collect constraints (all in scaled units)
        ceq: List[float] = []
        cineq: List[float] = []

        # 1) Event position equalities r_j == r_body(t_j)
        for j, body_id in enumerate(self.seq):
            # residual in scaled units
            ceq.extend(((r[j] - rb_list[j]) / L_REF).tolist())

        # 2) Time ordering: t[j+1]-t[j] >= 0, scaled
        for j in range(self.M):
            cineq.append(float((t[j + 1] - t[j]) / T_REF))

        # 3) Per half-leg propagation + mid-point match
        for j in range(self.M):
            tm = 0.5 * (t[j] + t[j + 1])
            # Controls
            ctrls_fwd = self._get_ctrls(x, self.idx_ctrl_fwd[j])
            ctrls_bwd = self._get_ctrls(x, self.idx_ctrl_bwd[j])
            # Forward from (r_j, v_post_j) to tm
            y0_f = np.hstack([r[j], vpost[j]])
            yf, _, _ = self.prop.propagate_piecewise(float(t[j]), y0_f, float(tm), ctrls_fwd)
            # Backward from (r_{j+1}, v_pre_{j+1}) to tm
            y0_b = np.hstack([r[j + 1], vpre[j + 1]])
            yb, _, _ = self.prop.propagate_piecewise(float(t[j + 1]), y0_b, float(tm), ctrls_bwd)
            # Midpoint equality (6): yf == yb, residuals in scaled units
            ceq.extend(((yf[:3] - yb[:3]) / L_REF).tolist())
            ceq.extend(((yf[3:] - yb[3:]) / V_REF).tolist())

        # 4) Same-body spacing dt - T/3 >= 0, scale time
        for j in range(self.M):
            if self.seq[j] == self.seq[j + 1]:
                T = self.cat.orbital_period(self.seq[j])
                cineq.append(float((t[j + 1] - t[j] - self.popts.same_body_gap_factor * T) / T_REF))

        # 5) Flyby constraints
        for j, body_id in enumerate(self.seq):
            if self.is_planet[j] and (vpre[j] is not None) and (vpost[j] is not None):
                # Planetary flyby: v_inf magnitude equality only
                vb_phys = vb_list[j]
                vinm, vinp, vm, vp = vinf_vectors(vb_phys, vpre[j], vpost[j])
                ceq.append((vp - vm) / V_REF)  # v_inf magnitude equality (scaled)
                # Altitude window via delta implied by angle between v_inf vectors
                delta = turn_angle(vinm, vinp)
                rp = rp_from_turn(vm, self.cat.body_mu(body_id), delta)
                R = self.cat.body_radius(body_id)
                # periapsis window, scaled
                cineq.append(rp / L_REF - (R + 0.1 * R) / L_REF)           # rp >= 1.1 R
                cineq.append((R + 100.0 * R) / L_REF - rp / L_REF)         # rp <= 101 R
            elif (not self.is_planet[j]) and (vpre[j] is not None) and (vpost[j] is not None):
                # Massless body: velocity continuity (3 eq), scaled
                ceq.extend(((vpost[j] - vpre[j]) / V_REF).tolist())

        # (Optional) Initial arrival constraints — TODO: not enforced by default
        # if self.popts.enforce_initial_arrival:
        #     ceq.extend(self.enforce_initial_arrival_constraints(r[0], vpre[0], vpost[0], t[0]))

        # Objective: feasibility (sum squares of scaled equalities) + mild control regularization
        eq = np.array(ceq)
        ineq_penalty = np.sum(np.minimum(0.0, np.array(cineq)) ** 2)
        ctrl_reg = 0.0
        for j in range(self.M):
            for idx_pair in (self.idx_ctrl_fwd[j], self.idx_ctrl_bwd[j]):
                lo, hi = idx_pair
                raw = x[lo:hi]
                # small penalty to keep angles moderate; encourages alpha near 0..pi/2 but smooth
                ctrl_reg += 1e-6 * float(np.dot(raw, raw))
        f0 = 0 #float(np.dot(eq, eq) + 100.0 * ineq_penalty + ctrl_reg)

        # Progress reporting
        self._eval_ctr += 1
        if self.progress_every > 0 and (self._eval_ctr % self.progress_every == 0):
            # Print concise progress: eval count, f0, ||eq||, ineq penalty
            eq_norm = float(np.linalg.norm(eq))
            print(
                f"[fit] eval {self._eval_ctr:6d}  f0={f0:.4e}  ||eq||={eq_norm:.2e}  ineq_pen={ineq_penalty:.2e}",
                flush=True,
            )

        return [f0] + ceq + cineq

    # -----------------------
    # Problem meta
    # -----------------------
    def get_nobj(self) -> int:  # pygmo API
        return 1

    def gradient(self, x: List[float]) -> List[float]:
        # Use pygmo's built-in finite-difference Jacobian estimator on the full fitness vector.
        # Default to estimate_gradient (central differences). Switch to estimate_gradient_h by toggling below if desired.
        if pg is None:
            raise RuntimeError("pygmo is required for gradient estimation")
        # Convert to numpy array for safety; pygmo accepts sequence-like
        x_np = np.asarray(x, dtype=float)
        # Central-difference estimator
        grad = pg.estimate_gradient(lambda dv: self.fitness(dv), x_np)
        return grad.tolist()

    def has_gradient(self) -> bool:
        return True

    def get_name(self) -> str:  # pygmo API
        return "GTOC13 multiple-shooting tour (sail)"

    def get_extra_info(self) -> str:  # pygmo API
        return (
            f"legs={self.M}, nseg_half={self.leg_ctrl.nseg_half}"
        )


# ---------------------------
# Convenience builder / example
# ---------------------------

def make_initial_guess(udp: GTOC13TourUDP, body_sequence: List[int], t_guess_s: List[float]) -> np.ndarray:
    """Construct a coarse initial guess vector x consistent with catalogs.
    - Sets event positions to body positions at t_j (scaled).
    - Sets incoming/outgoing velocities to body velocities (zero v_inf) as a neutral guess (scaled).
    - Sets all controls to alpha=pi/2 (ballistic), sigma=0.
    - Times written as scaled (t_j / T_REF).
    """
    x = np.zeros(udp.nx)
    # Times (write scaled)
    for k, tk in zip(udp.idx_t, t_guess_s):
        x[k] = tk / T_REF  # scale to nondimensional
    # States at events (positions and velocities scaled)
    for j, body in enumerate(body_sequence):
        i0, i1 = udp.idx_r[j]
        rb, vb = udp.cat.state(body, t_guess_s[j])
        x[i0:i1] = rb / L_REF  # scale positions
        if udp.idx_vpre[j] is not None:
            p0, p1 = udp.idx_vpre[j]
            x[p0:p1] = vb / V_REF  # scale velocities
        if udp.idx_vpost[j] is not None:
            q0, q1 = udp.idx_vpost[j]
            x[q0:q1] = vb / V_REF  # scale velocities
    # Controls
    for j in range(udp.M):
        for (lo, hi) in (udp.idx_ctrl_fwd[j], udp.idx_ctrl_bwd[j]):
            raw = x[lo:hi]
            for k in range(0, len(raw), 2):
                raw[k] = 0.5 * math.pi  # alpha = 90 deg (ballistic)
            raw[k + 1] = 0.0        # sigma = 0
        x[lo:hi] = raw
    return x


def plot_solution_2d(udp: GTOC13TourUDP, x: np.ndarray) -> None:
    """Plot a 2-D (x-y) view of the trajectory implied by decision vector x."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        print("matplotlib not available; skipping trajectory plot.")
        return

    x = np.asarray(x, dtype=float)
    # Unpack times and states (unscaled, same logic as in fitness)
    t = x[udp.idx_t] * T_REF
    r_list: List[np.ndarray] = []
    vpre_list: List[Optional[np.ndarray]] = []
    vpost_list: List[Optional[np.ndarray]] = []
    for j in range(len(udp.seq)):
        i0, i1 = udp.idx_r[j]
        r_list.append(x[i0:i1] * L_REF)
        if udp.idx_vpre[j] is not None:
            p0, p1 = udp.idx_vpre[j]
            vpre_list.append(x[p0:p1] * V_REF)
        else:
            vpre_list.append(None)
        if udp.idx_vpost[j] is not None:
            q0, q1 = udp.idx_vpost[j]
            vpost_list.append(x[q0:q1] * V_REF)
        else:
            vpost_list.append(None)

    # Body positions at event times
    body_pos = []
    for tj, body_id in zip(t, udp.seq):
        rb, _ = udp.cat.state(body_id, float(tj))
        body_pos.append(rb)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("GTOC13 Tour (heliocentric XY plane)")
    ax.set_xlabel("x [km]")
    ax.set_ylabel("y [km]")
    ax.grid(True, which="both", linestyle=":", linewidth=0.5)

    body_xy = np.array([[rb[0], rb[1]] for rb in body_pos])
    ax.scatter(body_xy[:, 0], body_xy[:, 1], c="tab:orange", marker="o", label="Body position")

    event_xy = np.array([[r[0], r[1]] for r in r_list])
    ax.scatter(event_xy[:, 0], event_xy[:, 1], c="tab:blue", marker="*", s=120, label="Event state")

    match_points: List[np.ndarray] = []

    for j in range(udp.M):
        ctrls_fwd = udp._get_ctrls(x, udp.idx_ctrl_fwd[j])
        ctrls_bwd = udp._get_ctrls(x, udp.idx_ctrl_bwd[j])
        tm = 0.5 * (t[j] + t[j + 1])

        y0_f = np.hstack([r_list[j], vpost_list[j]])
        yf, samples_f, _ = udp.prop.propagate_piecewise(
            float(t[j]),
            y0_f,
            float(tm),
            ctrls_fwd,
            collect_samples=True,
            samples_per_segment=300,
        )
        f_pts = np.array(samples_f) if samples_f else np.empty((0, 3))

        y0_b = np.hstack([r_list[j + 1], vpre_list[j + 1]])
        yb, samples_b, _ = udp.prop.propagate_piecewise(
            float(t[j + 1]),
            y0_b,
            float(tm),
            ctrls_bwd,
            collect_samples=True,
            samples_per_segment=300,
        )
        b_pts = np.array(samples_b) if samples_b else np.empty((0, 3))

        match_r = 0.5 * (yf[:3] + yb[:3])
        match_points.append(match_r)

        leg_xy: List[np.ndarray] = []
        if f_pts.size:
            leg_xy.extend(f_pts[:, :2])
        if b_pts.size:
            # Reverse to go from match point outwards and drop duplicate midpoint if needed
            b_xy = b_pts[::-1, :2]
            if leg_xy and b_xy.size and np.allclose(leg_xy[-1], b_xy[0]):
                leg_xy.extend(b_xy[1:])
            else:
                leg_xy.extend(b_xy)

        if leg_xy:
            leg_arr = np.asarray(leg_xy)
            ax.plot(leg_arr[:, 0], leg_arr[:, 1], linewidth=1.5, label=f"Leg {j+1}")

    if match_points:
        mp_xy = np.array([[mp[0], mp[1]] for mp in match_points])
        ax.scatter(mp_xy[:, 0], mp_xy[:, 1], c="tab:green", marker="x", s=80, label="Match point")

    ax.legend(loc="best")
    plt.show()


# ---------------------------
# Minimal usage sketch (not executed here)
# ---------------------------
if __name__ == "__main__":  # pragma: no cover
    if pg is None:
        raise SystemExit("pygmo is required to run this module")

    # Problem setup
    MU_STAR = 1.32712440018e11  # example: Sun mu in km^3/s^2 (replace with Altaira mu!)
    A0_1AU = 0.00025 / 1000.0  # example: 0.25 mm/s^2 at 1 AU -> km/s^2

    catalog = Catalog(mu_star_km3_s2=MU_STAR)
    sail = SailParams(a0_1au_km_s2=A0_1AU)

    # Example tour: [planet 1, asteroid 1001, planet 2]
    seq = [10, 9]
    t0 = 0.0
    dt1 = 30*365 * 86400.0
    # dt2 = 180.0 * 86400.0
    # t_guess = [t0, t0 + dt1, t0 + dt1 + dt2]
    t_guess = [t0, t0 + dt1]

    # Set up options to print progress every 0 fitness evaluations
    popts = ProblemOptions(progress_every_evals=0)
    leg_ctrl = LegCtrlSpec(nseg_half=2)  # reduce DOF for bootstrapping
    udp = GTOC13TourUDP(seq, t_guess, catalog, MU_STAR, sail, leg_ctrl, opts=popts)
    print(udp)
    x0 = make_initial_guess(udp, seq, t_guess)

    # Set integrator options on the propagator
    int_opts = IntOptions()
    udp.prop.opts = int_opts

    prob = pg.problem(udp)
    print(prob)

    # Single-stage: IPOPT with adaptive settings
    pop = pg.population(prob, 0)
    pop.push_back(x0)
    if hasattr(pg, "ipopt"):
        ip = pg.ipopt()
        ip.set_string_option("mu_strategy", "adaptive")
        ip.set_string_option("linear_solver", "mumps")
        ip.set_numeric_option("tol", 1e-6)
        ip.set_numeric_option("acceptable_tol", 1e-4)
        ip.set_numeric_option("barrier_tol_factor", 0.1)
        ip.set_numeric_option("constr_viol_tol", 1e-5)
        ip.set_numeric_option("dual_inf_tol", 1e-4)
        ip.set_string_option("mu_oracle", "loqo")
        ip.set_integer_option("max_iter", 6)
        ip.set_integer_option("print_level", 3)
        ip.set_integer_option("mumps_mem_percent", 2000)
        algo = pg.algorithm(ip)
        algo.set_verbosity(1)
        print(algo)
        pop = algo.evolve(pop)
    else:
        raise SystemExit("IPOPT not available in this pagmo build")

    xf = pop.get_x()[0]
    print("Final f:", prob.fitness(xf)[0])
    plot_solution_2d(udp, xf)
