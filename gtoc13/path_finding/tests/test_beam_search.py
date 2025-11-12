import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose
from dataclasses import replace

from gtoc13.bodies import bodies_data, INTERSTELLAR_BODY_ID
from gtoc13.constants import DAY, KMPAU, YEAR
from gtoc13.path_finding import BeamSearch
from gtoc13.path_finding.beam.config import (
    BodyRegistry,
    BASE_BODY_WEIGHTS,
    BASE_SEMI_MAJOR_AXES,
    make_lambert_config,
)
from gtoc13.path_finding.beam.dv_limits import max_transfer_dv_solar_sail
from gtoc13.path_finding.beam.lambert import (
    Encounter,
    ephemeris_position,
    LambertLegMeta,
    _requires_low_perihelion,
    InfeasibleLeg,
    resolve_lambert_leg,
)
from gtoc13.path_finding.beam.scoring import score_leg_mission, score_leg_mission_raw
from gtoc13.path_finding.beam.pipeline import make_expand_fn, make_score_fn, key_fn


class TestBeamSearch(unittest.TestCase):

    def test_basic(self):
        # Ultra-brief smoke test using integers
        def expand_fn(s: int):              # cheap proposals
            return (s + d for d in (1, 2, 3))

        def score_fn(p: int, c: int):       # heavy scoring (mock)
            target = 12
            return -abs(target - c)         # higher is better

        bs = BeamSearch(
            expand_fn,
            score_fn,
            beam_width=4,
            max_depth=5,
            key_fn=lambda s: s,
            parallel_backend="thread",
        )
        final_beam = bs.run(0)
        best = max(final_beam, key=lambda n: n.cum_score)

        assert_allclose(best.cum_score, -9.0, rtol=1.0E-3, atol=1.0E-3)  # use allclose for floating point testing
        self.assertEqual(best.state, 3)
        self.assertEqual(bs.reconstruct_path(best), [0, 3])

    def test_global_top_tracks_shallow_best(self):
        def expand_fn(s: int):
            if s >= 3:
                return ()
            return (s + 1,)

        def score_fn(parent: int, child: int):
            return -1.0, child

        bs = BeamSearch(
            expand_fn,
            score_fn,
            beam_width=1,
            max_depth=4,
            key_fn=None,
            parallel_backend=None,
            return_top_k=2,
        )
        final_nodes = bs.run(0)
        self.assertEqual(len(final_nodes), 2)
        scores = [n.cum_score for n in final_nodes]
        # Shallow node (depth 1) retains higher score than deeper negative-scoring nodes.
        self.assertGreater(scores[0], scores[1])
        self.assertEqual(final_nodes[0].depth, 1)


class TestDynamicDvHelper(unittest.TestCase):

    def test_zero_factor_returns_zero(self):
        r = (1.0e5, 0.0, 0.0)
        cap = max_transfer_dv_solar_sail(r, r, tof_days=1.0, factor=0.0)
        self.assertEqual(cap, 0.0)

    def test_expected_nominal_value(self):
        # Simple geometry: rotate arrival position 60 degrees in-plane.
        au_km = 149_597_870.7
        theta = math.radians(60.0)
        r_depart = (au_km, 0.0, 0.0)
        r_arrive = (au_km * math.cos(theta), au_km * math.sin(theta), 0.0)
        factor = 0.25
        tof_days = 100.0

        cap = max_transfer_dv_solar_sail(r_depart, r_arrive, tof_days=tof_days, factor=factor)

        # Manual evaluation of the published bound for comparison.
        r1_au = np.linalg.norm(r_depart) / au_km
        r2_au = np.linalg.norm(r_arrive) / au_km
        avg_au = 0.5 * (r1_au + r2_au)
        a1au = 3.24156e-4
        seconds = tof_days * 86_400.0
        expected = (a1au / (avg_au**2)) * seconds * factor * 1e-3

        assert_allclose(cap, expected, rtol=1e-12, atol=0.0)


class TestPerihelionPruning(unittest.TestCase):
    @staticmethod
    def _elliptic_state(a_au: float, e: float, nu: float):
        from gtoc13.constants import MU_ALTAIRA, KMPAU

        a = a_au * KMPAU
        mu = MU_ALTAIRA
        r_mag = a * (1.0 - e**2) / (1.0 + e * math.cos(nu))
        r = np.array(
            [
                r_mag * math.cos(nu),
                r_mag * math.sin(nu),
                0.0,
            ],
            dtype=float,
        )
        h = math.sqrt(mu * a * (1.0 - e**2))
        v = np.array(
            [
                -mu / h * math.sin(nu),
                mu / h * (e + math.cos(nu)),
                0.0,
            ],
            dtype=float,
        )
        return r, v

    def test_multi_rev_prunes_only_when_peri_below_floor(self):
        from gtoc13.constants import KMPAU

        threshold = 0.05 * KMPAU

        # Orbit with perihelion below threshold: a = 1 AU, e = 0.96 -> rp = 0.04 AU
        r_low, v_low = self._elliptic_state(1.0, 0.96, math.pi)
        r_arrive = r_low.copy()
        self.assertTrue(
            _requires_low_perihelion(r_low, v_low, r_arrive, revolutions=1, rp_min_km=threshold)
        )

        # Orbit with perihelion safely above threshold: a = 1 AU, e = 0.1 -> rp = 0.9 AU
        r_high, v_high = self._elliptic_state(1.0, 0.1, math.pi)
        self.assertFalse(
            _requires_low_perihelion(r_high, v_high, r_arrive, revolutions=1, rp_min_km=threshold)
        )

    def test_zero_rev_only_prunes_when_segment_wraps_perihelion(self):
        from gtoc13.constants import KMPAU

        threshold = 0.05 * KMPAU
        a_au = 0.5
        e = 0.91

        # Same low-peri orbit; choose anomalies that do not wrap around perihelion.
        nu_start = math.radians(60.0)
        nu_end = math.radians(65.0)
        r1, v1 = self._elliptic_state(a_au, e, nu_start)
        r2, _ = self._elliptic_state(a_au, e, nu_end)
        self.assertFalse(
            _requires_low_perihelion(r1, v1, r2, revolutions=0, rp_min_km=threshold)
        )

        # Now pick anomalies that wrap past perihelion (e.g., 200 deg -> 40 deg).
        nu_start_wrap = math.radians(200.0)
        nu_end_wrap = math.radians(40.0)
        r1_wrap, v1_wrap = self._elliptic_state(a_au, e, nu_start_wrap)
        r2_wrap, _ = self._elliptic_state(a_au, e, nu_end_wrap)
        self.assertTrue(
            _requires_low_perihelion(r1_wrap, v1_wrap, r2_wrap, revolutions=0, rp_min_km=threshold)
        )


class TestKeyFn(unittest.TestCase):
    def test_tail_disambiguates_visit_order(self):
        base_encounter_args = dict(
            r=(0.0, 0.0, 0.0),
            vinf_in=None,
            vinf_in_vec=None,
            vinf_out=None,
            vinf_out_vec=None,
            flyby_valid=None,
            flyby_altitude=None,
            dv_periapsis=None,
            dv_periapsis_vec=None,
            dv_limit=None,
            J_total=0.0,
        )

        path_a = (
            Encounter(body=1, t=0.0, **base_encounter_args),
            Encounter(body=2, t=10.0, **base_encounter_args),
            Encounter(body=3, t=20.0, **base_encounter_args),
        )
        path_b = (
            Encounter(body=2, t=0.0, **base_encounter_args),
            Encounter(body=1, t=10.0, **base_encounter_args),
            Encounter(body=3, t=20.0, **base_encounter_args),
        )

        key_a = key_fn(path_a)
        key_b = key_fn(path_b)

        self.assertNotEqual(key_a, key_b)
        # Sanity-check that last body and visited set still match.
        self.assertEqual(key_a[0], key_b[0])
        self.assertEqual(key_a[1], key_b[1])

    def test_vinf_percentage_bins(self):
        def make_enc(vinf: float, t: float = 0.0) -> Encounter:
            return Encounter(
                body=1,
                t=t,
                r=(0.0, 0.0, 0.0),
                vinf_in=vinf,
                vinf_in_vec=None,
                vinf_out=None,
                vinf_out_vec=None,
                flyby_valid=None,
                flyby_altitude=None,
                dv_periapsis=None,
                dv_periapsis_vec=None,
                dv_limit=None,
                J_total=0.0,
            )

        key_low = key_fn((make_enc(0.5),))
        key_low_b = key_fn((make_enc(0.9),))
        self.assertEqual(key_low, key_low_b)

        key_baseline = key_fn((make_enc(1.0),))
        key_within_pct = key_fn((make_enc(1.05),))
        self.assertEqual(key_baseline, key_within_pct)

        key_outside_pct = key_fn((make_enc(1.3),))
        self.assertNotEqual(key_baseline, key_outside_pct)

        key_v10 = key_fn((make_enc(10.0),))
        key_v12 = key_fn((make_enc(12.0),))
        self.assertNotEqual(key_v10, key_v12)


class TestInterstellarBody(unittest.TestCase):

    def test_state_is_fixed(self):
        interstellar = bodies_data[INTERSTELLAR_BODY_ID]
        state_now = interstellar.get_state(0.0)
        state_future = interstellar.get_state(12345.6)

        r_expected = np.array([-200.0 * KMPAU, 0.0, 0.0], dtype=float)
        v_expected = np.array([1.0, 0.0, 0.0], dtype=float)

        # The Interstellar state is approximate because JAX defaults to float32 precision by default.
        assert_allclose(np.asarray(state_now.r, dtype=float), r_expected, rtol=1e-8, atol=1e-3)
        assert_allclose(np.asarray(state_now.v, dtype=float), v_expected, rtol=1e-8, atol=1e-6)
        assert_allclose(np.asarray(state_future.r, dtype=float), r_expected, rtol=1e-8, atol=1e-3)
        assert_allclose(np.asarray(state_future.v, dtype=float), v_expected, rtol=1e-8, atol=1e-6)

        state_au = interstellar.get_state(0.0, distance_units="AU")
        r_expected_au = np.array([-200.0, 0.0, 0.0], dtype=float)
        v_expected_au = np.array([1.0 / KMPAU, 0.0, 0.0], dtype=float)
        assert_allclose(np.asarray(state_au.r, dtype=float), r_expected_au, rtol=1e-7, atol=1e-9)
        assert_allclose(np.asarray(state_au.v, dtype=float), v_expected_au, rtol=1e-7, atol=1e-12)

    def test_expand_fn_uses_special_tof_bounds(self):
        config = make_lambert_config(
            dv_max=None,
            vinf_max=None,
            tof_max_days=None,
            dv_mode="fixed",
        )
        registry = BodyRegistry(
            body_ids=(1, 2),
            semi_major_axes={
                1: BASE_SEMI_MAJOR_AXES[1],
                2: BASE_SEMI_MAJOR_AXES[2],
            },
            weights={
                1: BASE_BODY_WEIGHTS[1],
                2: BASE_BODY_WEIGHTS[2],
            },
            tof_sample_count=4,
        )

        interstellar_state = bodies_data[INTERSTELLAR_BODY_ID].get_state(0.0)
        root = (
            Encounter(
                body=INTERSTELLAR_BODY_ID,
                t=0.0,
                r=tuple(float(x) for x in np.asarray(interstellar_state.r, dtype=float)),
                vinf_in=None,
                vinf_in_vec=None,
                vinf_out=None,
                vinf_out_vec=None,
                flyby_valid=None,
                flyby_altitude=None,
                dv_periapsis=None,
                dv_periapsis_vec=None,
                dv_limit=None,
                J_total=0.0,
            ),
        )
        expand = make_expand_fn(config, registry)
        proposals = list(expand(root))
        self.assertTrue(proposals)
        self.assertTrue(all(prop.body != INTERSTELLAR_BODY_ID for prop in proposals))

        tof_values = [prop.tof for prop in proposals]
        min_expected = 5.0 * YEAR / DAY
        max_expected = 100.0 * YEAR / DAY

        self.assertGreaterEqual(min(tof_values), min_expected)
        self.assertLessEqual(max(tof_values), max_expected)

        for body_id in (1, 2):
            body_samples = [prop.tof for prop in proposals if prop.body == body_id]
            self.assertEqual(len(body_samples), registry.tof_sample_count)

    def test_seed_expansion_generates_offsets(self):
        config = make_lambert_config(
            dv_max=None,
            vinf_max=None,
            tof_max_days=None,
            dv_mode="fixed",
        )
        registry = BodyRegistry(
            body_ids=(1,),
            semi_major_axes={1: BASE_SEMI_MAJOR_AXES[1]},
            weights={1: BASE_BODY_WEIGHTS[1]},
            tof_sample_count=3,
        )
        interstellar_state = bodies_data[INTERSTELLAR_BODY_ID].get_state(0.0)
        root = (
            Encounter(
                body=INTERSTELLAR_BODY_ID,
                t=0.0,
                r=tuple(float(x) for x in np.asarray(interstellar_state.r, dtype=float)),
                vinf_in=None,
                vinf_in_vec=None,
                vinf_out=None,
                vinf_out_vec=None,
                flyby_valid=None,
                flyby_altitude=None,
                dv_periapsis=None,
                dv_periapsis_vec=None,
                dv_limit=None,
                J_total=0.0,
            ),
        )

        expand = make_expand_fn(config, registry, seed_count=4)
        seeds = list(expand(root))
        self.assertEqual(len(seeds), 4)
        self.assertTrue(all(prop.is_seed for prop in seeds))
        offsets = {prop.seed_offset for prop in seeds}
        self.assertEqual(len(offsets), 4)

        score = make_score_fn(config, registry, mode="medium")
        inc, seeded_state = score(root, seeds[0])
        self.assertEqual(inc, 0.0)
        self.assertEqual(len(seeded_state), 1)
        self.assertEqual(seeded_state[-1].seed_offset, seeds[0].seed_offset)

        post_seed_props = list(expand(seeded_state))
        self.assertTrue(post_seed_props)
        self.assertTrue(all(not prop.is_seed for prop in post_seed_props))

    def test_key_fn_distinguishes_seed_offsets(self):
        base_state = Encounter(
            body=INTERSTELLAR_BODY_ID,
            t=0.0,
            r=(0.0, 0.0, 0.0),
            vinf_in=None,
            vinf_in_vec=None,
            vinf_out=None,
            vinf_out_vec=None,
            flyby_valid=None,
            flyby_altitude=None,
            dv_periapsis=None,
            dv_periapsis_vec=None,
            dv_limit=None,
            J_total=0.0,
        )
        offset_a = replace(
            base_state,
            r=(base_state.r[0], 10.0 * KMPAU, -5.0 * KMPAU),
            seed_offset=(10.0, -5.0),
        )
        offset_b = replace(
            base_state,
            r=(base_state.r[0], -10.0 * KMPAU, 5.0 * KMPAU),
            seed_offset=(-10.0, 5.0),
        )
        key_a = key_fn((offset_a,))
        key_b = key_fn((offset_b,))
        self.assertNotEqual(key_a, key_b)

    def test_seed_prunes_large_transverse_vinf(self):
        config = make_lambert_config(dv_max=None, vinf_max=None, tof_max_days=None, dv_mode="fixed")
        registry = BodyRegistry(
            body_ids=(1,),
            semi_major_axes={1: BASE_SEMI_MAJOR_AXES[1]},
            weights={1: BASE_BODY_WEIGHTS[1]},
            tof_sample_count=3,
        )
        parent = Encounter(
            body=INTERSTELLAR_BODY_ID,
            t=0.0,
            r=(0.0, 0.0, 0.0),
            seed_offset=(10.0, -5.0),
        )
        proposal = SimpleNamespace(body=1, tof=10.0)

        def score_stub(_config, _registry, _prefix, _parent, child, _meta):
            return 0.0, child

        with patch("gtoc13.path_finding.beam.lambert.body_state", return_value=(np.zeros(3), np.zeros(3))), \
             patch("gtoc13.path_finding.beam.lambert.evaluate_flyby", return_value=(None, None, None, None)), \
             patch(
                 "gtoc13.path_finding.beam.lambert.enumerate_lambert_solutions",
                 return_value=[
                     {
                         "v1": np.array([0.0, 4.0, 4.0]),
                         "v2": np.zeros(3),
                         "r1": np.zeros(3),
                         "r2": np.zeros(3),
                         "rev": 0,
                     }
                 ],
             ):
            with self.assertRaises(InfeasibleLeg):
                resolve_lambert_leg(config, registry, (parent,), proposal, score_stub)

    def test_seed_allows_small_transverse_vinf(self):
        config = make_lambert_config(dv_max=None, vinf_max=None, tof_max_days=None, dv_mode="fixed")
        registry = BodyRegistry(
            body_ids=(1,),
            semi_major_axes={1: BASE_SEMI_MAJOR_AXES[1]},
            weights={1: BASE_BODY_WEIGHTS[1]},
            tof_sample_count=3,
        )
        parent = Encounter(
            body=INTERSTELLAR_BODY_ID,
            t=0.0,
            r=(0.0, 0.0, 0.0),
            seed_offset=(5.0, 5.0),
        )
        proposal = SimpleNamespace(body=1, tof=10.0)

        def score_stub(_config, _registry, _prefix, _parent, child, _meta):
            return 0.0, child

        with patch("gtoc13.path_finding.beam.lambert.body_state", return_value=(np.zeros(3), np.zeros(3))), \
             patch("gtoc13.path_finding.beam.lambert.evaluate_flyby", return_value=(None, None, None, None)), \
             patch(
                 "gtoc13.path_finding.beam.lambert.enumerate_lambert_solutions",
                 return_value=[
                     {
                         "v1": np.array([0.0, 2.0, 1.0]),
                         "v2": np.zeros(3),
                         "r1": np.zeros(3),
                         "r2": np.zeros(3),
                         "rev": 0,
                     }
                 ],
             ):
            contrib, state = resolve_lambert_leg(config, registry, (parent,), proposal, score_stub)
        self.assertEqual(contrib, 0.0)
        self.assertEqual(state[0].body, INTERSTELLAR_BODY_ID)
        self.assertEqual(state[-1].body, proposal.body)

    def test_seed_limits_lambert_revolutions(self):
        config = make_lambert_config(dv_max=None, vinf_max=None, tof_max_days=None, dv_mode="fixed")
        registry = BodyRegistry(
            body_ids=(1,),
            semi_major_axes={1: BASE_SEMI_MAJOR_AXES[1]},
            weights={1: BASE_BODY_WEIGHTS[1]},
            tof_sample_count=3,
        )
        parent = Encounter(
            body=INTERSTELLAR_BODY_ID,
            t=0.0,
            r=(0.0, 0.0, 0.0),
            seed_offset=(3.0, -2.0),
        )
        proposal = SimpleNamespace(body=1, tof=20.0)

        def score_stub(_config, _registry, _prefix, _parent, child, _meta):
            return 0.0, child

        captured_max_rev: dict[str, int] = {}

        def fake_enum(_bd, _ba, _td, _ta, max_revs):
            captured_max_rev["value"] = max_revs
            return [
                {
                    "v1": np.array([0.0, 1.0, 1.0]),
                    "v2": np.zeros(3),
                    "r1": np.zeros(3),
                    "r2": np.zeros(3),
                    "rev": 0,
                }
            ]

        with patch("gtoc13.path_finding.beam.lambert.body_state", return_value=(np.zeros(3), np.zeros(3))), \
             patch("gtoc13.path_finding.beam.lambert.evaluate_flyby", return_value=(None, None, None, None)), \
             patch("gtoc13.path_finding.beam.lambert.enumerate_lambert_solutions", side_effect=fake_enum):
            resolve_lambert_leg(config, registry, (parent,), proposal, score_stub)

        self.assertEqual(captured_max_rev.get("value"), 0)


class TestMissionRawScoring(unittest.TestCase):
    def test_raw_mode_skips_time_scaling(self):
        config = make_lambert_config(dv_max=None, vinf_max=None, tof_max_days=100.0)
        registry = BodyRegistry(body_ids=(), semi_major_axes={}, weights={}, tof_sample_count=1)

        parent = Encounter(body=1, t=94.0, r=(0.0, 0.0, 0.0), J_total=0.0)
        child = Encounter(body=2, t=99.0, r=(0.0, 0.0, 0.0))
        prefix = (parent,)
        meta = LambertLegMeta(
            proposal=SimpleNamespace(tof=6.0),
            lambert_solution={},
            vinf_out_vec=(0.0, 0.0, 0.0),
            vinf_in_vec=(0.0, 0.0, 0.0),
            vinf_out=0.0,
            vinf_in=0.0,
        )

        with patch("gtoc13.path_finding.beam.scoring.mission_score", return_value=100.0):
            contrib_scaled, child_scaled = score_leg_mission(
                config, registry, prefix, parent, child, meta
            )
            contrib_raw, child_raw = score_leg_mission_raw(
                config, registry, prefix, parent, child, meta
            )

        self.assertAlmostEqual(child_raw.J_total, 100.0)
        self.assertAlmostEqual(contrib_raw, 100.0)
        self.assertAlmostEqual(child_scaled.J_total, 100.0 / 6.0)
        self.assertAlmostEqual(contrib_scaled, 100.0 / 6.0)
        self.assertGreater(child_raw.J_total, child_scaled.J_total)


if __name__ == '__main__':
    unittest.main()
