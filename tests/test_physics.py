"""Tests for the physics layer. Run: python -m unittest discover tests -v"""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import physics  # noqa: E402


class AngleTests(unittest.TestCase):
    def test_cosine_values(self):
        self.assertAlmostEqual(physics.angle_factor(0), 1.0)
        self.assertAlmostEqual(physics.angle_factor(30), math.cos(math.radians(30)))
        self.assertAlmostEqual(physics.angle_factor(60), 0.5)
        self.assertEqual(physics.angle_factor(90), 0.0)  # exactly zero, no 6e-17 residue

    def test_monotonic_non_negative(self):
        vals = [physics.angle_factor(a) for a in range(0, 91)]
        self.assertTrue(all(v >= 0 for v in vals))
        self.assertTrue(all(a >= b for a, b in zip(vals, vals[1:])))

    def test_invalid_angles(self):
        for bad in (-1, 90.5, float("nan"), None):
            with self.assertRaises(ValueError):
                physics.angle_factor(bad)


class ShadingTests(unittest.TestCase):
    def test_endpoints_for_every_model(self):
        for model in ("linear", "bypass_diode", "power_law"):
            self.assertAlmostEqual(physics.shading_factor(0, model=model), 1.0, msg=model)
            self.assertAlmostEqual(physics.shading_factor(100, model=model), 0.0, msg=model)

    def test_linear_values(self):
        for pct in (0, 25, 50, 75, 100):
            self.assertAlmostEqual(physics.shading_factor(pct, model="linear"), 1 - pct / 100)

    def test_bypass_is_nonlinear_and_monotonic(self):
        vals = [physics.shading_factor(s, model="bypass_diode") for s in range(0, 101)]
        self.assertTrue(all(0 <= v <= 1 for v in vals))
        self.assertTrue(all(a >= b - 1e-12 for a, b in zip(vals, vals[1:])))
        self.assertNotAlmostEqual(physics.shading_factor(50, model="bypass_diode"), 0.5)

    def test_all_requested_levels_in_range(self):
        for model in ("linear", "bypass_diode", "power_law"):
            for pct in (0, 25, 50, 75, 100):
                self.assertTrue(0.0 <= physics.shading_factor(pct, model=model) <= 1.0)

    def test_invalid(self):
        for bad in (-5, 100.1):
            with self.assertRaises(ValueError):
                physics.shading_factor(bad)
        with self.assertRaises(ValueError):
            physics.shading_factor(10, model="magic")


class ScenarioTests(unittest.TestCase):
    def test_zero_angle_zero_shade_equals_baseline(self):
        sc = physics.compute_scenario(15.0, 0, 0, 1000, 1.6)
        self.assertAlmostEqual(sc["effective_pce"], 15.0)
        self.assertAlmostEqual(sc["power_w"], 1000 * 1.6 * 0.15)
        self.assertAlmostEqual(sc["power_w"], sc["reference_power_w"])
        self.assertAlmostEqual(sc["abs_change_pp"], 0.0)

    def test_formula_matches_documentation(self):
        a = physics.load_assumptions()
        sc = physics.compute_scenario(20.0, 60, 25, 800, 2.0, a)
        expected = 20.0 * 0.5 * physics.shading_factor(25, a)
        self.assertAlmostEqual(sc["effective_pce"], expected)
        self.assertAlmostEqual(sc["power_w"], 800 * 2.0 * expected / 100)

    def test_grid_has_no_negative_or_nan(self):
        for ang in (0, 30, 60, 90):
            for shd in (0, 25, 50, 75, 100):
                sc = physics.compute_scenario(18.5, ang, shd, 1000, 1.0)
                for key in ("angle_adjusted_pce", "shading_adjusted_pce", "effective_pce", "power_w"):
                    self.assertTrue(math.isfinite(sc[key]) and sc[key] >= 0, (ang, shd, key))
                self.assertLessEqual(sc["effective_pce"], 18.5 + 1e-9)

    def test_zero_baseline_does_not_divide_by_zero(self):
        sc = physics.compute_scenario(0.0, 30, 50, 1000, 1.0)
        self.assertEqual(sc["rel_change_pct"], 0.0)
        self.assertEqual(sc["power_w"], 0.0)

    def test_factors_applied_once(self):
        # 90 deg or 100% shade must zero the output; applying a factor twice would not change that, so also check 60 deg
        sc = physics.compute_scenario(10.0, 60, 0, 1000, 1.0)
        self.assertAlmostEqual(sc["effective_pce"], 5.0)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            physics.compute_scenario(-1, 0, 0, 1000, 1)
        with self.assertRaises(ValueError):
            physics.compute_scenario(15, 0, 0, -10, 1)
        with self.assertRaises(ValueError):
            physics.compute_scenario(15, 0, 0, 1000, 0)

    def test_curves(self):
        angles, pce = physics.angle_curve(15.0, 0)
        self.assertAlmostEqual(pce[0], 15.0)
        self.assertEqual(pce[-1], 0.0)
        levels, pce2 = physics.shading_curve(15.0, 0)
        self.assertAlmostEqual(pce2[0], 15.0)
        self.assertAlmostEqual(pce2[-1], 0.0)


if __name__ == "__main__":
    unittest.main()
