"""
Tests for the trained model and its input handling, using the real saved
pipeline and the real dataset. Run: python -m unittest discover tests -v
"""

import json
import os
import sys
import unittest

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import preprocessing as pre  # noqa: E402
from predict import (  # noqa: E402
    SunPredictor, default_selection, load_ui_options, predict_baseline_pce, validate_selection,
)

MODELS = os.path.join(ROOT, "models")


class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = pre.load_raw_dataset()

    def test_real_columns_present(self):
        needed = [pre.TARGET_COL] + pre.LEAKAGE_COLS + [
            "Cell_architecture", "ETL_stack_sequence", "HTL_stack_sequence", "Perovskite_composition_short_form",
            "Perovskite_additives_compounds", "Perovskite_thickness",
            "Perovskite_deposition_thermal_annealing_temperature",
        ]
        for col in needed:
            self.assertIn(col, self.df.columns)

    def test_no_leakage_columns_in_features(self):
        used = set(pre.CATEGORICAL_FEATURES + pre.NUMERICAL_FEATURES)
        for col in ("JV_default_Voc", "JV_default_Jsc", "JV_default_FF", "JV_measured", pre.TARGET_COL):
            self.assertNotIn(col, used)

    def test_feature_engineering_on_real_rows(self):
        valid = self.df[self.df[pre.TARGET_COL].notna()].head(500)
        eng = pre.engineer_features(valid)
        for col in pre.CATEGORICAL_FEATURES + pre.NUMERICAL_FEATURES:
            self.assertIn(col, eng.columns)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = SunPredictor(models_dir=MODELS)
        cls.options = load_ui_options(MODELS)
        with open(os.path.join(MODELS, "model_metadata.json")) as f:
            cls.meta = json.load(f)

    def test_default_prediction_is_valid(self):
        res = predict_baseline_pce(self.predictor, default_selection(self.options), self.options)
        self.assertTrue(0 <= res["predicted_pce"] <= 45)
        self.assertEqual(res["warnings"], [])

    def test_every_dropdown_option_predicts(self):
        base = default_selection(self.options)
        for key, items in self.options["categorical"].items():
            if key not in base:
                continue
            for item in items[:3]:
                sel = dict(base, **{key: item["value"]})
                res = predict_baseline_pce(self.predictor, sel, self.options)
                self.assertTrue(0 <= res["predicted_pce"] <= 45, (key, item))

    def test_quenching_and_architecture_change_prediction(self):
        base = default_selection(self.options)
        a = predict_baseline_pce(self.predictor, dict(base, quenching=True), self.options)["predicted_pce"]
        b = predict_baseline_pce(self.predictor, dict(base, quenching=False), self.options)["predicted_pce"]
        self.assertNotEqual(a, b)

    def test_out_of_range_warns(self):
        sel = dict(default_selection(self.options), annealing_temp=2000.0)
        res = predict_baseline_pce(self.predictor, sel, self.options)
        self.assertTrue(any("outside the full range" in w for w in res["warnings"]))

    def test_unsupported_category_warns(self):
        sel = dict(default_selection(self.options), HTL_stack_sequence="Unobtainium")
        errors, warnings = validate_selection(sel, self.options)
        self.assertEqual(errors, [])
        self.assertTrue(any("Unobtainium" in w for w in warnings))
        self.assertTrue(0 <= predict_baseline_pce(self.predictor, sel, self.options)["predicted_pce"] <= 45)

    def test_invalid_inputs_raise(self):
        base = default_selection(self.options)
        for bad in (dict(base, annealing_temp=-5), dict(base, perovskite_thickness=0),
                    dict(base, annealing_time=float("nan")), dict(base, Cell_architecture="")):
            with self.assertRaises(ValueError):
                predict_baseline_pce(self.predictor, bad, self.options)

    def test_feature_importance(self):
        fi = self.predictor.get_feature_importance(10)
        self.assertEqual(len(fi), 10)
        self.assertAlmostEqual(sum(d["importance"] for d in self.predictor.get_feature_importance(1000)), 1.0, places=6)

    def test_saved_metrics_match_fresh_evaluation(self):
        """Recompute test-set metrics from the saved pipeline; they must equal the stored numbers."""
        from sklearn.metrics import mean_absolute_error, r2_score
        data = pre.prepare_data(random_state=42)
        preds = self.predictor.pipeline.predict(data["X_test"])
        self.assertAlmostEqual(mean_absolute_error(data["y_test"], preds), self.meta["test_metrics"]["mae"], places=3)
        self.assertAlmostEqual(r2_score(data["y_test"], preds), self.meta["test_metrics"]["r2"], places=3)


if __name__ == "__main__":
    unittest.main()
