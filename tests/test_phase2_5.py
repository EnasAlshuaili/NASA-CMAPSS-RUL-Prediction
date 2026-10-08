import json
import unittest

import numpy as np
import pandas as pd

from src.data_loader import add_training_rul, load_train_data
from src.phase2_5_test_like_validation import (
    COMPARISON_FILE,
    RESULTS_FILE,
    V1_FEATURES,
    comparison_table,
    engine_bootstrap,
    holdout_predictions,
    observed_history,
    random_fractions,
    score_samples,
    truncation_points,
    test_like_features as build_test_like_features,
)
from src.phase2_experiments import RAW_DATA, Experiment, load_v1_reference


DATA = add_training_rul(load_train_data(RAW_DATA / "train_FD001.txt"))
TEMPORAL = Experiment("all temporal", rolling=True, trend=True)
SENSORS = ["sensor_11", "sensor_4"]


class TruncationTests(unittest.TestCase):
    def test_truncation_cycle_and_label(self) -> None:
        points = truncation_points(DATA, [1, 2], fractions=(0.6, 0.9))
        life = DATA.groupby("engine_id")["cycle"].max()
        for p in points.itertuples():
            with self.subTest(engine=p.engine_id, fraction=p.fraction):
                self.assertEqual(p.max_cycle, life[p.engine_id])
                self.assertEqual(p.truncation_cycle, int(np.floor(p.fraction * p.max_cycle + 0.5)))
                self.assertEqual(p.true_rul, p.max_cycle - p.truncation_cycle)
                # The label equals the training RUL recorded at that cycle.
                row = DATA[(DATA["engine_id"] == p.engine_id) & (DATA["cycle"] == p.truncation_cycle)]
                self.assertEqual(p.true_rul, int(row["RUL"].iloc[0]))

    def test_truncation_is_clamped_strictly_before_failure(self) -> None:
        points = truncation_points(DATA, [3], fractions=(0.0, 1.0))
        max_cycle = int(DATA.loc[DATA["engine_id"] == 3, "cycle"].max())
        self.assertEqual(points["truncation_cycle"].tolist(), [1, max_cycle - 1])
        self.assertTrue((points["true_rul"] >= 1).all())

    def test_truncation_is_deterministic(self) -> None:
        engines = load_v1_reference()[2]
        pd.testing.assert_frame_equal(truncation_points(DATA, engines), truncation_points(DATA, engines))
        self.assertEqual(random_fractions(engines), random_fractions(engines))
        a, b = truncation_points(DATA, engines, random_fractions(engines)), truncation_points(DATA, engines, random_fractions(engines))
        pd.testing.assert_frame_equal(a, b)


class LeakageTests(unittest.TestCase):
    def test_observed_history_contains_only_past_rows_of_one_engine(self) -> None:
        history = observed_history(DATA, 5, 120)
        self.assertEqual(set(history["engine_id"]), {5})
        self.assertEqual(history["cycle"].max(), 120)
        self.assertNotIn("RUL", history.columns)

    def test_future_cycles_do_not_change_features(self) -> None:
        points = truncation_points(DATA, [5, 11], fractions=(0.6, 0.8))
        corrupted = DATA.copy()
        for p in points.itertuples():
            future = (corrupted["engine_id"] == p.engine_id) & (corrupted["cycle"] > p.truncation_cycle)
            if p.fraction == 0.6:  # corrupt only after the earliest cut so later cuts stay valid
                corrupted.loc[future, SENSORS] = 1e6
                corrupted.loc[future, "RUL"] = -999
        for experiment in (V1_FEATURES, TEMPORAL):
            clean, columns = build_test_like_features(DATA, points[points["fraction"] == 0.6], experiment, SENSORS)
            dirty, _ = build_test_like_features(corrupted, points[points["fraction"] == 0.6], experiment, SENSORS)
            with self.subTest(experiment=experiment.name):
                pd.testing.assert_frame_equal(clean[columns], dirty[columns])

    def test_other_engines_do_not_change_features(self) -> None:
        points = truncation_points(DATA, [5], fractions=(0.7,))
        others = DATA.copy()
        others.loc[others["engine_id"] != 5, SENSORS] = -1e6
        clean, columns = build_test_like_features(DATA, points, TEMPORAL, SENSORS)
        dirty, _ = build_test_like_features(others, points, TEMPORAL, SENSORS)
        pd.testing.assert_frame_equal(clean[columns], dirty[columns])

    def test_feature_row_is_at_truncation_cycle(self) -> None:
        points = truncation_points(DATA, [5, 11, 13])
        features, _ = build_test_like_features(DATA, points, TEMPORAL, SENSORS)
        self.assertEqual(features["cycle"].tolist(), points["truncation_cycle"].tolist())
        self.assertEqual(features["engine_id"].tolist(), points["engine_id"].tolist())


class CapTests(unittest.TestCase):
    def test_cap_limits_training_labels_but_not_validation_labels(self) -> None:
        _, train_engines, validation_engines = load_v1_reference()
        points = truncation_points(DATA, validation_engines)
        _, samples = holdout_predictions(DATA, train_engines, points, cap=75)
        # Validation labels are the true, uncapped RUL ...
        self.assertEqual(samples["actual_rul"].tolist(), points["true_rul"].astype(float).tolist())
        self.assertGreater(samples["actual_rul"].max(), 75)
        # ... while a model trained on labels capped at 75 cannot learn to predict far above 75.
        self.assertLessEqual(samples["predicted_rul"].max(), 76)
        self.assertTrue(set(train_engines).isdisjoint(validation_engines))


class ReproducibilityTests(unittest.TestCase):
    def test_bootstrap_is_deterministic_and_paired_by_engine(self) -> None:
        points = truncation_points(DATA, [1, 2, 3])
        ref = score_samples(points, points["true_rul"] + 10)
        cand = score_samples(points, points["true_rul"] + 4)
        first, second = engine_bootstrap(cand, ref), engine_bootstrap(cand, ref)
        self.assertEqual(first, second)
        self.assertEqual(first["mae_difference"], -6.0)
        self.assertEqual(first["ci95"], [-6.0, -6.0])

    def test_saved_results_match_protocol_and_table(self) -> None:
        report = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
        validation_engines = load_v1_reference()[2]
        expected = truncation_points(DATA, validation_engines)
        self.assertEqual(report["truncation_points"], json.loads(expected.to_json(orient="records")))
        self.assertTrue(report["v1_reproduction"]["saved_v1_matches_no_cap_refit"])
        self.assertEqual([e["name"] for e in report["experiments"]],
                         ["V1 / No Cap", "Cap 150", "Cap 125", "Cap 100", "Cap 75"])
        saved_table = pd.read_csv(COMPARISON_FILE, keep_default_na=False, dtype=str)
        rebuilt = comparison_table(report).astype(str).replace({"None": "", "nan": ""})
        self.assertEqual(saved_table["Model"].tolist(), rebuilt["Model"].tolist())
        self.assertEqual(saved_table["MAE"].astype(float).tolist(), rebuilt["MAE"].astype(float).tolist())
        # Only the 60-90 % schedule is scored per engine: exactly 4 samples per validation engine.
        counts = {e["engine_id"]: e["n"] for e in report["experiments"][0]["holdout"]["by_engine"]}
        self.assertEqual(counts, {e: 4 for e in validation_engines})


if __name__ == "__main__":
    unittest.main()
