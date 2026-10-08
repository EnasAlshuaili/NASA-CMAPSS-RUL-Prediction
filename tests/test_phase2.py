import json
import unittest

import numpy as np
import pandas as pd

from src.data_loader import add_training_rul, load_train_data
from src.evaluate_official_test import METRICS_FILE as V1_OFFICIAL_FILE
from src.model_service import FEATURE_NAMES
from src.phase2_experiments import (
    EXPERIMENTS_FILE,
    RAW_DATA,
    Experiment,
    add_rolling_features,
    add_trend_features,
    build_features,
    fit_predict,
    load_v1_reference,
    official_test_scores,
    select_temporal_sensors,
    training_target,
)


DATA = add_training_rul(load_train_data(RAW_DATA / "train_FD001.txt"))
SENSORS = ["sensor_11", "sensor_4"]
TEMPORAL = Experiment("all temporal", rolling=True, trend=True)


class NoLeakageTests(unittest.TestCase):
    def test_features_use_only_past_and_current_cycles(self) -> None:
        """Truncating an engine's future must not change any feature at earlier cycles."""
        full, columns = build_features(DATA, TEMPORAL, SENSORS)
        for cut in (1, 3, 12, 80):
            truncated_data = DATA[(DATA["engine_id"] != 7) | (DATA["cycle"] <= cut)]
            truncated, _ = build_features(truncated_data, TEMPORAL, SENSORS)
            with self.subTest(cut=cut):
                pd.testing.assert_frame_equal(
                    full.loc[(full["engine_id"] == 7) & (full["cycle"] <= cut), columns].reset_index(drop=True),
                    truncated.loc[truncated["engine_id"] == 7, columns].reset_index(drop=True),
                )

    def test_features_are_isolated_per_engine(self) -> None:
        """An engine's features are identical whether or not other engines are present."""
        full, columns = build_features(DATA, TEMPORAL, SENSORS)
        for engine in (1, 2, 100):
            alone, _ = build_features(DATA[DATA["engine_id"] == engine], TEMPORAL, SENSORS)
            with self.subTest(engine=engine):
                pd.testing.assert_frame_equal(
                    full.loc[full["engine_id"] == engine, columns].reset_index(drop=True),
                    alone[columns].reset_index(drop=True),
                )

    def test_feature_values_on_a_known_sequence(self) -> None:
        engine = pd.DataFrame({"engine_id": 1, "cycle": [1, 2, 3, 4],
                               **{f: 1.0 for f in FEATURE_NAMES}})
        engine["sensor_11"] = [1.0, 3.0, 5.0, 9.0]
        rolled = add_rolling_features(engine, ["sensor_11"], windows=(2,))
        trended = add_trend_features(engine, ["sensor_11"], window=2)
        np.testing.assert_allclose(rolled["sensor_11_mean_2"], [1, 2, 4, 7])
        np.testing.assert_allclose(rolled["sensor_11_std_2"], [0, 1, 1, 2])
        np.testing.assert_allclose(trended["sensor_11_diff_1"], [0, 2, 2, 4])
        np.testing.assert_allclose(trended["sensor_11_slope_2"], [0, 2, 2, 4])
        np.testing.assert_allclose(trended["sensor_11_change_2"], [0, 2, 4, 6])

    def test_unordered_data_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "ordered"):
            add_rolling_features(DATA.iloc[::-1], SENSORS)

    def test_temporal_sensors_are_chosen_from_training_engines_only(self) -> None:
        _, train_engines, validation_engines = load_v1_reference()
        sensors = select_temporal_sensors(DATA[DATA["engine_id"].isin(train_engines)])
        self.assertEqual(len(sensors), 6)
        self.assertTrue(set(sensors) <= set(FEATURE_NAMES))
        self.assertTrue(set(train_engines).isdisjoint(validation_engines))


class TargetAndBaselineTests(unittest.TestCase):
    def test_cap_applies_to_training_target_only(self) -> None:
        rul = pd.Series([0, 50, 125, 300])
        self.assertEqual(training_target(rul, 100).tolist(), [0, 50, 100, 100])
        self.assertIs(training_target(rul, None), rul)

    def test_baseline_refit_reproduces_saved_v1_model(self) -> None:
        bundle, train_engines, validation_engines = load_v1_reference()
        train = DATA[DATA["engine_id"].isin(train_engines)]
        validation = DATA[DATA["engine_id"].isin(validation_engines)]
        _, refit = fit_predict(train, validation, list(FEATURE_NAMES), cap=None)
        saved = np.clip(bundle.model.predict(validation[FEATURE_NAMES]), 0, None)
        np.testing.assert_allclose(refit, saved, rtol=0, atol=1e-4)

    def test_official_protocol_reproduces_v1_official_results(self) -> None:
        scores = official_test_scores(load_v1_reference()[0].model, Experiment("V1 Baseline"), SENSORS)
        saved = json.loads(V1_OFFICIAL_FILE.read_text(encoding="utf-8"))
        self.assertEqual(scores["official_last_cycle"]["metrics"], saved["official_last_cycle"]["metrics"])
        self.assertEqual(scores["all_test_rows"]["metrics"], saved["all_test_rows"]["metrics"])


class SavedResultsTests(unittest.TestCase):
    def test_saved_experiments_record_v1_reproduction(self) -> None:
        report = json.loads(EXPERIMENTS_FILE.read_text(encoding="utf-8"))
        reproduction = report["experiment_0_reproduction"]
        self.assertTrue(reproduction["refit_matches_saved"])
        self.assertTrue(reproduction["saved_matches_recorded"])
        self.assertEqual(report["experiments"][0]["name"], "V1 Baseline")
        self.assertEqual({k: report["experiments"][0]["validation"][k] for k in ("MAE", "RMSE", "R2")},
                         {"MAE": 25.8786, "RMSE": 35.1955, "R2": 0.7126})


if __name__ == "__main__":
    unittest.main()
