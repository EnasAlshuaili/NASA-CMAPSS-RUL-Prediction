import json
import shutil
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

from src.data_loader import add_training_rul, load_train_data
from src.model_service import (
    FEATURE_NAMES,
    ModelArtifactError,
    _sha256,
    check_manual_values,
    load_model_bundle,
    predict_rul,
    sensor_ranges,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
RAW_DATA = PROJECT_ROOT / "data" / "raw"


class ModelArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = load_model_bundle(MODELS_DIR)
        cls.data = add_training_rul(load_train_data(RAW_DATA / "train_FD001.txt"))

    def test_served_model_is_the_groupkfold_tuned_xgboost(self) -> None:
        model = self.bundle.model
        self.assertEqual(type(model).__name__, "XGBRegressor")
        self.assertEqual(model.n_features_in_, 15)
        self.assertEqual([str(f) for f in model.feature_names_in_], FEATURE_NAMES)
        self.assertEqual(joblib.load(MODELS_DIR / "feature_names.pkl"), FEATURE_NAMES)
        params = model.get_params()
        for name, value in self.bundle.metadata["hyperparameters"].items():
            self.assertEqual(params[name], value, name)
        self.assertEqual(params["n_estimators"], 400)
        self.assertEqual(params["learning_rate"], 0.01)

    def test_reported_metrics_reproduce_from_raw_data(self) -> None:
        """Rebuild the notebook's engine-level split independently and re-score the saved model."""
        X, y = self.data[FEATURE_NAMES], self.data["RUL"]
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
        train_idx, val_idx = next(splitter.split(X, y, groups=self.data["engine_id"]))

        validation = self.bundle.metadata["validation"]
        self.assertEqual(sorted(self.data.iloc[val_idx]["engine_id"].unique()), validation["validation_engines"])
        self.assertEqual(sorted(self.data.iloc[train_idx]["engine_id"].unique()), validation["train_engines"])
        self.assertFalse(set(validation["train_engines"]) & set(validation["validation_engines"]))

        predictions = self.bundle.model.predict(X.iloc[val_idx])
        y_val = y.iloc[val_idx]
        reproduced = {
            "MAE": round(float(mean_absolute_error(y_val, predictions)), 4),
            "RMSE": round(float(np.sqrt(mean_squared_error(y_val, predictions))), 4),
            "R2": round(float(r2_score(y_val, predictions)), 4),
        }
        self.assertEqual(reproduced, self.bundle.metrics)
        self.assertEqual(reproduced, {"MAE": 25.8786, "RMSE": 35.1955, "R2": 0.7126})

    def test_predict_rul_uses_named_features_in_model_order(self) -> None:
        row = self.data[(self.data["engine_id"] == 5) & (self.data["cycle"] == 250)].iloc[0]
        values = {f: float(row[f]) for f in reversed(FEATURE_NAMES)}  # dict order must not matter
        expected = float(self.bundle.model.predict(row[FEATURE_NAMES].to_frame().T.astype(float))[0])
        self.assertAlmostEqual(predict_rul(self.bundle.model, values), max(0.0, expected), places=5)

    def test_predict_rul_rejects_missing_or_non_finite_values(self) -> None:
        values = {f: 1.0 for f in FEATURE_NAMES}
        with self.assertRaisesRegex(ValueError, "Missing"):
            predict_rul(self.bundle.model, {k: v for k, v in values.items() if k != "sensor_11"})
        with self.assertRaisesRegex(ValueError, "finite"):
            predict_rul(self.bundle.model, {**values, "sensor_11": float("nan")})


class ModelLoaderSafetyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        for name in ("xgboost_rul_model.pkl", "feature_names.pkl", "model_metadata.json", "scaler.pkl"):
            shutil.copy(MODELS_DIR / name, self.tmp / name)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _metadata(self) -> dict:
        return json.loads((self.tmp / "model_metadata.json").read_text(encoding="utf-8"))

    def _write_metadata(self, metadata: dict) -> None:
        (self.tmp / "model_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    def test_copy_of_valid_artifacts_loads(self) -> None:
        self.assertEqual(load_model_bundle(self.tmp).metrics, load_model_bundle(MODELS_DIR).metrics)

    def test_missing_model_file_is_an_error_not_a_fallback(self) -> None:
        (self.tmp / "xgboost_rul_model.pkl").unlink()
        with self.assertRaisesRegex(ModelArtifactError, "missing"):
            load_model_bundle(self.tmp)

    def test_metrics_for_a_different_model_file_are_rejected(self) -> None:
        metadata = self._metadata()
        metadata["model_sha256"] = "0" * 64
        self._write_metadata(metadata)
        with self.assertRaisesRegex(ModelArtifactError, "fingerprint"):
            load_model_bundle(self.tmp)

    def test_non_xgboost_object_is_rejected(self) -> None:
        shutil.copy(self.tmp / "scaler.pkl", self.tmp / "xgboost_rul_model.pkl")
        metadata = self._metadata()
        metadata["model_sha256"] = _sha256(self.tmp / "xgboost_rul_model.pkl")
        self._write_metadata(metadata)
        with self.assertRaisesRegex(ModelArtifactError, "expected XGBRegressor"):
            load_model_bundle(self.tmp)

    def test_wrong_feature_order_is_rejected(self) -> None:
        joblib.dump(list(reversed(FEATURE_NAMES)), self.tmp / "feature_names.pkl")
        with self.assertRaisesRegex(ModelArtifactError, "Feature order"):
            load_model_bundle(self.tmp)


class ManualValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ranges = sensor_ranges(load_train_data(RAW_DATA / "train_FD001.txt"))

    def test_training_medians_are_valid(self) -> None:
        values = self.ranges["median"].to_dict()
        self.assertEqual(check_manual_values(values, self.ranges), ([], []))

    def test_zero_and_negative_values_are_invalid(self) -> None:
        values = {f: 0.0 for f in FEATURE_NAMES}
        invalid, out_of_range = check_manual_values(values, self.ranges)
        self.assertEqual(invalid, FEATURE_NAMES)
        self.assertEqual(out_of_range, [])

    def test_positive_values_outside_training_range_are_flagged(self) -> None:
        values = self.ranges["median"].to_dict()
        values["sensor_11"] = float(self.ranges.loc["sensor_11", "max"]) + 1.0
        self.assertEqual(check_manual_values(values, self.ranges), ([], ["sensor_11"]))


if __name__ == "__main__":
    unittest.main()
