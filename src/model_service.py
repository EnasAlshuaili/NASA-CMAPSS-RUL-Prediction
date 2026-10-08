"""Load, validate, and query the final RUL model without any UI dependencies.

The app and scripts share this module so that model loading, feature order,
prediction, and manual-input checks are implemented exactly once.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import joblib
import pandas as pd
from xgboost import XGBRegressor


FEATURE_NAMES = [
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_6",
    "sensor_7",
    "sensor_8",
    "sensor_9",
    "sensor_11",
    "sensor_12",
    "sensor_13",
    "sensor_14",
    "sensor_15",
    "sensor_17",
    "sensor_20",
    "sensor_21",
]

MODEL_FILE = "xgboost_rul_model.pkl"
FEATURES_FILE = "feature_names.pkl"
METADATA_FILE = "model_metadata.json"
REQUIRED_METRICS = ("MAE", "RMSE", "R2")


class ModelArtifactError(RuntimeError):
    """Raised when the saved model artifacts are missing or inconsistent."""


@dataclass(frozen=True)
class ModelBundle:
    model: XGBRegressor
    metadata: dict
    model_path: Path

    @property
    def metrics(self) -> dict:
        return self.metadata["metrics"]

    @property
    def validation_engines(self) -> set[int]:
        return set(self.metadata["validation"]["validation_engines"])

    @property
    def train_engines(self) -> list[int]:
        return list(self.metadata["validation"]["train_engines"])


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_model_bundle(models_dir: str | Path) -> ModelBundle:
    """Load the final XGBoost model and verify it against its saved metadata.

    Only the explicitly named model file is ever loaded. The metadata's SHA-256
    fingerprint guarantees that the displayed metrics belong to this exact file.
    """
    models_dir = Path(models_dir)
    model_path = models_dir / MODEL_FILE
    features_path = models_dir / FEATURES_FILE
    metadata_path = models_dir / METADATA_FILE

    for path in (model_path, features_path, metadata_path):
        if not path.is_file():
            raise ModelArtifactError(f"Required model artifact is missing: {path.name}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("model_sha256") != _sha256(model_path):
        raise ModelArtifactError(
            f"{MODEL_FILE} does not match the fingerprint in {METADATA_FILE}. "
            "The metrics on record belong to a different model file."
        )

    model = joblib.load(model_path)
    if not isinstance(model, XGBRegressor):
        raise ModelArtifactError(
            f"{MODEL_FILE} contains {type(model).__name__}, expected XGBRegressor."
        )
    if model.n_features_in_ != len(FEATURE_NAMES):
        raise ModelArtifactError(
            f"Model expects {model.n_features_in_} features, expected {len(FEATURE_NAMES)}."
        )

    model_features = [str(name) for name in getattr(model, "feature_names_in_", [])]
    saved_features = list(joblib.load(features_path))
    for source, features in (
        ("model.feature_names_in_", model_features),
        (FEATURES_FILE, saved_features),
        (METADATA_FILE, list(metadata.get("features", []))),
    ):
        if features != FEATURE_NAMES:
            raise ModelArtifactError(f"Feature order in {source} does not match FEATURE_NAMES.")

    missing_metrics = [m for m in REQUIRED_METRICS if m not in metadata.get("metrics", {})]
    if missing_metrics:
        raise ModelArtifactError(f"{METADATA_FILE} is missing metrics: {missing_metrics}")

    return ModelBundle(model=model, metadata=metadata, model_path=model_path)


def predict_rul(model: XGBRegressor, sensor_values: Mapping[str, float]) -> float:
    """Predict RUL in cycles from raw (unscaled) values of the 15 model features."""
    missing = [name for name in FEATURE_NAMES if name not in sensor_values]
    if missing:
        raise ValueError(f"Missing sensor values: {missing}")

    row = [float(sensor_values[name]) for name in FEATURE_NAMES]
    if not all(math.isfinite(value) for value in row):
        raise ValueError("Sensor values must be finite numbers.")

    prediction = float(model.predict(pd.DataFrame([row], columns=FEATURE_NAMES))[0])
    # RUL cannot be negative; the regressor can slightly undershoot near failure.
    return max(0.0, prediction)


def sensor_ranges(data: pd.DataFrame) -> pd.DataFrame:
    """Observed min / median / max of each model feature in ``data``."""
    return data[FEATURE_NAMES].agg(["min", "median", "max"]).T


def check_manual_values(
    sensor_values: Mapping[str, float], ranges: pd.DataFrame
) -> tuple[list[str], list[str]]:
    """Split features into (invalid, out_of_training_range).

    Every model feature is a physical temperature, pressure, speed, ratio, or flow,
    so a value of zero or below is invalid. A positive value outside the range seen
    during training is allowed, but the model has never seen such inputs and will
    not extrapolate meaningfully.
    """
    invalid, out_of_range = [], []
    for name in FEATURE_NAMES:
        value = float(sensor_values[name])
        if not math.isfinite(value) or value <= 0:
            invalid.append(name)
        elif not ranges.loc[name, "min"] <= value <= ranges.loc[name, "max"]:
            out_of_range.append(name)
    return invalid, out_of_range
