"""Load and validate NASA C-MAPSS data without modifying the raw files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas.api.types import is_numeric_dtype


COLUMN_NAMES = [
    "engine_id",
    "cycle",
    "setting_1",
    "setting_2",
    "setting_3",
    "sensor_1",
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_5",
    "sensor_6",
    "sensor_7",
    "sensor_8",
    "sensor_9",
    "sensor_10",
    "sensor_11",
    "sensor_12",
    "sensor_13",
    "sensor_14",
    "sensor_15",
    "sensor_16",
    "sensor_17",
    "sensor_18",
    "sensor_19",
    "sensor_20",
    "sensor_21",
]


def _require_integer_values(series: pd.Series, column: str) -> None:
    """Raise when a numeric identifier column contains fractional values."""
    if not (series % 1 == 0).all():
        raise ValueError(f"{column} must contain whole-number values.")


def validate_cmapss_data(data: pd.DataFrame) -> None:
    """Validate the schema and per-engine time-series structure of C-MAPSS data.

    The source records must be ordered by engine and then by contiguous cycle,
    starting from cycle one for every engine. These checks make a malformed or
    reordered file fail early instead of silently corrupting later features.
    """
    if list(data.columns) != COLUMN_NAMES:
        raise ValueError(
            "Unexpected C-MAPSS columns. "
            f"Expected {len(COLUMN_NAMES)} canonical columns."
        )
    if data.empty:
        raise ValueError("C-MAPSS data must contain at least one observation.")
    if not all(is_numeric_dtype(data[column]) for column in COLUMN_NAMES):
        raise ValueError("All C-MAPSS columns must be numeric.")
    if data.isna().any().any():
        missing_columns = data.columns[data.isna().any()].tolist()
        raise ValueError(f"C-MAPSS data contains missing values in {missing_columns}.")
    if data.duplicated().any():
        raise ValueError("C-MAPSS data contains duplicate rows.")

    for column in ("engine_id", "cycle"):
        _require_integer_values(data[column], column)
        if (data[column] < 1).any():
            raise ValueError(f"{column} values must be positive.")

    if not data["engine_id"].is_monotonic_increasing:
        raise ValueError("Rows must be ordered by engine_id.")

    cycle_step = data.groupby("engine_id", sort=False)["cycle"].diff()
    first_cycles = data.groupby("engine_id", sort=False)["cycle"].first()
    if not first_cycles.eq(1).all() or not cycle_step.dropna().eq(1).all():
        raise ValueError(
            "Each engine must start at cycle 1 and have contiguous, ascending cycles."
        )


def _load_measurements(path: str | Path, *, validate: bool) -> pd.DataFrame:
    data = pd.read_csv(path, sep=r"\s+", header=None)
    if data.shape[1] != len(COLUMN_NAMES):
        raise ValueError(
            f"Expected {len(COLUMN_NAMES)} columns in {path}, found {data.shape[1]}."
        )
    data.columns = COLUMN_NAMES
    if validate:
        validate_cmapss_data(data)
    return data


def load_train_data(path: str | Path, *, validate: bool = True) -> pd.DataFrame:
    """Load a run-to-failure C-MAPSS training file."""
    return _load_measurements(path, validate=validate)


def load_test_data(path: str | Path, *, validate: bool = True) -> pd.DataFrame:
    """Load a truncated C-MAPSS test file."""
    return _load_measurements(path, validate=validate)


def load_rul_labels(path: str | Path) -> pd.Series:
    """Load official terminal RUL labels in their test-engine order."""
    labels = pd.read_csv(path, sep=r"\s+", header=None)
    if labels.shape[1] != 1:
        raise ValueError(f"Expected one RUL-label column in {path}, found {labels.shape[1]}.")

    rul = labels.iloc[:, 0].rename("RUL")
    if rul.empty:
        raise ValueError("RUL labels must contain at least one value.")
    if not is_numeric_dtype(rul) or rul.isna().any():
        raise ValueError("RUL labels must be numeric and non-missing.")
    _require_integer_values(rul, "RUL labels")
    if (rul < 0).any():
        raise ValueError("RUL labels must be non-negative.")
    return rul.astype("int64")


def add_training_rul(data: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of training data with the run-to-failure RUL target.

    The input is never modified. This target is valid only for the training
    set, where every engine's observed final cycle is its failure cycle.
    """
    if "RUL" in data.columns:
        raise ValueError("Input data already has an RUL column.")
    validate_cmapss_data(data)

    result = data.copy()
    result["RUL"] = result.groupby("engine_id")["cycle"].transform("max") - result["cycle"]
    return result


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    data = load_train_data(project_root / "data" / "raw" / "train_FD001.txt")
    labeled_data = add_training_rul(data)
    print("Dataset shape:", data.shape)
    print("Engines:", data["engine_id"].nunique())
    print("Training RUL range:", f"{labeled_data['RUL'].min()} to {labeled_data['RUL'].max()}")
