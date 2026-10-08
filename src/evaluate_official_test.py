"""Evaluate the served RUL model on the official NASA C-MAPSS FD001 test set.

Run from the project root:

    python -m src.evaluate_official_test

Read-only with respect to the model, data, and app. Writes only:
    results/official_test_metrics.json
    results/official_test_predictions.csv
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data_loader import load_rul_labels, load_test_data, load_train_data
from src.model_service import FEATURE_NAMES, load_model_bundle


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_FILE = RESULTS_DIR / "official_test_metrics.json"
PREDICTIONS_FILE = RESULTS_DIR / "official_test_predictions.csv"

BANDS = [-np.inf, 25, 50, 100, 150, 200, np.inf]
BAND_LABELS = ["<=25", "25-50", "50-100", "100-150", "150-200", ">200"]
PREDICTION_COLUMNS = ["engine_id", "cycle", "actual_rul", "predicted_rul", "absolute_error", "is_last_cycle"]


def _metrics(frame: pd.DataFrame) -> dict:
    y, p, ae = frame["actual_rul"], frame["predicted_rul"], frame["absolute_error"]
    return {
        "n": int(len(frame)),
        "MAE": round(float(mean_absolute_error(y, p)), 4),
        "RMSE": round(float(np.sqrt(mean_squared_error(y, p))), 4),
        "R2": round(float(r2_score(y, p)), 4),
        "median_absolute_error": round(float(ae.median()), 4),
        "min_absolute_error": round(float(ae.min()), 4),
        "max_absolute_error": round(float(ae.max()), 4),
        "mean_error_bias": round(float(frame["error"].mean()), 4),
    }


def _by_band(frame: pd.DataFrame) -> list[dict]:
    band = pd.cut(frame["actual_rul"], bins=BANDS, labels=BAND_LABELS)
    rows = []
    for label, group in frame.groupby(band, observed=False):
        empty = group.empty
        rows.append({
            "band": str(label),
            "n": int(len(group)),
            "MAE": None if empty else round(float(group["absolute_error"].mean()), 4),
            "RMSE": None if empty else round(float(np.sqrt((group["error"] ** 2).mean())), 4),
            "mean_error_bias": None if empty else round(float(group["error"].mean()), 4),
        })
    return rows


def evaluate() -> tuple[dict, pd.DataFrame]:
    """Score the served model on test_FD001; return (metrics dict, per-row predictions)."""
    # Exact model served by the app (fingerprint, type, and 15-feature order are verified).
    bundle = load_model_bundle(MODELS_DIR)
    model = bundle.model

    test = load_test_data(RAW_DATA / "test_FD001.txt")
    labels = load_rul_labels(RAW_DATA / "RUL_FD001.txt")

    # Label row i belongs to test engine i (engine ids 1..N in file order).
    engine_ids = test["engine_id"].unique()
    if list(engine_ids) != list(range(1, len(engine_ids) + 1)):
        raise ValueError("Test engine ids are not 1..N in file order.")
    if len(engine_ids) != len(labels):
        raise ValueError(f"{len(engine_ids)} test engines vs {len(labels)} RUL labels.")
    label_by_engine = pd.Series(labels.values, index=engine_ids)

    last_cycle = test.groupby("engine_id")["cycle"].max()
    implied_life = last_cycle + label_by_engine
    train_life = load_train_data(RAW_DATA / "train_FD001.txt").groupby("engine_id")["cycle"].max()

    # RUL_FD001.txt gives the RUL at each engine's LAST observed cycle; earlier cycles
    # add the cycles remaining until that last observation.
    test["actual_rul"] = test["engine_id"].map(label_by_engine) + (test["engine_id"].map(last_cycle) - test["cycle"])
    test["is_last_cycle"] = test["cycle"].eq(test["engine_id"].map(last_cycle))
    if not (test.loc[test["is_last_cycle"], "actual_rul"].values == labels.values).all():
        raise ValueError("Last-cycle RUL does not match RUL_FD001.txt.")

    # Raw features in the verified order (no scaler), clipped at 0 exactly as in the app.
    test["predicted_rul"] = np.clip(model.predict(test[FEATURE_NAMES]), 0, None).astype(float)
    test["error"] = test["predicted_rul"] - test["actual_rul"]
    test["absolute_error"] = test["error"].abs()

    last = test[test["is_last_cycle"]]
    result = {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": {
            "file": bundle.model_path.name,
            "sha256": bundle.metadata["model_sha256"],
            "name": bundle.metadata["model_name"],
            "hyperparameters": bundle.metadata["hyperparameters"],
            "features": FEATURE_NAMES,
            "input_scaling": "none (raw sensor values)",
            "predictions_clipped_at_zero": True,
        },
        "data": {
            "test_file": "data/raw/test_FD001.txt",
            "label_file": "data/raw/RUL_FD001.txt",
            "test_engines": int(len(engine_ids)),
            "test_rows": int(len(test)),
            "rul_labels": int(len(labels)),
            "alignment": "RUL_FD001.txt row i = RUL at the last observed cycle of test engine i "
                         "(engine ids 1..100, verified)",
            "observed_cycles_per_engine": {"min": int(last_cycle.min()), "max": int(last_cycle.max())},
            "label_rul_range": {"min": int(labels.min()), "max": int(labels.max())},
            "implied_total_life": {"min": int(implied_life.min()), "max": int(implied_life.max())},
            "train_total_life": {"min": int(train_life.min()), "max": int(train_life.max())},
        },
        "official_last_cycle": {
            "description": "Standard FD001 benchmark: one prediction per test engine at its last observed cycle, "
                           "compared with RUL_FD001.txt.",
            "metrics": _metrics(last),
            "error_by_true_rul": _by_band(last),
        },
        "all_test_rows": {
            "description": "Every test row; actual RUL = label + (last observed cycle - cycle). Supplementary.",
            "metrics": _metrics(test),
            "error_by_true_rul": _by_band(test),
        },
        "validation_reference": bundle.metrics,
    }

    predictions = test[PREDICTION_COLUMNS].assign(
        predicted_rul=test["predicted_rul"].round(4),
        absolute_error=test["absolute_error"].round(4),
    )
    return result, predictions


def main() -> None:
    result, predictions = evaluate()
    RESULTS_DIR.mkdir(exist_ok=True)
    METRICS_FILE.write_text(json.dumps(result, indent=2), encoding="utf-8")
    predictions.to_csv(PREDICTIONS_FILE, index=False)

    for key in ("official_last_cycle", "all_test_rows"):
        print(f"== {key} ==")
        print(json.dumps(result[key]["metrics"], indent=2))
        print(pd.DataFrame(result[key]["error_by_true_rul"]).to_string(index=False), "\n")
    print(f"Saved {METRICS_FILE.relative_to(PROJECT_ROOT)} and {PREDICTIONS_FILE.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
