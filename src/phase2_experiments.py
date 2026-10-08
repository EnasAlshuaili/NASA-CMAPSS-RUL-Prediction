"""Phase 2: controlled experiments to improve on the V1 RUL model.

Run from the project root:

    python -m src.phase2_experiments

Every experiment changes exactly one thing relative to V1 (the target or the
feature set) and keeps everything else identical: the same 80/20 engine-level
split, the same XGBoost hyperparameters, and predictions clipped at zero. The V1
model and its results are only read, never written.

Writes:
    results/phase2_experiments.json      every experiment, its metrics and the selection decision
    results/phase2_comparison.csv        one-row-per-experiment comparison table
    results/phase2_official_test_metrics.json   official FD001 test for V1 vs the selected candidate
    models/phase2_best_model.pkl          only if a candidate meaningfully beats V1
    models/phase2_model_metadata.json     only if a candidate meaningfully beats V1
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

from src.data_loader import add_training_rul, load_rul_labels, load_test_data, load_train_data
from src.evaluate_official_test import _by_band, _metrics
from src.model_service import FEATURE_NAMES, load_model_bundle


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

EXPERIMENTS_FILE = RESULTS_DIR / "phase2_experiments.json"
COMPARISON_FILE = RESULTS_DIR / "phase2_comparison.csv"
OFFICIAL_FILE = RESULTS_DIR / "phase2_official_test_metrics.json"
CANDIDATE_MODEL_FILE = MODELS_DIR / "phase2_best_model.pkl"
CANDIDATE_METADATA_FILE = MODELS_DIR / "phase2_model_metadata.json"

RUL_CAPS = (125, 100, 75)
ROLLING_WINDOWS = (5, 10)
TREND_WINDOW = 10
N_TEMPORAL_SENSORS = 6
CV_FOLDS = 5

# Decision rules, fixed before any experiment was run.
# A change is "useful" if it lowers hold-out MAE and RMSE versus V1 and also lowers CV MAE.
# A candidate replaces nothing automatically; it is "meaningful" only if hold-out MAE and RMSE
# both improve by at least this fraction and R² does not drop.
MEANINGFUL_IMPROVEMENT = 0.03


@dataclass(frozen=True)
class Experiment:
    name: str
    rolling: bool = False
    trend: bool = False
    cap: int | None = None
    notes: str = ""


# ---------------------------------------------------------------------------
# Data and features
# ---------------------------------------------------------------------------

def load_v1_reference():
    """V1 model bundle plus the exact engine split recorded when V1 was trained."""
    bundle = load_model_bundle(MODELS_DIR)
    validation = bundle.metadata["validation"]
    return bundle, list(validation["train_engines"]), list(validation["validation_engines"])


def v1_model() -> XGBRegressor:
    """An unfitted XGBoost with exactly the V1 hyperparameters."""
    bundle = load_model_bundle(MODELS_DIR)
    return XGBRegressor(objective="reg:squarederror", random_state=42, n_jobs=-1,
                        **bundle.metadata["hyperparameters"])


def select_temporal_sensors(train: pd.DataFrame, n: int = N_TEMPORAL_SENSORS) -> list[str]:
    """Sensors most correlated with RUL, ranked on the TRAINING engines only.

    Temporal features multiply the feature count, so they are built for a small,
    informative subset instead of all 15 sensors.
    """
    corr = train[FEATURE_NAMES].corrwith(train["RUL"]).abs().sort_values(ascending=False)
    return list(corr.index[:n])


def _check_ordered(data: pd.DataFrame) -> None:
    # Rolling/shift operations assume each engine's rows are contiguous and in cycle order.
    if not data["engine_id"].is_monotonic_increasing or not data.groupby("engine_id")["cycle"].diff().dropna().eq(1).all():
        raise ValueError("Data must be ordered by engine_id and contiguous cycle.")


def add_rolling_features(data: pd.DataFrame, sensors: list[str], windows=ROLLING_WINDOWS) -> pd.DataFrame:
    """Trailing rolling mean / std per engine (current and past cycles only).

    Rationale: single-cycle readings are noisy; smoothing over recent cycles should
    reduce cycle-to-cycle prediction jitter. min_periods=1 means early cycles use
    whatever history exists; ddof=0 gives std 0 for a single observation.
    """
    _check_ordered(data)
    out = data.copy()
    groups = out.groupby("engine_id", sort=False)
    for sensor in sensors:
        for window in windows:
            rolling = groups[sensor].rolling(window, min_periods=1)
            out[f"{sensor}_mean_{window}"] = rolling.mean().reset_index(level=0, drop=True)
            out[f"{sensor}_std_{window}"] = rolling.std(ddof=0).reset_index(level=0, drop=True)
    return out


def add_trend_features(data: pd.DataFrame, sensors: list[str], window: int = TREND_WINDOW) -> pd.DataFrame:
    """Per-engine trend features using current and past cycles only.

    Rationale: degradation shows up as drift, so the direction and speed of recent
    change may say more about remaining life than the current level alone.
      diff_1      change since the previous cycle (0 on an engine's first cycle)
      slope_w     least-squares slope over the last w cycles (0 with a single point)
      change_w    change over the last w cycles (since cycle 1 while history is shorter)
    """
    _check_ordered(data)
    out = data.copy()
    groups = out.groupby("engine_id", sort=False)
    t = out["cycle"].astype(float)
    for sensor in sensors:
        x = out[sensor].astype(float)
        out[f"{sensor}_diff_1"] = groups[sensor].diff().fillna(0.0)

        helper = pd.DataFrame({"engine_id": out["engine_id"], "t": t, "x": x, "tx": t * x, "tt": t * t})
        means = (helper.groupby("engine_id", sort=False)[["t", "x", "tx", "tt"]]
                 .rolling(window, min_periods=1).mean().reset_index(level=0, drop=True))
        cov = means["tx"] - means["t"] * means["x"]
        var = means["tt"] - means["t"] ** 2
        out[f"{sensor}_slope_{window}"] = np.where(var > 1e-9, cov / var.where(var > 1e-9, 1.0), 0.0)

        past = groups[sensor].shift(window).fillna(groups[sensor].transform("first"))
        out[f"{sensor}_change_{window}"] = x - past
    return out


def build_features(data: pd.DataFrame, experiment: Experiment, sensors: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Return (frame with feature columns, feature column names) for an experiment."""
    frame, columns = data, list(FEATURE_NAMES)
    if experiment.rolling:
        frame = add_rolling_features(frame, sensors)
        columns += [f"{s}_{stat}_{w}" for s in sensors for w in ROLLING_WINDOWS for stat in ("mean", "std")]
    if experiment.trend:
        frame = add_trend_features(frame, sensors)
        columns += [f"{s}_{kind}" for s in sensors for kind in ("diff_1", f"slope_{TREND_WINDOW}", f"change_{TREND_WINDOW}")]
    return frame, columns


def training_target(rul: pd.Series, cap: int | None) -> pd.Series:
    """Training target. Capping applies to TRAINING labels only; evaluation always uses true RUL."""
    return rul if cap is None else rul.clip(upper=cap)


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

def _scores(y_true, y_pred) -> dict:
    return {
        "MAE": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "R2": round(float(r2_score(y_true, y_pred)), 4),
    }


def _prediction_frame(actual, predicted) -> pd.DataFrame:
    frame = pd.DataFrame({"actual_rul": np.asarray(actual, dtype=float), "predicted_rul": np.asarray(predicted, dtype=float)})
    frame["error"] = frame["predicted_rul"] - frame["actual_rul"]
    frame["absolute_error"] = frame["error"].abs()
    return frame


def fit_predict(train: pd.DataFrame, evaluate: pd.DataFrame, columns: list[str], cap: int | None):
    model = v1_model().fit(train[columns], training_target(train["RUL"], cap))
    return model, np.clip(model.predict(evaluate[columns]), 0, None)


def run_experiment(experiment: Experiment, data: pd.DataFrame, train_engines, validation_engines, sensors) -> dict:
    """Hold-out score on the V1 validation engines plus GroupKFold CV on the training engines."""
    frame, columns = build_features(data, experiment, sensors)
    train = frame[frame["engine_id"].isin(train_engines)]
    validation = frame[frame["engine_id"].isin(validation_engines)]

    _, predictions = fit_predict(train, validation, columns, experiment.cap)
    holdout = _prediction_frame(validation["RUL"], predictions)

    fold_scores = []
    for fold_train, fold_eval in GroupKFold(n_splits=CV_FOLDS).split(train, groups=train["engine_id"]):
        _, fold_pred = fit_predict(train.iloc[fold_train], train.iloc[fold_eval], columns, experiment.cap)
        fold_scores.append(_scores(train.iloc[fold_eval]["RUL"], fold_pred))
    cv = {k: round(float(np.mean([s[k] for s in fold_scores])), 4) for k in ("MAE", "RMSE", "R2")}

    return {
        **asdict(experiment),
        "features": len(columns),
        "validation": {**_scores(holdout["actual_rul"], holdout["predicted_rul"]),
                       "median_absolute_error": round(float(holdout["absolute_error"].median()), 4),
                       "mean_error_bias": round(float(holdout["error"].mean()), 4)},
        "validation_by_rul_band": _by_band(holdout),
        "cv_train_engines": cv,
    }


def improves(result: dict, reference: dict) -> bool:
    """Useful: lower hold-out MAE and RMSE than the reference, confirmed by lower CV MAE."""
    v, r = result["validation"], reference["validation"]
    return v["MAE"] < r["MAE"] and v["RMSE"] < r["RMSE"] and result["cv_train_engines"]["MAE"] < reference["cv_train_engines"]["MAE"]


def is_meaningful(result: dict, reference: dict) -> bool:
    v, r = result["validation"], reference["validation"]
    return (v["MAE"] <= r["MAE"] * (1 - MEANINGFUL_IMPROVEMENT)
            and v["RMSE"] <= r["RMSE"] * (1 - MEANINGFUL_IMPROVEMENT)
            and v["R2"] >= r["R2"])


# ---------------------------------------------------------------------------
# Official test (same protocol as src/evaluate_official_test.py)
# ---------------------------------------------------------------------------

def official_test_frame() -> pd.DataFrame:
    """Test rows with true RUL, built exactly as in src/evaluate_official_test.evaluate()."""
    test = load_test_data(RAW_DATA / "test_FD001.txt")
    labels = load_rul_labels(RAW_DATA / "RUL_FD001.txt")
    engine_ids = test["engine_id"].unique()
    if list(engine_ids) != list(range(1, len(engine_ids) + 1)) or len(engine_ids) != len(labels):
        raise ValueError("Test engines and RUL labels are not aligned.")
    label_by_engine = pd.Series(labels.values, index=engine_ids)
    last_cycle = test.groupby("engine_id")["cycle"].max()
    test["actual_rul"] = test["engine_id"].map(label_by_engine) + (test["engine_id"].map(last_cycle) - test["cycle"])
    test["is_last_cycle"] = test["cycle"].eq(test["engine_id"].map(last_cycle))
    return test


def official_test_scores(model, experiment: Experiment, sensors: list[str]) -> dict:
    """Official benchmark (last observed cycle per engine) plus all-rows view for one model."""
    test = official_test_frame()
    frame, columns = build_features(test, experiment, sensors)
    frame["predicted_rul"] = np.clip(model.predict(frame[columns]), 0, None).astype(float)
    frame["error"] = frame["predicted_rul"] - frame["actual_rul"]
    frame["absolute_error"] = frame["error"].abs()
    last = frame[frame["is_last_cycle"]]
    return {
        "official_last_cycle": {"metrics": _metrics(last), "error_by_true_rul": _by_band(last)},
        "all_test_rows": {"metrics": _metrics(frame), "error_by_true_rul": _by_band(frame)},
    }


# ---------------------------------------------------------------------------
# Experiment plan
# ---------------------------------------------------------------------------

def run_all() -> dict:
    bundle, train_engines, validation_engines = load_v1_reference()
    data = add_training_rul(load_train_data(RAW_DATA / "train_FD001.txt"))
    sensors = select_temporal_sensors(data[data["engine_id"].isin(train_engines)])

    # Experiment 0: the saved V1 model must still score exactly as recorded, and refitting
    # it with this module's training function must reproduce it (proves the harness = V1).
    validation = data[data["engine_id"].isin(validation_engines)]
    saved_scores = _scores(validation["RUL"], np.clip(bundle.model.predict(validation[FEATURE_NAMES]), 0, None))
    baseline = run_experiment(Experiment("V1 Baseline", notes="Saved production model's configuration, refit"),
                              data, train_engines, validation_engines, sensors)
    reproduction = {
        "saved_v1_model_validation": saved_scores,
        "recorded_v1_metrics": bundle.metrics,
        "refit_matches_saved": {k: baseline["validation"][k] for k in ("MAE", "RMSE", "R2")} == saved_scores,
        "saved_matches_recorded": saved_scores == bundle.metrics,
    }

    results = [baseline]

    # Experiment 1: capping the training target. Early-life sensors barely move, so asking the
    # model to tell RUL 250 from RUL 150 adds noise; a cap teaches it "healthy" instead.
    caps = [run_experiment(Experiment(f"RUL Cap {c}", cap=c, notes="Training labels capped; validation labels are true RUL"),
                           data, train_engines, validation_engines, sensors) for c in RUL_CAPS]
    results += caps

    # Experiment 2: rolling mean/std to smooth single-cycle sensor noise.
    rolling = run_experiment(Experiment("Rolling Features", rolling=True,
                                        notes=f"Trailing mean/std, windows {list(ROLLING_WINDOWS)}, per engine"),
                             data, train_engines, validation_engines, sensors)
    results.append(rolling)

    # Experiment 3: trend features to capture the direction and rate of degradation.
    trend = run_experiment(Experiment("Trend Features", trend=True,
                                      notes=f"diff_1, slope_{TREND_WINDOW}, change_{TREND_WINDOW}, per engine"),
                           data, train_engines, validation_engines, sensors)
    results.append(trend)

    # Experiment 4: combine only what improved on its own.
    useful_caps = [r for r in caps if improves(r, baseline)]
    best_cap = min(useful_caps, key=lambda r: r["validation"]["MAE"])["cap"] if useful_caps else None
    use_rolling, use_trend = improves(rolling, baseline), improves(trend, baseline)
    combined_parts = (sum([use_rolling, use_trend]) + (best_cap is not None))
    if combined_parts >= 2:
        results.append(run_experiment(
            Experiment("Best Combined Model", rolling=use_rolling, trend=use_trend, cap=best_cap,
                       notes="Only components that improved individually"),
            data, train_engines, validation_engines, sensors))

    # Selection: the best experiment that is useful versus V1; flagged meaningful or not.
    contenders = [r for r in results[1:] if improves(r, baseline)]
    best = min(contenders, key=lambda r: (r["validation"]["MAE"], r["validation"]["RMSE"])) if contenders else None
    for r in results:
        r["improves_on_v1"] = r is not baseline and improves(r, baseline)
    decision = {
        "rules": {
            "useful": "lower hold-out MAE and RMSE than V1, and lower GroupKFold CV MAE",
            "meaningful": f"hold-out MAE and RMSE each at least {MEANINGFUL_IMPROVEMENT:.0%} lower than V1, R2 not lower",
        },
        "temporal_sensors": sensors,
        "components_used_in_combined": {"best_cap": best_cap, "rolling": use_rolling, "trend": use_trend},
        "best_experiment": best["name"] if best else None,
        "meaningful_improvement": bool(best and is_meaningful(best, baseline)),
    }
    return {"experiment_0_reproduction": reproduction, "experiments": results, "decision": decision,
            "split": {"train_engines": train_engines, "validation_engines": validation_engines}}


def comparison_table(report: dict) -> pd.DataFrame:
    rows = []
    for r in report["experiments"]:
        parts = ["15 raw sensors"] + (["rolling mean/std"] if r["rolling"] else []) + (["trends"] if r["trend"] else [])
        rows.append({
            "Model / Experiment": r["name"], "Features": f"{' + '.join(parts)} ({r['features']})",
            "RUL Cap": r["cap"] if r["cap"] else "none",
            "MAE": r["validation"]["MAE"], "RMSE": r["validation"]["RMSE"], "R2": r["validation"]["R2"],
            "Bias": r["validation"]["mean_error_bias"],
            "CV MAE": r["cv_train_engines"]["MAE"], "CV RMSE": r["cv_train_engines"]["RMSE"],
            "Improves on V1": r["improves_on_v1"], "Notes": r["notes"],
        })
    return pd.DataFrame(rows)


def main() -> None:
    report = run_all()
    RESULTS_DIR.mkdir(exist_ok=True)
    comparison_table(report).to_csv(COMPARISON_FILE, index=False)

    decision = report["decision"]
    best = next((r for r in report["experiments"] if r["name"] == decision["best_experiment"]), None)
    if best is not None:
        # The official test is touched only now, after selection, and never feeds back into it.
        _, train_engines, _ = load_v1_reference()
        data = add_training_rul(load_train_data(RAW_DATA / "train_FD001.txt"))
        experiment = Experiment(best["name"], rolling=best["rolling"], trend=best["trend"], cap=best["cap"])
        sensors = decision["temporal_sensors"]
        frame, columns = build_features(data, experiment, sensors)
        train = frame[frame["engine_id"].isin(train_engines)]
        model = v1_model().fit(train[columns], training_target(train["RUL"], experiment.cap))

        v1_bundle = load_model_bundle(MODELS_DIR)
        official = {
            "protocol": "Same as src/evaluate_official_test.py: RUL_FD001 label at each test engine's last cycle; "
                        "all-rows view uses label + (last cycle - cycle). Used only after model selection.",
            "v1": official_test_scores(v1_bundle.model, Experiment("V1 Baseline"), sensors),
            "phase2_candidate": {"name": best["name"],
                                 **official_test_scores(model, experiment, sensors)},
        }
        OFFICIAL_FILE.write_text(json.dumps(official, indent=2), encoding="utf-8")
        report["official_test_file"] = OFFICIAL_FILE.name

        if decision["meaningful_improvement"]:
            joblib.dump(model, CANDIDATE_MODEL_FILE)
            CANDIDATE_METADATA_FILE.write_text(json.dumps({
                "model_file": CANDIDATE_MODEL_FILE.name,
                "model_sha256": hashlib.sha256(CANDIDATE_MODEL_FILE.read_bytes()).hexdigest(),
                "status": "Phase 2 candidate; NOT the production model",
                "experiment": best["name"], "features": columns, "temporal_sensors": sensors,
                "rul_cap": experiment.cap, "hyperparameters": v1_bundle.metadata["hyperparameters"],
                "validation": best["validation"], "cv_train_engines": best["cv_train_engines"],
            }, indent=2), encoding="utf-8")
            report["candidate_model_file"] = CANDIDATE_MODEL_FILE.name

    EXPERIMENTS_FILE.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(comparison_table(report).drop(columns=["Notes"]).to_string(index=False))
    print("\nDecision:", json.dumps(decision, indent=2))
    print("Experiment 0:", json.dumps(report["experiment_0_reproduction"]))


if __name__ == "__main__":
    main()
