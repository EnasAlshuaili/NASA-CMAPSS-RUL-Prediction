"""Phase 2.5: does RUL capping help under a test-like (truncated-engine) validation?

Run from the project root:

    python -m src.phase2_5_test_like_validation

Why: V1 / Phase 2 validation scores every cycle of each validation engine's full life
(RUL up to ~360), while the official FD001 benchmark scores ONE prediction per engine at
the last cycle of a truncated trajectory (RUL <= ~150). This experiment re-creates that
setting on the held-out V1 validation engines and re-tests RUL capping there.

Test-like validation protocol (fixed before any result was computed):
  * Engines: the 20 V1 validation engines only; models train on the 80 V1 training engines.
  * Truncation: each validation engine is cut at fixed fractions of its observed life
    (60/70/80/90 %): truncation_cycle = round_half_up(fraction * max_cycle), kept in
    [1, max_cycle - 1]. The schedule is deterministic and never looks at predictions.
  * Observation: only rows with cycle <= truncation_cycle are passed to feature
    construction; the model predicts once, at the truncation cycle.
  * Label: max_cycle - truncation_cycle (the true, UNCAPPED RUL), used only for scoring.
  * Each engine contributes exactly one sample per fraction (4 per engine, 80 in total);
    the engine-level bootstrap below resamples engines, not samples, so repeated samples
    from one engine are never treated as independent.
  * Robustness: the same fitted models are also scored under two other fixed schedules
    (cuts at 30-90 % of life, and 5 seeded uniform cuts in 30-95 % of life per engine).

Read-only with respect to V1 and Phase 2 artifacts and the official test files.
Writes:
    results/phase2_5_test_like_validation.json
    results/phase2_5_comparison.csv
and, only if a cap clearly wins under test-like validation:
    models/phase2_5_candidate_model.pkl
    models/phase2_5_candidate_metadata.json
    results/phase2_5_official_test_metrics.json
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.data_loader import add_training_rul, load_train_data
from src.evaluate_official_test import _by_band, _metrics
from src.model_service import FEATURE_NAMES
from src.phase2_experiments import (
    CANDIDATE_METADATA_FILE as PHASE2_METADATA_FILE,
    CANDIDATE_MODEL_FILE as PHASE2_MODEL_FILE,
    MODELS_DIR,
    RAW_DATA,
    RESULTS_DIR,
    Experiment,
    _prediction_frame,
    _scores,
    build_features,
    fit_predict,
    load_v1_reference,
    official_test_frame,
    training_target,
    v1_model,
)


RESULTS_FILE = RESULTS_DIR / "phase2_5_test_like_validation.json"
COMPARISON_FILE = RESULTS_DIR / "phase2_5_comparison.csv"
OFFICIAL_FILE = RESULTS_DIR / "phase2_5_official_test_metrics.json"
CANDIDATE_MODEL_FILE = MODELS_DIR / "phase2_5_candidate_model.pkl"
CANDIDATE_METADATA_FILE = MODELS_DIR / "phase2_5_candidate_metadata.json"
CANDIDATE_STATUS = "Experimental candidate — NOT production"

TRUNCATION_FRACTIONS = (0.6, 0.7, 0.8, 0.9)
# Alternative truncation schedules (robustness only; models are not refit for them).
WIDE_FRACTIONS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
RANDOM_SCHEDULE = {"per_engine": 5, "low": 0.3, "high": 0.95, "seed": 42}
RUL_CAPS = (None, 150, 125, 100, 75)
CV_FOLDS = 5
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 42

BANDS = [-np.inf, 25, 50, 100, 150, np.inf]
BAND_LABELS = ["<=25", "25-50", "50-100", "100-150", ">150"]

# Selection rules, fixed before any experiment was run. A cap "clearly wins" over no cap
# only if ALL of these hold on test-like validation:
MIN_MAE_IMPROVEMENT = 0.03       # hold-out MAE at least 3 % lower
MIN_FRACTIONS_IMPROVED = 3       # MAE lower at >= 3 of the 4 truncation fractions
MAX_LOW_RUL_DEGRADATION = 2.0    # MAE for true RUL <= 25 at most 2 cycles worse
MAX_BAND_DEGRADATION = 2.0       # no RUL band (n >= MIN_BAND_SAMPLES) more than 2 cycles worse
MIN_BAND_SAMPLES = 5
# The first run used only the rules up to "low_rul_safety"; every cap passed and the tie-break
# chose Cap 75, whose MAE for true RUL 100-150 was worse than no cap (only 4 samples there,
# because the 60-90 % schedule tops out at RUL 108). Its auto-generated official-test output
# was deleted unread. The last two rules were added from validation evidence only, before
# the alternative schedules below were scored.
SELECTION_RULES = {
    "primary": "hold-out test-like MAE at least 3% lower than no cap",
    "secondary": "hold-out test-like RMSE lower than no cap",
    "supporting": "hold-out test-like R2 not lower than no cap",
    "robust_across_truncation": "MAE lower at >= 3 of 4 truncation fractions",
    "robust_across_engines": "engine-level bootstrap 95% CI of the MAE difference entirely below 0",
    "confirmed_on_training_engines": "test-like 5-fold GroupKFold CV MAE on the 80 training engines lower than no cap",
    "low_rul_safety": "MAE for true RUL <= 25 at most 2 cycles worse than no cap",
    "robust_to_truncation_schedule": "MAE lower than no cap under BOTH alternative schedules "
                                     "(fractions 30-90 %, and 5 seeded uniform cuts in 30-95 % of life)",
    "no_band_degradation": "pooled over all three schedules, no RUL band with >= 5 samples has MAE "
                           "more than 2 cycles worse than no cap",
    "tie_break": "lowest hold-out test-like MAE among caps that pass every rule",
}


def cap_label(cap: int | None) -> str:
    return "none" if cap is None else str(cap)


# ---------------------------------------------------------------------------
# Test-like truncation
# ---------------------------------------------------------------------------

def truncation_points(data: pd.DataFrame, engines, fractions=TRUNCATION_FRACTIONS) -> pd.DataFrame:
    """One deterministic truncation point per (engine, fraction).

    `fractions` is either one sequence used for every engine or a mapping engine -> sequence.
    The cut depends only on the engine's observed length and the fixed fraction, never on
    model output. Rounding is half-up, then clamped so at least one cycle is observed and
    at least one cycle remains (RUL >= 1, i.e. strictly before the final cycle).
    """
    life = data[data["engine_id"].isin(engines)].groupby("engine_id")["cycle"].max()
    rows = []
    for engine in sorted(int(e) for e in engines):
        max_cycle = int(life.loc[engine])
        for fraction in (fractions[engine] if isinstance(fractions, dict) else fractions):
            cut = int(np.floor(fraction * max_cycle + 0.5))
            cut = min(max(cut, 1), max_cycle - 1)
            rows.append({"engine_id": engine, "fraction": fraction, "max_cycle": max_cycle,
                         "truncation_cycle": cut, "true_rul": max_cycle - cut})
    return pd.DataFrame(rows)


def random_fractions(engines, per_engine=RANDOM_SCHEDULE["per_engine"], low=RANDOM_SCHEDULE["low"],
                     high=RANDOM_SCHEDULE["high"], seed=RANDOM_SCHEDULE["seed"]) -> dict:
    """Seeded uniform lifetime fractions, drawn in sorted engine order (reproducible)."""
    rng = np.random.default_rng(seed)
    return {e: tuple(round(float(f), 4) for f in rng.uniform(low, high, per_engine))
            for e in sorted(int(e) for e in engines)}


def schedules(data: pd.DataFrame, engines) -> dict[str, pd.DataFrame]:
    return {
        "fractions_60_90": truncation_points(data, engines, TRUNCATION_FRACTIONS),
        "fractions_30_90": truncation_points(data, engines, WIDE_FRACTIONS),
        "uniform_30_95_seed42": truncation_points(data, engines, random_fractions(engines)),
    }


def observed_history(data: pd.DataFrame, engine: int, truncation_cycle: int) -> pd.DataFrame:
    """Telemetry a test-like observer would have: one engine, cycles <= truncation, no RUL column."""
    history = data[(data["engine_id"] == engine) & (data["cycle"] <= truncation_cycle)]
    return history.drop(columns=["RUL"], errors="ignore")


def test_like_features(data: pd.DataFrame, points: pd.DataFrame, experiment: Experiment,
                       sensors: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Feature row at each truncation cycle, built from that engine's observed history only."""
    rows, columns = [], None
    for point in points.itertuples(index=False):
        frame, columns = build_features(observed_history(data, point.engine_id, point.truncation_cycle),
                                        experiment, sensors)
        last = frame.iloc[[-1]]
        if int(last["cycle"].iloc[0]) != point.truncation_cycle:
            raise ValueError("Feature row is not at the truncation cycle.")
        rows.append(last)
    features = pd.concat(rows, ignore_index=True)
    return features, columns


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _band_rows(frame: pd.DataFrame) -> list[dict]:
    band = pd.cut(frame["actual_rul"], bins=BANDS, labels=BAND_LABELS)
    rows = []
    for label, group in frame.groupby(band, observed=False):
        empty = group.empty
        rows.append({
            "band": str(label), "n": int(len(group)),
            "MAE": None if empty else round(float(group["absolute_error"].mean()), 4),
            "RMSE": None if empty else round(float(np.sqrt((group["error"] ** 2).mean())), 4),
            "mean_error_bias": None if empty else round(float(group["error"].mean()), 4),
        })
    return rows


def overall_scores(samples: pd.DataFrame) -> dict:
    return {"n_samples": int(len(samples)), "n_engines": int(samples["engine_id"].nunique()),
            **_scores(samples["actual_rul"], samples["predicted_rul"]),
            "median_absolute_error": round(float(samples["absolute_error"].median()), 4),
            "mean_error_bias": round(float(samples["error"].mean()), 4)}


def summarize(samples: pd.DataFrame) -> dict:
    """Overall, per-fraction, per-band and per-engine scores for one model's test-like predictions."""
    overall = overall_scores(samples)
    by_fraction = [{"fraction": float(f), "n": int(len(g)),
                    **_scores(g["actual_rul"], g["predicted_rul"]),
                    "mean_error_bias": round(float(g["error"].mean()), 4),
                    "true_rul_range": [int(g["actual_rul"].min()), int(g["actual_rul"].max())]}
                   for f, g in samples.groupby("fraction")]
    by_engine = [{"engine_id": int(e), "n": int(len(g)),
                  "MAE": round(float(g["absolute_error"].mean()), 4),
                  "mean_error_bias": round(float(g["error"].mean()), 4)}
                 for e, g in samples.groupby("engine_id")]
    return {"overall": overall, "by_truncation_fraction": by_fraction,
            "by_rul_band": _band_rows(samples), "by_engine": by_engine}


def score_samples(points: pd.DataFrame, predictions) -> pd.DataFrame:
    samples = pd.concat([points.reset_index(drop=True),
                         _prediction_frame(points["true_rul"], predictions)], axis=1)
    return samples


def engine_bootstrap(candidate: pd.DataFrame, reference: pd.DataFrame,
                     n: int = BOOTSTRAP_SAMPLES, seed: int = BOOTSTRAP_SEED) -> dict:
    """Paired, engine-level bootstrap of MAE(candidate) - MAE(reference).

    Engines are resampled with replacement and all of a drawn engine's samples come with it,
    so the 4 correlated samples per engine are never counted as independent evidence.
    """
    key = ["engine_id", "fraction"]
    paired = candidate[key + ["absolute_error"]].merge(reference[key + ["absolute_error"]],
                                                       on=key, suffixes=("_c", "_r"), validate="one_to_one")
    per_engine = paired.groupby("engine_id")[["absolute_error_c", "absolute_error_r"]].agg(["sum", "count"])
    sums_c = per_engine[("absolute_error_c", "sum")].to_numpy()
    sums_r = per_engine[("absolute_error_r", "sum")].to_numpy()
    counts = per_engine[("absolute_error_c", "count")].to_numpy()
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(counts), size=(n, len(counts)))
    diffs = (sums_c[draws].sum(axis=1) - sums_r[draws].sum(axis=1)) / counts[draws].sum(axis=1)
    observed = float((sums_c.sum() - sums_r.sum()) / counts.sum())
    return {"mae_difference": round(observed, 4),
            "ci95": [round(float(np.percentile(diffs, 2.5)), 4), round(float(np.percentile(diffs, 97.5)), 4)],
            "share_of_resamples_improved": round(float((diffs < 0).mean()), 4),
            "resamples": n, "seed": seed}


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

V1_FEATURES = Experiment("V1 features")  # raw 15 sensors, no temporal features


def holdout_predictions(data, train_engines, points, cap):
    """Fit on the full trajectories of the training engines; predict at each truncation point."""
    train = data[data["engine_id"].isin(train_engines)]
    features, columns = test_like_features(data, points, V1_FEATURES, [])
    model, predictions = fit_predict(train, features, columns, cap)
    return model, score_samples(points, predictions)


def predict_points(model, data, points) -> pd.DataFrame:
    features, columns = test_like_features(data, points, V1_FEATURES, [])
    return score_samples(points, np.clip(model.predict(features[columns]), 0, None))


def cv_predictions(data, train_engines, cap) -> pd.DataFrame:
    """Test-like GroupKFold on the 80 training engines (fold engines are truncated like validation)."""
    train = data[data["engine_id"].isin(train_engines)].reset_index(drop=True)
    pooled = []
    for fold_train, fold_eval in GroupKFold(n_splits=CV_FOLDS).split(train, groups=train["engine_id"]):
        eval_engines = train.iloc[fold_eval]["engine_id"].unique()
        points = truncation_points(train, eval_engines)
        features, columns = test_like_features(train, points, V1_FEATURES, [])
        _, predictions = fit_predict(train.iloc[fold_train], features, columns, cap)
        pooled.append(score_samples(points, predictions))
    return pd.concat(pooled, ignore_index=True).sort_values(["engine_id", "fraction"], ignore_index=True)


def phase2_reference(data, points) -> tuple[pd.DataFrame, dict]:
    """Saved Phase 2 candidate under the same protocol (reference only; never a contender here)."""
    metadata = json.loads(PHASE2_METADATA_FILE.read_text(encoding="utf-8"))
    if hashlib.sha256(PHASE2_MODEL_FILE.read_bytes()).hexdigest() != metadata["model_sha256"]:
        raise ValueError("Phase 2 candidate model does not match its recorded fingerprint.")
    model = joblib.load(PHASE2_MODEL_FILE)
    experiment = Experiment(metadata["experiment"], rolling=True, trend=True, cap=metadata["rul_cap"])
    features, columns = test_like_features(data, points, experiment, metadata["temporal_sensors"])
    if columns != metadata["features"]:
        raise ValueError("Phase 2 feature columns do not match its metadata.")
    predictions = np.clip(model.predict(features[columns]), 0, None)
    return score_samples(points, predictions), metadata


def clearly_wins(result: dict, reference: dict) -> dict:
    """Evaluate every pre-registered selection rule for one cap against no cap."""
    h, r = result["holdout"]["overall"], reference["holdout"]["overall"]
    fractions_improved = sum(c["MAE"] < b["MAE"] for c, b in zip(result["holdout"]["by_truncation_fraction"],
                                                                  reference["holdout"]["by_truncation_fraction"]))
    low_c = result["holdout"]["by_rul_band"][0]["MAE"]
    low_r = reference["holdout"]["by_rul_band"][0]["MAE"]
    alternatives = [s for s in result["schedules"] if s != "fractions_60_90"]
    band_pairs = zip(result["pooled_schedules_by_rul_band"], reference["pooled_schedules_by_rul_band"])
    checks = {
        "primary": h["MAE"] <= r["MAE"] * (1 - MIN_MAE_IMPROVEMENT),
        "secondary": h["RMSE"] < r["RMSE"],
        "supporting": h["R2"] >= r["R2"],
        "robust_across_truncation": fractions_improved >= MIN_FRACTIONS_IMPROVED,
        "robust_across_engines": result["bootstrap_vs_no_cap"]["ci95"][1] < 0,
        "confirmed_on_training_engines": result["cv_train_engines"]["overall"]["MAE"]
                                         < reference["cv_train_engines"]["overall"]["MAE"],
        "low_rul_safety": low_c is not None and low_r is not None and low_c <= low_r + MAX_LOW_RUL_DEGRADATION,
        "robust_to_truncation_schedule": all(result["schedules"][s]["MAE"] < reference["schedules"][s]["MAE"]
                                             for s in alternatives),
        "no_band_degradation": all(c["MAE"] <= r["MAE"] + MAX_BAND_DEGRADATION
                                   for c, r in band_pairs if r["n"] >= MIN_BAND_SAMPLES),
    }
    return {**{k: bool(v) for k, v in checks.items()}, "fractions_improved": int(fractions_improved),
            "clearly_wins": bool(all(checks.values()))}


def run_all() -> tuple[dict, dict]:
    bundle, train_engines, validation_engines = load_v1_reference()
    data = add_training_rul(load_train_data(RAW_DATA / "train_FD001.txt"))
    points = truncation_points(data, validation_engines)
    schedule_points = schedules(data, validation_engines)

    results, samples, models = [], {}, {}
    for cap in RUL_CAPS:
        model, holdout = holdout_predictions(data, train_engines, points, cap)
        cv = cv_predictions(data, train_engines, cap)
        name = "V1 / No Cap" if cap is None else f"Cap {cap}"
        samples[name], models[name] = holdout, model
        by_schedule = {s: predict_points(model, data, p) for s, p in schedule_points.items()}
        results.append({"name": name, "rul_cap": cap, "features": len(FEATURE_NAMES),
                        "holdout": summarize(holdout), "cv_train_engines": summarize(cv),
                        "schedules": {s: overall_scores(f) for s, f in by_schedule.items()},
                        "pooled_schedules_by_rul_band": _band_rows(pd.concat(by_schedule.values()))})

    # V1 reproduction: the saved production model gives the same test-like predictions as the refit.
    v1_features, _ = test_like_features(data, points, V1_FEATURES, [])
    saved = np.clip(bundle.model.predict(v1_features[FEATURE_NAMES]), 0, None)
    reproduction = {"saved_v1_matches_no_cap_refit":
                    bool(np.allclose(saved, samples["V1 / No Cap"]["predicted_rul"], rtol=0, atol=1e-4))}

    baseline = results[0]
    for result in results:
        result["bootstrap_vs_no_cap"] = engine_bootstrap(samples[result["name"]], samples[baseline["name"]])
    for result in results[1:]:
        result["selection_checks"] = clearly_wins(result, baseline)

    phase2_samples, phase2_meta = phase2_reference(data, points)
    phase2 = {"name": "Phase 2 candidate (reference)", "rul_cap": phase2_meta["rul_cap"],
              "features": len(phase2_meta["features"]), "holdout": summarize(phase2_samples),
              "bootstrap_vs_no_cap": engine_bootstrap(phase2_samples, samples[baseline["name"]]),
              "notes": "Saved Phase 2 model, features rebuilt from truncated history; reference only, not a contender"}

    winners = [r for r in results[1:] if r["selection_checks"]["clearly_wins"]]
    best = min(winners, key=lambda r: r["holdout"]["overall"]["MAE"]) if winners else None
    decision = {"rules": SELECTION_RULES, "caps_that_clearly_win": [r["name"] for r in winners],
                "selected_candidate": best["name"] if best else None,
                "official_test_justified": best is not None}

    report = {
        "protocol": {
            "validation_engines": validation_engines, "train_engines": train_engines,
            "truncation_fractions": list(TRUNCATION_FRACTIONS),
            "truncation_cycle": "round_half_up(fraction * max_cycle), clamped to [1, max_cycle - 1]",
            "label": "max_cycle - truncation_cycle (true, uncapped RUL; evaluation only)",
            "features": "built from the engine's own rows with cycle <= truncation_cycle; one prediction at the truncation cycle",
            "training": "full trajectories of the 80 V1 training engines, V1 hyperparameters; cap applied to training labels only",
            "samples": f"{len(points)} ({len(validation_engines)} engines x {len(TRUNCATION_FRACTIONS)} fractions)",
            "cv": f"{CV_FOLDS}-fold GroupKFold over the training engines, held-out fold engines truncated the same way",
            "bootstrap": f"paired, engine-level, {BOOTSTRAP_SAMPLES} resamples, seed {BOOTSTRAP_SEED}",
            "alternative_schedules": {"fractions_30_90": list(WIDE_FRACTIONS),
                                      "uniform_30_95_seed42": RANDOM_SCHEDULE,
                                      "use": "robustness checks only; same fitted models, no refit"},
            "rul_bands": BAND_LABELS,
        },
        "truncation_points": points.to_dict(orient="records"),
        "v1_reproduction": reproduction,
        "experiments": results,
        "phase2_reference": phase2,
        "decision": decision,
    }
    return report, {"models": models, "samples": samples, "data": data, "train_engines": train_engines}


def comparison_table(report: dict) -> pd.DataFrame:
    rows = []
    for r in report["experiments"] + [report["phase2_reference"]]:
        o, bands = r["holdout"]["overall"], {b["band"]: b for b in r["holdout"]["by_rul_band"]}
        checks = r.get("selection_checks")
        if checks is None:
            notes = r.get("notes", "Baseline: V1 configuration, refit (identical to the saved V1 model)")
        else:
            failed = [k for k, v in checks.items() if v is False]
            notes = "Clearly wins" if checks["clearly_wins"] else "Does not clearly win; fails: " + ", ".join(failed)
        rows.append({
            "Model": r["name"], "RUL Cap": cap_label(r["rul_cap"]),
            "Number of validation samples": o["n_samples"],
            "MAE": o["MAE"], "RMSE": o["RMSE"], "R2": o["R2"],
            "Median Absolute Error": o["median_absolute_error"], "Average Bias": o["mean_error_bias"],
            "MAE <=25": bands["<=25"]["MAE"], "MAE 25-50": bands["25-50"]["MAE"],
            "MAE 50-100": bands["50-100"]["MAE"], "MAE 100-150": bands["100-150"]["MAE"],
            "Bias 50-100": bands["50-100"]["mean_error_bias"], "Bias 100-150": bands["100-150"]["mean_error_bias"],
            "CV MAE (train engines)": r["cv_train_engines"]["overall"]["MAE"] if "cv_train_engines" in r else None,
            "MAE 30-90% schedule": r["schedules"]["fractions_30_90"]["MAE"] if "schedules" in r else None,
            "MAE random schedule": r["schedules"]["uniform_30_95_seed42"]["MAE"] if "schedules" in r else None,
            "MAE diff vs no cap [95% CI]": f"{r['bootstrap_vs_no_cap']['mae_difference']} {r['bootstrap_vs_no_cap']['ci95']}",
            "Notes": notes,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Official test: only for a candidate selected above, scored once, never fed back
# ---------------------------------------------------------------------------

def official_scores(model, experiment: Experiment, sensors: list[str]) -> dict:
    test = official_test_frame()
    frame, columns = build_features(test, experiment, sensors)
    frame["predicted_rul"] = np.clip(model.predict(frame[columns]), 0, None).astype(float)
    frame["error"] = frame["predicted_rul"] - frame["actual_rul"]
    frame["absolute_error"] = frame["error"].abs()
    last = frame[frame["is_last_cycle"]]
    return {"official_last_cycle": {"metrics": _metrics(last), "error_by_true_rul": _by_band(last),
                                    "error_by_phase2_5_band": _band_rows(last)},
            "all_test_rows": {"metrics": _metrics(frame)}}


def run_official(report: dict, context: dict) -> dict:
    name = report["decision"]["selected_candidate"]
    selected = next(r for r in report["experiments"] if r["name"] == name)
    bundle, _, _ = load_v1_reference()
    phase2_meta = json.loads(PHASE2_METADATA_FILE.read_text(encoding="utf-8"))
    return {
        "protocol": "Same as src/evaluate_official_test.py: RUL_FD001 label at each test engine's last cycle. "
                    "Run once, after selection on test-like validation; no tuning afterwards.",
        "v1": official_scores(bundle.model, V1_FEATURES, []),
        "phase2_candidate": official_scores(joblib.load(PHASE2_MODEL_FILE),
                                            Experiment("Phase 2", rolling=True, trend=True),
                                            phase2_meta["temporal_sensors"]),
        "phase2_5_candidate": {"name": name, "rul_cap": selected["rul_cap"],
                               **official_scores(context["models"][name], V1_FEATURES, [])},
    }


def main() -> None:
    report, context = run_all()
    RESULTS_DIR.mkdir(exist_ok=True)
    table = comparison_table(report)
    table.to_csv(COMPARISON_FILE, index=False)

    decision = report["decision"]
    if decision["selected_candidate"]:
        name = decision["selected_candidate"]
        selected = next(r for r in report["experiments"] if r["name"] == name)
        joblib.dump(context["models"][name], CANDIDATE_MODEL_FILE)
        bundle, _, _ = load_v1_reference()
        CANDIDATE_METADATA_FILE.write_text(json.dumps({
            "model_file": CANDIDATE_MODEL_FILE.name,
            "model_sha256": hashlib.sha256(CANDIDATE_MODEL_FILE.read_bytes()).hexdigest(),
            "status": CANDIDATE_STATUS, "experiment": name, "rul_cap": selected["rul_cap"],
            "features": list(FEATURE_NAMES), "hyperparameters": bundle.metadata["hyperparameters"],
            "test_like_validation": selected["holdout"]["overall"],
        }, indent=2), encoding="utf-8")
        OFFICIAL_FILE.write_text(json.dumps(run_official(report, context), indent=2), encoding="utf-8")
        report["candidate_model_file"] = CANDIDATE_MODEL_FILE.name
        report["official_test_file"] = OFFICIAL_FILE.name

    RESULTS_FILE.write_text(json.dumps(report, indent=2), encoding="utf-8")

    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(table.drop(columns=["Notes"]).to_string(index=False))
    for r in report["experiments"][1:]:
        print(f"{r['name']}: {r['selection_checks']}")
    print("Decision:", json.dumps(decision["caps_that_clearly_win"]), "->", decision["selected_candidate"])
    print("V1 reproduction:", report["v1_reproduction"])


if __name__ == "__main__":
    main()
