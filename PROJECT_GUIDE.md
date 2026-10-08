# NASA C-MAPSS FD001 — Predictive Maintenance & RUL Prediction

## 1. Project Overview

This project uses the NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) dataset to build a predictive maintenance system for aircraft engine health.

The main objective is:

> Predict the Remaining Useful Life (RUL) of an engine from its sensor measurements and operating conditions.

RUL means the estimated number of operating cycles remaining before an engine reaches its failure/end-of-life point.

This is a regression problem because the target is a continuous number.

The project starts with FD001, the simplest C-MAPSS subset, before potentially expanding to FD002, FD003, and FD004.

---

## 2. Why This Project Matters

Traditional maintenance can be:
- Corrective: repair after failure.
- Preventive: repair according to a fixed schedule.
- Predictive: estimate equipment condition from data and plan maintenance before unexpected failure.

The same predictive-maintenance workflow can be applied to pumps, compressors, turbines, motors, engines, and other industrial assets.

---

## 3. Core Problem

Each engine is monitored over multiple operating cycles. At every cycle, the dataset records the engine ID, cycle number, operating settings, and sensor measurements.

Training engines continue until failure.

For a training observation:

`RUL = Maximum cycle of that engine - Current cycle`

Example: if an engine reaches cycle 200, then cycle 1 has RUL 199, cycle 100 has RUL 100, and cycle 200 has RUL 0.

---

## 4. Dataset

The project currently focuses on NASA C-MAPSS FD001.

Current raw files:

```text
data/raw/train_FD001.txt
data/raw/test_FD001.txt
data/raw/RUL_FD001.txt
```

The original archive is:

```text
data/CMAPSSData.zip
```

FD001 training data contains:

- 20,631 rows
- 26 columns
- 100 engines

The 26 columns are:

```text
engine_id
cycle
setting_1
setting_2
setting_3
sensor_1
sensor_2
sensor_3
sensor_4
sensor_5
sensor_6
sensor_7
sensor_8
sensor_9
sensor_10
sensor_11
sensor_12
sensor_13
sensor_14
sensor_15
sensor_16
sensor_17
sensor_18
sensor_19
sensor_20
sensor_21
```

---

## 5. Column Meaning

### engine_id

Identifies the engine. There are 100 engines in FD001.

### cycle

The operating-cycle/time index for an engine.

Each engine starts at cycle 1 and continues until its observed final cycle in the training data.

### setting_1, setting_2, setting_3

Operating conditions/configurations under which the engine operates.

These are not direct health measurements, but they can influence sensor readings.

### sensor_1 ... sensor_21

Simulated engine measurements.

C-MAPSS contains measurements representing different engine operating/health quantities. Do not assume that every sensor is useful. EDA should determine which sensors vary, which are constant, and which appear related to degradation.

---

## 6. Project Structure

The full, current tree is in the README. Responsibilities:

- `data/`: datasets.
- `data/raw/`: original raw files. Do not overwrite them.
- `notebooks/`: analysis and experiments.
- `src/`: reusable Python code.
- `models/`: trained models and model artifacts.
- `results/`: plots, metrics, tables, and experiment outputs.
- `tests/`: automated tests.

---

## 7. Python Environment

The project targets Python 3.12 in a project-local virtual environment (`.venv`).

- `requirements.txt`: pinned runtime dependencies for the app, `src/` scripts, and tests.
- `requirements-notebooks.txt`: additional packages for the notebooks.

Versions are pinned because the saved `.pkl` models must be loaded with the same
scikit-learn / XGBoost versions that created them. Setup steps are in the README.

---

## 8. Data Loading

`src/data_loader.py` defines the canonical 26 columns, loads training and test
measurements plus official RUL labels, and validates the expected data structure.

The source files are whitespace-delimited, so loading typically uses:

```python
pd.read_csv(file_path, sep=r"\s+", header=None)
```

The loader supports:
- Training data
- Test data
- RUL labels

It validates:
- Expected column count
- Numeric types
- Missing values
- Duplicate rows
- Engine IDs
- Cycle ordering

---

## 9. Initial Data Quality Findings

Initial inspection found:

```text
Training rows: 20,631
Columns: 26
Unique engines: 100
Overall cycle range: 1–362
Duplicate complete rows: 0
Missing values: none detected
```

Each engine starts at cycle 1 and has contiguous cycles.

These facts should still be checked programmatically whenever the pipeline runs.

---

## 10. EDA — Exploratory Data Analysis

EDA is the first major analytical stage.

The goal is to understand the data before building a model.

Important questions:

### Dataset level
- How large is the dataset?
- Are there missing values?
- Are there duplicates?
- What are the data types?
- What are the value ranges?

### Engine level
- How many engines exist?
- How many cycles does each engine have?
- How different are engine lifetimes?
- What is the lifetime distribution?

### Sensor level
- Which sensors are constant?
- Which have little variation?
- Which change over time?
- Which appear to contain degradation information?

### Relationship level
- Which sensors are correlated?
- Which sensors correlate with cycle?
- Which measurements may help predict RUL?

---

## 11. Degradation

Degradation is the central concept of the project.

An engine can move from a relatively healthy state toward failure over its operating cycles.

Sensor degradation may appear as:

- Increasing trend
- Decreasing trend
- Non-linear trend
- Increasing variability
- Distribution shifts

Not every sensor will show a clear degradation trend. Multiple sensors and modeling methods may therefore be necessary.

---

## 12. RUL Target

For every training engine, the final observed cycle represents its failure/end-of-life point.

For an engine with maximum cycle `C_max`:

```text
RUL = C_max - current_cycle
```

Example:

```text
Maximum cycle = 192

cycle 1   -> RUL 191
cycle 50  -> RUL 142
cycle 100 -> RUL 92
cycle 150 -> RUL 42
cycle 191 -> RUL 1
cycle 192 -> RUL 0
```

Training RUL can be constructed with:

```python
max_cycle = df.groupby("engine_id")["cycle"].transform("max")
df["RUL"] = max_cycle - df["cycle"]
```

Do not overwrite the raw dataset.

---

## 13. Train vs Test

This distinction is critical.

### Training data

Training engines are observed until failure, so their maximum cycle is known.

### Test data

Test engines are truncated before failure.

Their true failure point is not directly present in the test sensor sequence.

`RUL_FD001.txt` provides the true RUL for the final observed point of each test engine.

Conceptually:

```text
TRAIN
history ------------------> failure
                            known

TEST
history ------------> unknown failure
                      |
                      +-- RUL_FD001 gives remaining life
```

---

## 14. Time-Series Considerations

This is not ordinary independent tabular data.

Each engine is a sequence:

```text
Engine 1:
cycle 1 -> cycle 2 -> cycle 3 -> ... -> failure

Engine 2:
cycle 1 -> cycle 2 -> cycle 3 -> ... -> failure
```

Randomly splitting individual rows can cause leakage because rows from the same engine can appear in both training and validation.

Where appropriate, split by `engine_id` so validation represents unseen engines.

---

## 15. Data Leakage

Avoid using information that would not be available at prediction time.

When predicting RUL at cycle 100, the model must not use:

- Future sensor measurements
- Future cycles
- The actual failure cycle
- Features calculated using future observations

The model should use only information available up to the prediction point.

---

## 16. Feature Engineering

Possible future features include:

### Raw sensor values

Original measurements.

### Rolling statistics

For example:

```text
rolling_mean
rolling_std
```

### Sensor trends

For example:

```text
current_value - previous_value
```

or a slope over a recent window.

Feature engineering should be justified by EDA rather than added randomly.

---

## 17. Candidate Models

After EDA and preparation, possible regression models include:

### Linear Regression
Useful as a simple baseline.

### Random Forest Regressor
Can model non-linear relationships and provide feature importance.

### Gradient Boosting
Can capture complex non-linear relationships.

### XGBoost
A strong gradient-boosting option for tabular regression.

### Support Vector Regression
Can model non-linear relationships with suitable kernels and scaling.

### Neural Network
Can learn complex relationships but requires careful preprocessing and tuning.

Do not assume a model is best before evaluating it on the same validation setup.

---

## 18. Evaluation Metrics

### MAE

Mean Absolute Error.

Interpretation:

> On average, how many cycles away is the prediction from the true RUL?

Lower is better.

### RMSE

Root Mean Squared Error.

Large errors are penalized more strongly.

Lower is better.

### R²

Measures explained variance.

Higher is generally better, but it should not be used alone.

### NASA-style scoring

C-MAPSS RUL research commonly uses an asymmetric scoring concept where early and late predictions can receive different penalties.

If implemented, this should be documented separately from MAE/RMSE.

---

## 19. Modeling Pipeline

```text
Raw Dataset
     ↓
Data Loading
     ↓
Data Validation
     ↓
EDA
     ↓
RUL Target Construction
     ↓
Feature Engineering
     ↓
Engine-level Train/Validation Split
     ↓
Scaling when required
     ↓
Baseline Models
     ↓
Hyperparameter Tuning
     ↓
Evaluation
     ↓
Error Analysis
     ↓
Model Selection
     ↓
Save Model
     ↓
Prediction / Deployment
```

---

## 20. Development Stages

These are the development stages of the original plan. (The "Phase 2" and "Phase 2.5"
experiments described in the README came after these stages and are separate from them.)

### Stage 1 — Data Setup

Completed: Python environment, C-MAPSS archive, FD001 extraction, training data verified.

### Stage 2 — EDA

Completed: data-quality checks, engine lifetime analysis, sensor and operating-setting
analysis, correlation analysis, degradation visualization, initial RUL analysis.

### Stage 3 — Data Preparation

Completed:

- Robust loading and validation for training, test, and official RUL-label files (`src/data_loader.py`)
- Reusable training RUL construction that does not alter raw data
- Sensor selection: the 6 constant sensors were removed, leaving 15 sensors
- Engine-level train/validation split (`GroupShuffleSplit` by `engine_id`, 80 / 20 engines)

### Stage 4 — Modeling

Completed (`notebooks/02_EDA_FD001_cleaned.ipynb`):

- Linear Regression baseline, Random Forest, and XGBoost compared on the same engine-level split
- XGBoost tuned with `RandomizedSearchCV` and 3-fold `GroupKFold` on the training engines
- Final model exported with metadata and a SHA-256 fingerprint (`models/model_metadata.json`)

### Stage 5 — Evaluation and Interpretation

Completed:

- Validation metrics and error by RUL band
- Feature importance and actual-vs-predicted analysis
- One-time evaluation on the official NASA FD001 test set (`src/evaluate_official_test.py`)

### Stage 6 — Deployment

Completed: a two-page Streamlit app (`app.py`) that serves the verified V1 model:

```text
Sensor Input
     ↓
Input Validation
     ↓
Trained RUL Model (V1)
     ↓
Predicted Remaining Useful Life
     ↓
Health / Maintenance Information
```

---

## 21. What a Good Final Project Should Demonstrate

The project should demonstrate an end-to-end predictive-maintenance workflow:

```text
Industrial Sensor Data
        ↓
Data Understanding
        ↓
Data Cleaning
        ↓
Degradation Analysis
        ↓
RUL Construction
        ↓
Feature Engineering
        ↓
Machine Learning
        ↓
Evaluation
        ↓
Interpretability
        ↓
Prediction System
```

The objective is not simply to obtain the lowest error.

The project should demonstrate understanding of:

- Industrial sensor data
- Degradation
- Time-dependent behavior
- Remaining Useful Life
- Predictive maintenance
- Data leakage
- Feature engineering
- Regression
- Evaluation
- Explainability
- Deployment

---

## 22. Working Principles

Rules followed throughout development:

- Reuse existing code and avoid unnecessary duplication.
- Keep raw data unchanged; never modify the raw C-MAPSS files.
- Validate results after every change and keep notebooks reproducible.
- Do not invent sensor meanings.
- Do not claim a model is good without evaluation.
- Do not randomly split rows when this causes leakage; split by engine.
- Never train or tune using test labels.
- Never use future information to predict current RUL.

---

## 23. Documentation Rule

For every important experiment, document:

1. What was done.
2. Why it was done.
3. What result was obtained.
4. What the result means.
5. What decision follows from it.

Example:

```text
Experiment:
Compare Random Forest and Gradient Boosting.

Why:
Determine whether a non-linear tree ensemble improves RUL prediction.

Result:
Evaluate both models using the same engine-level validation set.

Interpretation:
Compare MAE, RMSE, R², and error behavior.

Decision:
Continue with the strongest validated candidate and perform error analysis.
```

---

## 24. Current Status

```text
Environment         ✅
Dataset             ✅
Data loading        ✅
EDA                 ✅
RUL construction    ✅
Feature selection   ✅
Modeling            ✅
Evaluation          ✅
Interpretability    ✅
Deployment          ✅
```

The V1 XGBoost model is the production model. Later experiments (Phase 2 and
Phase 2.5) are documented in the README and did not replace V1.

---

## 25. One-Sentence Project Goal

> Build a reproducible machine-learning pipeline that uses NASA C-MAPSS engine sensor data to estimate Remaining Useful Life and demonstrate an end-to-end industrial predictive-maintenance workflow.

---

## 26. Important Principle

Verified facts and future findings must be kept separate.

Currently verified:

```text
Training rows: 20,631
Columns: 26
Engines: 100
Cycle range: 1–362
Duplicate rows: 0
Missing values: none detected
```

Claims about the best sensors, features, model, final error, and deployment
performance come from the analysis and experiments recorded in the notebooks,
`models/model_metadata.json`, and `results/`.

Never present an assumption as an experimental result.
