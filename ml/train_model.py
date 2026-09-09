"""
Train and compare ML models for power line fault classification.

Pipeline:
    1. Load the 5 real Simulink-exported CSVs from data/raw/.
    2. Slice each single simulation run into several overlapping
       time-windows (since one CSV = one run, but we need many labeled
       rows to train a classifier).
    3. Run the shared extract_features() on each window -> one training
       row per window, labeled with the fault class from the filename.
    4. Evaluate each candidate model with stratified k-fold cross-validation
       (see evaluate_with_cv for why, instead of one fixed 80/20 split).
    5. Refit the winning model on the FULL labeled dataset (cross-validation
       is for honestly comparing/reporting performance; the model that
       actually goes live should be trained on every labeled example
       available, not just the folds that happened to be "training" folds).
    6. Save the comparison table to ml/model_comparison.json, and the
       refit best model (plus its fitted StandardScaler) to ml/model.joblib.

Run from the project root:
    python ml/train_model.py
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Allow running this script directly (python ml/train_model.py) as well as
# as a module, by making sure the ml/ directory is importable.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_extraction import FAULT_WINDOWS, FEATURE_NAMES, extract_features  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
MODEL_PATH = Path(__file__).resolve().parent / "model.joblib"
METRICS_PATH = Path(__file__).resolve().parent / "model_comparison.json"

# Filename -> fault class code. These are the exact, fixed filenames
# expected in data/raw/ (see project spec - no distance variants).
FAULT_FILES = {
    "NoFault": "NoFault.csv",
    "LL": "LL.csv",
    "LG": "LG.csv",
    "LLG": "LLG.csv",
    "LLLG": "LLLG.csv",
}

REQUIRED_COLUMNS = ["time", "Ia", "Ib", "Ic", "Va", "Vb", "Vc"]

# Windowing parameters: each run is sliced into overlapping windows so
# that a single simulation produces enough labeled rows to train on.
NUM_WINDOWS_PER_FILE = 12
WINDOW_FRACTION = 0.2  # each window spans 20% of the run's total duration

# Number of folds for cross-validated evaluation. With 12 rows/class, 5
# folds means each fold holds out ~2-3 rows per class for testing while
# training on the rest - a reasonable split for a dataset this size.
N_CV_FOLDS = 5


def check_data_files_exist() -> None:
    """Fail fast with a clear message if any required CSV is missing."""
    missing = [
        fname for fname in FAULT_FILES.values() if not (DATA_DIR / fname).exists()
    ]
    if missing:
        print("ERROR: Missing required data file(s) in data/raw/:")
        for fname in missing:
            print(f"  - {fname}")
        print(
            "\nPlease add the real MATLAB/Simulink-exported CSVs to "
            f"{DATA_DIR} before running training."
        )
        sys.exit(1)


def load_and_validate(fault_code: str, filename: str) -> pd.DataFrame:
    path = DATA_DIR / filename
    df = pd.read_csv(path)
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        print(
            f"ERROR: {filename} is missing required column(s): {missing_cols}. "
            f"Expected columns: {REQUIRED_COLUMNS}"
        )
        sys.exit(1)
    return df


def make_windows(df: pd.DataFrame, fault_window: tuple[float, float] | None) -> list[pd.DataFrame]:
    """
    Slice a single simulation run into overlapping time-windows, drawn ONLY
    from the period during which the file's label is actually true:

      - Fault classes: windows are drawn only from `fault_window` (when the
        fault is active), not the full file. Every fault-case CSV also
        contains a healthy pre-fault lead-in (before the fault activates) -
        windowing across the whole file would label some perfectly healthy
        slices as e.g. "LG", teaching the model that faults sometimes look
        exactly like NoFault.
      - NoFault: there is no fault window: the entire run is healthy, so
        windows are drawn from its full duration.

    Windows are spread evenly across that range (so they naturally overlap,
    since window length > spacing between window starts), giving each fault
    case multiple labeled feature rows despite coming from a single
    simulation.
    """
    if fault_window is not None:
        t_start, t_end = fault_window
        df = df[(df["time"] >= t_start) & (df["time"] <= t_end)]
    else:
        t_start, t_end = df["time"].min(), df["time"].max()

    if df.empty:
        return []

    total_duration = t_end - t_start
    window_len = total_duration * WINDOW_FRACTION

    last_start = t_end - window_len
    starts = np.linspace(t_start, last_start, NUM_WINDOWS_PER_FILE)

    windows = []
    for start in starts:
        end = start + window_len
        window_df = df[(df["time"] >= start) & (df["time"] <= end)]
        if len(window_df) >= 2:  # need at least 2 samples for a meaningful RMS
            windows.append(window_df)
    return windows


def build_dataset() -> tuple[pd.DataFrame, pd.Series]:
    rows = []
    labels = []

    for fault_code, filename in FAULT_FILES.items():
        df = load_and_validate(fault_code, filename)
        windows = make_windows(df, FAULT_WINDOWS[fault_code])
        print(f"  {filename}: {len(df)} rows -> {len(windows)} feature windows")
        for window_df in windows:
            rows.append(extract_features(window_df))
            labels.append(fault_code)

    X = pd.DataFrame(rows, columns=FEATURE_NAMES)
    y = pd.Series(labels, name="fault_type")
    return X, y


def evaluate_with_cv(model, X, y, cv) -> dict:
    """
    Evaluate a model with stratified k-fold cross-validation instead of one
    fixed train/test split.

    With only ~60 labeled rows total, a single 80/20 split tests on just
    ~12 of them - too few for the resulting accuracy/precision/recall/F1 to
    mean much, and prone to ties (several models scoring a clean 100%
    simply because the held-out 12 rows happened to be easy). Cross-
    validation instead rotates which rows are "held out": every one of the
    ~60 rows gets predicted exactly once, by a fold that did NOT see it
    during training, so the reported metrics are computed over the full
    dataset instead of a twelfth of it - a more honest and more
    discriminating comparison between models.

    `model` is wrapped in a Pipeline([scaler, model]) by the caller so that
    StandardScaler is fit fresh on each fold's training portion only - if
    scaling were fit once on the whole dataset up front (as a single-split
    evaluation typically does), statistics from each fold's held-out rows
    would leak into what "normal" looks like during training, quietly
    inflating every model's score.
    """
    y_pred = cross_val_predict(model, X, y, cv=cv)
    accuracy = accuracy_score(y, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, y_pred, average="macro", zero_division=0
    )
    return {
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
    }


def main():
    print("Checking for required data files...")
    check_data_files_exist()

    print("\nBuilding windowed dataset from raw simulation CSVs...")
    X, y = build_dataset()
    print(f"\nTotal training rows: {len(X)} across {y.nunique()} classes")
    print(y.value_counts().to_string())

    cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=42)

    candidates = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "SVM (RBF kernel)": SVC(kernel="rbf", probability=True, random_state=42),
        "MLP Classifier": MLPClassifier(
            hidden_layer_sizes=(32, 16), max_iter=2000, random_state=42
        ),
    }

    results = {}

    print(f"\nEvaluating models with {N_CV_FOLDS}-fold stratified cross-validation...\n")
    for name, model in candidates.items():
        pipeline = Pipeline([("scaler", StandardScaler()), ("clf", clone(model))])
        metrics = evaluate_with_cv(pipeline, X, y, cv)
        results[name] = metrics
        print(
            f"  {name:22s} "
            f"accuracy={metrics['accuracy']:.4f}  "
            f"precision={metrics['precision']:.4f}  "
            f"recall={metrics['recall']:.4f}  "
            f"f1={metrics['f1_score']:.4f}"
        )

    best_name = max(results, key=lambda n: results[n]["accuracy"])
    print(f"\nBest model (by cross-validated accuracy): {best_name}")

    METRICS_PATH.write_text(json.dumps(results, indent=2))
    print(f"Saved model comparison metrics to {METRICS_PATH}")

    # Cross-validation above is for honestly comparing/reporting model
    # performance. The model that actually goes live should be trained on
    # every labeled example we have, not just whatever folds happened to
    # be "training" folds during evaluation - so refit the winner (and a
    # fresh scaler) on the full dataset here.
    scaler = StandardScaler()
    X_scaled_full = scaler.fit_transform(X)
    best_model = clone(candidates[best_name])
    best_model.fit(X_scaled_full, y)

    joblib.dump({"model": best_model, "scaler": scaler, "model_name": best_name}, MODEL_PATH)
    print(f"Saved best model + scaler (refit on full dataset) to {MODEL_PATH}")


if __name__ == "__main__":
    main()
