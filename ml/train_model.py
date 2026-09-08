"""
Train and compare ML models for power line fault classification.

Pipeline:
    1. Load the 5 real Simulink-exported CSVs from data/raw/.
    2. Slice each single simulation run into several overlapping
       time-windows (since one CSV = one run, but we need many labeled
       rows to train a classifier).
    3. Run the shared extract_features() on each window -> one training
       row per window, labeled with the fault class from the filename.
    4. Stratified 80/20 train/test split.
    5. Train + evaluate several classifiers, print a comparison table,
       save it to ml/model_comparison.json, and persist the best model
       (plus its fitted StandardScaler) to ml/model.joblib.

Run from the project root:
    python ml/train_model.py
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
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


def evaluate_model(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

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
    fitted_models = {}

    print("\nTraining and evaluating models...\n")
    for name, model in candidates.items():
        model.fit(X_train_scaled, y_train)
        metrics = evaluate_model(model, X_test_scaled, y_test)
        results[name] = metrics
        fitted_models[name] = model
        print(
            f"  {name:22s} "
            f"accuracy={metrics['accuracy']:.4f}  "
            f"precision={metrics['precision']:.4f}  "
            f"recall={metrics['recall']:.4f}  "
            f"f1={metrics['f1_score']:.4f}"
        )

    best_name = max(results, key=lambda n: results[n]["accuracy"])
    best_model = fitted_models[best_name]
    print(f"\nBest model: {best_name} (accuracy={results[best_name]['accuracy']:.4f})")

    METRICS_PATH.write_text(json.dumps(results, indent=2))
    print(f"Saved model comparison metrics to {METRICS_PATH}")

    joblib.dump({"model": best_model, "scaler": scaler, "model_name": best_name}, MODEL_PATH)
    print(f"Saved best model + scaler to {MODEL_PATH}")


if __name__ == "__main__":
    main()
