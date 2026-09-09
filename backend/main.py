"""
FastAPI backend for the power line fault detection demo.

Serves:
  - fault type metadata
  - live simulation + ML prediction (loads a raw CSV, extracts features
    with the SAME feature_extraction module used at training time, and
    runs the saved model)
  - model comparison metrics for the dashboard
  - read-only CSV browsing for the raw data viewer modal

Run from the project root:
    uvicorn backend.main:app --reload
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Make ml/ importable so we reuse the exact same feature extraction code
# that was used during training - critical for prediction correctness.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "ml"))
from feature_extraction import FAULT_WINDOWS, FEATURE_NAMES, extract_features  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data" / "raw"
MODEL_PATH = PROJECT_ROOT / "ml" / "model.joblib"
METRICS_PATH = PROJECT_ROOT / "ml" / "model_comparison.json"

# Fixed, known fault classes. Human-readable names shown in the UI.
FAULT_TYPES = [
    {"code": "NoFault", "name": "Healthy / No Fault"},
    {"code": "LL", "name": "Line-to-Line Fault (A-B)"},
    {"code": "LG", "name": "Line-to-Ground Fault (A-G)"},
    {"code": "LLG", "name": "Double Line-to-Ground Fault (A,B-G)"},
    {"code": "LLLG", "name": "Three-Phase-to-Ground Fault (A,B,C-G)"},
]
FAULT_CODES = {ft["code"] for ft in FAULT_TYPES}

# The exact 5 filenames allowed for CSV access. Used to strictly validate
# the `filename` query param and prevent any path-traversal.
ALLOWED_FILENAMES = {f"{code}.csv" for code in FAULT_CODES}

app = FastAPI(title="Power Line Fault Detection Demo API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_model_bundle = None


@app.on_event("startup")
def load_model_on_startup():
    global _model_bundle
    if not MODEL_PATH.exists():
        print(
            "\n*** WARNING: ml/model.joblib not found. ***\n"
            "Run `python ml/train_model.py` first to train and save a model "
            "before using /api/simulate.\n"
        )
        _model_bundle = None
        return
    _model_bundle = joblib.load(MODEL_PATH)
    print(f"Loaded model: {_model_bundle.get('model_name', 'unknown')}")


def _load_raw_csv(fault_code: str) -> pd.DataFrame:
    path = DATA_DIR / f"{fault_code}.csv"
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Data file for '{fault_code}' not found at {path}. "
            "Add the real Simulink-exported CSVs to data/raw/.",
        )
    return pd.read_csv(path)


# The raw CSVs come straight off Simulink's variable-step solver, which
# takes far finer time-steps right around the fault switching instant than
# during quiet periods (30k-150k+ rows for a 0.3s run). Sending every row to
# the browser and drawing it as one Plotly line packs huge numbers of points
# into a handful of pixel columns, which both (a) tanks rendering
# performance and (b) can visually alias into wobble/ringing patterns that
# aren't actually in the signal.
#
# We fix this by decimating in TIME (not by row index, since row spacing is
# wildly uneven) into a fixed number of bins, keeping - per bin - the single
# real sample with the largest combined deviation across the bin's columns.
# Unlike naive "take every Nth row" decimation, this can never smooth away a
# brief spike/sag: whichever instant swung furthest from normal within a bin
# is exactly the instant kept.
def _decimate_preserve_extremes(df: pd.DataFrame, columns: list[str], max_points: int = 2000) -> pd.DataFrame:
    n = len(df)
    if n <= max_points:
        return df

    time = df["time"].to_numpy()
    # Normalize each column by its own peak magnitude first, so columns on
    # very different scales (e.g. thousands of volts vs. single-digit amps)
    # contribute comparably to picking the "most extreme" sample per bin.
    composite = np.zeros(n)
    for col in columns:
        values = df[col].to_numpy(dtype=float)
        scale = np.max(np.abs(values))
        if scale > 0:
            composite += (values / scale) ** 2

    bin_edges = np.linspace(time[0], time[-1], max_points + 1)
    bin_idx = np.clip(np.searchsorted(bin_edges, time, side="right") - 1, 0, max_points - 1)

    picks = (
        pd.DataFrame({"bin": bin_idx, "composite": composite})
        .groupby("bin")["composite"]
        .idxmax()
        .to_numpy()
    )
    picks = np.sort(picks)  # keep chronological order for line plotting
    return df.iloc[picks]


@app.get("/api/fault-types")
def get_fault_types():
    return FAULT_TYPES


@app.get("/api/simulate")
def simulate(fault_type: str = Query(...)):
    if fault_type not in FAULT_CODES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown fault_type '{fault_type}'. Must be one of {sorted(FAULT_CODES)}.",
        )
    if _model_bundle is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run `python ml/train_model.py` first.",
        )

    df = _load_raw_csv(fault_type)

    # Predict using the FULL run's features (not a sub-window) so the
    # single prediction reflects the whole simulated event.
    features = extract_features(df)
    feature_vector = [[features[name] for name in FEATURE_NAMES]]

    scaler = _model_bundle["scaler"]
    model = _model_bundle["model"]
    scaled_vector = scaler.transform(feature_vector)

    predicted = model.predict(scaled_vector)[0]

    confidence = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(scaled_vector)[0]
        class_index = list(model.classes_).index(predicted)
        confidence = float(proba[class_index])

    # Decimate voltage and current independently for charting - each keeps
    # the real sample that deviates most within its own bin, so a current
    # spike can't get discarded just because voltage swings more (or vice
    # versa). Predictions above already ran on the full, non-decimated data.
    voltage_df = _decimate_preserve_extremes(df, ["Va", "Vb", "Vc"])
    current_df = _decimate_preserve_extremes(df, ["Ia", "Ib", "Ic"])

    return {
        "fault_type_actual": fault_type,
        "fault_type_predicted": predicted,
        "confidence": confidence,
        "correct": predicted == fault_type,
        "features": features,
        "time_voltage": voltage_df["time"].tolist(),
        "Va": voltage_df["Va"].tolist(),
        "Vb": voltage_df["Vb"].tolist(),
        "Vc": voltage_df["Vc"].tolist(),
        "time_current": current_df["time"].tolist(),
        "Ia": current_df["Ia"].tolist(),
        "Ib": current_df["Ib"].tolist(),
        "Ic": current_df["Ic"].tolist(),
        "fault_window": FAULT_WINDOWS[fault_type],
    }


@app.get("/api/model-metrics")
def get_model_metrics():
    if not METRICS_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Model metrics not found. Run `python ml/train_model.py` first.",
        )
    return json.loads(METRICS_PATH.read_text())


@app.get("/api/csv-list")
def get_csv_list():
    files = []
    for code in FAULT_TYPES:
        fault_code = code["code"]
        path = DATA_DIR / f"{fault_code}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        files.append(
            {
                "filename": f"{fault_code}.csv",
                "row_count": len(df),
                "columns": list(df.columns),
            }
        )
    return files


@app.get("/api/csv-content")
def get_csv_content(filename: str = Query(...)):
    # Strict allowlist check - only the 5 known filenames are servable.
    # This prevents any path-traversal or arbitrary file read.
    if filename not in ALLOWED_FILENAMES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid filename. Must be one of {sorted(ALLOWED_FILENAMES)}.",
        )

    path = DATA_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{filename} not found in data/raw/.")

    df = pd.read_csv(path)
    return {
        "filename": filename,
        "columns": list(df.columns),
        "row_count": len(df),
        "rows": df.to_dict(orient="records"),
    }

app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")