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

    return {
        "fault_type_actual": fault_type,
        "fault_type_predicted": predicted,
        "confidence": confidence,
        "correct": predicted == fault_type,
        "features": features,
        "time": df["time"].tolist(),
        "Va": df["Va"].tolist(),
        "Vb": df["Vb"].tolist(),
        "Vc": df["Vc"].tolist(),
        "Ia": df["Ia"].tolist(),
        "Ib": df["Ib"].tolist(),
        "Ic": df["Ic"].tolist(),
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