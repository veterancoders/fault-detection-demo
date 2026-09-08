# Power Line Fault Detection & Diagnosis — ML Demo

A local web application demonstrating ML-based fault detection and
classification for a three-phase power line, using real data exported from
a MATLAB/Simulink Simscape fault simulation. The classification approach
(RMS feature extraction from voltage/current signals, comparison of
multiple classical ML models) follows the methodology described in Lim Zhan
Rui et al., "Fault Detection and Diagnosis in Power Systems Using Machine
Learning," IEEE ICGEA 2025.

The app classifies each simulation run into one of 5 fixed fault classes
(no distance variants — a single fixed line configuration):

| Code | Description |
|---|---|
| `NoFault` | Healthy / no fault |
| `LL` | Line-to-Line fault (Phase A–B) |
| `LG` | Line-to-Ground fault (Phase A–Ground) |
| `LLG` | Double Line-to-Ground fault (Phase A, B–Ground) |
| `LLLG` | Three-Phase-to-Ground fault (all phases–Ground) |

`data/raw/*.csv` are real Simulink-exported simulation results (not
synthetic), and `frontend/assets/simulink_model.png` is a screenshot of the
actual Simulink model used to generate them.

## Folder structure

```
fault-detection-demo/
├── data/
│   └── raw/                         # NoFault.csv, LL.csv, LG.csv, LLG.csv, LLLG.csv
├── ml/
│   ├── feature_extraction.py        # shared RMS feature-computation function
│   ├── train_model.py               # builds dataset, trains + compares models
│   ├── model.joblib                 # best trained classifier + scaler (generated)
│   └── model_comparison.json        # accuracy table for dashboard (generated)
├── backend/
│   ├── main.py                      # FastAPI app
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── app.js
│   ├── style.css
│   └── assets/
│       └── simulink_model.png       # screenshot of the Simulink model
└── README.md
```

## Setup instructions

From the project root:

```bash
# 1. Create and activate a virtual environment (Windows)
python -m venv venv
venv\Scripts\activate

# 2. Install dependencies
pip install -r backend/requirements.txt

# 3. Add your data (if not already present):
#    - Place NoFault.csv, LL.csv, LG.csv, LLG.csv, LLLG.csv in data/raw/
#    - Place a screenshot of the Simulink model as
#      frontend/assets/simulink_model.png

# 4. Train the models (creates ml/model.joblib and ml/model_comparison.json)
python ml/train_model.py

# 5. Start the backend API (from the project root)
uvicorn backend.main:app --reload

# 6. Serve the frontend (in a separate terminal, from the frontend/ folder)
cd frontend
python -m http.server 5500

# 7. Open the app
#    http://localhost:5500
```

The backend runs on `http://localhost:8000` and the frontend calls it
directly via `fetch()` (CORS is enabled on the API for this).

## How it works

- **Feature extraction** (`ml/feature_extraction.py`): computes 8 features
  from a window of `time, Ia, Ib, Ic, Va, Vb, Vc` data — RMS of each voltage
  and current phase, peak absolute current, and the RMS of the zero-sequence
  current (`(Ia+Ib+Ic)/3`, which detects whether a ground path is involved).
  This exact function is imported by both the training script and the live
  API, so predictions are always computed the same way the model was trained.
- **Training** (`ml/train_model.py`): since each fault type is only a
  single simulation run, the script slices each run into several
  overlapping time-windows to generate enough labeled rows, then trains and
  compares Random Forest, Gradient Boosting, KNN, SVM, and an MLP
  classifier, saving the best one.
- **Backend** (`backend/main.py`): loads the saved model, and on
  `/api/simulate` extracts features from the full raw CSV for the selected
  fault type, predicts the class, and returns the waveform data alongside
  the prediction for charting.
- **Frontend**: vanilla HTML/CSS/JS with Plotly.js (via CDN) for the
  voltage/current charts, including a shaded region marking the
  fault-active time window (`fault_window`, `null` for `NoFault`).
