"""
Shared feature-extraction logic for the fault detection ML pipeline.

This module is imported IDENTICALLY by ml/train_model.py (offline training)
and backend/main.py (live prediction API), so that the features fed into
the model at training time and at inference time are computed by the exact
same code path. Any divergence here would silently break prediction
accuracy without raising an error.
"""

import numpy as np
import pandas as pd

# Column names expected in every raw simulation CSV.
VOLTAGE_COLUMNS = ["Va", "Vb", "Vc"]
CURRENT_COLUMNS = ["Ia", "Ib", "Ic"]

# Final, ordered list of feature names produced by extract_features().
# Keeping this as an explicit constant lets both training and the API
# build feature vectors in a guaranteed-consistent order.
FEATURE_NAMES = [
    "rms_Va", "rms_Vb", "rms_Vc",
    "rms_Ia", "rms_Ib", "rms_Ic",
    "peak_I",
    "rms_I0",
]

# Time window (seconds) during which each fault type is actually active in
# its simulation run - fixed by how the Simulink model was built. NoFault
# has no such window (the whole run is healthy).
#
# Shared between training and the API for two reasons: the API uses it to
# shade the fault-active region on the frontend charts, and training uses it
# to restrict labeled windows to the genuinely fault-affected portion of
# each file - the first ~0.1s of every fault-case CSV is a healthy pre-fault
# lead-in, so windows drawn from it would carry a fault label despite
# looking exactly like NoFault, poisoning the training set.
FAULT_WINDOWS = {
    "NoFault": None,
    "LL": (0.1, 0.3),
    "LG": (0.1, 0.3),
    "LLG": (0.1, 0.3),
    "LLLG": (0.1, 0.3),
}


def _rms(series: pd.Series) -> float:
    """Root-mean-square of a signal window."""
    values = series.to_numpy(dtype=float)
    return float(np.sqrt(np.mean(np.square(values))))


def extract_features(df: pd.DataFrame) -> dict:
    """
    Compute the fixed 8-feature vector used by the classifier from a window
    of simulation data.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns: time, Ia, Ib, Ic, Va, Vb, Vc. Only the six
        electrical signal columns are used; `time` is not part of the
        feature set (features must be independent of window position).

    Returns
    -------
    dict
        Keys match FEATURE_NAMES exactly:
        rms_Va, rms_Vb, rms_Vc, rms_Ia, rms_Ib, rms_Ic, peak_I, rms_I0
    """
    features = {}

    for col in VOLTAGE_COLUMNS:
        features[f"rms_{col}"] = _rms(df[col])

    for col in CURRENT_COLUMNS:
        features[f"rms_{col}"] = _rms(df[col])

    # Peak absolute current across all three phases in this window -
    # a single scalar that captures fault severity regardless of which
    # phase(s) are involved.
    current_values = df[CURRENT_COLUMNS].to_numpy(dtype=float)
    features["peak_I"] = float(np.max(np.abs(current_values)))

    # Zero-sequence current: in a healthy or purely phase-to-phase fault
    # (no ground path, e.g. LL) the three phase currents sum to ~0 at every
    # instant, since whatever flows out one phase flows back in through
    # another. Whenever a ground path is involved (LG, LLG, LLLG) some
    # current returns through ground instead, so the instantaneous sum
    # stops cancelling to zero. Its RMS is therefore near-zero for ungrounded
    # faults and clearly non-zero for grounded ones - the one signal in this
    # feature set that can actually tell LL and LLG apart.
    zero_seq = (df["Ia"] + df["Ib"] + df["Ic"]) / 3.0
    features["rms_I0"] = _rms(zero_seq)

    return features
