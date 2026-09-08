# How This App Works (Plain-Language Explanation)

This is written so you can walk your supervisor through the app without
needing to read code. It covers: what the data is, how the charts are
drawn (including the colored fault region), what the RMS features are,
how the different ML models work, and how training actually happens.

---

## 1. The big picture

1. A MATLAB/Simulink model simulates a three-phase power line — normal
   operation, and four different fault conditions.
2. Each simulation run is exported as a CSV: a time series of 6 electrical
   signals (3 voltages, 3 currents).
3. From each run, we compute 8 summary numbers ("features") that describe
   the shape/severity of the signals.
4. Those 8 numbers are fed into a trained ML model, which predicts which
   of the 5 fault classes the run belongs to.
5. The web app lets you pick a fault type, re-run this whole pipeline live,
   and see the waveform, the 8 features, and the model's prediction.

The key idea to communicate to your supervisor: **the ML model never looks
at the raw waveform picture.** It only ever sees the 8 numbers. The chart
is for *humans* to visually confirm what happened; the model works purely
on the numeric fingerprint.

---

## 2. The raw data — what's in each CSV

Each `data/raw/<FaultType>.csv` has these columns:

| Column | Meaning |
|---|---|
| `time` | seconds, since the start of the simulation |
| `Va, Vb, Vc` | instantaneous voltage on phase A, B, C (volts) |
| `Ia, Ib, Ic` | instantaneous current on phase A, B, C (amps) |

In healthy operation these are clean 50/60 Hz sine waves, 120° apart from
each other (that's what "three-phase" means). A fault physically disturbs
this: depending on which phases are shorted together or shorted to ground,
some phases see a **voltage sag** (drop) and a **current surge** (spike),
while unaffected phases keep behaving normally.

The 5 classes differ in *which* phases are affected:

| Code | What's physically happening |
|---|---|
| `NoFault` | nothing — clean healthy waveform, whole run |
| `LL` | Phase A and B short together — both sag/surge, C stays normal |
| `LG` | Phase A shorts to ground — A is disturbed, B and C stay normal |
| `LLG` | Phase A and B short to ground — both disturbed, C stays normal |
| `LLLG` | All three phases short to ground — everything disturbed |

For the fault files, the disturbance is only active roughly between
**t = 0.1s and t = 0.3s** — before and after that window, the signal looks
just as healthy as `NoFault`.

---

## 3. The waveform charts (Voltage / Current vs Time)

These are the two big line charts you see after clicking **Run
Simulation**.

- **X-axis** = `time`, taken directly from the CSV's `time` column, in
  seconds.
- **Y-axis** = the instantaneous signal value — volts for the voltage
  chart, amps for the current chart.
- **The three colored lines** are the three phases (A = blue, B = orange,
  C = aqua in this app — chosen to still be tellable apart under
  colorblindness, unlike the more obvious red/green choice), plotted from
  the `Va/Vb/Vc` or `Ia/Ib/Ic` columns directly. Nothing is computed here —
  it's a literal plot of the CSV.

This is intentionally the "raw evidence" view: your supervisor can see
with their own eyes that, say, an `LG` fault makes Phase A's voltage
collapse while B and C keep humming along normally.

**Two things you'll notice that are real, not chart bugs:**
- **The current chart looks almost identical across every fault type.**
  In this dataset, `Ia/Ib/Ic` barely move regardless of which fault is
  active — RMS current stays around 1.4–1.5A in every single file,
  including `NoFault`. That's a property of wherever this current is
  measured in the Simulink model (likely upstream of the fault, behind a
  stiff source or current limiting), not a plotting problem.
- **`LL` and `LLG`'s voltage charts look nearly identical.** They
  genuinely are: both short phases A and B together, and that
  phase-to-phase short dominates the voltage sag on its own — the extra
  ground path in `LLG` barely changes the voltage picture. (`Va` differs
  by ~10%, `Vb`/`Vc` by under 1%, during the fault.) This is exactly why
  `rms_I0` (§ 4) had to be added: it's the one feature that separates
  these two even when the voltage and current *waveforms* can't visually
  be told apart.

### The colored (shaded) region — what it is and why it changes

This is the reddish band that appears over part of the chart when you run
a fault case, and is **absent** for `NoFault`.

- The backend knows, for each fault type, the time window during which the
  fault is actually active (`[0.1, 0.3]` seconds — baked in because that's
  how the Simulink model was built). For `NoFault` there is no such window,
  so this value is `null`.
- The frontend chart (using the Plotly.js charting library) draws this as
  a **rectangle shape** on top of the chart:
  - Its left/right edges (`x0`, `x1`) are the fault's start/end time —
    tied to the real time axis, just like the data.
  - Its top/bottom edges span the **entire height of the chart**
    (achieved by anchoring it to the plot's paper coordinates rather than
    a data value, so it always fully covers whatever the y-axis range
    happens to be).
  - It's drawn semi-transparent and placed **below** the data lines
    (`layer: "below"`), so it acts like a highlighter behind the waveform
    rather than covering it up.
  - When `fault_window` is `null` (the `NoFault` case), the frontend simply
    skips adding this rectangle — nothing is drawn, which is why a healthy
    run shows a plain, unshaded chart.
- **Why it matters visually:** it lets you *see* the exact moment the
  fault starts and ends, and visually correlate it with the waveform
  changing shape inside that band (voltage sagging, current spiking) and
  returning to normal right at the edge of the band. It's a visual aid
  only — it plays no role in the model's prediction, which is computed
  separately from the numeric features (see below).

---

## 4. "Extracted RMS Features (model input)" — what this bar chart is

This is the actual input to the machine learning model. Everything else
on the page is supporting visualization; **this is what the model sees.**

### What is RMS?

RMS = **Root Mean Square**. An AC voltage/current constantly swings
between positive and negative every cycle, so its plain average is always
close to zero — that tells you nothing about its strength. RMS fixes this
by:
1. **Squaring** every sample (this makes everything positive, and
   emphasizes large swings more than small ones),
2. **Averaging** those squared values,
3. Taking the **square root** to bring it back to the original units.

The result is a single number representing the "effective size" of an
oscillating signal — this is the same RMS your multimeter reports when you
measure household mains voltage (e.g. "230V" or "120V" is an RMS value,
even though the instantaneous voltage is constantly swinging much higher
and lower than that).

### The 8 features

For each run, we compute:

| Feature | Meaning |
|---|---|
| `rms_Va`, `rms_Vb`, `rms_Vc` | RMS voltage of each phase over the run |
| `rms_Ia`, `rms_Ib`, `rms_Ic` | RMS current of each phase over the run |
| `peak_I` | the single largest absolute current spike seen on *any* phase |
| `rms_I0` | RMS of the **zero-sequence current**, `(Ia + Ib + Ic) / 3` |

This turns a waveform with hundreds/thousands of raw time points into
**one fixed-size fingerprint of just 8 numbers** — regardless of how long
the run is. That fingerprint looks very different per fault class: e.g. an
`LG` fault produces a low `rms_Va` (phase A sagged) and a high `rms_Ia` /
`peak_I` (phase A surged), while `rms_Vb`, `rms_Vc`, `rms_Ib`, `rms_Ic`
stay near their healthy values. An `LLLG` fault instead disturbs all three
`rms_V*` and `rms_I*` values at once. This is exactly the kind of pattern
a classifier can learn to distinguish.

`rms_I0` deserves a special mention because it's the one feature that
isn't a simple per-phase or per-severity number — it specifically detects
whether **ground is involved**. In healthy operation, and in any fault
that stays purely phase-to-phase (like `LL`), whatever current leaves one
phase returns through another phase — the three instantaneous currents
sum to ~0 at every moment, so `rms_I0` is ~0. The moment a fault involves a
path to ground (`LG`, `LLG`, `LLLG`), some current returns through ground
instead of through the other phases, so that sum stops cancelling and
`rms_I0` becomes clearly non-zero. This was added specifically because
`LL` and `LLG` disturb the exact same two phases (A and B) and were
getting confused by the model without it (see § 7) — `rms_I0` is the one
number in the whole feature set that tells those two apart.

One subtlety worth knowing if it comes up: a **balanced** `LLLG` fault
(all three phases equally affected) can also show `rms_I0` near 0, even
though ground is very much involved — because the three equal, evenly
120°-apart fault currents still cancel each other out when summed,
regardless of where they return to. That's not a flaw in the feature, it's
correct three-phase theory (only an *unbalanced* fault produces a net
zero-sequence current) — and it's not a problem for classification here
because `LLLG` is already unmistakable from its voltage features alone
(all three phases collapse at once, unlike any other class).

- **X-axis** of each bar chart = the feature names above (split into a
  voltage chart and a current chart — see the note on the page itself
  about why they're not combined into one).
- **Y-axis** = their computed value (volts for the 3 RMS voltage bars,
  amps for the RMS/peak current bars). Note the units differ between
  voltage and current features, so bar heights are only meant to be
  compared within their own chart, not against each other.

---

## 5. The 5 machine learning models, and why compare several

There is no single "best" algorithm for every dataset — different
algorithms make different assumptions about how the data is shaped, so the
standard practice (and what the referenced IEEE paper also does) is to
train several and empirically pick whichever scores best on data it
wasn't trained on. That's what the **Model Performance Comparison** table
at the bottom of the page shows.

| Model | Plain-language idea |
|---|---|
| **Random Forest** | Builds many simple decision trees (each one just asks a sequence of yes/no threshold questions like "is `rms_Ia` > 40?"), then has all the trees vote on the answer. Averaging many trees smooths out the mistakes any single tree would make. |
| **Gradient Boosting** | Also builds decision trees, but one at a time — each new tree is trained specifically to correct the errors the previous trees made. Often very accurate, but can overfit small datasets more easily than Random Forest. |
| **K-Nearest Neighbors (KNN)** | Doesn't really "learn" a rule at all — it just remembers every training example. To classify a new run, it finds the K most similar training examples (by distance in the 8-feature space) and takes a majority vote among them. |
| **SVM (RBF kernel)** | Tries to draw the best possible dividing boundary between classes in the 8-feature space, maximizing the margin (gap) between classes. The "RBF kernel" lets that boundary bend into curved/non-linear shapes instead of only straight lines. |
| **MLP Classifier** | A small artificial neural network (a couple of layers of simple math units called "neurons"). It learns non-linear combinations of the 8 input features through many rounds of adjusting internal weights (backpropagation). |

Each model is trained on **identical data** and scored with the same
metrics, so the comparison is fair:

- **Accuracy** — % of test cases classified correctly overall.
- **Precision** — of the cases the model *labeled* as a given fault, what
  fraction actually were that fault (measures false alarms).
- **Recall** — of the cases that *actually were* a given fault, what
  fraction the model caught (measures missed detections).
- **F1-score** — a single number balancing precision and recall together.

The model with the best accuracy is the one automatically saved and used
for live predictions in the app (shown as the "best model" when you run
`train_model.py`).

---

## 6. How training actually works

This is the part worth explaining carefully, because there's a
non-obvious trick involved: **there are only 5 real simulation runs total
(one CSV per fault class)** — nowhere near enough rows to train a
classifier on directly, since a classifier needs many labeled examples per
class.

The training script (`ml/train_model.py`) solves this with **windowing**:

1. For each CSV, instead of treating the whole run as a single example, it
   slices the run into **~12 overlapping time-windows** — like taking 12
   snapshots of slightly different (overlapping) time-ranges across the
   same run.
2. It computes the same 8 RMS features independently for *each* window.
3. Each of those windows becomes one labeled training row, tagged with
   that file's fault class.

So 5 runs × ~12 windows ≈ **60 labeled training rows** — enough to
meaningfully train and test a classifier, without needing dozens of
separate Simulink simulations.

From there, standard ML practice:

4. **80/20 split**: 80% of the ~60 rows are used for training, 20% held
   back purely for testing. The split is *stratified*, meaning each class
   keeps the same proportion in both the training set and the test set, so
   testing isn't accidentally skewed toward one fault type.
5. **Feature scaling**: the 8 features are standardized (subtract the mean,
   divide by standard deviation) using a `StandardScaler` fitted only on
   the training rows. This matters because raw voltage values (hundreds)
   and current values (tens) live on very different numeric scales, and
   several of these algorithms (SVM, KNN, MLP) are sensitive to that.
6. **Train**: all 5 models are fit on the scaled 80% training rows.
7. **Evaluate**: each model then predicts labels for the 20% test rows it
   never saw during training, and its predictions are compared against the
   true labels to compute accuracy/precision/recall/F1.
8. **Save**: the single best-performing model, together with the fitted
   scaler (needed later so live predictions get scaled the exact same
   way), is saved to `ml/model.joblib`. The comparison table itself is
   saved to `ml/model_comparison.json`.

### What happens when you click "Run Simulation" (live prediction)

This reuses the *exact same* feature-extraction function as training (not
a copy — the literal same Python function, imported by both), which is
important: if training and live prediction computed features even
slightly differently, the model's predictions would be meaningless.

1. The backend loads the full CSV for the fault type you picked.
2. It computes the 8 RMS features over the *entire* run (not a small
   window this time — we want the model's best-informed single guess for
   the whole event).
3. Those 8 numbers are scaled using the **same scaler saved during
   training**.
4. The saved model predicts a fault class from the scaled features, and
   (for models that support it) also reports a **confidence** — the
   probability it assigned to its predicted class.
5. The app compares the prediction to the actual fault type (which we know
   because that's literally which CSV we loaded) and shows a ✓/✗ badge.

---

## 7. A debugging story worth telling your supervisor: `LL` vs `LLG`

Earlier versions of this app (before the 8th feature, `rms_I0`, was added)
would sometimes get exactly this kind of result:

```
Actual Fault:     LLG
Predicted Fault:  LL
Confidence:       51.0%
Result:           ✗ Incorrect
```

This is worth understanding even though it's now fixed, because it's a
good story about *why* a feature set matters, not just tuning a model
harder — and it's the kind of thing worth walking a supervisor through.

**What each field means, concretely:**
- **Actual Fault** — the ground truth. Known for certain because it's
  literally which CSV file was loaded (`LLG.csv`).
- **Predicted Fault** — whatever class `model.predict()` returned.
- **Confidence** — the probability the model assigned to *its own chosen*
  class, not to the correct one. It is not a measure of how right the
  model turned out to be.
- **✗ Incorrect** — simply `predicted != actual`.

**Why `LLG` and `LL` specifically got confused:**

Physically, `LL` (Phase A–B short) and `LLG` (Phase A, B short *and* to
ground) disturb the exact same two phases — A and B — while Phase C stays
essentially normal in both cases. The *only* physical difference between
them is the extra path to ground in `LLG`. With only the original 7
features (per-phase RMS voltage/current + peak current), nothing in the
feature set specifically measured "how much current is returning through
the ground" — so an `LLG` event and an `LL` event landed very close
together in feature space, and confidences around 50% (a near coin-flip)
showed the model was genuinely unsure, not confidently wrong.

**The fix:** adding `rms_I0` — the RMS of the zero-sequence current,
`(Ia + Ib + Ic) / 3` (see § 4) — gave the model exactly the missing signal.
It's ~0 whenever current has nowhere to go but back through the other
phases (healthy operation, or `LL`), and clearly non-zero the moment a
ground path is involved (`LG`, `LLG`, `LLLG`). Retraining with this one
extra feature took accuracy on the held-out test set from 66–75% up to a
perfect 100% on 4 of the 5 models, and the `LL`/`LLG` predictions are now
correct with confidences in the 80–95% range instead of hovering near 50%.

**Why this is worth explaining rather than hiding:** it demonstrates the
actual ML workflow — a model isn't "wrong" in the abstract, it's missing
information; the fix was diagnosing *which* physical signal was missing
and adding it, not just throwing more compute at the same features. That's
a stronger thing to show a supervisor than a system that simply worked
the first time.

**One honest caveat:** "100% accuracy" here is measured on a test set of
only ~12 rows (roughly 2–3 per class), because there are still only 5 real
Simulink runs behind all of this (see § 6). It's a genuine result — the
zero-sequence signal really is that clean a discriminator for this
problem — but it's not a claim that the model would score 100% against
a much larger, more varied set of real fault recordings. Say "100% on our
current held-out test set" to your supervisor, not just "100% accurate."

**A note on confidence being a suspiciously round number (like 40%).**
If the saved model happens to be K-Nearest Neighbors with `k=5`, its
confidence is always a multiple of 20% (0, 20, 40, 60, 80, 100), because
"confidence" for KNN is just "how many of the 5 nearest training examples
voted for this class." 40% means only 2 of 5 agreed — a weak plurality,
not a strong majority. Random Forest or the MLP, by contrast, produce
continuous-looking confidences (e.g. 76.7%) because they're averaging
across many trees / a learned probability curve rather than counting
discrete votes. Which style you see just depends on which model won that
training run.

---

## 8. Clearing the "cache" in this app

There's no database and no server-side response caching here, so when
something looks stale, it's almost always one of these two specific
things — not a generic "cache" in the browser-extension sense.

**1. The backend holds the trained model in memory, loaded once at
startup.** `backend/main.py` loads `ml/model.joblib` a single time when the
server process starts (`@app.on_event("startup")`), and reuses that same
in-memory object for every `/api/simulate` request afterward. If you run
`python ml/train_model.py` again — because you added/changed data — the
new `ml/model.joblib` file is written to disk, but the *already-running*
server process has no way to know that happened. It keeps serving
predictions from the old model it loaded at startup until you restart it.

**Fix:** stop the server (Ctrl+C in its terminal) and start it again:
```bash
uvicorn backend.main:app --reload
```
`--reload` watches for changes and restarts automatically for *code*
edits, but don't rely on it noticing a retrained `.joblib` file — always
restart by hand after running `train_model.py`, to be certain.

**2. The browser caches the static frontend files** (`index.html`,
`app.js`, `style.css`). If you edit those and don't see the change, it's
the browser serving its cached copy instead of re-fetching.

**Fix:** hard-refresh instead of a normal reload:
- Windows/Linux: `Ctrl + Shift + R` (or `Ctrl + F5`)
- Mac: `Cmd + Shift + R`

Or open DevTools (`F12`) → Network tab → check "Disable cache" while
DevTools stays open, which is the more reliable option while you're
actively making frontend changes.

There is nothing else in this app that caches — no Redis, no HTTP cache
headers set on the API responses, nothing in localStorage. If a restart +
hard refresh doesn't fix a stale-looking result, the cause is somewhere
else (wrong model file, wrong CSV, etc.), not caching.

---

## 9. One-paragraph summary for your supervisor

"We simulate three-phase power line faults in Simulink and export the
voltage/current waveforms as CSVs. From each waveform we compute 8 RMS
features that numerically summarize how disturbed each phase's voltage and
current were — including a zero-sequence current feature that specifically
detects whether a ground path is involved, which was necessary to tell
apart faults that disturb the same phases with and without grounding.
Because we only have one simulation run per fault type, we slice each run
into overlapping windows to generate enough labeled training examples,
then train and compare five different classical ML algorithms on those
features. The best-performing model is saved and used for live
predictions: when you pick a fault type in the app, it reloads that run's
real data, recomputes the same 8 features, and the saved model predicts
the fault class with a confidence score — which the app then checks
against the known correct answer."
