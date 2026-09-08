// Frontend logic for the Power Line Fault Detection demo.
// Vanilla JS, no build step - everything talks to the FastAPI backend
// at API_BASE via fetch().

const API_BASE = "http://localhost:8000";

// Fixed phase identity colors, reused everywhere a phase is drawn (waveform
// lines and feature bars alike) so "Phase A" always means the same color.
// Chosen from a colorblind-validated categorical palette - the more obvious
// blue/green/red choice puts red and green adjacent, which is very hard to
// tell apart under deuteranopia/protanopia.
const PHASE_COLORS = { A: "#2a78d6", B: "#eb6834", C: "#1baf7a" };
const PEAK_COLOR = "#eda100"; // distinct from phase colors - not a phase, an aggregate
const ZERO_SEQ_COLOR = "#e87ba4"; // distinct again - not a phase or a peak, a ground-path indicator

const faultSelect = document.getElementById("fault-select");
const runBtn = document.getElementById("run-btn");
const statusMsg = document.getElementById("status-msg");
const resultsSection = document.getElementById("results-section");

// ---------- Init ----------

async function init() {
  await loadFaultTypes();
  await loadModelMetrics();
  setupModals();
}

async function loadFaultTypes() {
  try {
    const res = await fetch(`${API_BASE}/api/fault-types`);
    const types = await res.json();
    faultSelect.innerHTML = types
      .map((t) => `<option value="${t.code}">${t.name}</option>`)
      .join("");
  } catch (err) {
    setStatus("Could not reach backend API. Is it running on port 8000?", true);
  }
}

// ---------- Run simulation ----------

runBtn.addEventListener("click", runSimulation);

async function runSimulation() {
  const faultType = faultSelect.value;
  if (!faultType) return;

  setStatus("Running simulation and ML prediction...");
  runBtn.disabled = true;

  try {
    const res = await fetch(`${API_BASE}/api/simulate?fault_type=${encodeURIComponent(faultType)}`);
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Request failed (${res.status})`);
    }
    const data = await res.json();
    renderResults(data);
    setStatus("");
    resultsSection.hidden = false;
  } catch (err) {
    setStatus(err.message, true);
  } finally {
    runBtn.disabled = false;
  }
}

function setStatus(msg, isError = false) {
  statusMsg.textContent = msg;
  statusMsg.classList.toggle("error", isError);
}

// ---------- Rendering results ----------

function renderResults(data) {
  renderWaveformChart(
    "voltage-chart",
    data.time,
    [
      { name: "Va", y: data.Va, color: PHASE_COLORS.A },
      { name: "Vb", y: data.Vb, color: PHASE_COLORS.B },
      { name: "Vc", y: data.Vc, color: PHASE_COLORS.C },
    ],
    "Voltage (V)",
    data.fault_window
  );

  renderWaveformChart(
    "current-chart",
    data.time,
    [
      { name: "Ia", y: data.Ia, color: PHASE_COLORS.A },
      { name: "Ib", y: data.Ib, color: PHASE_COLORS.B },
      { name: "Ic", y: data.Ic, color: PHASE_COLORS.C },
    ],
    "Current (A)",
    data.fault_window
  );

  renderFeaturesCharts(data.features);
  renderResultCard(data);
}

function renderWaveformChart(divId, time, series, yLabel, faultWindow) {
  const traces = series.map((s) => ({
    x: time,
    y: s.y,
    name: s.name,
    mode: "lines",
    line: { color: s.color, width: 1.5 },
  }));

  const layout = {
    margin: { t: 10, r: 20, b: 40, l: 55 },
    xaxis: { title: "Time (s)" },
    yaxis: { title: yLabel },
    legend: { orientation: "h", y: -0.2 },
    shapes: [],
  };

  // Shade the fault-active region so it's visually obvious where the
  // disturbance occurs. Skipped entirely for NoFault (fault_window: null).
  if (faultWindow) {
    layout.shapes.push({
      type: "rect",
      xref: "x",
      yref: "paper",
      x0: faultWindow[0],
      x1: faultWindow[1],
      y0: 0,
      y1: 1,
      fillcolor: "rgba(220, 38, 38, 0.12)",
      line: { width: 0 },
      layer: "below",
    });
  }

  Plotly.newPlot(divId, traces, layout, { responsive: true, displaylogo: false });
}

// Voltage (hundreds of volts) and current (tens of amps) features are
// plotted as two separate bar charts, each with its own correctly-scaled
// y-axis - putting both on one shared linear axis (as a single combined
// chart) squashes the current/peak bars down to nearly nothing next to the
// much larger voltage bars.
function renderFeaturesCharts(features) {
  const voltageBars = [
    { label: "rms_Va", value: features.rms_Va, color: PHASE_COLORS.A },
    { label: "rms_Vb", value: features.rms_Vb, color: PHASE_COLORS.B },
    { label: "rms_Vc", value: features.rms_Vc, color: PHASE_COLORS.C },
  ];
  const currentBars = [
    { label: "rms_Ia", value: features.rms_Ia, color: PHASE_COLORS.A },
    { label: "rms_Ib", value: features.rms_Ib, color: PHASE_COLORS.B },
    { label: "rms_Ic", value: features.rms_Ic, color: PHASE_COLORS.C },
    { label: "peak_I", value: features.peak_I, color: PEAK_COLOR },
    { label: "rms_I0", value: features.rms_I0, color: ZERO_SEQ_COLOR },
  ];

  renderFeatureBarChart("voltage-features-chart", voltageBars, "Volts (RMS)");
  renderFeatureBarChart("current-features-chart", currentBars, "Amps (RMS / peak)");
}

function renderFeatureBarChart(divId, bars, yLabel) {
  const trace = {
    x: bars.map((b) => b.label),
    y: bars.map((b) => b.value),
    type: "bar",
    marker: { color: bars.map((b) => b.color) },
    // Direct value labels on each bar - some of these fill colors (aqua,
    // yellow) don't have strong contrast against a white background, so
    // the exact value shouldn't depend on perceiving the fill precisely.
    text: bars.map((b) => b.value.toFixed(2)),
    textposition: "outside",
  };

  const layout = {
    margin: { t: 20, r: 20, b: 50, l: 55 },
    yaxis: { title: yLabel },
    bargap: 0.35,
  };

  Plotly.newPlot(divId, [trace], layout, { responsive: true, displaylogo: false });
}

function renderResultCard(data) {
  document.getElementById("result-actual").textContent = data.fault_type_actual;
  document.getElementById("result-predicted").textContent = data.fault_type_predicted;

  const confidencePct =
    data.confidence !== null && data.confidence !== undefined
      ? `${(data.confidence * 100).toFixed(1)}%`
      : "N/A";
  document.getElementById("result-confidence").textContent = confidencePct;

  const badge = document.getElementById("result-badge");
  if (data.correct) {
    badge.textContent = "✓ Correct";
    badge.className = "badge correct";
  } else {
    badge.textContent = "✗ Incorrect";
    badge.className = "badge incorrect";
  }
}

// ---------- Model metrics table ----------

async function loadModelMetrics() {
  try {
    const res = await fetch(`${API_BASE}/api/model-metrics`);
    if (!res.ok) return; // model not trained yet - leave table empty
    const metrics = await res.json();
    const tbody = document.querySelector("#metrics-table tbody");
    tbody.innerHTML = Object.entries(metrics)
      .map(
        ([name, m]) => `
        <tr>
          <td>${name}</td>
          <td>${(m.accuracy * 100).toFixed(2)}%</td>
          <td>${(m.precision * 100).toFixed(2)}%</td>
          <td>${(m.recall * 100).toFixed(2)}%</td>
          <td>${(m.f1_score * 100).toFixed(2)}%</td>
        </tr>`
      )
      .join("");
  } catch (err) {
    // Backend unreachable - already surfaced by loadFaultTypes().
  }
}

// ---------- Modals ----------

function setupModals() {
  const modelModal = document.getElementById("model-modal");
  const dataModal = document.getElementById("data-modal");

  document.getElementById("view-model-btn").addEventListener("click", () => {
    modelModal.hidden = false;
  });

  document.getElementById("browse-data-btn").addEventListener("click", () => {
    dataModal.hidden = false;
    showCsvList();
    loadCsvList();
  });

  document.querySelectorAll("[data-close-modal]").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.getElementById(btn.dataset.closeModal).hidden = true;
    });
  });

  // Click-outside-to-close for both modals.
  [modelModal, dataModal].forEach((overlay) => {
    overlay.addEventListener("click", (e) => {
      if (e.target === overlay) overlay.hidden = true;
    });
  });

  document.getElementById("csv-back-btn").addEventListener("click", showCsvList);
}

function showCsvList() {
  document.getElementById("csv-list").hidden = false;
  document.getElementById("csv-viewer").hidden = true;
}

async function loadCsvList() {
  const listEl = document.getElementById("csv-list");
  listEl.innerHTML = "<p>Loading...</p>";
  try {
    const res = await fetch(`${API_BASE}/api/csv-list`);
    const files = await res.json();
    if (files.length === 0) {
      listEl.innerHTML = "<p>No CSV files found in data/raw/.</p>";
      return;
    }
    listEl.innerHTML = files
      .map(
        (f) => `
        <div class="csv-list-item" data-filename="${f.filename}">
          <span class="fname">${f.filename}</span>
          <span class="meta">${f.row_count} rows · ${f.columns.join(", ")}</span>
        </div>`
      )
      .join("");

    listEl.querySelectorAll(".csv-list-item").forEach((item) => {
      item.addEventListener("click", () => loadCsvContent(item.dataset.filename));
    });
  } catch (err) {
    listEl.innerHTML = "<p>Failed to load file list.</p>";
  }
}

async function loadCsvContent(filename) {
  document.getElementById("csv-list").hidden = true;
  const viewer = document.getElementById("csv-viewer");
  viewer.hidden = false;
  document.getElementById("csv-viewer-title").textContent = "Loading " + filename + "...";
  document.getElementById("csv-table").innerHTML = "";

  try {
    const res = await fetch(`${API_BASE}/api/csv-content?filename=${encodeURIComponent(filename)}`);
    if (!res.ok) throw new Error("Failed to load file");
    const data = await res.json();

    document.getElementById("csv-viewer-title").textContent =
      `${data.filename} — ${data.row_count} rows`;

    const table = document.getElementById("csv-table");
    const thead = `<thead><tr>${data.columns.map((c) => `<th>${c}</th>`).join("")}</tr></thead>`;
    const tbody = `<tbody>${data.rows
      .map(
        (row) =>
          `<tr>${data.columns.map((c) => `<td>${row[c]}</td>`).join("")}</tr>`
      )
      .join("")}</tbody>`;
    table.innerHTML = thead + tbody;
  } catch (err) {
    document.getElementById("csv-viewer-title").textContent = "Failed to load " + filename;
  }
}

init();
