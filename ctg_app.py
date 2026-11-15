from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from scipy.io import loadmat
from scipy.ndimage import median_filter

# ---------------------------------------------------------------------
# CTG Research Evaluator — interactive Streamlit interface
# Model artifact expected from CTG_analysis.ipynb:
# {"model": fitted sklearn Pipeline, "feature_columns": [...], "classes": [...]}
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="CTG Research Evaluator",
    page_icon="〰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

FEATURE_COLUMNS = [
    "fhr_mean", "fhr_variance", "baseline_fhr",
    "acceleration_count", "deceleration_count",
    "toco_mean", "toco_variance",
]
CLASS_ORDER = ["Normal", "Suspect", "Pathologic"]
EVENT_STYLE = {
    "Acceleration": {"color": "#159D83", "fill": "rgba(21,157,131,0.18)"},
    "Deceleration": {"color": "#D95757", "fill": "rgba(217,87,87,0.18)"},
}

st.markdown("""
<style>
:root { --ctg-blue:#315A9B; }
.block-container {max-width: 1580px; padding-top: 1.35rem; padding-bottom: 3rem;}
[data-testid="stMetric"] {
  background: linear-gradient(135deg, rgba(49,90,155,.07), rgba(49,90,155,.015));
  border: 1px solid rgba(120,140,165,.22); padding: 13px 15px; border-radius: 13px;
}
[data-testid="stMetricLabel"] {font-size:.82rem;}
[data-testid="stMetricValue"] {font-size:1.55rem;}
section[data-testid="stSidebar"] {border-right:1px solid rgba(120,140,165,.18);}
.ctg-muted {color:#718096; font-size:.88rem;}
.ctg-step {font-size:.76rem; letter-spacing:.08em; text-transform:uppercase;
  color:#6b7d93; font-weight:700;}
.ctg-card {border:1px solid rgba(120,140,165,.25); border-radius:13px; padding:14px 16px;}
</style>
""", unsafe_allow_html=True)


def safe_float(value):
    try:
        value = float(value)
        return value if np.isfinite(value) else None
    except Exception:
        return None


def get_api_key() -> str:
    try:
        return str(st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))).strip()
    except Exception:
        return os.getenv("GROQ_API_KEY", "").strip()


def extract_vector(mat: dict, key: str) -> np.ndarray:
    if key not in mat:
        raise ValueError(f"MAT file does not contain the required '{key}' variable.")
    arr = np.asarray(mat[key]).squeeze()
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    if arr.size < 2:
        raise ValueError(f"'{key}' must contain at least two samples.")
    return arr.astype(float)


def load_mat_recording(upload):
    raw = loadmat(io.BytesIO(upload.getvalue()), squeeze_me=True, struct_as_record=False)
    fhr = extract_vector(raw, "fhr")
    toco = extract_vector(raw, "toco")
    n = min(len(fhr), len(toco))
    return fhr[:n], toco[:n]


def find_runs(mask: np.ndarray, fs: float, min_duration: float, label: str):
    padded = np.r_[False, mask, False]
    changes = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    found = []
    for start, end in zip(starts, ends):
        duration = (end - start) / fs
        if duration >= min_duration:
            found.append({
                "event": label,
                "start_idx": int(start),
                "end_idx": int(end),
                "start_seconds": float(start / fs),
                "end_seconds": float(end / fs),
                "duration_seconds": float(duration),
            })
    return found


def extract_ctg_features(fhr, toco, fs=4.0):
    """Feature calculation follows the uploaded notebook's 10-minute median baseline."""
    fhr = np.asarray(fhr, dtype=float).flatten()
    toco = np.asarray(toco, dtype=float).flatten()
    valid = np.isfinite(fhr)
    fhr_valid = fhr[valid]
    if not len(fhr_valid):
        raise ValueError("FHR has no finite samples.")

    filled = fhr.copy()
    if (~valid).any():
        if len(fhr_valid) == 1:
            filled[~valid] = fhr_valid[0]
        else:
            filled[~valid] = np.interp(
                np.flatnonzero(~valid), np.flatnonzero(valid), fhr_valid
            )

    # Match notebook: 10 minutes × 60 seconds × sampling rate; odd-sized filter.
    window_samples = int(10 * 60 * fs)
    if window_samples % 2 == 0:
        window_samples += 1
    baseline = median_filter(filled, size=window_samples, mode="nearest")

    acc_mask = valid & (filled >= baseline + 15)
    dec_mask = valid & (filled <= baseline - 15)
    events = find_runs(acc_mask, fs, 15, "Acceleration")
    events += find_runs(dec_mask, fs, 15, "Deceleration")
    events.sort(key=lambda e: e["start_seconds"])

    toco_valid = toco[np.isfinite(toco)]
    features = {
        "fhr_mean": float(np.mean(fhr_valid)),
        "fhr_variance": float(np.var(fhr_valid)),
        "baseline_fhr": float(np.median(baseline[valid])),
        "acceleration_count": int(sum(e["event"] == "Acceleration" for e in events)),
        "deceleration_count": int(sum(e["event"] == "Deceleration" for e in events)),
        "toco_mean": float(np.mean(toco_valid)) if len(toco_valid) else float("nan"),
        "toco_variance": float(np.var(toco_valid)) if len(toco_valid) else float("nan"),
    }
    return baseline, events, features


def build_chart(fhr, toco, baseline, events, fs, start_s, end_s,
                show_baseline=True, show_acc=True, show_dec=True,
                show_toco=True):
    n = len(fhr)
    lo = max(0, int(start_s * fs))
    hi = min(n, max(lo + 2, int(end_s * fs)))
    # Keep charts responsive for very long traces while preserving the full-resolution data
    # in the underlying analysis and event table.
    stride = max(1, int(np.ceil((hi - lo) / 18000)))
    idx = np.arange(lo, hi, stride)
    t = idx / fs

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.075,
        row_heights=[0.64, 0.36],
        subplot_titles=("Fetal heart rate", "Uterine activity (TOCO)"),
    )
    fig.add_trace(go.Scattergl(
        x=t, y=fhr[idx], mode="lines", name="FHR",
        line=dict(color="#315A9B", width=1.35),
        hovertemplate="Time: %{x:.1f} s<br>FHR: %{y:.1f} bpm<extra></extra>",
    ), row=1, col=1)

    if show_baseline:
        fig.add_trace(go.Scattergl(
            x=t, y=baseline[idx], mode="lines", name="10-min median baseline",
            line=dict(color="#D49B24", width=1.2, dash="dash"),
            hovertemplate="Time: %{x:.1f} s<br>Baseline: %{y:.1f} bpm<extra></extra>",
        ), row=1, col=1)

    for event_name, visible in [("Acceleration", show_acc), ("Deceleration", show_dec)]:
        if not visible:
            continue
        color = EVENT_STYLE[event_name]["color"]
        for event in events:
            if event["event"] != event_name:
                continue
            a, b = event["start_seconds"], event["end_seconds"]
            if b < start_s or a > end_s:
                continue
            fig.add_vrect(
                x0=max(a, start_s), x1=min(b, end_s),
                fillcolor=EVENT_STYLE[event_name]["fill"],
                line_width=0, row=1, col=1,
                annotation_text=event_name if a >= start_s and a <= end_s else None,
                annotation_font_size=10,
            )

    if show_toco:
        fig.add_trace(go.Scattergl(
            x=t, y=toco[idx], mode="lines", name="TOCO",
            line=dict(color="#8055A5", width=1.25),
            hovertemplate="Time: %{x:.1f} s<br>TOCO: %{y:.1f}<extra></extra>",
        ), row=2, col=1)

    fig.update_yaxes(title_text="FHR (bpm)", row=1, col=1)
    fig.update_yaxes(title_text="TOCO (a.u.)", row=2, col=1)
    fig.update_xaxes(title_text="Time (seconds)", row=2, col=1, range=[start_s, end_s])
    fig.update_layout(
        height=660, margin=dict(l=20, r=25, t=65, b=25),
        hovermode="x unified", template="plotly_white",
        legend=dict(orientation="h", y=1.08, x=0),
        font=dict(family="Arial", size=12),
    )
    return fig


def load_model(upload):
    if upload is not None:
        artifact = joblib.load(io.BytesIO(upload.getvalue()))
        source = upload.name
    else:
        local = Path(__file__).with_name("ctg_random_forest.joblib")
        if not local.exists():
            return None, None, None
        artifact = joblib.load(local)
        source = local.name

    if isinstance(artifact, dict) and "model" in artifact:
        model = artifact["model"]
        columns = artifact.get("feature_columns", FEATURE_COLUMNS)
        classes = artifact.get("classes", None)
    else:
        model, columns, classes = artifact, FEATURE_COLUMNS, None
    return model, list(columns), classes


def predict(model, columns, features):
    frame = pd.DataFrame([{c: features.get(c, np.nan) for c in columns}])
    label = str(model.predict(frame)[0])
    probs = {}
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(frame)[0]
        clf = getattr(model, "named_steps", {}).get("classifier") if hasattr(model, "named_steps") else None
        classes = getattr(clf, "classes_", getattr(model, "classes_", None))
        if classes is not None:
            probs = {str(c): float(v) for c, v in zip(classes, p)}
    return label, probs


def run_agents(payload: dict[str, Any]):
    from groq import Groq
    key = get_api_key()
    if not key:
        raise RuntimeError("No GROQ_API_KEY found. Add it to your environment or Streamlit secrets.")
    client = Groq(api_key=key)
    model_name = "openai/gpt-oss-120b"

    interpreter_prompt = """You are a CTG signal interpretation assistant for a research prototype.
Explain the ML output using only supplied numerical features, waveform anomalies and probabilities.
Distinguish measurements from predictions; do not diagnose or infer causes; identify signal-quality
concerns and uncertainty; do not invent clinical history or events. Return JSON with prediction_summary,
supporting_features (list), waveform_observations (list), uncertainties (list), clinical_caveat."""
    risk_prompt = """You are a clinical safety and risk-review assistant in a research CTG prototype.
Do not diagnose, determine fetal status or assign a validated obstetric risk tier. Identify evidence
warranting qualified clinician review, gaps and uncertainty. Do not invent guideline thresholds.
Return JSON with review_priority (routine_review|prompt_review|urgent_clinical_review|indeterminate),
reasons (list), limitations (list), requires_clinician_assessment (boolean). This is not a validated
clinical risk score."""
    workflow_prompt = """You are a clinical workflow assistant for a CTG research demonstration.
Suggest checking signal quality, reviewing the complete CTG with maternal/pregnancy context, and
contacting the responsible obstetric team when indicated. Do not issue treatment orders or recommend
interventions. Return JSON with workflow_prompts (list), escalation_note,
requires_clinician_review (boolean), not_a_treatment_order (boolean)."""

    def call(prompt, data):
        response = client.chat.completions.create(
            model=model_name, temperature=0.2,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(data, allow_nan=False)},
            ],
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)

    interpreter = call(interpreter_prompt, payload)
    risk = call(risk_prompt, {"ctg_input": payload, "interpreter_output": interpreter})
    workflow = call(workflow_prompt, {
        "ctg_input": payload, "interpretation": interpreter, "risk_review": risk
    })
    return {"input": payload, "interpreter": interpreter,
            "risk_review": risk, "workflow": workflow,
            "disclaimer": "Research prototype only; not a diagnosis, validated risk classification or treatment order."}


def render_bullets(items):
    if not items:
        st.caption("No items returned.")
        return
    for item in items:
        st.markdown(f"- {item}")


# --------------------------- Sidebar ---------------------------------
with st.sidebar:
    st.markdown("## 〰️ CTG Evaluator")
    st.caption("Research analysis workspace")
    st.markdown("---")
    st.markdown("#### 1 · Recording")
    mat_upload = st.file_uploader(
        "Upload CTG recording", type=["mat"],
        help="MATLAB file must contain 1-D variables named fhr and toco.",
    )
    model_upload = st.file_uploader(
        "Trained Random Forest (optional)", type=["joblib", "pkl"],
        help="Upload the ctg_random_forest.joblib artifact saved by your notebook.",
    )
    st.markdown("---")
    st.markdown("#### 2 · Signal settings")
    fs = st.number_input("Sampling frequency (Hz)", 0.1, 100.0, 4.0, 0.5)
    st.caption("The supplied notebook uses 4 Hz.")
    with st.expander("Detection settings", expanded=False):
        st.caption("Notebook-aligned defaults")
        deviation = st.number_input("Baseline deviation (BPM)", 5, 40, 15, 1)
        min_duration = st.number_input("Minimum event duration (seconds)", 5, 120, 15, 5)
        baseline_minutes = st.number_input("Median baseline window (minutes)", 1, 20, 10, 1)
    st.markdown("---")
    st.markdown("#### 3 · AI review")
    ai_enabled = st.toggle("Enable multi-agent review", value=False)
    if ai_enabled:
        st.caption("Requires GROQ_API_KEY in environment or Streamlit secrets.")
    st.markdown("---")
    st.caption("Research prototype • Not for clinical decision-making")


# --------------------------- Header ----------------------------------
st.title("CTG Research Evaluator")
st.markdown(
    "Explore fetal heart-rate and uterine-activity signals, review the model output, "
    "and inspect the multi-agent research interpretation."
)
st.markdown(
    '<div class="ctg-step">WORKFLOW &nbsp; 01 Upload &nbsp; → &nbsp; 02 Explore signals '
    '&nbsp; → &nbsp; 03 Review prediction &nbsp; → &nbsp; 04 AI interpretation</div>',
    unsafe_allow_html=True,
)

if not mat_upload:
    st.markdown("")
    a, b, c = st.columns(3)
    with a:
        st.markdown("### <span style='color:#315A9B'>01</span> Upload", unsafe_allow_html=True)
        st.write("Choose a MATLAB recording containing `fhr` and `toco` arrays.")
    with b:
        st.markdown("### <span style='color:#159D83'>02</span> Explore", unsafe_allow_html=True)
        st.write("Inspect the dual trace, baseline and automatically detected episodes.")
    with c:
        st.markdown("### <span style='color:#8055A5'>03</span> Interpret", unsafe_allow_html=True)
        st.write("Review the trained classifier and optional three-agent report.")
    st.info("Start by uploading a `.mat` recording in the left sidebar.")
    st.markdown("**Expected variables:** `fhr` = fetal heart rate; `toco` = uterine activity.")
    st.stop()

# --------------------------- Load / analyze ---------------------------
try:
    fhr, toco = load_mat_recording(mat_upload)
    valid_fhr = np.isfinite(fhr)
    if valid_fhr.sum() == 0:
        raise ValueError("FHR contains no finite samples.")
    # The notebook's defaults are retained; the UI exposes settings for exploration.
    filled = fhr.copy()
    if (~valid_fhr).any():
        good = np.flatnonzero(valid_fhr)
        filled[~valid_fhr] = np.interp(np.flatnonzero(~valid_fhr), good, fhr[valid_fhr])
    win = int(baseline_minutes * 60 * fs)
    if win % 2 == 0:
        win += 1
    baseline = median_filter(filled, size=max(1, win), mode="nearest")
    acc_mask = valid_fhr & (filled >= baseline + deviation)
    dec_mask = valid_fhr & (filled <= baseline - deviation)
    events = find_runs(acc_mask, fs, min_duration, "Acceleration")
    events += find_runs(dec_mask, fs, min_duration, "Deceleration")
    events.sort(key=lambda e: e["start_seconds"])
    features = {
        "fhr_mean": float(np.nanmean(fhr)),
        "fhr_variance": float(np.nanvar(fhr)),
        "baseline_fhr": float(np.nanmedian(baseline[valid_fhr])),
        "acceleration_count": sum(e["event"] == "Acceleration" for e in events),
        "deceleration_count": sum(e["event"] == "Deceleration" for e in events),
        "toco_mean": float(np.nanmean(toco)) if np.isfinite(toco).any() else float("nan"),
        "toco_variance": float(np.nanvar(toco)) if np.isfinite(toco).any() else float("nan"),
    }
except Exception as exc:
    st.error(f"Unable to read or analyse this recording: {exc}")
    st.info("Check that the MAT file contains numeric `fhr` and `toco` arrays. MATLAB v7.3 files may require conversion to an earlier MAT format.")
    st.stop()

duration = len(fhr) / fs
duration_min = duration / 60
signal_quality = 100 * valid_fhr.mean()

# Attempt model loading without interrupting signal review.
try:
    model, model_columns, model_classes = load_model(model_upload)
    model_error = None
except Exception as exc:
    model, model_columns, model_classes = None, None, None
    model_error = str(exc)

prediction, probabilities = None, {}
if model is not None:
    try:
        prediction, probabilities = predict(model, model_columns, features)
    except Exception as exc:
        model_error = str(exc)

# --------------------------- Summary metrics -------------------------
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Recording length", f"{duration_min:.1f} min")
m2.metric("Sampling rate", f"{fs:g} Hz")
m3.metric("FHR signal present", f"{signal_quality:.1f}%")
m4.metric("Accelerations", sum(e["event"] == "Acceleration" for e in events))
m5.metric("Decelerations", sum(e["event"] == "Deceleration" for e in events))

if signal_quality < 95:
    st.warning("Some FHR samples are missing or non-finite. Review signal quality before interpreting results.")
else:
    st.success("Recording loaded. Review the waveform and automated annotations below.")

# --------------------------- Main tabs -------------------------------
tab_signal, tab_events, tab_ml, tab_ai = st.tabs([
    "〰️ Signal explorer", "🏷️ Event explorer", "🧠 ML prediction", "🤖 AI interpretation"
])

with tab_signal:
    st.subheader("Dual-channel CTG waveform")
    st.caption("Drag to zoom • Double-click to reset • Hover for sample values")
    control1, control2, control3 = st.columns([1.3, 1, 1])
    with control1:
        end_default = min(duration, 10 * 60)
        if duration > 60:
            view = st.slider(
                "Visible time window (seconds)",
                min_value=0.0, max_value=float(duration),
                value=(0.0, float(end_default)),
                step=max(1.0, round(duration / 1000, 1)),
                key="time_window",
            )
        else:
            view = (0.0, float(duration))
            st.caption(f"Full recording shown: {duration:.1f} seconds")
    with control2:
        show_baseline = st.checkbox("Show estimated baseline", True)
        show_acc = st.checkbox("Show accelerations", True)
    with control3:
        show_dec = st.checkbox("Show decelerations", True)
        show_toco = st.checkbox("Show TOCO", True)

    fig = build_chart(
        fhr, toco, baseline, events, fs, view[0], view[1],
        show_baseline, show_acc, show_dec, show_toco,
    )
    st.plotly_chart(fig, use_container_width=True, config={
        "displaylogo": False, "scrollZoom": True, "modeBarButtonsToRemove": ["lasso2d", "select2d"]
    })
    st.markdown(
        '<span style="color:#159D83">● Acceleration</span> &nbsp;&nbsp; '
        '<span style="color:#D95757">● Deceleration</span> &nbsp;&nbsp; '
        '<span style="color:#D49B24">┄ Baseline</span> &nbsp;&nbsp; '
        '<span style="color:#8055A5">━ TOCO</span>',
        unsafe_allow_html=True,
    )
    with st.expander("How to read this view"):
        st.write(
            "The upper panel displays FHR and the estimated rolling-median baseline. "
            "Shaded regions indicate rule-detected periods above or below the baseline. "
            "The lower panel displays uterine activity (TOCO). These overlays are exploratory "
            "signal-processing annotations, not clinically validated event labels."
        )

with tab_events:
    st.subheader("Detected FHR event episodes")
    st.caption("Events are contiguous regions meeting the selected deviation and minimum-duration settings.")
    if events:
        event_df = pd.DataFrame(events)
        event_df.insert(0, "Event ID", [f"E{i+1:02d}" for i in range(len(event_df))])
        event_df["Start"] = event_df["start_seconds"].map(lambda x: f"{int(x//60):02d}:{x%60:05.2f}")
        event_df["End"] = event_df["end_seconds"].map(lambda x: f"{int(x//60):02d}:{x%60:05.2f}")
        event_df["Duration (s)"] = event_df["duration_seconds"].round(1)
        display_df = event_df[["Event ID", "event", "Start", "End", "Duration (s)"]]
        st.dataframe(
            display_df, use_container_width=True, hide_index=True,
            column_config={"event": st.column_config.TextColumn("Event type")},
        )
        selected_id = st.selectbox(
            "Inspect an event", options=event_df["Event ID"].tolist(),
            format_func=lambda x: (
                f"{x} · {event_df.loc[event_df['Event ID']==x, 'event'].iloc[0]} · "
                f"{event_df.loc[event_df['Event ID']==x, 'Start'].iloc[0]}–"
                f"{event_df.loc[event_df['Event ID']==x, 'End'].iloc[0]}"
            ),
        )
        selected_event = event_df.loc[event_df["Event ID"] == selected_id].iloc[0]
        e1, e2, e3 = st.columns(3)
        e1.metric("Event", selected_event["event"])
        e2.metric("Duration", f"{selected_event['duration_seconds']:.1f} s")
        e3.metric("Start time", f"{selected_event['start_seconds']:.1f} s")
        st.download_button(
            "Download event table (CSV)",
            data=display_df.to_csv(index=False).encode("utf-8"),
            file_name="ctg_detected_events.csv",
            mime="text/csv",
        )
    else:
        st.info("No events met the current detection settings.")
        st.caption("You can adjust the deviation and minimum duration in the sidebar. The default settings match the notebook's 15 BPM and 15-second criteria.")

with tab_ml:
    st.subheader("Machine-learning classification")
    if model_error:
        st.error(f"Model could not be loaded or applied: {model_error}")
    if prediction is None:
        st.warning("No usable trained model is available. Upload `ctg_random_forest.joblib` in the sidebar or place it beside `ctg_app.py`.")
        st.markdown(
            "The notebook saves a dictionary containing the fitted model, feature-column order and class names. "
            "The application intentionally does not generate a prediction without that artifact."
        )
    else:
        st.caption("Prediction from the supplied trained model. This is a model output, not a clinical diagnosis.")
        p1, p2 = st.columns([1, 1.4])
        with p1:
            st.markdown("#### Predicted class")
            st.markdown(f"### {prediction}")
            st.metric("Model confidence (highest class score)", f"{max(probabilities.values()):.1%}" if probabilities else "Not available")
        with p2:
            st.markdown("#### Class probabilities")
            if probabilities:
                order = [c for c in CLASS_ORDER if c in probabilities] + [c for c in probabilities if c not in CLASS_ORDER]
                bar = go.Figure(go.Bar(
                    x=[probabilities[c] for c in order], y=order, orientation="h",
                    marker_color=["#159D83" if c == "Normal" else "#D49B24" if c == "Suspect" else "#D95757" for c in order],
                    text=[f"{probabilities[c]:.1%}" for c in order], textposition="outside",
                ))
                bar.update_layout(
                    height=240, xaxis=dict(range=[0, 1.08], tickformat=".0%", title="Model-estimated probability"),
                    yaxis=dict(autorange="reversed"), margin=dict(l=10, r=45, t=10, b=35),
                    template="plotly_white",
                )
                st.plotly_chart(bar, use_container_width=True, config={"displayModeBar": False})
            else:
                st.info("The loaded estimator does not expose class probabilities.")
        st.markdown("#### Features supplied to the model")
        feat_df = pd.DataFrame([
            {"Feature": k, "Value": features.get(k)} for k in (model_columns or FEATURE_COLUMNS)
        ])
        st.dataframe(feat_df, use_container_width=True, hide_index=True)
    with st.expander("All extracted signal features"):
        st.json({k: safe_float(v) if safe_float(v) is not None else v for k, v in features.items()})

with tab_ai:
    st.subheader("Clinical AI multi-agent review")
    st.caption("Three sequential agents: signal interpreter → safety/risk reviewer → workflow assistant.")
    if not ai_enabled:
        st.info("Enable **Multi-agent review** in the sidebar to activate the AI report.")
    elif prediction is None:
        st.warning("A working trained ML model is required before running the agents.")
    elif not get_api_key():
        st.warning("Add your Groq API key as `GROQ_API_KEY` before running the agents.")
        st.code('export GROQ_API_KEY="your_api_key"', language="bash")
    else:
        if st.button("Run multi-agent interpretation", type="primary", use_container_width=False):
            payload = {
                "ml_prediction": prediction,
                "ml_probabilities": probabilities,
                "clinical_features": {k: safe_float(v) if safe_float(v) is not None else v for k, v in features.items()},
                "waveform_anomalies": [
                    {k: e[k] for k in ("event", "start_seconds", "end_seconds", "duration_seconds")}
                    | {"detection_method": "baseline_deviation_rule", "verified": False}
                    for e in events
                ],
                "recording_metadata": {
                    "sampling_rate_hz": float(fs),
                    "duration_seconds": float(duration),
                    "signal_completeness_percent": float(signal_quality),
                },
                "purpose": "research_prototype",
            }
            try:
                with st.spinner("Agents are reviewing the recording and model output…"):
                    st.session_state["ctg_agent_report"] = run_agents(payload)
                st.success("Multi-agent report generated.")
            except Exception as exc:
                st.error(f"Agent workflow failed: {exc}")

        report = st.session_state.get("ctg_agent_report")
        if report:
            interpreter = report.get("interpreter", {})
            risk = report.get("risk_review", {})
            workflow = report.get("workflow", {})
            priority = str(risk.get("review_priority", "indeterminate")).replace("_", " ").title()
            r1, r2, r3 = st.columns(3)
            r1.metric("Workflow review label", priority)
            r2.metric("Clinician assessment", "Flagged" if risk.get("requires_clinician_assessment") else "Not flagged")
            r3.metric("Workflow review", "Required" if workflow.get("requires_clinician_review") else "Not flagged")

            with st.expander("Agent 1 · Signal interpreter", expanded=True):
                st.markdown("**Prediction summary**")
                st.write(interpreter.get("prediction_summary", "Not provided."))
                st.markdown("**Supporting features**")
                render_bullets(interpreter.get("supporting_features", []))
                st.markdown("**Waveform observations**")
                render_bullets(interpreter.get("waveform_observations", []))
                st.markdown("**Uncertainties**")
                render_bullets(interpreter.get("uncertainties", []))
                st.info(interpreter.get("clinical_caveat", "Research prototype only."))
            with st.expander("Agent 2 · Risk reviewer", expanded=True):
                st.markdown("**Reasons for review**")
                render_bullets(risk.get("reasons", []))
                st.markdown("**Limitations**")
                render_bullets(risk.get("limitations", []))
            with st.expander("Agent 3 · Workflow assistant", expanded=True):
                render_bullets(workflow.get("workflow_prompts", []))
                st.write(workflow.get("escalation_note", ""))
                st.caption("Workflow prompts only; not treatment orders.")
            st.download_button(
                "Download complete AI report (JSON)",
                data=json.dumps(report, indent=2, ensure_ascii=False),
                file_name="ctg_ai_evaluation_report.json",
                mime="application/json",
            )

st.divider()
with st.expander("Analysis settings and methodology"):
    st.markdown(
        f"- Sampling frequency: **{fs:g} Hz**\n"
        f"- Baseline: **{baseline_minutes} minute rolling median**\n"
        f"- Event deviation: **±{deviation} BPM**\n"
        f"- Minimum event duration: **{min_duration} seconds**\n"
        "- Model features: FHR mean, FHR variance, median baseline, acceleration count, "
        "deceleration count, TOCO mean and TOCO variance."
    )
    st.caption("Changing event settings changes the event counts supplied to the model. For valid model use, retain the training settings from the notebook (4 Hz, 10-minute baseline, ±15 BPM, 15 seconds).")

st.warning(
    "Clinical safety notice: This is a research/evaluation prototype, not a medical device or "
    "validated clinical decision-support system. Rule-based event overlays and ML outputs are "
    "not diagnoses or validated fetal-status assessments. Do not use this interface for triage "
    "or treatment. A qualified clinician must review the complete trace and clinical context."
)
