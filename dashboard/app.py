"""
Dashboard -- a thin visual layer over the engine, nothing more.

DESIGN RULE (see README's "data contract for a dashboard" section): this
file only ever calls engine.step() and reads the dict it returns, plus
FrameRenderer.render() for the visual camera panel. It never reaches into
Kalman/camera/motion internals directly. If you need more data on screen,
add a field to engine.step()'s return dict -- don't bypass it here.

Run:
    streamlit run dashboard/app.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
import numpy as np
import pandas as pd
import streamlit as st

from fsoc_tracker import TrackingEngine, FrameRenderer, RenderConfig
from fsoc_tracker.scenarios import MOTION_PROFILES, DISTURBANCE_LEVELS, build_scenario

st.set_page_config(page_title="FSOC Coarse-Alignment Tracker", layout="wide")

ATMOSPHERES = ["clear", "haze", "fog", "rain", "low_light"]
NOISE_CHOICES = {
    "none": ("none",),
    "gaussian": ("gaussian",),
    "salt & pepper": ("salt_pepper",),
    "gaussian + salt & pepper": ("gaussian", "salt_pepper"),
}

page = st.sidebar.radio("View", ["Live Tracking Simulation", "AI Detector Benchmark"])

# ============================================================
# PAGE 1 -- live simulation, driven entirely by engine.step()
# ============================================================
if page == "Live Tracking Simulation":
    st.title("FSOC Coarse-Alignment -- Live Tracking Simulation")
    st.caption("PS 26169 -- every number on this page comes from one call: `engine.step()`.")

    with st.sidebar:
        st.header("Scenario")
        motion = st.selectbox("Motion profile", MOTION_PROFILES, index=3)
        severity = st.selectbox("Disturbance level", list(DISTURBANCE_LEVELS.keys()), index=2)
        duration_s = st.slider("Duration (s)", 3, 30, 12)
        seed = st.number_input("Random seed", value=42, step=1)

        st.header("Camera feed look (cosmetic only)")
        st.caption("Purely visual -- doesn't affect the numeric tracking, which uses its own disturbance model.")
        atmosphere = st.selectbox("Atmosphere", ATMOSPHERES, index=0)
        noise_choice = st.selectbox("Sensor noise", list(NOISE_CHOICES.keys()), index=1)

        run_clicked = st.button("Run simulation", type="primary")

    if run_clicked:
        scenario = build_scenario(motion, severity, duration_s=duration_s)
        engine = TrackingEngine(scenario.config, seed=int(seed))
        renderer = FrameRenderer(
            RenderConfig(noise_types=NOISE_CHOICES[noise_choice], atmosphere=atmosphere),
            seed=int(seed),
        )

        n_frames = int(duration_s / engine.dt)

        col_video, col_charts = st.columns([1, 1.3])
        with col_video:
            st.subheader("Camera feed")
            video_slot = st.empty()
            status_slot = st.empty()
        with col_charts:
            st.subheader("Tracking error (mrad)")
            error_chart_slot = st.empty()
            st.subheader("Lock status")
            lock_chart_slot = st.empty()

        metrics_slot = st.empty()

        history = {"t": [], "theta_e_mrad": [], "locked": []}
        FRAME_STRIDE = max(1, n_frames // 150)  # cap redraws for long runs

        for i in range(n_frames):
            state = engine.step()
            history["t"].append(state["t"])
            history["theta_e_mrad"].append(state["theta_e_mrad"] if state["theta_e_mrad"] is not None else np.nan)
            history["locked"].append(1 if state["locked"] else 0)

            if i % FRAME_STRIDE == 0 or i == n_frames - 1:
                # visual panel: render what the camera sees using the
                # engine's own noisy measurement / filtered estimate,
                # purely for display
                u = state.get("meas_u")
                v = state.get("meas_v")
                frame = renderer.render(u, v)
                frame_rgb = np.stack([frame] * 3, axis=-1)

                if "filt_u" in state:
                    fu, fv = int(state["filt_u"]), int(state["filt_v"])
                    r = 10
                    h, w = frame_rgb.shape[:2]
                    y0, y1 = max(0, fv - r), min(h, fv + r)
                    x0, x1 = max(0, fu - r), min(w, fu + r)
                    frame_rgb[y0:y1, max(0, x0):max(0, x0) + 1] = [0, 120, 255]
                    frame_rgb[y0:y1, min(w - 1, x1):min(w - 1, x1) + 1] = [0, 120, 255]
                    frame_rgb[max(0, y0):max(0, y0) + 1, x0:x1] = [0, 120, 255]
                    frame_rgb[min(h - 1, y1):min(h - 1, y1) + 1, x0:x1] = [0, 120, 255]

                video_slot.image(frame_rgb, caption=f"t={state['t']:.2f}s  (blue box = filtered estimate)", use_container_width=True)
                status_slot.markdown(
                    f"**Detected:** {state['detected']}  |  **Locked:** {state['locked']}  |  "
                    f"**Error:** {state['theta_e_mrad']:.2f} mrad" if state["theta_e_mrad"] is not None
                    else f"**Detected:** {state['detected']}  |  **Locked:** {state['locked']}  |  **Error:** --"
                )

                df = pd.DataFrame(history).set_index("t")
                error_chart_slot.line_chart(df[["theta_e_mrad"]])
                lock_chart_slot.line_chart(df[["locked"]])

                snap = state["metrics"]
                if snap:
                    metrics_slot.markdown(
                        f"**FPS:** {snap.get('fps', 0):.0f}   |   "
                        f"**Avg error:** {snap.get('avg_error_mrad', 0):.2f} mrad   |   "
                        f"**Max error:** {snap.get('max_error_mrad', 0):.2f} mrad   |   "
                        f"**Lock retention:** {snap.get('lock_retention_rate', 0):.1%}   |   "
                        f"**Acquisition time:** {snap.get('acquisition_time_s')}"
                    )

        final = engine.metrics.final_report()
        st.success("Run complete.")
        st.json(final)
    else:
        st.info("Pick a scenario in the sidebar and click **Run simulation**.")

# ============================================================
# PAGE 2 -- surfaces the Part 4b detector benchmark results
# ============================================================
else:
    st.title("AI Detector Benchmark -- Naive vs. Trained Classifier")
    st.caption("Results from experiments/benchmark_detector.py (Part 4b). Run it first if you don't see data below.")

    out_dir = os.path.join(os.path.dirname(__file__), "..", "experiments", "output")
    summary_path = os.path.join(out_dir, "detector_benchmark_summary.csv")
    chart_path = os.path.join(out_dir, "detector_benchmark_chart.png")
    fl_chart_path = os.path.join(out_dir, "detector_false_lock_chart.png")
    train_report_path = os.path.join(out_dir, "detector_training_report.txt")

    if os.path.exists(summary_path):
        df = pd.read_csv(summary_path)
        st.subheader("Overall summary")
        st.dataframe(df, use_container_width=True)

        st.markdown(
            "**Reading this honestly:** raw success rate is usually near-identical between "
            "detectors -- the real difference is in *how* each one fails. The naive detector "
            "never says \"not found\"; every miss becomes a confident, silent lock onto a noise "
            "blob. The trained classifier converts most of those into a safe \"no detection\" "
            "instead. See the false-lock chart below."
        )

        col1, col2 = st.columns(2)
        if os.path.exists(chart_path):
            col1.image(chart_path, caption="Success rate by atmosphere")
        if os.path.exists(fl_chart_path):
            col2.image(fl_chart_path, caption="False-lock rate by atmosphere (the dangerous failure mode)")

        if os.path.exists(train_report_path):
            with st.expander("Classifier training report"):
                st.text(open(train_report_path).read())
    else:
        st.warning(
            "No benchmark results found yet. Run these first:\n\n"
            "```\npython experiments/train_detector.py\npython experiments/benchmark_detector.py\n```"
        )
