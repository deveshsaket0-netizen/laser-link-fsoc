# FSOC Coarse-Alignment Tracking Engine (headless core)

A dashboard-agnostic implementation of the math framework (see
`FSOC_Tracking_Math_Framework.md`). Every module maps 1:1 to a section:

| File | Math framework section |
|---|---|
| `fsoc_tracker/motion.py` | §4.3 beacon trajectory generators |
| `fsoc_tracker/camera.py` | §2 camera & coordinate model |
| `fsoc_tracker/disturbances.py` | §4.4 disturbance models |
| `fsoc_tracker/kalman.py` | §5 Kalman filter |
| `fsoc_tracker/metrics.py` | §3.4 performance metrics |
| `fsoc_tracker/engine.py` | orchestration + §6 controller |

## Install
```
pip install -r requirements.txt
```
No `filterpy` dependency — the Kalman filter is a small vendored
implementation in `kalman.py` (`_LinearKalmanFilter`), same F/H/Q/R math,
no external filtering library needed.

## Run the headless demo
```
python examples/run_demo.py
```
This runs 20 seconds of simulated tracking with no GUI, prints the final
performance report, exports `performance_log.csv` +
`performance_summary.json` (your Performance Log deliverable), and saves a
sanity-check plot.

## The data contract for a dashboard

Everything a UI needs comes from one call:

```python
from fsoc_tracker import TrackingEngine

engine = TrackingEngine()
state = engine.step()   # call this every frame, in your UI's render loop
```

`state` is a plain dict, e.g.:

```python
{
  "t": 1.233,
  "frame": 37,
  "true_theta_x_mrad": 12.4,
  "true_theta_y_mrad": 3.1,
  "detected": True,
  "meas_u": 342.1, "meas_v": 210.7,      # raw noisy detection
  "filt_u": 340.8, "filt_v": 209.9,      # Kalman-filtered estimate
  "theta_e_mrad": 3.2,                   # current angular tracking error
  "locked": True,
  "gimbal_pan_deg": 0.71, "gimbal_tilt_deg": 0.18,
  "metrics": { ...snapshot() from MetricsTracker... }
}
```

A dashboard (PyQt, Streamlit, a web frontend via websockets, whatever) should:
1. Loop calling `engine.step()` on a timer (or as fast as possible for offline replay).
2. Draw a dot at `(meas_u, meas_v)` and `(filt_u, filt_v)` on a canvas sized `camera.width x camera.height`.
3. Plot `theta_e_mrad` over time and show `metrics` fields as live stat readouts.
4. Never reach into `fsoc_tracker` internals beyond this dict — if you need
   more data, add a field to `engine.step()`'s return, don't bypass it.

## Tuning knobs (all in `fsoc_tracker/config.py`)
- `MotionConfig.profile`: `"constant_rate" | "sinusoidal" | "random_walk" | "coordinated_turn"` — swap to stress-test the filter.
- `DisturbanceConfig`: toggle turbulence/vibration/sensor-noise/dropout independently for your test methodology section.
- `KalmanConfig.process_noise_q`: increase if the filter lags behind fast motion; decrease if it's too jittery on noisy input.
- `GimbalConfig.kp` / `max_slew_rate_dps`: controller aggressiveness.

## Stress testing
`fsoc_tracker/scenarios.py` + `experiments/run_stress_test.py` +
`experiments/plot_degradation.py` run the engine across a motion x
disturbance grid and turn the results into a severity-vs-performance chart.
`experiments/run_recovery_test.py` + `experiments/plot_recovery.py` test
recovery from a briefly blocked camera against the SIH re-acquisition-time
spec. See **`EXPERIMENTS.md`** for what each does, how to run them, and how
to extend them.

## AI-based detection (Part 4 — in progress)
The renderer (`fsoc_tracker/render.py`) produces actual pixel images with
the beacon, noise, and atmospheric effects — the first step toward
replacing direct pixel injection with a real trained detector. See
**`AI_DETECTOR.md`**.

## What's deliberately NOT here yet
- No trained detector yet — `fsoc_tracker/render.py` renders real frames
  (see `AI_DETECTOR.md`), but nothing reads them yet; the engine still
  uses direct pixel injection with noise for its detection step. Swap-in
  point: replace that injection in `engine.py` with rendering a frame and
  running the detector on it.
- No GUI. That's the point — build it as a thin consumer of `step()`.

## Dashboard

```
pip install -r requirements.txt
streamlit run dashboard/app.py
```

Two views:
- **Live Tracking Simulation** — pick a motion profile + disturbance level
  (same six profiles / four levels as the stress-test grid), run it, and
  watch tracking error, lock status, and a rendered camera panel update as
  it goes. Built entirely on `engine.step()`'s output dict — the dashboard
  file never touches Kalman/camera/motion internals directly.
- **AI Detector Benchmark** — surfaces the actual Part 4b results
  (naive vs. trained-classifier comparison, false-lock chart, training
  report). Run `experiments/train_detector.py` and
  `experiments/benchmark_detector.py` first if this page is empty.

The camera panel's noise/atmosphere controls are cosmetic only — they
control what the *rendered frame* looks like, not the actual disturbance
math driving the numbers (that's the scenario's disturbance level). This
is intentional (see AI_DETECTOR.md's note on the two separate paths) but
worth knowing so the two don't get confused for the same thing.
