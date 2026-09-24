# Stress Testing

This is Part 1 of hardening the demo for judges: instead of running the
simulation once under nice conditions, run it many times across a grid of
motion patterns x disturbance severities, and log the numbers every time.
Parts 2 (degradation chart), 3 (recovery/occlusion test) and 6 (before/after
table) all build on the CSVs this produces.

## Files

| File | What it does |
|---|---|
| `fsoc_tracker/scenarios.py` | Defines the test grid: 6 motion profiles x 4 disturbance levels (clean/light/moderate/severe) = 24 scenarios |
| `experiments/run_stress_test.py` | Runs every scenario x 5 random seeds (120 simulations), logs final metrics for each, aggregates per scenario |
| `experiments/plot_degradation.py` | Reads the summary CSV, plots lock retention & avg error vs. disturbance severity per motion profile, and finds each profile's breaking point |
| `experiments/output/stress_test_raw.csv` | One row per (scenario, seed) run |
| `experiments/output/stress_test_summary.csv` | One row per scenario, averaged across seeds |
| `experiments/output/degradation_chart.png` | Two-panel chart: lock retention and avg error vs. severity, one line per motion profile |
| `experiments/output/degradation_breakpoints.csv` | Per motion profile, the first disturbance level where lock retention drops below 0.80 |

## Run it
```
python experiments/run_stress_test.py
python experiments/plot_degradation.py
```
The second script reads the first script's summary CSV, so run them in that order. Together they take a few seconds.

## The grid

**Motion profiles** (from `fsoc_tracker/motion.py`): `stationary`, `linear`,
`circular`, `lissajous`, `sudden`, `random_walk`.

**Disturbance levels** (from `fsoc_tracker/scenarios.py`): each level scales
turbulence, vibration, sensor noise, and dropout probability together.
- `clean` — everything off (best case, sanity check)
- `light` — mild jitter, ~1% dropout
- `moderate` — the engine's original defaults, ~2% dropout
- `severe` — heavy turbulence/noise, ~5% dropout

Crossing them gives 24 scenarios. Each runs 5 seeds so the numbers aren't
luck-of-the-draw from one random trajectory.

## Output columns

`stress_test_summary.csv`:
- `avg_error_mrad_mean/std` — mean tracking error, averaged over **detected
  frames only** (see the metrics.py note below), averaged again across seeds
- `max_error_mrad_mean` — worst-case error per run, averaged across seeds
- `lock_retention_rate_mean/std` — fraction of ALL frames (including missed
  detections) where the tracker was locked
- `acquisition_time_s_mean` — time to first sustained lock

## A metrics.py fix that shipped alongside this

`MetricsTracker.snapshot()` used to average `theta_e_mrad` over *every*
frame, including frames with no detection (logged as `inf`). One dropout
frame anywhere in a run made `avg_error_mrad`/`max_error_mrad` come out as
`inf` for the whole run — invisible in the original single clean demo run,
but it broke every scenario here that has dropout enabled. Fixed to average
over detected frames only; `lock_retention_rate` is untouched and still
correctly divides by all frames (a missed detection should hurt that number).

## Adding a new scenario or level
Edit `fsoc_tracker/scenarios.py`:
- New motion profile: add it to `motion.py` first (a new `BeaconMotion`
  method + `MotionConfig` fields), then add its name to `MOTION_PROFILES`.
- New disturbance level: add a key to `DISTURBANCE_LEVELS` with a
  `DisturbanceConfig`.
Both are picked up automatically by `build_all_scenarios()` — no changes
needed in `run_stress_test.py`.

## The degradation chart (Part 2)

`experiments/plot_degradation.py` plots each motion profile's lock retention
rate and average error against disturbance severity, so instead of a single
"it works" data point you get a curve showing where each profile starts to
struggle. It also computes a **breaking point**: the first disturbance level
at which lock retention drops below 0.80 (tune `BREAK_THRESHOLD` in the
script if you want a stricter/looser bar).

Current breakpoints (from the run included in this repo):

| Motion profile | Breaks at | Lock rate @ severe |
|---|---|---|
| stationary | never (within tested range) | 0.93 |
| circular | never (within tested range) | 0.87 |
| linear | severe | 0.68 |
| random_walk | severe | 0.77 |
| sudden | **clean** | 0.55 |
| lissajous | **clean** | 0.15 |

Read this carefully before quoting it in the report: `sudden` and
`lissajous` fail the 0.80 bar even with **zero** injected disturbance. That
is not a noise-robustness problem — it means the constant-velocity Kalman
filter and/or the proportional controller can't keep up with those two
trajectory shapes regardless of noise (abrupt direction changes for
`sudden`; the controller's gain vs. the lissajous amplitude/frequency for
that one). This is the concrete, data-backed case for adding an EKF or
retuning the controller — see the "future improvements" discussion in the
main conversation. Don't present these two rows as "disturbance-broken" in
the report; present them as "motion-model-limited, disturbance is a smaller
secondary effect here."

`circular` and `stationary` hold up across the whole severity range tested
— good evidence the disturbance-handling side of the pipeline itself is
solid; the profiles that struggle are struggling for a different reason.

## Recovery testing (Part 3)

The stress test measures how often the tracker stays locked; this measures
how well it **comes back** when it briefly can't see the target at all —
e.g. something passes in front of the camera. This is the direct test of
a **re-acquisition target of <= 1 sec**.

> **Unverified:** the 1s figure is an engineering assumption this project
> adopted as a reasonable target, not a confirmed value from the official
> SIH parameters table — the problem statement we were given had a blank
> table there. Check the real spec sheet and update `SPEC_LIMIT_S` in
> `run_recovery_test.py` (and re-run) if the actual number differs.

New config: `DisturbanceConfig.enable_occlusion` / `occlusion_windows`
(`fsoc_tracker/config.py`, `fsoc_tracker/disturbances.py`). Unlike
`enable_dropout` (random per-frame misses), an occlusion window is a
scripted, contiguous "target hidden for exactly N seconds starting at time
T" event — `occlusion_windows = ((5.0, 1.0), (10.0, 2.0))` blocks the
target from t=5-6s and t=10-12s. `engine.step()`'s `detected` flag goes
`False` for the whole window regardless of geometry; the Kalman filter
keeps predicting (no update) through the gap, same as it already does for
dropout.

| File | What it does |
|---|---|
| `experiments/run_recovery_test.py` | For each motion profile x occlusion duration (0.3s / 1.0s / 2.0s) x 5 seeds, runs a 20s sim with 3 scripted occlusion events, and measures the time from occlusion-end to next re-lock for each event |
| `experiments/plot_recovery.py` | Bar chart of mean re-acquisition time per motion profile x occlusion duration, against the 1s spec line |
| `experiments/output/recovery_test_raw.csv` | One row per occlusion event (270 rows: 6 motions x 3 durations x 5 seeds x 3 events) |
| `experiments/output/recovery_test_summary.csv` | One row per (motion, duration), averaged across seeds/events |
| `experiments/output/recovery_chart.png` | The bar chart |

Re-acquisition is measured the same way the stress test measures initial
acquisition: first frame after the occlusion ends where `locked` is `True`
(angular error back under the lock threshold) — not just "detected again."
That's a fair definition since the filter needs a beat to reconverge after
a gap, and re-acquisition should mean "usable track," not just "saw a
pixel." Background disturbance is fixed at the `moderate` level with
dropout turned off, so the numbers reflect occlusion recovery specifically,
not a mix of occlusion and random dropout noise.

## Run it
```
python experiments/run_recovery_test.py
python experiments/plot_recovery.py
```

## Current results

Every motion profile eventually re-locks after every occlusion tested (no
permanent target loss). But "eventually" isn't the spec — the spec is
<=1s, and two profiles miss it at longer occlusion durations:

| Motion profile | Worst case tested | Meets 1s spec? |
|---|---|---|
| stationary, circular, random_walk | <=0.35s at 2.0s occlusion | Yes, comfortably |
| linear, sudden (short/moderate occlusion) | <=0.35s | Yes |
| **sudden**, 2.0s occlusion | 0.62s mean, spec_pass_rate 0.80 | Mostly, but degrading |
| **lissajous**, 1.0s occlusion | 0.79s mean, spec_pass_rate 0.93 | Borderline |
| **lissajous**, 2.0s occlusion | 1.15s mean, spec_pass_rate 0.33 | **No — fails majority of trials** |

Same story as the degradation chart: `lissajous` and `sudden` are the two
profiles where the constant-velocity Kalman filter struggles, and here it
shows up as slow reconvergence after a gap, not just steady-state error.
Another data point for the EKF discussion in "future improvements" —
worth citing alongside the degradation-chart breakpoints rather than
separately, since they're pointing at the same underlying limitation.

## Not done yet (see the plan)
- Part 4: replace the geometric detector with a trained CV/CNN detector
  (non-negotiable — this is what makes "AI-Based" true).
- Part 6: the final before/after table for the report.
