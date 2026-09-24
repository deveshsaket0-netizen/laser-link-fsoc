"""
Recovery test -- Part 3 of the hardening plan. Briefly blocks the beacon
(a scripted occlusion window, like something passing in front of the
camera) and times how long the tracker takes to re-lock afterward. This is
the direct test of a re-acquisition target of <= 1 sec.

UNVERIFIED: the 1s limit below is an engineering assumption this project
adopted as a reasonable target, not a confirmed number from the official
SIH parameters table (that table was blank in the problem statement we
were given). Update SPEC_LIMIT_S once the real spec sheet is checked.

Every real tracker loses the target sometimes; the question graders care
about is whether -- and how fast -- it recovers. That's what this measures,
separately from the stress test's aggregate lock_retention_rate (which
tells you *how often* things go wrong, not *how well it comes back*).

Run: python experiments/run_recovery_test.py
"""

import sys
import os
import csv
import dataclasses

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fsoc_tracker import TrackingEngine, TrackerConfig, MotionConfig
from fsoc_tracker.scenarios import MOTION_PROFILES, DISTURBANCE_LEVELS

SEEDS = [1, 2, 3, 4, 5]
OCCLUSION_DURATIONS_S = [0.3, 1.0, 2.0]   # short / spec-boundary / long
OCCLUSION_STARTS_S = [5.0, 10.0, 15.0]    # 3 events per run, spaced to fully recover between
SIM_DURATION_S = 20.0
FPS_ASSUMED = 30.0
MAX_WAIT_S = 3.0          # how long we'll wait for re-lock before calling it "lost"
SPEC_LIMIT_S = 1.0        # ASSUMED target, not a confirmed SIH spec value -- verify

RAW_FIELDS = [
    "motion_profile", "occlusion_duration_s", "seed", "occlusion_start_s",
    "occlusion_end_s", "reacquired", "reacquisition_time_s", "meets_spec",
]
SUMMARY_FIELDS = [
    "motion_profile", "occlusion_duration_s", "n_events",
    "reacquired_rate", "reacquisition_time_s_mean", "reacquisition_time_s_max",
    "spec_pass_rate",
]


def build_config(motion_profile: str, occlusion_duration: float) -> TrackerConfig:
    base_disturbance = DISTURBANCE_LEVELS["moderate"]
    disturbance = dataclasses.replace(
        base_disturbance,
        enable_dropout=False,   # isolate occlusion recovery from random dropout noise
        enable_occlusion=True,
        occlusion_windows=tuple((start, occlusion_duration) for start in OCCLUSION_STARTS_S),
    )
    return TrackerConfig(
        motion=MotionConfig(profile=motion_profile),
        disturbance=disturbance,
    )


def run_one(motion_profile: str, occlusion_duration: float, seed: int) -> list:
    """Returns one result row per occlusion event in this run."""
    cfg = build_config(motion_profile, occlusion_duration)
    engine = TrackingEngine(cfg, seed=seed)
    n_steps = int(SIM_DURATION_S * FPS_ASSUMED)
    log = list(engine.run(n_steps))

    rows = []
    for start in OCCLUSION_STARTS_S:
        end = start + occlusion_duration
        search_limit = end + MAX_WAIT_S
        reacquired_at = None
        for s in log:
            if s["t"] < end:
                continue
            if s["t"] > search_limit:
                break
            if s["locked"]:
                reacquired_at = s["t"]
                break

        reacquired = reacquired_at is not None
        reacq_time = (reacquired_at - end) if reacquired else None
        rows.append({
            "motion_profile": motion_profile,
            "occlusion_duration_s": occlusion_duration,
            "seed": seed,
            "occlusion_start_s": start,
            "occlusion_end_s": end,
            "reacquired": reacquired,
            "reacquisition_time_s": reacq_time,
            "meets_spec": bool(reacquired and reacq_time <= SPEC_LIMIT_S),
        })
    return rows


def _mean(values):
    values = list(values)
    return sum(values) / len(values) if values else None


def aggregate(raw_rows: list) -> list:
    grouped = {}
    for row in raw_rows:
        key = (row["motion_profile"], row["occlusion_duration_s"])
        grouped.setdefault(key, []).append(row)

    summary = []
    for (motion, duration), rows in grouped.items():
        reacq_times = [r["reacquisition_time_s"] for r in rows if r["reacquired"]]
        summary.append({
            "motion_profile": motion,
            "occlusion_duration_s": duration,
            "n_events": len(rows),
            "reacquired_rate": sum(r["reacquired"] for r in rows) / len(rows),
            "reacquisition_time_s_mean": _mean(reacq_times),
            "reacquisition_time_s_max": max(reacq_times) if reacq_times else None,
            "spec_pass_rate": sum(r["meets_spec"] for r in rows) / len(rows),
        })
    return summary


def main():
    print(f"Running recovery test: {len(MOTION_PROFILES)} motion profiles x "
          f"{len(OCCLUSION_DURATIONS_S)} occlusion durations x {len(SEEDS)} seeds "
          f"x {len(OCCLUSION_STARTS_S)} events/run...")

    raw_rows = []
    for motion in MOTION_PROFILES:
        for duration in OCCLUSION_DURATIONS_S:
            for seed in SEEDS:
                raw_rows.extend(run_one(motion, duration, seed))
        print(f"  done: {motion}")

    summary_rows = aggregate(raw_rows)
    summary_rows.sort(key=lambda r: (r["motion_profile"], r["occlusion_duration_s"]))

    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)

    raw_path = os.path.join(out_dir, "recovery_test_raw.csv")
    with open(raw_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RAW_FIELDS)
        writer.writeheader()
        writer.writerows(raw_rows)

    summary_path = os.path.join(out_dir, "recovery_test_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nWrote {len(raw_rows)} occlusion events to {raw_path}")
    print(f"Wrote {len(summary_rows)} scenario summaries to {summary_path}")

    print(f"\n=== Re-acquisition time (spec: <= {SPEC_LIMIT_S}s) ===")
    header = f"{'motion':15s} {'occl_s':>7s} {'reacq_rate':>11s} {'mean_s':>8s} {'max_s':>8s} {'spec_pass':>10s}"
    print(header)
    for r in summary_rows:
        mean_s = f"{r['reacquisition_time_s_mean']:.2f}" if r["reacquisition_time_s_mean"] is not None else "n/a"
        max_s = f"{r['reacquisition_time_s_max']:.2f}" if r["reacquisition_time_s_max"] is not None else "n/a"
        print(f"{r['motion_profile']:15s} {r['occlusion_duration_s']:7.1f} "
              f"{r['reacquired_rate']:11.2f} {mean_s:>8s} {max_s:>8s} {r['spec_pass_rate']:10.2f}")


if __name__ == "__main__":
    main()
