"""
Batch stress test -- runs the full motion x disturbance grid (fsoc_tracker/
scenarios.py), several random seeds each, and logs the final performance
metrics of every run. This is the raw data behind the degradation chart and
the before/after summary table (see EXPERIMENTS.md).

Run: python experiments/run_stress_test.py
"""

import sys
import os
import csv
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fsoc_tracker import TrackingEngine
from fsoc_tracker.scenarios import build_all_scenarios

SEEDS = [1, 2, 3, 4, 5]
FPS_ASSUMED = 30.0

RAW_FIELDS = [
    "scenario", "motion_profile", "disturbance_level", "seed",
    "avg_error_mrad", "max_error_mrad", "lock_retention_rate",
    "acquisition_time_s", "frame_count",
]
def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values)


def _pstdev(values) -> float:
    values = list(values)
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / len(values))


SUMMARY_FIELDS = [
    "scenario", "motion_profile", "disturbance_level", "n_seeds",
    "avg_error_mrad_mean", "avg_error_mrad_std",
    "max_error_mrad_mean",
    "lock_retention_rate_mean", "lock_retention_rate_std",
    "acquisition_time_s_mean",
]


def run_one(scenario, seed: int) -> dict:
    engine = TrackingEngine(scenario.config, seed=seed)
    n_steps = int(scenario.duration_s * FPS_ASSUMED)
    for _ in engine.run(n_steps):
        pass
    report = engine.metrics.final_report()
    return {
        "scenario": scenario.name,
        "motion_profile": scenario.motion_profile,
        "disturbance_level": scenario.disturbance_level,
        "seed": seed,
        "avg_error_mrad": report.get("avg_error_mrad"),
        "max_error_mrad": report.get("max_error_mrad"),
        "lock_retention_rate": report.get("lock_retention_rate"),
        "acquisition_time_s": report.get("acquisition_time_s"),
        "frame_count": report.get("frame_count"),
    }


def aggregate(raw_rows: list) -> list:
    by_scenario = {}
    for row in raw_rows:
        by_scenario.setdefault(row["scenario"], []).append(row)

    summary_rows = []
    for name, rows in by_scenario.items():
        acq_times = [r["acquisition_time_s"] for r in rows if r["acquisition_time_s"] is not None]
        summary_rows.append({
            "scenario": name,
            "motion_profile": rows[0]["motion_profile"],
            "disturbance_level": rows[0]["disturbance_level"],
            "n_seeds": len(rows),
            "avg_error_mrad_mean": _mean(r["avg_error_mrad"] for r in rows),
            "avg_error_mrad_std": _pstdev(r["avg_error_mrad"] for r in rows),
            "max_error_mrad_mean": _mean(r["max_error_mrad"] for r in rows),
            "lock_retention_rate_mean": _mean(r["lock_retention_rate"] for r in rows),
            "lock_retention_rate_std": _pstdev(r["lock_retention_rate"] for r in rows),
            "acquisition_time_s_mean": _mean(acq_times) if acq_times else None,
        })
    return summary_rows


def main():
    scenarios = build_all_scenarios(duration_s=15.0)
    print(f"Running {len(scenarios)} scenarios x {len(SEEDS)} seeds = "
          f"{len(scenarios) * len(SEEDS)} simulations...")

    raw_rows = []
    for scenario in scenarios:
        for seed in SEEDS:
            raw_rows.append(run_one(scenario, seed))
        print(f"  done: {scenario.name}")

    summary_rows = aggregate(raw_rows)
    summary_rows.sort(key=lambda r: (r["motion_profile"], r["disturbance_level"]))

    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)

    raw_path = os.path.join(out_dir, "stress_test_raw.csv")
    with open(raw_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RAW_FIELDS)
        writer.writeheader()
        writer.writerows(raw_rows)

    summary_path = os.path.join(out_dir, "stress_test_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nWrote {len(raw_rows)} raw runs to {raw_path}")
    print(f"Wrote {len(summary_rows)} scenario summaries to {summary_path}")

    print("\n=== Summary (avg error mrad, lock retention) ===")
    header = f"{'scenario':30s} {'avg_err':>9s} {'max_err':>9s} {'lock_rate':>10s} {'acq_s':>8s}"
    print(header)
    for r in summary_rows:
        acq = f"{r['acquisition_time_s_mean']:.2f}" if r["acquisition_time_s_mean"] is not None else "n/a"
        print(f"{r['scenario']:30s} {r['avg_error_mrad_mean']:9.2f} {r['max_error_mrad_mean']:9.2f} "
              f"{r['lock_retention_rate_mean']:10.2f} {acq:>8s}")


if __name__ == "__main__":
    main()
