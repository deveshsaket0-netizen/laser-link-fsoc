"""
Recovery chart -- bar chart of mean re-acquisition time per motion profile
x occlusion duration, against an assumed 1s re-acquisition target (this
figure is an engineering assumption, not a confirmed official SIH spec
value -- see run_recovery_test.py). Reads
experiments/output/recovery_test_summary.csv -- run run_recovery_test.py
first.

Run: python experiments/plot_recovery.py
"""

import sys
import os
import csv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.pyplot as plt
import numpy as np

SPEC_LIMIT_S = 1.0  # ASSUMED target, not a confirmed SIH spec value -- verify


def load_summary(path: str) -> list:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            row["occlusion_duration_s"] = float(row["occlusion_duration_s"])
            row["reacquisition_time_s_mean"] = float(row["reacquisition_time_s_mean"]) \
                if row["reacquisition_time_s_mean"] not in ("", "None") else None
            rows.append(row)
        return rows


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "output")
    summary_path = os.path.join(out_dir, "recovery_test_summary.csv")
    if not os.path.exists(summary_path):
        raise SystemExit(f"{summary_path} not found -- run experiments/run_recovery_test.py first.")

    rows = load_summary(summary_path)
    motions = sorted(set(r["motion_profile"] for r in rows))
    durations = sorted(set(r["occlusion_duration_s"] for r in rows))

    by_key = {(r["motion_profile"], r["occlusion_duration_s"]): r["reacquisition_time_s_mean"] for r in rows}

    x = np.arange(len(motions))
    width = 0.8 / len(durations)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, duration in enumerate(durations):
        values = [by_key.get((m, duration)) or 0.0 for m in motions]
        ax.bar(x + i * width - 0.4 + width / 2, values, width, label=f"{duration:.1f}s occlusion")

    ax.axhline(SPEC_LIMIT_S, color="red", linestyle="--", linewidth=1.2,
               label=f"assumed target ({SPEC_LIMIT_S}s, unverified)")
    ax.set_xticks(x)
    ax.set_xticklabels(motions)
    ax.set_ylabel("mean re-acquisition time (s)")
    ax.set_title("Re-acquisition time after scripted occlusion")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    chart_path = os.path.join(out_dir, "recovery_chart.png")
    plt.savefig(chart_path, dpi=130)
    print(f"Saved recovery chart to {chart_path}")


if __name__ == "__main__":
    main()
