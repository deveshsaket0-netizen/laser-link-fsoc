"""
Degradation chart -- turns experiments/output/stress_test_summary.csv into
the "how bad can it get before it breaks" picture: performance plotted
against disturbance severity, one line per motion profile, so the judges
see a curve (solid -> wobbling -> falling apart) instead of a single
"it works" data point.

Also derives each motion profile's breaking point: the first disturbance
level at which lock retention drops below a threshold.

Requires stress_test_summary.csv to already exist -- run
experiments/run_stress_test.py first.

Run: python experiments/plot_degradation.py
"""

import sys
import os
import csv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.pyplot as plt

LEVEL_ORDER = ["clean", "light", "moderate", "severe"]
BREAK_THRESHOLD = 0.80   # lock_retention_rate below this counts as "broken"


def load_summary(path: str) -> list:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            for key in ("avg_error_mrad_mean", "max_error_mrad_mean",
                        "lock_retention_rate_mean", "n_seeds"):
                row[key] = float(row[key])
            rows.append(row)
        return rows


def by_motion_profile(rows: list) -> dict:
    grouped = {}
    for row in rows:
        grouped.setdefault(row["motion_profile"], {})[row["disturbance_level"]] = row
    return grouped


def find_breakpoints(grouped: dict) -> list:
    """First disturbance level per motion profile where lock retention < threshold."""
    results = []
    for motion, levels in grouped.items():
        break_level = None
        for level in LEVEL_ORDER:
            if level not in levels:
                continue
            if levels[level]["lock_retention_rate_mean"] < BREAK_THRESHOLD:
                break_level = level
                break
        results.append({
            "motion_profile": motion,
            "breaks_at": break_level or "never (within tested range)",
            "lock_rate_at_severe": levels.get("severe", {}).get("lock_retention_rate_mean"),
        })
    return results


def plot(grouped: dict, out_path: str):
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))

    for motion, levels in sorted(grouped.items()):
        xs, lock_rates, errors = [], [], []
        for level in LEVEL_ORDER:
            if level not in levels:
                continue
            xs.append(level)
            lock_rates.append(levels[level]["lock_retention_rate_mean"])
            errors.append(levels[level]["avg_error_mrad_mean"])
        axs[0].plot(xs, lock_rates, marker="o", label=motion)
        axs[1].plot(xs, errors, marker="o", label=motion)

    axs[0].axhline(BREAK_THRESHOLD, color="gray", linestyle="--", linewidth=1,
                   label=f"break threshold ({BREAK_THRESHOLD})")
    axs[0].set_title("Lock retention vs. disturbance severity")
    axs[0].set_ylabel("lock retention rate")
    axs[0].set_ylim(0, 1.05)
    axs[0].legend(fontsize=8)
    axs[0].grid(alpha=0.3)

    axs[1].set_title("Avg tracking error vs. disturbance severity")
    axs[1].set_ylabel("avg error (mrad)")
    axs[1].legend(fontsize=8)
    axs[1].grid(alpha=0.3)

    for ax in axs:
        ax.set_xlabel("disturbance level")

    plt.tight_layout()
    plt.savefig(out_path, dpi=130)


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "output")
    summary_path = os.path.join(out_dir, "stress_test_summary.csv")
    if not os.path.exists(summary_path):
        raise SystemExit(
            f"{summary_path} not found -- run experiments/run_stress_test.py first."
        )

    rows = load_summary(summary_path)
    grouped = by_motion_profile(rows)

    chart_path = os.path.join(out_dir, "degradation_chart.png")
    plot(grouped, chart_path)
    print(f"Saved degradation chart to {chart_path}")

    breakpoints = find_breakpoints(grouped)
    breakpoints.sort(key=lambda r: r["motion_profile"])

    bp_path = os.path.join(out_dir, "degradation_breakpoints.csv")
    with open(bp_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["motion_profile", "breaks_at", "lock_rate_at_severe"])
        writer.writeheader()
        writer.writerows(breakpoints)
    print(f"Saved breakpoints to {bp_path}")

    print(f"\n=== Breaking point per motion profile (lock retention < {BREAK_THRESHOLD}) ===")
    for r in breakpoints:
        print(f"  {r['motion_profile']:15s} breaks at: {r['breaks_at']:10s}  "
              f"(lock rate @ severe: {r['lock_rate_at_severe']:.2f})")


if __name__ == "__main__":
    main()
