"""
Naive (classical-only) vs. Learned (trained model) detector benchmark.

This is the "before/after AI" evidence: does the trained classifier stage
actually improve detection, or is it decoration? Run on held-out frames
(different seed from training) across the same noise/atmosphere grid.

Three outcomes per frame, not just right/wrong:
  - correct:        detector's pick is within tolerance of the true beacon
  - false_lock:      detector confidently returned something, but it's WRONG
                      (this is the dangerous case -- silently tracking noise)
  - no_detection:    detector correctly reported nothing found

Run: python experiments/benchmark_detector.py
(requires experiments/train_detector.py to have been run first)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import csv
import numpy as np
import matplotlib.pyplot as plt

from fsoc_tracker.render import FrameRenderer, RenderConfig
from fsoc_tracker.detector import NaiveDetector, LearnedDetector

NOISE_COMBOS = [("none",), ("gaussian",), ("salt_pepper",), ("poisson",), ("gaussian", "salt_pepper")]
ATMOSPHERES = ["clear", "haze", "fog", "rain", "low_light"]
FRAMES_PER_COMBO = 30
TOLERANCE_PX = 7
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "fsoc_tracker", "models", "detector_mlp.joblib")


def classify_result(result, true_u, true_v):
    if result is None:
        return "no_detection", None
    u, v = result
    dist = ((u - true_u) ** 2 + (v - true_v) ** 2) ** 0.5
    return ("correct", dist) if dist <= TOLERANCE_PX else ("false_lock", dist)


def run_benchmark(seed=999):
    rng = np.random.default_rng(seed)
    naive = NaiveDetector()
    learned = LearnedDetector(MODEL_PATH)

    rows = []
    for noise_types in NOISE_COMBOS:
        for atmosphere in ATMOSPHERES:
            cfg = RenderConfig(noise_types=noise_types, atmosphere=atmosphere)
            renderer = FrameRenderer(cfg, seed=int(rng.integers(0, 1_000_000)))

            for _ in range(FRAMES_PER_COMBO):
                margin = cfg.target_size_px
                u = rng.uniform(margin, cfg.width - margin)
                v = rng.uniform(margin, cfg.height - margin)
                frame = renderer.render(u, v)

                for name, det in [("naive", naive), ("learned", learned)]:
                    outcome, dist = classify_result(det.detect(frame), u, v)
                    rows.append({
                        "noise": "+".join(noise_types), "atmosphere": atmosphere,
                        "detector": name, "outcome": outcome,
                        "error_px": dist if dist is not None else "",
                    })
    return rows


def summarize(rows):
    summary = {}
    for name in ["naive", "learned"]:
        sub = [r for r in rows if r["detector"] == name]
        n = len(sub)
        correct = sum(1 for r in sub if r["outcome"] == "correct")
        false_lock = sum(1 for r in sub if r["outcome"] == "false_lock")
        no_det = sum(1 for r in sub if r["outcome"] == "no_detection")
        errs = [r["error_px"] for r in sub if r["outcome"] == "correct"]
        summary[name] = {
            "n": n,
            "success_rate": correct / n,
            "false_lock_rate": false_lock / n,
            "no_detection_rate": no_det / n,
            "mean_error_px_when_correct": (sum(errs) / len(errs)) if errs else float("nan"),
        }
    return summary


def main():
    print("Running naive vs. learned detector benchmark on held-out frames...")
    rows = run_benchmark()

    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "detector_benchmark_raw.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["noise", "atmosphere", "detector", "outcome", "error_px"])
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    print("\n=== Overall (all conditions pooled) ===")
    for name, s in summary.items():
        print(f"{name:8s}  success={s['success_rate']:.3f}  false_lock={s['false_lock_rate']:.3f}  "
              f"no_detection={s['no_detection_rate']:.3f}  mean_err_px={s['mean_error_px_when_correct']:.2f}")

    with open(os.path.join(out_dir, "detector_benchmark_summary.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["detector", "n", "success_rate", "false_lock_rate", "no_detection_rate", "mean_error_px_when_correct"])
        for name, s in summary.items():
            writer.writerow([name, s["n"], s["success_rate"], s["false_lock_rate"], s["no_detection_rate"], s["mean_error_px_when_correct"]])

    conditions = sorted(set((r["noise"], r["atmosphere"]) for r in rows))
    per_cond = {}
    for noise, atm in conditions:
        for name in ["naive", "learned"]:
            sub = [r for r in rows if r["noise"] == noise and r["atmosphere"] == atm and r["detector"] == name]
            n = len(sub)
            correct = sum(1 for r in sub if r["outcome"] == "correct")
            per_cond[(noise, atm, name)] = correct / n if n else float("nan")

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(ATMOSPHERES))
    width = 0.35

    naive_rates = [np.mean([per_cond[(n, a, "naive")] for n in ["none", "gaussian", "salt_pepper", "poisson", "gaussian+salt_pepper"]]) for a in ATMOSPHERES]
    learned_rates = [np.mean([per_cond[(n, a, "learned")] for n in ["none", "gaussian", "salt_pepper", "poisson", "gaussian+salt_pepper"]]) for a in ATMOSPHERES]

    ax.bar(x - width / 2, naive_rates, width, label="Naive (classical only)", color="tab:gray")
    ax.bar(x + width / 2, learned_rates, width, label="Learned (trained classifier)", color="tab:blue")
    ax.set_xticks(x)
    ax.set_xticklabels(ATMOSPHERES)
    ax.set_ylabel("Detection success rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Detector success rate by atmosphere (averaged across noise types)")
    ax.legend()
    plt.tight_layout()
    fig_path = os.path.join(out_dir, "detector_benchmark_chart.png")
    plt.savefig(fig_path, dpi=120)
    print(f"\nSaved chart to {fig_path}")

    # --- second chart: this is the one that actually shows the AI stage's
    # value. Raw success rate comes out identical between the two detectors
    # (see chart above) -- the real difference is HOW they fail. Naive
    # either finds the right blob or confidently locks onto the wrong one
    # (false_lock). Learned trades some of that false-lock rate for a safe
    # "no_detection" instead -- silently tracking noise vs. correctly
    # saying "not found" are very different failure modes for a coarse-
    # alignment system, even at equal success rate. ---
    naive_fl = [np.mean([1 - per_cond[(n, a, "naive")] for n in ["none", "gaussian", "salt_pepper", "poisson", "gaussian+salt_pepper"]]) for a in ATMOSPHERES]
    fl_rows_by_atm = {name: {a: [] for a in ATMOSPHERES} for name in ["naive", "learned"]}
    for r in rows:
        fl_rows_by_atm[r["detector"]][r["atmosphere"]].append(1 if r["outcome"] == "false_lock" else 0)
    naive_fl_rate = [np.mean(fl_rows_by_atm["naive"][a]) for a in ATMOSPHERES]
    learned_fl_rate = [np.mean(fl_rows_by_atm["learned"][a]) for a in ATMOSPHERES]

    fig2, ax2 = plt.subplots(figsize=(9, 5))
    ax2.bar(x - width / 2, naive_fl_rate, width, label="Naive (classical only)", color="tab:red", alpha=0.75)
    ax2.bar(x + width / 2, learned_fl_rate, width, label="Learned (trained classifier)", color="tab:orange", alpha=0.9)
    ax2.set_xticks(x)
    ax2.set_xticklabels(ATMOSPHERES)
    ax2.set_ylabel("False-lock rate (confidently wrong)")
    ax2.set_title("False-lock rate by atmosphere -- the dangerous failure mode\n(silently tracking noise instead of the beacon)")
    ax2.legend()
    plt.tight_layout()
    fig2_path = os.path.join(out_dir, "detector_false_lock_chart.png")
    plt.savefig(fig2_path, dpi=120)
    print(f"Saved false-lock chart to {fig2_path}")


if __name__ == "__main__":
    main()
