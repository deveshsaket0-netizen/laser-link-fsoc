"""
Headless demo -- proves the core engine works end-to-end with no GUI.

This is exactly the loop a dashboard would run: call engine.step()
every frame and read the returned dict. Here we just collect the dicts
and plot them at the end; a live dashboard would render each one as
it arrives instead.

Run: python examples/run_demo.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.pyplot as plt
from fsoc_tracker import TrackingEngine, TrackerConfig, MotionConfig, DisturbanceConfig


def main():
    cfg = TrackerConfig(
        motion=MotionConfig(profile="lissajous", lissajous_amp_deg=6.0, lissajous_freq_hz=0.15),
        disturbance=DisturbanceConfig(
            enable_turbulence=True, turbulence_sigma_urad=40.0,
            enable_vibration=True,
            enable_sensor_noise=True, sensor_noise_px=1.5,
        ),
    )
    engine = TrackingEngine(cfg, seed=7)

    n_frames = 600  # 20 seconds at 30 FPS
    log = list(engine.run(n_frames))

    final = engine.metrics.final_report()
    print("=== Final Performance Report ===")
    for k, v in final.items():
        print(f"  {k}: {v}")

    # Export deliverable-ready files
    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)
    engine.metrics.export_csv(os.path.join(out_dir, "performance_log.csv"))
    engine.metrics.export_summary_json(os.path.join(out_dir, "performance_summary.json"))
    print(f"\nExported performance_log.csv and performance_summary.json to {out_dir}")

    # --- quick visual sanity check ---
    t = [s["t"] for s in log]
    true_x = [s["true_theta_x_mrad"] for s in log]
    err = [s["theta_e_mrad"] if s["theta_e_mrad"] is not None else float("nan") for s in log]
    pan = [s["gimbal_pan_deg"] for s in log]

    fig, axs = plt.subplots(3, 1, figsize=(9, 8), sharex=True)

    axs[0].plot(t, true_x, label="true beacon angle (mrad)")
    axs[0].plot(t, [p * 1000 * 3.14159 / 180 for p in pan], label="gimbal pan (converted, mrad)", alpha=0.7)
    axs[0].legend()
    axs[0].set_ylabel("mrad")
    axs[0].set_title("Beacon motion vs. gimbal pointing")

    axs[1].plot(t, err, color="tab:red")
    axs[1].axhline(cfg.lock_threshold_mrad, color="gray", linestyle="--", label="lock threshold")
    axs[1].set_ylabel("theta_e (mrad)")
    axs[1].set_title("Angular tracking error")
    axs[1].legend()

    locked = [1 if s["locked"] else 0 for s in log]
    axs[2].plot(t, locked, drawstyle="steps-post")
    axs[2].set_ylabel("locked")
    axs[2].set_xlabel("time (s)")
    axs[2].set_title("Lock status")

    plt.tight_layout()
    fig_path = os.path.join(out_dir, "tracking_performance.png")
    plt.savefig(fig_path, dpi=120)
    print(f"Saved plot to {fig_path}")


if __name__ == "__main__":
    main()
