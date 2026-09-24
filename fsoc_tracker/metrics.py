"""
Metrics & performance logging -- section 3.4 of the math framework.

Tracks exactly the quantities the SIH deliverable list asks for:
simulation duration, FPS, acquisition time, average/max tracking error,
lock retention rate. A dashboard can poll `MetricsTracker.snapshot()`
every frame, or read the full log at the end for the "Performance Log"
deliverable (CSV/JSON export included).
"""

import time
import math
import csv
import json


class MetricsTracker:
    def __init__(self, lock_threshold_rad: float):
        self.lock_threshold_rad = lock_threshold_rad
        self.records = []          # per-frame dicts
        self._acquired_at = None   # sim time of first sustained lock
        self._start_wall = time.time()

    def log_frame(self, t: float, theta_e: float, locked: bool):
        self.records.append({"t": t, "theta_e_mrad": theta_e * 1000.0, "locked": locked})
        if locked and self._acquired_at is None:
            self._acquired_at = t

    def snapshot(self) -> dict:
        """
        Cheap incremental stats -- safe to call every frame for a live dashboard.

        avg/max/current error are computed over DETECTED frames only. A frame
        with no detection (occlusion/dropout) logs theta_e as inf, and it
        correctly still counts against lock_retention_rate (which divides by
        ALL frames), but including inf in an average/max would make one lost
        frame poison the entire run's error stats to infinity.
        """
        if not self.records:
            return {}
        n = len(self.records)
        locked_frames = sum(1 for r in self.records if r["locked"])
        finite_errors = [r["theta_e_mrad"] for r in self.records if math.isfinite(r["theta_e_mrad"])]
        elapsed_wall = time.time() - self._start_wall
        last_finite = finite_errors[-1] if finite_errors else None
        return {
            "frame_count": n,
            "detected_frame_count": len(finite_errors),
            "sim_duration_s": self.records[-1]["t"],
            "fps": n / elapsed_wall if elapsed_wall > 0 else 0.0,
            "current_error_mrad": last_finite,
            "avg_error_mrad": (sum(finite_errors) / len(finite_errors)) if finite_errors else None,
            "max_error_mrad": max(finite_errors) if finite_errors else None,
            "lock_retention_rate": locked_frames / n,
            "acquisition_time_s": self._acquired_at,
        }

    def final_report(self) -> dict:
        return self.snapshot()

    def export_csv(self, path: str):
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["t", "theta_e_mrad", "locked"])
            writer.writeheader()
            writer.writerows(self.records)

    def export_summary_json(self, path: str):
        with open(path, "w") as f:
            json.dump(self.final_report(), f, indent=2)
