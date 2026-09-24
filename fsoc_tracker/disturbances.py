"""
Disturbance models -- section 4.4 of the math framework.

Applied AFTER clean projection, BEFORE the "detector" sees the beacon.
Each disturbance is independently toggleable via DisturbanceConfig so you
can isolate their individual effect on tracking error for the report's
test methodology / performance analysis sections.
"""

import math
import random
from .config import DisturbanceConfig


class DisturbanceModel:
    def __init__(self, cfg: DisturbanceConfig, seed: int = None):
        self.cfg = cfg
        self._rng = random.Random(seed)

    def apply_angular(self, theta_x: float, theta_y: float, t: float) -> tuple:
        """
        Turbulence and vibration act at the angle-of-arrival level, i.e.
        before projection -- they perturb where the beacon *appears* to be
        in world-angle space (§4.4 table, rows 1-2).
        """
        dx, dy = 0.0, 0.0

        if self.cfg.enable_turbulence:
            sigma = self.cfg.turbulence_sigma_urad * 1e-6  # urad -> rad
            dx += self._rng.gauss(0, sigma)
            dy += self._rng.gauss(0, sigma)

        if self.cfg.enable_vibration:
            for amp_deg, f in zip(self.cfg.vib_amplitudes_deg, self.cfg.vib_freqs_hz):
                amp = math.radians(amp_deg)
                dx += amp * math.sin(2 * math.pi * f * t)
                dy += amp * math.cos(2 * math.pi * f * t * 1.3)

        return theta_x + dx, theta_y + dy

    def apply_pixel(self, u: float, v: float) -> tuple:
        """
        Sensor/centroiding noise acts at the pixel level, after projection
        (§4.4 table, row 3).
        """
        if self.cfg.enable_sensor_noise:
            u += self._rng.gauss(0, self.cfg.sensor_noise_px)
            v += self._rng.gauss(0, self.cfg.sensor_noise_px)
        return u, v

    def dropout(self) -> bool:
        """Returns True if this frame should simulate a missed detection."""
        if not self.cfg.enable_dropout:
            return False
        return self._rng.random() < self.cfg.dropout_prob

    def occluded(self, t: float) -> bool:
        """
        Returns True if `t` falls inside a scripted occlusion window --
        a deliberate, contiguous "something is blocking the camera" event,
        as opposed to dropout's random per-frame misses.
        """
        if not self.cfg.enable_occlusion:
            return False
        return any(start <= t < start + duration
                   for start, duration in self.cfg.occlusion_windows)
