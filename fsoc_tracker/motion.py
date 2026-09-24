"""
Beacon motion models -- section 4.3 / 4.1 of the math framework.

Generates GROUND TRUTH angular position of the beacon relative to the
tracker boresight, as a function of time. This is the answer key used to
compute tracking error, and it's also what the detector/filter are trying
to recover.

Five profiles ("stationary", "linear", "circular", "lissajous", "sudden")
match the team's shared Evaluation Plan scenario names (see the Member 6
analysis toolkit) so results reported here line up with the rest of the
team's tables and graphs. "random_walk" is kept as a sixth, unlabeled-in-
the-plan stress case, since it's motion none of the other five can produce.

All angles in radians. `step(t)` returns (theta_x, theta_y).
"""

import math
import random
from .config import MotionConfig


class BeaconMotion:
    def __init__(self, cfg: MotionConfig, seed: int = None):
        self.cfg = cfg
        self._rng = random.Random(seed)
        self._walk_x = 0.0
        self._walk_y = 0.0

    def step(self, t: float) -> tuple:
        profile = self.cfg.profile
        if profile == "stationary":
            return self._stationary(t)
        elif profile == "linear":
            return self._linear(t)
        elif profile == "circular":
            return self._circular(t)
        elif profile == "lissajous":
            return self._lissajous(t)
        elif profile == "sudden":
            return self._sudden(t)
        elif profile == "random_walk":
            return self._random_walk(t)
        else:
            raise ValueError(f"Unknown motion profile: {profile}")

    # -- baseline: target does not move --
    def _stationary(self, t: float) -> tuple:
        return 0.0, 0.0

    # -- constant angular rate: theta(t) = rate * t --
    def _linear(self, t: float) -> tuple:
        rate = math.radians(self.cfg.linear_rate_deg_s)
        return rate * t, 0.0

    # -- constant-radius, constant-rate circle in angle-space --
    def _circular(self, t: float) -> tuple:
        radius = math.radians(self.cfg.circular_radius_deg)
        rate = math.radians(self.cfg.circular_rate_deg_s)
        return radius * math.cos(rate * t), radius * math.sin(rate * t)

    # -- decoupled sinusoids on each axis --
    def _lissajous(self, t: float) -> tuple:
        amp = math.radians(self.cfg.lissajous_amp_deg)
        f = self.cfg.lissajous_freq_hz
        theta_x = amp * math.sin(2 * math.pi * f * t)
        theta_y = 0.6 * amp * math.sin(2 * math.pi * f * 0.7 * t + 0.8)
        return theta_x, theta_y

    # -- constant rate, then an abrupt change in rate at a fixed time --
    def _sudden(self, t: float) -> tuple:
        switch_t = self.cfg.sudden_switch_time_s
        rate1 = math.radians(self.cfg.sudden_rate1_deg_s)
        rate2 = math.radians(self.cfg.sudden_rate2_deg_s)
        if t <= switch_t:
            return rate1 * t, 0.0
        theta_at_switch = rate1 * switch_t
        return theta_at_switch + rate2 * (t - switch_t), 0.0

    # -- unpredictable drift: theta_{k+1} = theta_k + N(0, sigma^2) --
    def _random_walk(self, t: float) -> tuple:
        sigma = math.radians(self.cfg.random_walk_sigma_deg)
        self._walk_x += self._rng.gauss(0, sigma)
        self._walk_y += self._rng.gauss(0, sigma)
        return self._walk_x, self._walk_y
