"""
Named test scenarios for stress testing -- builds the motion x disturbance
grid that the batch runner (experiments/run_stress_test.py) executes.

Two independent axes:
  - motion profile: which of the six BeaconMotion profiles the target follows
  - disturbance level: how much turbulence/vibration/sensor noise/dropout
    is stacked on top of it (clean -> light -> moderate -> severe)

Crossing them gives a grid instead of a handful of hand-picked cases, which
is what turns "it works" into "here is exactly where it stops working" --
see EXPERIMENTS.md for how the results get turned into that chart.
"""

from dataclasses import dataclass
from .config import TrackerConfig, MotionConfig, DisturbanceConfig


MOTION_PROFILES = ["stationary", "linear", "circular", "lissajous", "sudden", "random_walk"]


# Each level scales turbulence / vibration / sensor noise / dropout together.
# "moderate" matches the engine's original defaults; the others scale from there.
DISTURBANCE_LEVELS = {
    "clean": DisturbanceConfig(
        enable_turbulence=False,
        enable_vibration=False,
        enable_sensor_noise=False,
        enable_dropout=False,
    ),
    "light": DisturbanceConfig(
        enable_turbulence=True, turbulence_sigma_urad=15.0,
        enable_vibration=True, vib_amplitudes_deg=(0.01, 0.005), vib_freqs_hz=(8.0, 21.0),
        enable_sensor_noise=True, sensor_noise_px=0.7,
        enable_dropout=True, dropout_prob=0.01,
    ),
    "moderate": DisturbanceConfig(
        enable_turbulence=True, turbulence_sigma_urad=50.0,
        enable_vibration=True, vib_amplitudes_deg=(0.03, 0.015), vib_freqs_hz=(8.0, 21.0),
        enable_sensor_noise=True, sensor_noise_px=1.5,
        enable_dropout=True, dropout_prob=0.02,
    ),
    "severe": DisturbanceConfig(
        enable_turbulence=True, turbulence_sigma_urad=120.0,
        enable_vibration=True, vib_amplitudes_deg=(0.08, 0.04), vib_freqs_hz=(8.0, 21.0),
        enable_sensor_noise=True, sensor_noise_px=3.5,
        enable_dropout=True, dropout_prob=0.05,
    ),
}


@dataclass
class Scenario:
    name: str
    motion_profile: str
    disturbance_level: str
    duration_s: float
    config: TrackerConfig


def build_scenario(motion_profile: str, disturbance_level: str, duration_s: float = 15.0) -> Scenario:
    if motion_profile not in MOTION_PROFILES:
        raise ValueError(f"Unknown motion profile: {motion_profile}")
    if disturbance_level not in DISTURBANCE_LEVELS:
        raise ValueError(f"Unknown disturbance level: {disturbance_level}")

    cfg = TrackerConfig(
        motion=MotionConfig(profile=motion_profile),
        disturbance=DISTURBANCE_LEVELS[disturbance_level],
    )
    name = f"{motion_profile}__{disturbance_level}"
    return Scenario(
        name=name,
        motion_profile=motion_profile,
        disturbance_level=disturbance_level,
        duration_s=duration_s,
        config=cfg,
    )


def build_all_scenarios(duration_s: float = 15.0) -> list:
    """Full motion x disturbance grid, one Scenario per combination."""
    return [
        build_scenario(motion, level, duration_s=duration_s)
        for motion in MOTION_PROFILES
        for level in DISTURBANCE_LEVELS
    ]
