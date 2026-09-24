"""
Central configuration for the FSOC coarse-alignment tracking engine.

Every number that shows up in the math framework (camera intrinsics, gimbal
limits, filter noise, disturbance strength) lives here as a single source of
truth. A dashboard layer should read/write THIS object, never hardcode
numbers into rendering or UI code.
"""

from dataclasses import dataclass, field


@dataclass
class CameraConfig:
    # Image sensor resolution (pixels)
    width: int = 640
    height: int = 480
    # Field of view (degrees, full angle) -> used to derive focal length in pixels
    fov_x_deg: float = 20.0
    fov_y_deg: float = 15.0

    @property
    def fx(self) -> float:
        import math
        return self.width / (2 * math.tan(math.radians(self.fov_x_deg) / 2))

    @property
    def fy(self) -> float:
        import math
        return self.height / (2 * math.tan(math.radians(self.fov_y_deg) / 2))

    @property
    def cx(self) -> float:
        return self.width / 2.0

    @property
    def cy(self) -> float:
        return self.height / 2.0

    @property
    def ifov_x(self) -> float:
        """radians per pixel, x axis"""
        import math
        return math.radians(self.fov_x_deg) / self.width

    @property
    def ifov_y(self) -> float:
        """radians per pixel, y axis"""
        import math
        return math.radians(self.fov_y_deg) / self.height


@dataclass
class GimbalConfig:
    pan_limit_deg: float = 90.0     # +/- range
    tilt_limit_deg: float = 45.0
    max_slew_rate_dps: float = 60.0  # deg/sec, caps controller output
    kp: float = 0.8                  # proportional gain, angular error -> slew command


@dataclass
class MotionConfig:
    """
    Parameters for synthetic beacon trajectory generators (see motion.py).

    profile: "stationary" | "linear" | "circular" | "lissajous" | "sudden" | "random_walk"
    The first five names match the team's shared Evaluation Plan scenarios
    (see the Member 6 analysis toolkit); random_walk is an extra stress
    case beyond that set. Each profile only reads its own fields below.
    """
    profile: str = "lissajous"

    linear_rate_deg_s: float = 2.0        # linear: constant angular rate

    circular_radius_deg: float = 5.0      # circular: angular radius
    circular_rate_deg_s: float = 3.0      # circular: constant angular rate

    lissajous_amp_deg: float = 5.0        # lissajous: amplitude
    lissajous_freq_hz: float = 0.2        # lissajous: base frequency

    sudden_switch_time_s: float = 10.0    # sudden: time of the direction change
    sudden_rate1_deg_s: float = 2.0       # sudden: angular rate before the switch
    sudden_rate2_deg_s: float = -3.0      # sudden: angular rate after the switch

    random_walk_sigma_deg: float = 0.05   # random_walk: per-step std dev


@dataclass
class DisturbanceConfig:
    enable_turbulence: bool = True
    turbulence_sigma_urad: float = 50.0   # angle-of-arrival jitter, micro-radians

    enable_vibration: bool = True
    vib_amplitudes_deg: tuple = (0.03, 0.015)
    vib_freqs_hz: tuple = (8.0, 21.0)

    enable_sensor_noise: bool = True
    sensor_noise_px: float = 1.5          # centroid noise, pixels (1-sigma)

    enable_dropout: bool = False          # missed-detection simulation
    dropout_prob: float = 0.02

    # Scripted occlusion: deliberate, contiguous "something blocks the
    # camera" windows -- distinct from dropout's random per-frame misses.
    # Each entry is (start_time_s, duration_s). Used by the recovery test
    # (experiments/run_recovery_test.py) to measure re-acquisition time
    # against a 1s re-acquisition target. NOTE: this 1s figure is an
    # engineering assumption we adopted, not a confirmed value from the
    # official SIH parameters table -- verify against the actual spec
    # sheet and update SPEC_LIMIT_S in run_recovery_test.py if it differs.
    enable_occlusion: bool = False
    occlusion_windows: tuple = ()


@dataclass
class KalmanConfig:
    dt: float = 1.0 / 30.0           # frame period, seconds (30 FPS)
    process_noise_q: float = 5.0     # acceleration-noise spectral density
    # measurement noise is derived from DisturbanceConfig.sensor_noise_px by default,
    # but can be overridden here for filter mistuning experiments
    meas_noise_override_px: float = None


@dataclass
class TrackerConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    gimbal: GimbalConfig = field(default_factory=GimbalConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    disturbance: DisturbanceConfig = field(default_factory=DisturbanceConfig)
    kalman: KalmanConfig = field(default_factory=KalmanConfig)
    lock_threshold_mrad: float = 2.0   # angular error below which we call it "locked"
