"""
Kalman filter -- section 5 of the math framework.

2-axis constant-velocity model tracking pixel position + velocity
(state = [u, v, u_dot, v_dot]). Uses a small vendored linear KF (see
_LinearKalmanFilter below, no external dependency) so the matrix algebra
matches sections 5.2-5.5 exactly; BeaconTracker just wires up F, H, Q, R
from config and exposes predict()/update() in terms of pixel measurements.

Swap-in point for later: replace this class with an EKF/IMM variant
(section 5.6/5.7) without touching engine.py, as long as predict()/update()
keep the same signature.
"""

import numpy as np
from .config import KalmanConfig, DisturbanceConfig


class _LinearKalmanFilter:
    """
    Minimal linear Kalman filter with the same F/H/Q/R/P/x attributes and
    predict()/update()/S interface as filterpy.kalman.KalmanFilter. Vendored
    so this package has zero external dependencies beyond numpy -- filterpy
    is not installable on offline/competition machines and this is ~15 lines
    of standard predict/update algebra, identical to what filterpy runs.
    """

    def __init__(self, dim_x: int, dim_z: int):
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.x = np.zeros(dim_x)
        self.F = np.eye(dim_x)
        self.H = np.zeros((dim_z, dim_x))
        self.Q = np.eye(dim_x)
        self.R = np.eye(dim_z)
        self.P = np.eye(dim_x)
        self.S = None

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z: np.ndarray):
        y = z - self.H @ self.x
        self.S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(self.S)
        self.x = self.x + K @ y
        i = np.eye(self.dim_x)
        self.P = (i - K @ self.H) @ self.P


def build_kalman_filter(kcfg: KalmanConfig, dcfg: DisturbanceConfig) -> _LinearKalmanFilter:
    dt = kcfg.dt
    kf = _LinearKalmanFilter(dim_x=4, dim_z=2)

    # State transition F -- section 5.2
    kf.F = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])

    # Measurement matrix H -- section 5.3 (position-only observation)
    kf.H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ])

    # Process noise Q -- section 5.2, discretized white-noise-acceleration model
    q = kcfg.process_noise_q
    dt2 = dt * dt
    dt3 = dt2 * dt
    kf.Q = q * np.array([
        [dt3 / 3, 0, dt2 / 2, 0],
        [0, dt3 / 3, 0, dt2 / 2],
        [dt2 / 2, 0, dt, 0],
        [0, dt2 / 2, 0, dt],
    ])

    # Measurement noise R -- section 5.3, from sensor noise unless overridden
    sigma_px = kcfg.meas_noise_override_px or dcfg.sensor_noise_px
    kf.R = np.eye(2) * (sigma_px ** 2)

    # Initial state covariance -- generous, filter converges quickly
    kf.P *= 500.0

    return kf


class BeaconTracker:
    """Thin, engine-friendly wrapper around the filterpy KalmanFilter."""

    def __init__(self, kcfg: KalmanConfig, dcfg: DisturbanceConfig,
                 init_u: float, init_v: float):
        self.kf = build_kalman_filter(kcfg, dcfg)
        self.kf.x = np.array([init_u, init_v, 0.0, 0.0])
        self.initialized = True

    def predict(self):
        """Section 5.4"""
        self.kf.predict()

    def update(self, u: float, v: float):
        """Section 5.5 -- call with a real detection."""
        self.kf.update(np.array([u, v]))

    @property
    def position(self) -> tuple:
        return float(self.kf.x[0]), float(self.kf.x[1])

    @property
    def velocity(self) -> tuple:
        return float(self.kf.x[2]), float(self.kf.x[3])

    @property
    def innovation_covariance_trace(self) -> float:
        """Useful for track-loss gating (section 5.7)."""
        return float(np.trace(self.kf.S)) if self.kf.S is not None else float("inf")
