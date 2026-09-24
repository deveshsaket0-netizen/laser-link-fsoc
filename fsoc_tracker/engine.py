"""
Simulation engine -- orchestrates motion -> disturbance -> projection ->
detection -> Kalman filter -> controller -> metrics, one frame at a time.

DESIGN INTENT: this file has zero rendering/GUI code. `TrackingEngine.step()`
returns a plain dict of everything happening this frame (true position,
measured pixel, filtered estimate, gimbal command, error, lock status).
A dashboard -- PyQt, Streamlit, a web frontend, whatever -- should treat
this dict as its entire data contract: poll `step()` in a loop and render
whatever fields it needs. Nothing here should ever import a GUI library.
"""

import math
from .config import TrackerConfig
from .motion import BeaconMotion
from .camera import CameraModel
from .disturbances import DisturbanceModel
from .kalman import BeaconTracker
from .metrics import MetricsTracker


class TrackingEngine:
    def __init__(self, config: TrackerConfig = None, seed: int = 42):
        self.cfg = config or TrackerConfig()
        self.camera = CameraModel(self.cfg.camera)
        self.motion = BeaconMotion(self.cfg.motion, seed=seed)
        self.disturbance = DisturbanceModel(self.cfg.disturbance, seed=seed)
        self.metrics = MetricsTracker(
            lock_threshold_rad=self.cfg.lock_threshold_mrad / 1000.0
        )

        self.t = 0.0
        self.dt = self.cfg.kalman.dt

        # gimbal starts centered on boresight
        self.gimbal_pan = 0.0
        self.gimbal_tilt = 0.0

        self.tracker = None   # BeaconTracker, created on first valid detection
        self.frame_idx = 0

    def _controller_command(self, dtheta_x: float, dtheta_y: float) -> tuple:
        """
        Proportional controller (section: build order step 6). Converts
        filtered angular error into a gimbal slew command, capped at the
        configured max slew rate. Deliberately simple -- swap for a
        PID/feedforward controller later without touching the rest of
        the engine.
        """
        kp = self.cfg.gimbal.kp
        max_step = math.radians(self.cfg.gimbal.max_slew_rate_dps) * self.dt

        cmd_x = max(-max_step, min(max_step, kp * dtheta_x))
        cmd_y = max(-max_step, min(max_step, kp * dtheta_y))
        return cmd_x, cmd_y

    def step(self) -> dict:
        """Advance the simulation by one frame. Returns a state dict."""
        t = self.t

        # 1. Ground truth beacon angular position (section 4)
        true_theta_x, true_theta_y = self.motion.step(t)

        # 2. Apply angle-level disturbances: turbulence + vibration (section 4.4)
        disturbed_theta_x, disturbed_theta_y = self.disturbance.apply_angular(
            true_theta_x, true_theta_y, t
        )

        # 3. Project into pixel space given current gimbal pointing (section 2)
        pixel = self.camera.angle_to_pixel(
            disturbed_theta_x, disturbed_theta_y, self.gimbal_pan, self.gimbal_tilt
        )

        detected = pixel is not None and not self.disturbance.dropout() and not self.disturbance.occluded(t)

        state = {
            "t": t,
            "frame": self.frame_idx,
            "true_theta_x_mrad": true_theta_x * 1000.0,
            "true_theta_y_mrad": true_theta_y * 1000.0,
            "detected": detected,
            "occluded": self.disturbance.occluded(t),
            "gimbal_pan_deg": math.degrees(self.gimbal_pan),
            "gimbal_tilt_deg": math.degrees(self.gimbal_tilt),
        }

        if detected:
            u, v = pixel
            # 4. Sensor/centroiding noise at pixel level (section 4.4)
            u_meas, v_meas = self.disturbance.apply_pixel(u, v)
            state["meas_u"] = u_meas
            state["meas_v"] = v_meas

            # 5. Kalman filter predict/update (section 5)
            if self.tracker is None:
                self.tracker = BeaconTracker(
                    self.cfg.kalman, self.cfg.disturbance, u_meas, v_meas
                )
            else:
                self.tracker.predict()
                self.tracker.update(u_meas, v_meas)

            filt_u, filt_v = self.tracker.position
            state["filt_u"] = filt_u
            state["filt_v"] = filt_v

            # 6. Pixel -> angular error (section 2.4 / 3.2), then controller
            dtheta_x, dtheta_y = self.camera.pixel_to_angle_error(filt_u, filt_v)
            theta_e = self.camera.angular_error_magnitude(dtheta_x, dtheta_y)

            cmd_x, cmd_y = self._controller_command(dtheta_x, dtheta_y)
            self.gimbal_pan += cmd_x
            self.gimbal_tilt += cmd_y

            locked = theta_e < (self.cfg.lock_threshold_mrad / 1000.0)
            state["theta_e_mrad"] = theta_e * 1000.0
            state["locked"] = locked

            self.metrics.log_frame(t, theta_e, locked)
        else:
            # predict-only when no detection this frame (occlusion/dropout)
            if self.tracker is not None:
                self.tracker.predict()
            state["theta_e_mrad"] = None
            state["locked"] = False
            self.metrics.log_frame(t, float("inf"), False)

        state["metrics"] = self.metrics.snapshot()

        self.t += self.dt
        self.frame_idx += 1
        return state

    def run(self, n_steps: int):
        """Convenience generator for headless / scripted runs."""
        for _ in range(n_steps):
            yield self.step()
