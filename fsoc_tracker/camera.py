"""
Camera & coordinate model -- section 2 of the math framework.

Pipeline: beacon angular offset (from boresight, world-referenced) ->
account for current gimbal pointing -> pinhole projection -> pixel coords.
And the inverse: pixel error -> angular error (section 2.4 / 3.2), which is
what the controller and Kalman filter actually consume.

We work directly in small-angle space (angles from boresight) rather than
carrying full 3D world points, since coarse-alignment tracking cares about
angular error, not absolute range. This keeps the math in this file a
direct, checkable implementation of sections 2.3-2.4 and 3.1-3.2.
"""

import math
from .config import CameraConfig


class CameraModel:
    def __init__(self, cfg: CameraConfig):
        self.cfg = cfg

    def angle_to_pixel(self, beacon_theta_x: float, beacon_theta_y: float,
                        gimbal_pan: float, gimbal_tilt: float) -> tuple:
        """
        Project a beacon at world angular position (beacon_theta_x, beacon_theta_y)
        into the image plane, given the camera is currently pointed at
        (gimbal_pan, gimbal_tilt). Returns (u, v) pixel coords, or None if the
        beacon falls outside the sensor (out of FOV).

        This implements the pinhole model of section 2.3, using the
        *relative* angle between beacon and boresight -- valid under the
        small-angle assumption stated in section 6.
        """
        rel_x = beacon_theta_x - gimbal_pan
        rel_y = beacon_theta_y - gimbal_tilt

        u = self.cfg.cx + self.cfg.fx * math.tan(rel_x)
        v = self.cfg.cy + self.cfg.fy * math.tan(rel_y)

        if 0 <= u < self.cfg.width and 0 <= v < self.cfg.height:
            return u, v
        return None

    def pixel_to_angle_error(self, u: float, v: float) -> tuple:
        """
        Section 2.4 / 3.2: convert pixel error (relative to image center,
        i.e. current boresight) into angular error using IFOV.
        """
        e_u = u - self.cfg.cx
        e_v = v - self.cfg.cy
        dtheta_x = e_u * self.cfg.ifov_x
        dtheta_y = e_v * self.cfg.ifov_y
        return dtheta_x, dtheta_y

    @staticmethod
    def angular_error_magnitude(dtheta_x: float, dtheta_y: float) -> float:
        """Section 3.2: theta_e = sqrt(dtheta_x^2 + dtheta_y^2)"""
        return math.hypot(dtheta_x, dtheta_y)
