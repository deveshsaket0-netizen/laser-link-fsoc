"""
Frame renderer -- produces an actual grayscale pixel image with the beacon
drawn on it, plus image-domain noise and atmospheric effects. The noise
types and atmospheric presets below (gaussian/salt-pepper/poisson,
haze/fog/rain/low_light) are reasonable engineering choices matching the
problem statement's general request for realistic disturbances -- they are
NOT confirmed against a numbered official spec item (unverified; check
the real spec sheet if/when available and adjust presets if it's specific).

This is the piece the codebase was missing entirely: everywhere else,
"detection" means the true (u, v) is injected directly with noise added in
coordinate space. Nothing renders a pixel array. This module does exactly
that, and nothing else -- it doesn't touch engine.py or the existing
angle/pixel disturbance model, which keep working exactly as before for the
fast headless simulation loop. This is a separate, additional sensing path
used to (a) generate training data for the detector (fsoc_tracker/detector.py,
Part 4b) and (b) let a dashboard show a literal video feed if it wants one.

Order of operations, matching how a real sensor forms an image:
  draw target on background -> apply atmosphere (contrast/brightness,
  affects the whole scene) -> apply sensor noise (added at the sensor,
  after the scene is formed) -> clip to uint8.
"""

from dataclasses import dataclass
import numpy as np


# Atmosphere presets: (contrast_factor, brightness_shift). Modeled as a
# contrast/brightness reduction rather than scene-level physical rendering
# (rain streaks, fog particulates) -- a simplification choice, not something
# copied from a confirmed spec definition. Revisit if the real spec turns
# out to require more physically detailed atmospheric modeling.
ATMOSPHERE_PRESETS = {
    "clear": (1.0, 0),
    "haze": (0.70, 30),
    "fog": (0.50, 55),
    "rain": (0.75, -10),
    "low_light": (0.55, -70),
}


@dataclass
class RenderConfig:
    width: int = 640
    height: int = 480
    background_level: int = 40        # base scene grayscale (0-255)
    target_size_px: int = 10          # square beacon side length (spec default)
    target_intensity: int = 255

    # Image-domain sensor noise -- one or more types selectable at once,
    # hence a tuple. Type choices are a reasonable engineering default,
    # not confirmed against an official spec list (unverified).
    noise_types: tuple = ("gaussian",)   # subset of "salt_pepper" | "gaussian" | "poisson"
    salt_pepper_amount: float = 0.05     # fraction of pixels flipped
    gaussian_noise_std: float = 12.0     # intensity units, 0-255 scale
    poisson_scale: float = 6.0           # higher = less shot-noise-visible

    atmosphere: str = "clear"            # key into ATMOSPHERE_PRESETS


class FrameRenderer:
    def __init__(self, cfg: RenderConfig, seed: int = None):
        self.cfg = cfg
        self._rng = np.random.default_rng(seed)

    def render(self, u: float, v: float) -> np.ndarray:
        """
        Render one frame. (u, v) is the beacon's pixel position, or None if
        it's not visible this frame (out of FOV / occluded / dropout) --
        in which case the frame is background + noise only, no target.
        Returns a (height, width) uint8 array.
        """
        img = np.full((self.cfg.height, self.cfg.width), self.cfg.background_level, dtype=np.float32)

        if u is not None and v is not None:
            self._draw_target(img, u, v)

        img = self._apply_atmosphere(img)
        img = self._apply_noise(img)
        return np.clip(img, 0, 255).astype(np.uint8)

    def _draw_target(self, img: np.ndarray, u: float, v: float):
        half = self.cfg.target_size_px // 2
        row0 = int(round(v)) - half
        col0 = int(round(u)) - half
        row1 = row0 + self.cfg.target_size_px
        col1 = col0 + self.cfg.target_size_px

        r0, r1 = max(0, row0), min(self.cfg.height, row1)
        c0, c1 = max(0, col0), min(self.cfg.width, col1)
        if r0 < r1 and c0 < c1:
            img[r0:r1, c0:c1] = self.cfg.target_intensity

    def _apply_atmosphere(self, img: np.ndarray) -> np.ndarray:
        contrast, brightness = ATMOSPHERE_PRESETS[self.cfg.atmosphere]
        return img * contrast + brightness

    def _apply_noise(self, img: np.ndarray) -> np.ndarray:
        for noise_type in self.cfg.noise_types:
            if noise_type == "gaussian":
                img = img + self._rng.normal(0, self.cfg.gaussian_noise_std, img.shape)
            elif noise_type == "salt_pepper":
                mask = self._rng.random(img.shape) < self.cfg.salt_pepper_amount
                salt = self._rng.random(img.shape) < 0.5
                img = np.where(mask & salt, 255, img)
                img = np.where(mask & ~salt, 0, img)
            elif noise_type == "poisson":
                scale = self.cfg.poisson_scale
                img = self._rng.poisson(np.clip(img, 0, None) * scale) / scale
            elif noise_type == "none":
                pass
            else:
                raise ValueError(f"Unknown noise type: {noise_type}")
        return img
