"""
Beacon detection -- Part 4b of the AI-detector work (see AI_DETECTOR.md).

Two-stage design, same shape as most real small-target detectors:
  1. Classical candidate proposal (cheap, high-recall, not smart) -- finds
     every blob that *could* be the beacon.
  2. A trained classifier scores each candidate and picks the best one --
     this is the stage that's actually "AI". It's what lets the system
     tell a real beacon apart from a noise blob that happens to be bright,
     which the classical stage alone cannot reliably do.

Two detector classes are provided so they can be benchmarked against each
other (see experiments/benchmark_detector.py):
  - NaiveDetector: classical-only baseline (largest/brightest blob, no
    learning). This is what the codebase would do WITHOUT the AI stage.
  - LearnedDetector: classical proposals + trained MLPClassifier scoring.

Both operate on rendered frames (fsoc_tracker/render.py), not on the
engine's fast coordinate-space path -- this module doesn't touch engine.py.
"""

from dataclasses import dataclass
import numpy as np
import cv2
import joblib


# Feature order is fixed and used identically at training and inference
# time. Deliberately excludes the blob's (u, v) position itself -- the
# classifier should learn to recognize the BEACON'S SHAPE, not "things
# near the middle of the frame", so it generalizes to any position.
FEATURE_NAMES = [
    "area", "bbox_w", "bbox_h", "extent", "aspect_ratio",
    "mean_intensity", "std_intensity", "contrast",
]


@dataclass
class Candidate:
    u: float
    v: float
    area: int
    bbox_w: int
    bbox_h: int
    extent: float           # area / bbox area -- ~1.0 for a solid square, low for scattered noise
    aspect_ratio: float
    mean_intensity: float
    std_intensity: float
    contrast: float         # mean_intensity - background estimate

    def features(self) -> np.ndarray:
        return np.array([
            self.area, self.bbox_w, self.bbox_h, self.extent, self.aspect_ratio,
            self.mean_intensity, self.std_intensity, self.contrast,
        ], dtype=np.float64)


def propose_candidates(frame: np.ndarray, min_area: int = 2, max_area: int = 400) -> list:
    """
    Classical stage: Otsu-threshold the frame, connected-component label
    the result, and return one Candidate per surviving blob. High recall
    by design -- this is meant to over-propose (including noise blobs)
    and let the classifier stage filter, not to be precise on its own.
    """
    background = float(np.median(frame))
    _, binary = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    candidates = []
    for i in range(1, num_labels):  # label 0 is background
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area or area > max_area:
            continue
        x, y, w, h = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                      stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
        cu, cv_ = centroids[i]
        mask = labels[y:y + h, x:x + w] == i
        region = frame[y:y + h, x:x + w][mask].astype(np.float64)

        candidates.append(Candidate(
            u=float(cu), v=float(cv_),
            area=int(area), bbox_w=int(w), bbox_h=int(h),
            extent=float(area) / (w * h),
            aspect_ratio=float(w) / h if h > 0 else 0.0,
            mean_intensity=float(region.mean()),
            std_intensity=float(region.std()),
            contrast=float(region.mean()) - background,
        ))
    return candidates


class NaiveDetector:
    """
    Classical-only baseline: no learning, just picks the largest bright
    blob. This is the "before AI" comparison point -- what the system
    would fall back to without the trained classifier.
    """

    def detect(self, frame: np.ndarray):
        candidates = propose_candidates(frame)
        if not candidates:
            return None
        best = max(candidates, key=lambda c: c.area)
        return best.u, best.v


class LearnedDetector:
    """
    Classical proposals + trained MLPClassifier picks the best one. This
    is the AI-attributable component: a real trained model, not a coordinate
    formula, deciding which candidate is actually the beacon.
    """

    def __init__(self, model_path: str, decision_threshold: float = 0.5):
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.scaler = bundle["scaler"]
        self.decision_threshold = decision_threshold

    def detect(self, frame: np.ndarray):
        candidates = propose_candidates(frame)
        if not candidates:
            return None

        X = np.stack([c.features() for c in candidates])
        X_scaled = self.scaler.transform(X)
        probs = self.model.predict_proba(X_scaled)[:, 1]  # P(is beacon)

        best_idx = int(np.argmax(probs))
        if probs[best_idx] < self.decision_threshold:
            return None  # nothing confidently looks like the beacon this frame

        best = candidates[best_idx]
        return best.u, best.v
