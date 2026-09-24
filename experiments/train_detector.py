"""
Trains the beacon-vs-noise classifier (Part 4b).

Self-labeling strategy: since the renderer always knows the true (u, v) it
drew the beacon at, every candidate blob the classical proposer finds can
be auto-labeled -- positive if it's close to the true position, negative
otherwise -- with no manual annotation required. This is what makes a
trained-model approach practical here without a real labeled dataset.

Run: python experiments/train_detector.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

from fsoc_tracker.render import FrameRenderer, RenderConfig
from fsoc_tracker.detector import propose_candidates, FEATURE_NAMES

NOISE_COMBOS = [("none",), ("gaussian",), ("salt_pepper",), ("poisson",), ("gaussian", "salt_pepper")]
ATMOSPHERES = ["clear", "haze", "fog", "rain", "low_light"]
FRAMES_PER_COMBO = 40
POSITIVE_TOLERANCE_PX = 7   # candidate centroid within this of true (u,v) -> positive
MAX_NEGATIVES_PER_FRAME = 3


def generate_dataset(seed: int = 0):
    rng = np.random.default_rng(seed)
    X_rows, y_rows, meta_rows = [], [], []

    for noise_types in NOISE_COMBOS:
        for atmosphere in ATMOSPHERES:
            cfg = RenderConfig(noise_types=noise_types, atmosphere=atmosphere)
            renderer = FrameRenderer(cfg, seed=int(rng.integers(0, 1_000_000)))

            for _ in range(FRAMES_PER_COMBO):
                margin = cfg.target_size_px
                u = rng.uniform(margin, cfg.width - margin)
                v = rng.uniform(margin, cfg.height - margin)

                frame = renderer.render(u, v)
                candidates = propose_candidates(frame)
                if not candidates:
                    continue

                positives, negatives = [], []
                for c in candidates:
                    dist = ((c.u - u) ** 2 + (c.v - v) ** 2) ** 0.5
                    (positives if dist <= POSITIVE_TOLERANCE_PX else negatives).append(c)

                for c in positives:
                    X_rows.append(c.features())
                    y_rows.append(1)
                    meta_rows.append((noise_types, atmosphere))

                if negatives:
                    chosen = rng.choice(len(negatives), size=min(MAX_NEGATIVES_PER_FRAME, len(negatives)), replace=False)
                    for i in chosen:
                        X_rows.append(negatives[i].features())
                        y_rows.append(0)
                        meta_rows.append((noise_types, atmosphere))

    return np.array(X_rows), np.array(y_rows), meta_rows


def main():
    print("Generating self-labeled training data from the renderer...")
    X, y, meta = generate_dataset(seed=0)
    print(f"  {len(X)} candidate samples  ({int(y.sum())} positive / {int((1 - y).sum())} negative)")

    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)
    import csv
    with open(os.path.join(out_dir, "detector_training_data.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(FEATURE_NAMES + ["label"])
        for row, label in zip(X, y):
            writer.writerow(list(row) + [int(label)])
    print(f"  Saved dataset to {out_dir}/detector_training_data.csv")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=2000, random_state=0)
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["noise/other", "beacon"])
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n=== Test set performance (held-out, {len(X_test)} samples) ===")
    print(f"Accuracy: {acc:.4f}")
    print(report)
    print("Confusion matrix (rows=true, cols=predicted):")
    print(cm)

    with open(os.path.join(out_dir, "detector_training_report.txt"), "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n{report}\nConfusion matrix:\n{cm}\n")

    models_dir = os.path.join(os.path.dirname(__file__), "..", "fsoc_tracker", "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "detector_mlp.joblib")
    joblib.dump({"model": model, "scaler": scaler, "feature_names": FEATURE_NAMES}, model_path)
    print(f"\nSaved trained model to {model_path}")


if __name__ == "__main__":
    main()
