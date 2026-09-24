"""
Renderer sanity check -- renders the beacon under every noise type and
atmospheric condition so you can eyeball that they look right before
building the detector on top of them.

Run: python experiments/render_demo.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.pyplot as plt
from fsoc_tracker import FrameRenderer, RenderConfig
from fsoc_tracker.render import ATMOSPHERE_PRESETS

U, V = 320, 240   # dead center, for a clean look


def main():
    noise_cases = [
        ("clean", ("none",)),
        ("gaussian", ("gaussian",)),
        ("salt_pepper", ("salt_pepper",)),
        ("poisson", ("poisson",)),
        ("gaussian+salt_pepper", ("gaussian", "salt_pepper")),
    ]
    atmosphere_cases = list(ATMOSPHERE_PRESETS.keys())

    fig, axs = plt.subplots(2, max(len(noise_cases), len(atmosphere_cases)), figsize=(16, 6))

    for i, (label, noise_types) in enumerate(noise_cases):
        cfg = RenderConfig(noise_types=noise_types)
        renderer = FrameRenderer(cfg, seed=1)
        img = renderer.render(U, V)
        axs[0, i].imshow(img, cmap="gray", vmin=0, vmax=255)
        axs[0, i].set_title(f"noise: {label}", fontsize=9)
        axs[0, i].axis("off")

    for i, atmosphere in enumerate(atmosphere_cases):
        cfg = RenderConfig(atmosphere=atmosphere, noise_types=("gaussian",))
        renderer = FrameRenderer(cfg, seed=1)
        img = renderer.render(U, V)
        axs[1, i].imshow(img, cmap="gray", vmin=0, vmax=255)
        axs[1, i].set_title(f"atmosphere: {atmosphere}", fontsize=9)
        axs[1, i].axis("off")

    for i in range(len(noise_cases), axs.shape[1]):
        axs[0, i].axis("off")
    for i in range(len(atmosphere_cases), axs.shape[1]):
        axs[1, i].axis("off")

    plt.tight_layout()
    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "render_samples.png")
    plt.savefig(out_path, dpi=130)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
