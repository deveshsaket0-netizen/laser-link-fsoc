# AI-Based Detection (Part 4)

This is the one non-negotiable item: replacing direct pixel injection with
an actual trained model looking at actual images. Two pieces:

- **4a — the renderer** (done)
- **4b — the trained classifier** (done, see below)

## Why a renderer was needed first

Nowhere in the original codebase does a pixel image exist. `engine.py`
computes the beacon's true (u, v) and hands it straight to the Kalman
filter with noise added in coordinate space — there's no picture to run
computer vision on. `fsoc_tracker/render.py` produces one: an actual
grayscale `numpy` array with the beacon drawn on it, plus image-domain
noise and atmospheric effects.

This also closes a second gap independently of the AI requirement: the
problem statement asks for disturbances to be introduced "in the virtual
camera feed" (an actual image feed), which only makes sense applied to a
real image. The specific noise types (salt & pepper / Gaussian / Poisson)
and atmospheric conditions (haze/fog/rain/low-light) implemented here are
reasonable engineering choices for that requirement, not values confirmed
against a numbered official spec item -- **unverified**, check the real
spec sheet if/when available.

Deliberately **not** wired into `engine.py` / the stress or recovery
tests — those stay exactly as they were, fast and headless, using the
existing angle/pixel disturbance model. The renderer is a separate,
additional path used to (a) generate training data for the detector and
(b) optionally give a dashboard a literal video feed to display.

## `fsoc_tracker/render.py`

`RenderConfig` — the knobs: image size, background level, target size
(spec default 10x10 px), which noise type(s) to apply (`"gaussian"` |
`"salt_pepper"` | `"poisson"`, or several at once), and `atmosphere`
(`"clear"` | `"haze"` | `"fog"` | `"rain"` | `"low_light"`).

`FrameRenderer.render(u, v)` — pass `None, None` for a frame where the
target isn't visible (out of FOV, occluded, dropout). Order of operations
mirrors how a real sensor forms an image: draw the target on the
background, apply atmosphere (modeled as contrast/brightness reduction —
a simplification choice, not scene-level fog/rain particle rendering),
then apply sensor noise on top, since noise is added at the sensor after
the scene is already formed.

```python
from fsoc_tracker import FrameRenderer, RenderConfig

cfg = RenderConfig(noise_types=("gaussian", "salt_pepper"), atmosphere="fog")
renderer = FrameRenderer(cfg, seed=1)
frame = renderer.render(u=320, v=240)   # (480, 640) uint8 array
```

## Sanity check
```
python experiments/render_demo.py
```
Saves `experiments/output/render_samples.png` — a grid of the beacon under
every noise type and every atmospheric condition, so you can eyeball that
they look right before anything is trained on them.

## Next: 4b, the trained classifier

Two-stage architecture (agreed approach, see the main conversation for the
reasoning): a classical, cheap first pass finds candidate bright blobs in
the frame; a small trained neural net (`sklearn.neural_network.MLPClassifier`)
looks at each candidate patch and scores whether it's the real beacon or
noise/clutter. The renderer's ground truth (u, v) makes every candidate
patch self-labeling — no manual annotation needed to build the training set.

## 4b — `fsoc_tracker/detector.py`: the trained classifier

Two-stage design:
1. **Classical candidate proposal** (`propose_candidates`) — Otsu-threshold
   the frame, connected-component label it, return one `Candidate` per
   blob with shape/intensity features (area, bbox, extent, aspect ratio,
   mean/std intensity, contrast). Deliberately high-recall and not smart:
   it proposes every blob that *could* be the beacon, including noise.
2. **Trained `MLPClassifier`** scores each candidate and picks the best
   one above a confidence threshold. This is the actual AI-attributable
   step — it's what tells a real beacon apart from a noise blob that
   happens to be bright, which stage 1 alone cannot do reliably.

Features deliberately exclude the candidate's `(u, v)` position — the
classifier learns to recognize the beacon's *shape*, not "things near the
middle of the frame", so it generalizes across the whole image rather than
overfitting to wherever training targets happened to be rendered.

**Training data is self-labeled**, not manually annotated: the renderer
always knows the true `(u, v)` it drew the beacon at, so every candidate
the classical stage proposes gets auto-labeled positive (close to truth)
or negative (not) with zero manual work. `experiments/train_detector.py`
generates ~2,500 labeled candidates across 5 noise combos × 5 atmospheres,
trains a small MLP (16, 8 hidden units), and saves it to
`fsoc_tracker/models/detector_mlp.joblib`.

**Held-out test accuracy: 98.4%** (100% precision / 95% recall on the
beacon class — the model essentially never calls noise a beacon, and
misses a real beacon only occasionally). Full report in
`experiments/output/detector_training_report.txt`.

### Does it actually help? `experiments/benchmark_detector.py`

Compares `NaiveDetector` (classical-only baseline: pick the largest blob,
no learning) against `LearnedDetector` on held-out frames.

**Honest finding: raw success rate is identical — 0.720 for both.** The
value isn't in finding the target more often. It's in *how each one
fails* when it doesn't find it:

| | Naive | Learned |
|---|---|---|
| Success rate | 0.720 | 0.720 |
| **False-lock rate** (confidently wrong) | **0.280** | **0.065** |
| No-detection rate (correctly says "not found") | 0.000 | 0.215 |

The naive detector never says "I don't know" — every miss becomes a
confident, silent lock onto a noise blob. The learned detector converts
most of that into a safe "no detection" instead, which a downstream
system can act on (hold last state, widen search) rather than silently
tracking garbage. For a coarse-alignment system, false-lock is the
dangerous failure mode, not no-detection — so this is a real improvement
even though it doesn't show up in raw success rate.

By atmosphere condition (`experiments/output/detector_false_lock_chart.png`):
false-lock rate drops sharply under `fog` (0.40 → 0.05) and `low_light`
(0.40 → 0.00), and moderately under `clear`/`haze` (0.20 → ~0.04). **One
honest exception: under `rain`, the improvement is negligible (0.20 →
0.19)** — worth investigating further rather than glossing over; the
current guess is that the rain preset's specific contrast/brightness
combination produces candidate blobs whose shape features overlap more
with the beacon's than the other conditions do.

### Running it
```
python experiments/train_detector.py      # generates data, trains, saves model
python experiments/benchmark_detector.py  # naive vs learned comparison + charts
```

### Not yet done
- Not wired into `engine.py`'s fast headless loop (by design — see 4a).
  Wiring a `LearnedDetector` in as an alternative to the coordinate-space
  disturbance model, so the stress/recovery tests could optionally run
  against real rendered frames, is a reasonable next step but a much
  slower one (image rendering + OpenCV + MLP inference per frame vs.
  pure coordinate math).
- No investigation yet into *why* `rain` doesn't improve — flagged above,
  not resolved.
