# Art Focus Finder

Classical-CV center-of-focus detector for art images (Rijksmuseum collection) — no AI/ML APIs.

## Requirements

- Python 3
- opencv-python
- numpy
- scipy

## Install

```bash
pip install opencv-python numpy scipy
```

## Usage

### CLI

```bash
python focus_detector.py [image_dir]
```

- `image_dir` defaults to `images/` next to `focus_detector.py`.
- Scans the directory for `.jpg`/`.png`/`.jpeg` files, runs detection on each, and writes results to `detection_results.json` (also next to the script).

### Programmatic

```python
from focus_detector import FocusDetector

detector = FocusDetector()
result = detector.detect("path/to/image.jpg")  # or a numpy array (BGR image)
print(result["x_pct"], result["y_pct"], result["method"])

# Optional: draw the heatmap + focus marker over the image
detector.visualize("path/to/image.jpg", result, output_path="overlay.jpg")
```

`detect()` accepts an image path (`str`) or an already-loaded BGR `numpy` array.

## Output format

`detection_results.json` is a list of per-image objects:

```json
{
  "filename": "rijks_000.jpg",
  "x_pct": 68.2,
  "y_pct": 47.1,
  "face_detected": true,
  "skin_detected": false,
  "body_detected": false,
  "method": "face_haar"
}
```

`x_pct`/`y_pct` locate the detected focal point as a percentage of image width/height. `method` is one of `face_haar`, `skin_color`, `upperbody`, or `saliency` — whichever signal drove the final result. On a per-image processing error, the entry is `{"filename": ..., "error": "..."}` instead.

`FocusDetector.detect()` returns the same fields plus `x`, `y` (pixel coords), `width`, `height`, and `heatmap` (a full-resolution saliency array, not serialized to JSON).

## How it works

- **CLAHE** (contrast-limited adaptive histogram equalization) preprocesses grayscale images before Haar detection, handling uneven painting lighting better than plain histogram equalization.
- **Multi-cascade Haar face detection** tries several frontal cascades plus a profile cascade, validated by eye count and aspect ratio to suppress false positives from painting textures.
- Multiple **saliency maps** (multi-scale spectral residual, frequency-tuned, local/color contrast, region distinctiveness, symmetry, skin color, upper-body) are computed and normalized.
- Signals are combined via a weighted blend, then refined with **median-of-peaks consensus fusion** — the peak of each map is taken and the median position pulls the final result toward agreement, which is robust to a single outlier (e.g. a false-positive face detection).
- A **center-bias prior** (Gaussian weighted toward upper-center plus rule-of-thirds points) and 15% border suppression reflect the tendency of focal points in art to sit centrally.


