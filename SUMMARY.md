# Center-of-Focus Detection for Art Images — Research & Implementation Summary

## Goal

Detect the center of focus (primary subject/focal point) in art images from the Rijksmuseum collection using **classical computer vision only** — no AI/ML APIs. Target: within 10% of image diagonal consistently.

## Dataset

100 images downloaded from the Rijksmuseum API spanning paintings, photographs, prints, and drawings. 10 benchmark ground-truth points established via multimodal vision analysis.

---

## Research Conducted

All sources were fetched and read in full during this session. The following summarizes what each source contributed:

### Primary Sources

**Viola & Jones, "Rapid Object Detection using a Boosted Cascade of Simple Features" (CVPR 2001)**
- The foundational Haar cascade paper. Describes the cascade structure where each stage is an ensemble of weak learners (decision stumps) trained by AdaBoost.
- The first two features selected by AdaBoost detect: (1) eye region darker than cheek region, (2) eyes darker than nose bridge. These are "somewhat insensitive to size and location of the face."
- Training used 24x24 pixel sub-windows with variance normalization to handle lighting differences.
- Describes integration of multiple detections: overlapping detections are grouped into one (§5.6).
- Average of only 8 features evaluated per sub-window due to early rejection in the cascade.

**Viola & Jones, "Robust Real-time Object Detection" (IJCV 2001)**
- Extended version with 32-layer cascade, 4297 total features.
- All training sub-windows were variance normalized — this is critical for handling lighting variation.
- During detection, normalization is achieved by post-multiplying feature values by the standard deviation of the sub-window.

### Tutorials & Educational Sources

**OpenCV Cascade Classifier Tutorial (docs.opencv.org)**
- Uses `haarcascade_frontalface_alt.xml` (NOT `default`) — the alt cascade is more robust.
- **Critically applies `equalizeHist()` on the grayscale image before detection.** This histogram equalization is the variance normalization the Viola-Jones paper describes. Our v2 code was missing this entirely.
- Also demonstrates sequential face→eye detection as a validation step.

**PyImageSearch (Adrian Rosebrock) — "Haar Cascades" and "Face Detection"**
- Haar cascades are "notoriously prone to false-positives."
- Increasing `minNeighbors` from 5 to 7 eliminated false positives while keeping true positives.
- Recommended starting point: `scaleFactor=1.05`, `minNeighbors=5`, `minSize=(30,30)`.
- Face detection is most stable; eye and mouth cascades fire many false positives.
- For facial feature extraction, facial landmarks are recommended over Haar cascades.

**MachineLearningMastery — "Using Haar Cascade for Object Detection"**
- `scaleFactor` controls the scale pyramid (1.01-1.3 typical; lower = more detections but slower).
- `minNeighbors` controls false positive filtering (higher = fewer detections but higher quality).
- Three types of Haar features: edges, lines, center-surrounded.

**Will Berger — "Deep Learning Haar Cascade Explained"**
- Each stage is an ensemble of weak learners (decision stumps).
- Low false negative rate is critical per stage — if a stage rejects a true face, it's gone forever.
- High false positive rate per stage is acceptable — subsequent stages correct it.
- Adding more stages reduces overall false positives but also reduces true positive rate.

**Chi-Feng Wang — "Haar-Feature Classifiers vs CNNs"**
- Haar features are manually determined (unlike CNN kernels which are learned).
- Haar cascades can only detect objects with clear edges and lines.
- They struggle with partially covered faces, tilted heads.
- **Key insight for paintings:** Haar cascades trained on photos may not work well on paintings because painting textures differ from photo textures. This was confirmed by our experiments.

**Wikipedia — "Haar-like feature"**
- Tilted (45°) Haar-like features were introduced by Lienhart and Maydt.
- These can detect edges at 45°, which could help with non-frontal faces.
- However, rotated features have rounding errors at low resolutions.

**Towards Data Science — "AdaBoost, Step-by-Step"**
- AdaBoost builds weak learners sequentially.
- Each subsequent weak learner focuses on misclassified examples from previous rounds.
- The "amount of say" (alpha) is inversely proportional to the error rate.
- Final prediction is a weighted sum of all weak learners.

**GeeksForGeeks — "Face Detection using Cascade Classifier"**
- Uses `scaleFactor=1.2`, `minNeighbors=5`.
- Demonstrates sequential face→eye detection pattern.

### Available Cascade Files

The sandbox has 18 Haar cascade XML files, including:
- `haarcascade_frontalface_alt2.xml` — more features than default, generally more robust
- `haarcascade_frontalface_alt.xml` — used by the OpenCV tutorial
- `haarcascade_frontalface_default.xml` — the standard cascade
- `haarcascade_profileface.xml` — for profile faces (common in paintings)
- `haarcascade_upperbody.xml` — upper body fallback
- `haarcascade_eye.xml` — for face validation
- `haarcascade_fullbody.xml`, `haarcascade_lowerbody.xml`, `haarcascade_smile.xml`, etc.

---

## Algorithm Evolution

### v1 (Initial)
- Multi-scale spectral residual saliency, frequency-tuned saliency, edge density, color contrast, brightness, symmetry, center bias
- Single Haar cascade (`haarcascade_frontalface_default.xml`) with `minNeighbors=5`
- Face detection weight: 3x
- **Result: Not formally benchmarked, but visually poor**

### v2 (Improved Saliency + Skin Color)
- Added HSV-based skin color detection
- Added multi-scale spectral residual (3 scales)
- Added local contrast via unsharp masking
- Added region distinctiveness via k-means
- Added adaptive weighting based on detection signals
- Added border suppression (10%)
- **Result: Average 23.8%, Median 20.9%, 0/10 within 10%**

### v3 (Research-Backed Haar Improvements)
- **Added `equalizeHist()` before Haar detection** (variance normalization per Viola-Jones)
- **Multi-cascade**: alt2, alt, default, profileface (with image flip for left-facing profiles)
- **Eye validation**: face detections confirmed by checking for eyes within the ROI
- **Detection grouping**: overlapping detections merged (Viola-Jones §5.6)
- Higher `minNeighbors` (8) to suppress false positives in painting textures
- Larger `minSize` (40px) since art subjects aren't tiny
- Upper body detection as fallback
- Connected component analysis for skin color (largest blob = likely face)
- **Result: Average 24.4%, Median 19.8%, 0/10 within 10%**
- **Finding: The research-backed Haar improvements didn't help.** equalizeHist + multi-cascade + eye validation changed which detections fired but didn't improve accuracy. The core issue is that Haar cascades fundamentally don't work well on art images.

### v4 (Consensus-Based Fusion)
- **Consensus-based fusion**: finds the peak of each individual map, then uses median-of-peaks (robust to outliers) blended with the weighted combination
- **Drastically reduced face detection weight** (2x instead of 6x) — face detection is useful when correct but catastrophically wrong when it false-positives on art textures
- **CLAHE** (Contrast Limited Adaptive Histogram Equalization) instead of simple `equalizeHist` — better for paintings with uneven lighting, limits contrast amplification to avoid amplifying noise
- **More aggressive border suppression** (15% instead of 10%)
- **Stronger, tighter center bias** — smaller sigma Gaussian, focal points in art are usually central
- **Stricter face validation**: require 2 eyes, check aspect ratio (0.7-1.4 w/h for frontal, 0.6-1.5 for profile)
- **Higher `minNeighbors`** (10) for even fewer false positives
- **Result: Average 18.0%, Median 15.9%, 0/10 within 10%**
- **24% improvement in average error over v2.** Two images near the 10% target (rijks_010 at 10.2%, rijks_015 at 12.0%).

### v4 Tuning Attempt (Reverted)
- Tried reducing consensus blend from 0.5 to 0.3 (less consensus pulling)
- Tried agreement boosting (face+skin peaks close → boost both to 4x/3x)
- **Result: Average 19.6%** — worse than original v4
- rijks_000 jumped from 18.4% to 37.9% because reducing consensus let false positive face detection dominate again
- **Reverted to original v4 settings**

---

## Failure Analysis (v4)

### Failure Mode 1: False Positive Face Detections (rijks_000, 001, 020)
Haar cascades fire on text patterns, store window displays, stadium lights — anything with face-like texture. Even with eye validation, CLAHE, and minNeighbors=10, false positives still occur. The consensus fusion helps dilute these but can't eliminate them.

**rijks_000** (photo of cluttered display): Face detected at (68%, 47%), GT is (45%, 35%). The face detection found a face-like pattern in the display items. Error: 18.4%.

**rijks_001** (civil rights protest photo): Face detected at (26%, 49%), GT is (40%, 32%). Multiple faces in the scene; detection picked the wrong one. Error: 15.4%.

**rijks_020** (nighttime stadium): Face detected at (72.5%, 35.5%), GT is (50%, 36%). False positive from stadium lights/structure. Error: 15.9%.

### Failure Mode 2: Saliency Picks Wrong Region (rijks_005, 003)
All saliency methods (spectral residual, frequency-tuned, local contrast) agree on the wrong location. This is a fundamental limitation of classical saliency — it can't understand semantic content, only image statistics.

**rijks_005** (landscape with watermill): Saliency points to (69.5%, 83.1%) — bottom-right. GT is (50.1%, 31.8%) — upper-center. The saliency methods are distracted by high-contrast foliage/water in the lower portion. Error: 38.8%.

**rijks_003**: Saliency points to (49%, 43.8%) — center. GT is (30%, 65%) — lower-left. Center bias pulls toward center when GT is off-center. Error: 20.1%.

### Failure Mode 3: Face Correct but GT Disagrees (rijks_025)
The face IS where the algorithm says it is, but the ground truth focal point is elsewhere. This reveals that "center of focus" is subjective — sometimes it's not the face.

**rijks_025** (portrait of a man): Face detected at (53.1%, 49.9%), GT is (65%, 65%). The face is actually at ~50%, 40% per vision analysis. The GT may represent the overall composition center rather than the face. Error: 14.0%.

### Near-Success Cases (rijks_010, 015)
**rijks_010** (street scene): Saliency at (41.2%, 56.9%), GT is (35%, 70%). Error: 10.2% — very close to target. No face detected (correctly), saliency worked well.

**rijks_015**: Face detected at (45.5%, 41.7%), GT is (35%, 55%). Error: 12.0% — close to target.

---

## Key Findings

1. **Haar cascades are the wrong tool for art images.** They were trained on photographs and fire false positives on painting textures, text patterns, and architectural features. Even with equalizeHist/CLAHE, eye validation, aspect ratio checks, and minNeighbors=10, false positives persist. The research confirmed this: Wang noted that Haar cascades "can only detect objects with clear edges and lines" and struggle with anything outside their training distribution.

2. **Consensus-based fusion is the biggest single improvement.** By finding the peak of each map and taking the median position, we get robustness to outlier signals (like a false positive face detection pointing to a corner). This reduced average error from 23.8% (v2) to 18.0% (v4).

3. **Face detection weight is the critical knob.** At 6x (v3), a single false positive face detection overwhelms everything. At 2x (v4), it's a useful but non-dominant signal. Reducing it was the single most impactful change.

4. **Saliency methods fail together.** When spectral residual, frequency-tuned, and local contrast all point to the wrong area, the consensus can't help — it just finds the median of wrong answers. This is a fundamental limitation of classical saliency.

5. **The 10% target is very ambitious for classical CV.** The best we achieved is 18.0% average, with the best single image at 10.2%. Classical methods can't understand semantic content — they only see edges, colors, and contrast patterns. Reaching 10% consistently would likely require semantic understanding (i.e., deep learning).

---

## Current State

| Version | Average Error | Median Error | Within 10% |
|---------|--------------|-------------|-----------|
| v2 | 23.8% | 20.9% | 0/10 |
| v3 | 24.4% | 19.8% | 0/10 |
| v4 | **18.0%** | **15.9%** | 0/10 |

Best single image: rijks_010 at 10.2% error.
Worst single image: rijks_005 at 38.8% error.

---

## Recommendations

1. **To reach the 10% target**: Classical CV alone is unlikely to get there consistently. The remaining errors come from semantic misunderstanding — saliency can't tell a watermill from foliage, and Haar cascades can't tell a real face from a face-like texture pattern. A lightweight on-device model (e.g., MediaPipe face detection, or a small CNN trained on art images) would likely solve the face detection problem. For non-face focal points, a saliency model trained on art composition would help.

2. **If staying with classical CV**: The most promising unexplored direction is **image type classification** — classifying images as portraits, landscapes, still lifes, or photographs, then using type-specific parameter sets. Portraits should use stronger face/skin detection; landscapes should use stronger center bias; photographs should use different saliency parameters.

3. **Improve ground truth**: The current 10-image benchmark is small and some ground truth points may be subjective (e.g., rijks_025 where the face is at 50%,40% but GT says 65%,65%). Expanding to 30+ images with multiple annotators would give more reliable evaluation.

4. **Try LBP cascades**: OpenCV includes `lbpcascade_frontalface.xml` (Local Binary Patterns). LBP features are more robust to lighting changes than Haar features and might perform better on paintings. This was mentioned in the OpenCV docs but not yet tested.
