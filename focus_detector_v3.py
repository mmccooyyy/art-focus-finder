"""
Center-of-Focus Detector v3 — Classical Computer Vision (no AI/ML services).

Incorporates findings from the Viola-Jones papers and all referenced sources:

1. Histogram equalization before Haar detection (variance normalization per Viola-Jones)
   — The OpenCV tutorial and both papers stress this. Our v2 was missing it entirely.
2. Multiple cascade files: alt2, alt, default, profileface, upperbody
   — alt2 is more robust than default (more features). profileface catches 3/4 views
   common in paintings. upperbody is a fallback when no face is found.
3. Higher minNeighbors (8-10) to suppress false positives in painting textures
   — PyImageSearch showed 5→7 eliminates FPs; paintings are noisier than photos.
4. Larger minSize (40px on 400px images) — main subjects in art are not tiny.
5. Eye validation: face detections are confirmed by checking for eyes within the ROI
   — Standard technique from OpenCV tutorial and PyImageSearch.
6. Detection grouping: overlapping detections are merged (Viola-Jones paper §5.6)
7. Skin color with connected component analysis — largest skin blob is likely the face
8. Multi-scale spectral residual, frequency-tuned saliency, local contrast
9. Border suppression, upper-third center bias for portraits
10. Adaptive weighting based on what was detected

Sources consulted:
- Viola & Jones, "Rapid Object Detection using a Boosted Cascade of Simple Features" (CVPR 2001)
- Viola & Jones, "Robust Real-time Object Detection" (IJCV 2001)
- OpenCV Cascade Classifier tutorial (docs.opencv.org/4.x/db/d28/tutorial_cascade_classifier.html)
- PyImageSearch: "OpenCV Haar Cascades" and "OpenCV Face detection with Haar cascades"
- MachineLearningMastery: "Using Haar Cascade for Object Detection"
- Wikipedia: "Haar-like feature"
- Will Berger: "Deep Learning Haar Cascade Explained"
- Chi-Feng Wang: "What's the Difference Between Haar-Feature Classifiers and CNNs?"
- Towards Data Science: "AdaBoost, Step-by-Step"
- GeeksForGeeks: "Face Detection using Cascade Classifier using OpenCV"
"""

import cv2
import numpy as np
import os
import json
from scipy import ndimage
from scipy.ndimage import gaussian_filter, label, center_of_mass


class FocusDetector:
    """Self-contained center-of-focus detector using classical CV."""

    def __init__(self, face_cascade_path=None):
        self._cascades = {}
        self._init_cascades()

    def _init_cascades(self):
        """Initialize multiple Haar cascades for robust face detection."""
        haardir = cv2.data.haarcascades

        cascade_configs = {
            "frontal_alt2": "haarcascade_frontalface_alt2.xml",
            "frontal_alt": "haarcascade_frontalface_alt.xml",
            "frontal_default": "haarcascade_frontalface_default.xml",
            "profile": "haarcascade_profileface.xml",
            "upperbody": "haarcascade_upperbody.xml",
            "eye": "haarcascade_eye.xml",
        }

        for name, filename in cascade_configs.items():
            path = os.path.join(haardir, filename)
            cascade = cv2.CascadeClassifier(path)
            if not cascade.empty():
                self._cascades[name] = cascade

    def detect(self, image_path_or_array):
        """
        Detect the center of focus in an image.

        Returns dict with x_pct, y_pct, x, y, heatmap, width, height,
        face_detected, skin_detected, method, and debug info.
        """
        # Load image
        if isinstance(image_path_or_array, str):
            image = cv2.imread(image_path_or_array)
            if image is None:
                raise ValueError(f"Could not load image: {image_path_or_array}")
        else:
            image = image_path_or_array.copy()

        h, w = image.shape[:2]

        # Work at standardized size
        max_dim = 400
        scale = max_dim / max(w, h)
        if scale < 1.0:
            proc_w, proc_h = int(w * scale), int(h * scale)
            proc_image = cv2.resize(image, (proc_w, proc_h))
        else:
            proc_image = image.copy()
            proc_w, proc_h = w, h

        ph, pw = proc_image.shape[:2]

        # ─── Compute all saliency maps ───
        maps = {}
        maps["spectral_residual"] = self._multi_scale_spectral_residual(proc_image)
        maps["frequency_tuned"] = self._frequency_tuned_saliency(proc_image)
        maps["edge_density"] = self._edge_density_map(proc_image)
        maps["color_contrast"] = self._local_color_contrast(proc_image)
        maps["brightness"] = self._brightness_map(proc_image)
        maps["local_contrast"] = self._local_contrast_map(proc_image)
        maps["symmetry"] = self._symmetry_map(proc_image)
        maps["center_bias"] = self._center_bias_map(pw, ph)
        maps["region_distinct"] = self._region_distinctiveness(proc_image)

        # Skin color detection with connected component analysis
        skin_result = self._skin_color_map_cc(proc_image)
        maps["skin"] = skin_result["map"]
        skin_detected = skin_result["detected"]
        skin_centroid = skin_result.get("centroid")

        # Face detection (multi-cascade with eye validation)
        face_result = self._face_detection_multi(proc_image)
        maps["face_detection"] = face_result["map"]
        face_detected = face_result["detected"]
        face_centroid = face_result.get("centroid")

        # Upper body detection (fallback)
        body_result = self._upperbody_detection(proc_image)
        maps["upperbody"] = body_result["map"]
        body_detected = body_result["detected"]

        # ─── Normalize all maps ───
        for key in maps:
            maps[key] = self._normalize(maps[key])

        # ─── Apply border suppression ───
        border_mask = self._border_suppression_mask(pw, ph)
        for key in maps:
            if key != "center_bias":
                maps[key] = maps[key] * border_mask

        # ─── Adaptive weight combination ───
        weights = self._compute_adaptive_weights(
            face_detected, skin_detected, body_detected, skin_result.get("ratio", 0)
        )

        combined = np.zeros((ph, pw), dtype=np.float64)
        total_weight = 0.0
        for key, weight in weights.items():
            if key in maps and weight > 0:
                combined += weight * maps[key]
                total_weight += weight

        if total_weight > 0:
            combined /= total_weight

        # ─── Strong-signal fusion ───
        if face_detected or skin_detected:
            strong_signal = np.zeros((ph, pw), dtype=np.float64)
            if face_detected:
                strong_signal = np.maximum(strong_signal, maps["face_detection"])
            if skin_detected:
                strong_signal = np.maximum(strong_signal, maps["skin"])
            combined = 0.55 * combined + 0.45 * strong_signal
        elif body_detected:
            combined = 0.7 * combined + 0.3 * maps["upperbody"]

        # Smoothing
        combined = gaussian_filter(combined, sigma=4)
        combined = self._normalize(combined)

        # Find peak
        peak_idx = np.unravel_index(np.argmax(combined), combined.shape)
        peak_y_proc, peak_x_proc = peak_idx

        # Scale back
        inv_scale = 1.0 / scale if scale < 1.0 else 1.0
        peak_x = int(peak_x_proc * inv_scale)
        peak_y = int(peak_y_proc * inv_scale)
        peak_x = max(0, min(w - 1, peak_x))
        peak_y = max(0, min(h - 1, peak_y))

        heatmap_full = cv2.resize(combined, (w, h))

        # Determine method
        if face_detected:
            method = "face_haar"
        elif skin_detected:
            method = "skin_color"
        elif body_detected:
            method = "upperbody"
        else:
            method = "saliency"

        return {
            "x_pct": (peak_x / w) * 100,
            "y_pct": (peak_y / h) * 100,
            "x": peak_x,
            "y": peak_y,
            "heatmap": heatmap_full,
            "width": w,
            "height": h,
            "face_detected": bool(face_detected),
            "skin_detected": bool(skin_detected),
            "body_detected": bool(body_detected),
            "method": method,
        }

    def _compute_adaptive_weights(self, face_detected, skin_detected, body_detected, skin_ratio):
        """Compute weights based on what was detected."""
        weights = {
            "spectral_residual": 1.0,
            "frequency_tuned": 1.0,
            "edge_density": 0.4,
            "color_contrast": 0.8,
            "brightness": 0.3,
            "local_contrast": 1.2,
            "symmetry": 0.2,
            "skin": 0.0,
            "face_detection": 0.0,
            "upperbody": 0.0,
            "center_bias": 0.8,
            "region_distinct": 0.8,
        }

        if face_detected:
            weights["face_detection"] = 6.0
            weights["center_bias"] = 0.2
            weights["spectral_residual"] = 0.3
            weights["frequency_tuned"] = 0.3
            weights["local_contrast"] = 0.5

        if skin_detected and skin_ratio < 0.3:
            weights["skin"] = 3.5
            if not face_detected:
                weights["center_bias"] = 0.4
                weights["spectral_residual"] = 0.5

        if body_detected and not face_detected and not skin_detected:
            weights["upperbody"] = 2.0
            weights["center_bias"] = 0.6

        if not face_detected and not skin_detected and not body_detected:
            weights["center_bias"] = 1.2
            weights["local_contrast"] = 1.5
            weights["region_distinct"] = 1.0
            weights["spectral_residual"] = 1.2

        return weights

    # ─── Face detection (multi-cascade with eye validation) ───

    def _face_detection_multi(self, image):
        """
        Multi-cascade face detection with eye validation.

        Key improvements from research:
        - Uses equalizeHist() before detection (variance normalization per Viola-Jones)
        - Tries multiple cascade files (alt2, alt, default, profile)
        - Higher minNeighbors (8) to suppress false positives in painting textures
        - Larger minSize (40px) since art subjects aren't tiny
        - Validates faces by checking for eyes within the face ROI
        - Groups overlapping detections (Viola-Jones §5.6)
        """
        h, w = image.shape[:2]
        face_map = np.zeros((h, w), dtype=np.float32)
        detected = False
        all_detections = []

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # CRITICAL: Histogram equalization for variance normalization
        # This is what the OpenCV tutorial does and what Viola-Jones describe.
        # Without this, lighting variation in paintings causes missed/false detections.
        gray_eq = cv2.equalizeHist(gray)

        # Try each frontal cascade with eye validation
        frontal_cascades = ["frontal_alt2", "frontal_alt", "frontal_default"]
        for cascade_name in frontal_cascades:
            if cascade_name not in self._cascades:
                continue
            cascade = self._cascades[cascade_name]

            # Higher minNeighbors (8) to suppress false positives in painting textures
            # PyImageSearch showed 5→7 helps; paintings are noisier so we go higher
            faces = cascade.detectMultiScale(
                gray_eq,
                scaleFactor=1.05,
                minNeighbors=8,
                minSize=(40, 40),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            if len(faces) > 0:
                # Validate each face with eye detection
                for (fx, fy, fw, fh) in faces:
                    face_roi = gray_eq[fy:fy+fh, fx:fx+fw]
                    if face_roi.size == 0:
                        continue

                    # Check for eyes within the face (reduces false positives)
                    has_eyes = self._validate_eyes(face_roi)

                    # Also check if the detection is in a reasonable position
                    # (not in the extreme corners of the image)
                    cx = fx + fw / 2.0
                    cy = fy + fh / 2.0
                    in_bounds = (cx > w * 0.1 and cx < w * 0.9 and
                                 cy > h * 0.05 and cy < h * 0.85)

                    if has_eyes and in_bounds:
                        all_detections.append((fx, fy, fw, fh, 1.0))  # validated
                        detected = True
                    elif in_bounds and fw * fh > (w * h * 0.01):
                        # Large detection in reasonable position even without eyes
                        all_detections.append((fx, fy, fw, fh, 0.5))  # unvalidated
                        detected = True

                if detected:
                    break  # Found faces with this cascade, no need to try others

        # Try profile face cascade if no frontal faces found
        if not detected and "profile" in self._cascades:
            profile_cascade = self._cascades["profile"]
            # Profile faces: also try flipped image for left-facing profiles
            for img_to_check, flip_label in [(gray_eq, "normal"), (cv2.flip(gray_eq, 1), "flipped")]:
                faces = profile_cascade.detectMultiScale(
                    img_to_check,
                    scaleFactor=1.05,
                    minNeighbors=8,
                    minSize=(40, 40),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                if len(faces) > 0:
                    for (fx, fy, fw, fh) in faces:
                        if flip_label == "flipped":
                            fx = w - fx - fw  # unflip x coordinate
                        cx = fx + fw / 2.0
                        cy = fy + fh / 2.0
                        in_bounds = (cx > w * 0.1 and cx < w * 0.9 and
                                     cy > h * 0.05 and cy < h * 0.85)
                        if in_bounds and fw * fh > (w * h * 0.01):
                            all_detections.append((fx, fy, fw, fh, 0.7))
                            detected = True
                    if detected:
                        break

        # Group overlapping detections (Viola-Jones §5.6)
        if len(all_detections) > 1:
            all_detections = self._group_detections(all_detections)

        # Build face map from grouped detections
        centroid = None
        if all_detections:
            # Use the highest-confidence detection
            best = max(all_detections, key=lambda d: d[4])
            fx, fy, fw, fh, conf = best
            cx = fx + fw / 2.0
            cy = fy + fh / 2.0
            centroid = (cx / w * 100, cy / h * 100)

            for (fx, fy, fw, fh, conf) in all_detections:
                cx = fx + fw / 2.0
                cy = fy + fh / 2.0
                sigma = max(fw, fh) / 1.5
                y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
                blob = conf * np.exp(-((x - cx)**2 / (2 * sigma**2) + (y - cy)**2 / (2 * sigma**2)))
                face_map += blob

        return {"map": face_map, "detected": detected, "centroid": centroid}

    def _validate_eyes(self, face_roi):
        """Check if eyes are detected within a face ROI (false positive filter)."""
        if "eye" not in self._cascades:
            return True  # Can't validate, assume valid

        if face_roi.shape[0] < 20 or face_roi.shape[1] < 20:
            return True  # Too small to validate

        eye_cascade = self._cascades["eye"]
        eyes = eye_cascade.detectMultiScale(
            face_roi,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(10, 10),
            maxSize=(face_roi.shape[1] // 3, face_roi.shape[0] // 3)
        )

        # At least one eye detected confirms this is likely a real face
        return len(eyes) >= 1

    def _group_detections(self, detections):
        """Group overlapping detections (Viola-Jones §5.6)."""
        if len(detections) <= 1:
            return detections

        groups = []
        used = set()

        for i, (fx, fy, fw, fh, conf) in enumerate(detections):
            if i in used:
                continue
            group = [(fx, fy, fw, fh, conf)]
            used.add(i)

            for j, (jx, jy, jw, jh, jconf) in enumerate(detections):
                if j in used:
                    continue
                # Check overlap
                if self._rects_overlap((fx, fy, fw, fh), (jx, jy, jw, jh)):
                    group.append((jx, jy, jw, jh, jconf))
                    used.add(j)

            # Average the group
            avg_x = sum(g[0] for g in group) / len(group)
            avg_y = sum(g[1] for g in group) / len(group)
            avg_w = sum(g[2] for g in group) / len(group)
            avg_h = sum(g[3] for g in group) / len(group)
            max_conf = max(g[4] for g in group)
            groups.append((avg_x, avg_y, avg_w, avg_h, max_conf))

        return groups

    @staticmethod
    def _rects_overlap(r1, r2):
        """Check if two rectangles overlap."""
        x1, y1, w1, h1 = r1
        x2, y2, w2, h2 = r2
        return not (x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1)

    def _upperbody_detection(self, image):
        """Upper body detection as fallback when no face is found."""
        h, w = image.shape[:2]
        body_map = np.zeros((h, w), dtype=np.float32)
        detected = False

        if "upperbody" not in self._cascades:
            return {"map": body_map, "detected": detected}

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_eq = cv2.equalizeHist(gray)

        bodies = self._cascades["upperbody"].detectMultiScale(
            gray_eq,
            scaleFactor=1.05,
            minNeighbors=5,
            minSize=(50, 50),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        if len(bodies) > 0:
            # Take the largest detection
            best = max(bodies, key=lambda b: b[2] * b[3])
            bx, by, bw, bh = best
            cx = bx + bw / 2.0
            cy = by + bh / 2.0
            sigma = max(bw, bh) / 1.5

            y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
            body_map = np.exp(-((x - cx)**2 / (2 * sigma**2) + (y - cy)**2 / (2 * sigma**2)))
            detected = True

        return {"map": body_map, "detected": detected}

    # ─── Skin color detection with connected components ───

    def _skin_color_map_cc(self, image):
        """
        HSV-based skin color detection with connected component analysis.

        Instead of just blurring the skin mask, we find the largest connected
        component of skin-colored pixels. The centroid of the largest blob
        is likely the face in a portrait.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, w = image.shape[:2]

        # Skin color ranges in HSV (broad for paintings)
        lower1 = np.array([0, 20, 40])
        upper1 = np.array([25, 200, 255])
        lower2 = np.array([160, 20, 40])
        upper2 = np.array([180, 200, 255])

        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        skin_mask = mask1 + mask2

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)

        skin_ratio = np.sum(skin_mask > 0) / (h * w)
        detected = 0.005 < skin_ratio < 0.5

        skin_map = np.zeros((h, w), dtype=np.float32)
        centroid = None

        if detected:
            # Connected component analysis — find the largest skin blob
            labeled, num_features = label(skin_mask > 0)
            if num_features > 0:
                # Find the largest component (excluding background label 0)
                sizes = ndimage.sum(skin_mask, labeled, range(1, num_features + 1))
                if len(sizes) > 0:
                    largest_label = np.argmax(sizes) + 1
                    largest_blob = (labeled == largest_label).astype(np.float32)

                    # Get centroid of the largest blob
                    cy, cx = center_of_mass(largest_blob)
                    centroid = (cx / w * 100, cy / h * 100)

                    # Create a Gaussian centered on the largest blob centroid
                    sigma = max(w, h) / 8
                    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
                    skin_map = np.exp(-((x - cx)**2 / (2 * sigma**2) + (y - cy)**2 / (2 * sigma**2)))

                    # Also add the raw blob (smoothed) for spatial accuracy
                    blob_smoothed = gaussian_filter(largest_blob, sigma=5)
                    skin_map = 0.6 * skin_map + 0.4 * blob_smoothed

        return {"map": skin_map, "detected": detected, "ratio": skin_ratio, "centroid": centroid}

    # ─── Saliency methods (same as v2, kept for completeness) ───

    def _multi_scale_spectral_residual(self, image):
        """Multi-scale spectral residual saliency."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        results = []
        for downsample in [1, 2, 4]:
            if downsample > 1:
                h, w = gray.shape
                small = cv2.resize(gray, (max(1, w // downsample), max(1, h // downsample)))
            else:
                small = gray
            sr = self._spectral_residual_single(small)
            if downsample > 1:
                sr = cv2.resize(sr, (gray.shape[1], gray.shape[0]))
            results.append(sr)
        combined = np.mean(results, axis=0)
        combined = gaussian_filter(combined, sigma=5)
        return combined

    def _spectral_residual_single(self, gray):
        h, w = gray.shape
        if h < 8 or w < 8:
            return np.zeros_like(gray)
        fft = np.fft.fft2(gray)
        fft_shifted = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shifted)
        phase = np.angle(fft_shifted)
        log_magnitude = np.log(magnitude + 1e-8)
        kernel_size = max(3, min(h, w) // 8)
        if kernel_size % 2 == 0:
            kernel_size += 1
        avg_filter = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
        avg_log = ndimage.convolve(log_magnitude, avg_filter, mode="reflect")
        spectral_residual = log_magnitude - avg_log
        new_magnitude = np.exp(spectral_residual)
        new_fft = new_magnitude * np.exp(1j * phase)
        new_fft_shifted = np.fft.ifftshift(new_fft)
        saliency = np.abs(np.fft.ifft2(new_fft_shifted))
        saliency = saliency ** 2
        saliency = gaussian_filter(saliency, sigma=5)
        return saliency

    def _frequency_tuned_saliency(self, image):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        L, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
        L_mean, a_mean, b_mean = np.mean(L), np.mean(a), np.mean(b)
        L_blur = gaussian_filter(L, sigma=5)
        a_blur = gaussian_filter(a, sigma=5)
        b_blur = gaussian_filter(b, sigma=5)
        return (L_blur - L_mean) ** 2 + (a_blur - a_mean) ** 2 + (b_blur - b_mean) ** 2

    def _edge_density_map(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        h, w = edges.shape
        border = max(5, min(h, w) // 20)
        edges[:border, :] = 0
        edges[-border:, :] = 0
        edges[:, :border] = 0
        edges[:, -border:] = 0
        return gaussian_filter(edges.astype(np.float32), sigma=12)

    def _local_color_contrast(self, image):
        img_float = image.astype(np.float32)
        local_mean = np.zeros_like(img_float)
        for c in range(3):
            local_mean[:, :, c] = gaussian_filter(img_float[:, :, c], sigma=20)
        diff = img_float - local_mean
        contrast = np.sqrt(np.sum(diff ** 2, axis=2))
        return gaussian_filter(contrast, sigma=5)

    def _brightness_map(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray_norm = gray / gray.max() if gray.max() > 0 else gray
        return gaussian_filter(gray_norm, sigma=12)

    def _local_contrast_map(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        blurred = gaussian_filter(gray, sigma=8)
        contrast = np.abs(gray - blurred)
        return gaussian_filter(contrast, sigma=6)

    def _symmetry_map(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        h_flip = np.fliplr(gray)
        h_diff = np.abs(gray - h_flip)
        h_sym = 1.0 / (1.0 + h_diff / 50.0)
        h_sym = gaussian_filter(h_sym, sigma=8)
        v_flip = np.flipud(gray)
        v_diff = np.abs(gray - v_flip)
        v_sym = 1.0 / (1.0 + v_diff / 50.0)
        v_sym = gaussian_filter(v_sym, sigma=8)
        return h_sym * 0.7 + v_sym * 0.3

    def _center_bias_map(self, w, h):
        y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        cx, cy = w / 2.0, h * 0.42  # upper-center bias for portraits
        sigma_x, sigma_y = w / 2.5, h / 2.5
        center_gauss = np.exp(-((x - cx)**2 / (2 * sigma_x**2) + (y - cy)**2 / (2 * sigma_y**2)))
        third_x = [w / 3.0, 2 * w / 3.0]
        third_y = [h / 3.0, 2 * h / 3.0]
        thirds_map = np.zeros((h, w))
        for tx in third_x:
            for ty in third_y:
                sigma_t = min(w, h) / 5.0
                thirds_map += 0.2 * np.exp(-((x - tx)**2 / (2 * sigma_t**2) + (y - ty)**2 / (2 * sigma_t**2)))
        return center_gauss + thirds_map

    def _region_distinctiveness(self, image):
        h, w = image.shape[:2]
        small_w, small_h = max(50, w // 2), max(50, h // 2)
        small = cv2.resize(image, (small_w, small_h))
        data = small.reshape((-1, 3)).astype(np.float32)
        k = 5
        _, labels, centers = cv2.kmeans(
            data, k, None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0),
            3, cv2.KMEANS_PP_CENTERS
        )
        mean_color = np.mean(data, axis=0)
        distinct_map = np.zeros((small_h, small_w), dtype=np.float32)
        for i in range(k):
            cluster_pixels = data[labels.flatten() == i]
            if len(cluster_pixels) > 0:
                cluster_mean = np.mean(cluster_pixels, axis=0)
                dist = np.linalg.norm(cluster_mean - mean_color)
                size_ratio = len(cluster_pixels) / len(data)
                size_weight = 1.0 / (1.0 + abs(size_ratio - 0.15) * 5)
                mask = (labels.reshape((small_h, small_w)) == i).astype(np.float32)
                distinct_map += dist * size_weight * mask
        distinct_map = gaussian_filter(distinct_map, sigma=5)
        return cv2.resize(distinct_map, (w, h))

    def _border_suppression_mask(self, w, h):
        y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        border = min(w, h) // 10
        dist_x = np.minimum(x, w - 1 - x)
        dist_y = np.minimum(y, h - 1 - y)
        dist = np.minimum(dist_x, dist_y)
        return np.clip(dist / border, 0, 1).astype(np.float32)

    @staticmethod
    def _normalize(arr):
        min_val, max_val = np.min(arr), np.max(arr)
        if max_val - min_val > 1e-8:
            return (arr - min_val) / (max_val - min_val)
        return np.zeros_like(arr)

    def visualize(self, image_path, result, output_path=None):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        h, w = image.shape[:2]
        x, y = result["x"], result["y"]
        heatmap = result["heatmap"]
        heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(image, 0.6, heatmap_color, 0.4, 0)
        cv2.drawMarker(overlay, (x, y), (0, 255, 0), cv2.MARKER_CROSS, 30, 2)
        cv2.circle(overlay, (x, y), 12, (0, 255, 0), 2)
        label = f"({result['x_pct']:.1f}%, {result['y_pct']:.1f}%) [{result['method']}]"
        cv2.putText(overlay, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if output_path:
            cv2.imwrite(output_path, overlay)
        return overlay


def batch_detect(image_dir, output_path=None):
    detector = FocusDetector()
    results = []
    image_files = sorted(
        [f for f in os.listdir(image_dir) if f.endswith((".jpg", ".png", ".jpeg"))]
    )
    for fname in image_files:
        fpath = os.path.join(image_dir, fname)
        try:
            result = detector.detect(fpath)
            results.append({
                "filename": fname,
                "x_pct": result["x_pct"],
                "y_pct": result["y_pct"],
                "face_detected": bool(result["face_detected"]),
                "skin_detected": bool(result["skin_detected"]),
                "body_detected": bool(result.get("body_detected", False)),
                "method": result["method"],
            })
            print(f"  {fname}: ({result['x_pct']:.1f}%, {result['y_pct']:.1f}%)"
                  f"  method={result['method']}")
        except Exception as e:
            print(f"  {fname}: ERROR - {e}")
            results.append({"filename": fname, "error": str(e)})
    if output_path:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
    return results


if __name__ == "__main__":
    import sys
    image_dir = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
    output = os.path.join(os.path.dirname(os.path.abspath(__file__)), "detection_results_v3.json")
    print(f"Running focus detection v3 on images in {image_dir}...")
    results = batch_detect(image_dir, output)
    print(f"\nResults saved to {output}")
    print(f"Processed {len(results)} images")
