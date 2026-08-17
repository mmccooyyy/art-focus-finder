"""
Center-of-Focus Detector v4 — Classical Computer Vision (no AI/ML services).

Key changes from v3 based on failure analysis of 10 benchmark images:

1. CONSENSUS-BASED FUSION instead of weighted average
   — v3's 6x face weight meant a single false positive face detection
     overwhelmed all other signals. v4 finds the peak of each map,
     then uses median-of-peaks (robust to outliers) combined with
     a spatial consensus vote.

2. Drastically reduced face detection weight (2x instead of 6x)
   — Face detection is useful when correct but catastrophically wrong
     when it false-positives on art textures. It's now a moderate signal.

3. CLAHE (Contrast Limited Adaptive Histogram Equalization) instead of
   simple equalizeHist — better for paintings with uneven lighting.
   CLAHE limits contrast amplification to avoid amplifying noise.

4. More aggressive border suppression (15% instead of 10%)
   — Saliency methods were still picking up border artifacts.

5. Stronger, tighter center bias — for art, the focal point is almost
   always in the central 50% of the image. Smaller sigma Gaussian.

6. Stricter face validation: require 2 eyes, check aspect ratio
   — Faces are roughly square (0.7-1.3 w/h ratio). Filter out
     elongated false positives.

7. Higher minNeighbors (10) for even fewer Haar false positives.

8. Median-of-peaks fusion: find each map's peak, take the median
   position. This is inherently robust to outlier signals (like
   a false positive face detection pointing to a corner).

All research-backed improvements from v3 are retained:
- Histogram equalization (now CLAHE) before Haar detection
- Multi-cascade: alt2, alt, default, profileface, upperbody
- Eye validation, detection grouping
- Connected component skin analysis
- Multi-scale spectral residual, frequency-tuned saliency
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
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self._init_cascades()

    def _init_cascades(self):
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
        """Detect the center of focus in an image."""
        if isinstance(image_path_or_array, str):
            image = cv2.imread(image_path_or_array)
            if image is None:
                raise ValueError(f"Could not load image: {image_path_or_array}")
        else:
            image = image_path_or_array.copy()

        h, w = image.shape[:2]
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

        # Skin color with connected components
        skin_result = self._skin_color_map_cc(proc_image)
        maps["skin"] = skin_result["map"]
        skin_detected = skin_result["detected"]

        # Face detection
        face_result = self._face_detection_multi(proc_image)
        maps["face_detection"] = face_result["map"]
        face_detected = face_result["detected"]

        # Upper body
        body_result = self._upperbody_detection(proc_image)
        maps["upperbody"] = body_result["map"]
        body_detected = body_result["detected"]

        # ─── Normalize all maps ───
        for key in maps:
            maps[key] = self._normalize(maps[key])

        # ─── Aggressive border suppression (15%) ───
        border_mask = self._border_suppression_mask(pw, ph, border_frac=0.15)
        for key in maps:
            if key != "center_bias":
                maps[key] = maps[key] * border_mask

        # ─── CONSENSUS-BASED FUSION ───
        # Step 1: Find the peak of each individual map
        peaks = {}  # key -> (x, y, confidence)
        for key, m in maps.items():
            peak_idx = np.unravel_index(np.argmax(m), m.shape)
            peaks[key] = (peak_idx[1], peak_idx[0], m[peak_idx])  # (x, y, conf)

        # Step 2: Weighted combination (but with much lower face weight)
        weights = self._compute_weights(face_detected, skin_detected, body_detected,
                                        skin_result.get("ratio", 0))

        combined = np.zeros((ph, pw), dtype=np.float64)
        total_weight = 0.0
        for key, weight in weights.items():
            if key in maps and weight > 0:
                combined += weight * maps[key]
                total_weight += weight
        if total_weight > 0:
            combined /= total_weight

        # Step 3: Strong-signal boost (but moderate)
        if face_detected or skin_detected:
            strong = np.zeros((ph, pw), dtype=np.float64)
            if face_detected:
                strong = np.maximum(strong, maps["face_detection"])
            if skin_detected:
                strong = np.maximum(strong, maps["skin"])
            combined = 0.65 * combined + 0.35 * strong
        elif body_detected:
            combined = 0.75 * combined + 0.25 * maps["upperbody"]

        # Step 4: Median-of-peaks consensus
        # Collect peak positions from reliable signals
        consensus_peaks = []
        for key in ["spectral_residual", "frequency_tuned", "local_contrast",
                     "color_contrast", "region_distinct", "center_bias"]:
            if key in peaks:
                consensus_peaks.append((peaks[key][0], peaks[key][1]))

        if face_detected:
            consensus_peaks.append((peaks["face_detection"][0], peaks["face_detection"][1]))
        if skin_detected:
            consensus_peaks.append((peaks["skin"][0], peaks["skin"][1]))
        if body_detected and not face_detected and not skin_detected:
            consensus_peaks.append((peaks["upperbody"][0], peaks["upperbody"][1]))

        # Compute median peak position
        if consensus_peaks:
            median_x = np.median([p[0] for p in consensus_peaks])
            median_y = np.median([p[1] for p in consensus_peaks])

            # Create a Gaussian at the median position and blend with combined map
            # This pulls the result toward the consensus position
            sigma_consensus = min(pw, ph) / 6
            y_grid, x_grid = np.meshgrid(np.arange(ph), np.arange(pw), indexing="ij")
            consensus_gauss = np.exp(-((x_grid - median_x)**2 / (2 * sigma_consensus**2) +
                                       (y_grid - median_y)**2 / (2 * sigma_consensus**2)))
            consensus_gauss = self._normalize(consensus_gauss)
            combined = 0.5 * combined + 0.5 * consensus_gauss

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

        # Method
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

    def _compute_weights(self, face_detected, skin_detected, body_detected, skin_ratio):
        """Moderate weights — face detection is no longer dominant."""
        weights = {
            "spectral_residual": 1.0,
            "frequency_tuned": 1.0,
            "edge_density": 0.3,
            "color_contrast": 0.8,
            "brightness": 0.2,
            "local_contrast": 1.2,
            "symmetry": 0.15,
            "skin": 0.0,
            "face_detection": 0.0,
            "upperbody": 0.0,
            "center_bias": 1.0,
            "region_dist": 0.8,
        }

        # Face detection: moderate weight (2x, not 6x)
        if face_detected:
            weights["face_detection"] = 2.0
            weights["center_bias"] = 0.5
            weights["spectral_residual"] = 0.5
            weights["frequency_tuned"] = 0.5

        # Skin color: moderate weight
        if skin_detected and skin_ratio < 0.3:
            weights["skin"] = 2.0
            if not face_detected:
                weights["center_bias"] = 0.6

        # Upper body fallback
        if body_detected and not face_detected and not skin_detected:
            weights["upperbody"] = 1.5
            weights["center_bias"] = 0.8

        # No strong signals: lean on saliency + center bias
        if not face_detected and not skin_detected and not body_detected:
            weights["center_bias"] = 1.5
            weights["local_contrast"] = 1.5
            weights["region_distinct"] = 1.0
            weights["spectral_residual"] = 1.2

        return weights

    # ─── Face detection ───

    def _face_detection_multi(self, image):
        """Multi-cascade face detection with strict validation."""
        h, w = image.shape[:2]
        face_map = np.zeros((h, w), dtype=np.float32)
        detected = False
        all_detections = []

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # CLAHE instead of equalizeHist — better for paintings
        gray_eq = self._clahe.apply(gray)

        # Try each frontal cascade
        frontal_cascades = ["frontal_alt2", "frontal_alt", "frontal_default"]
        for cascade_name in frontal_cascades:
            if cascade_name not in self._cascades:
                continue
            cascade = self._cascades[cascade_name]

            # minNeighbors=10 for very few false positives
            faces = cascade.detectMultiScale(
                gray_eq,
                scaleFactor=1.05,
                minNeighbors=10,
                minSize=(40, 40),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            if len(faces) > 0:
                for (fx, fy, fw, fh) in faces:
                    face_roi = gray_eq[fy:fy+fh, fx:fx+fw]
                    if face_roi.size == 0:
                        continue

                    # Strict validation: 2 eyes + aspect ratio check
                    eye_count = self._count_eyes(face_roi)
                    aspect_ok = 0.7 < (fw / fh) < 1.4  # faces are roughly square

                    cx = fx + fw / 2.0
                    cy = fy + fh / 2.0
                    in_bounds = (cx > w * 0.1 and cx < w * 0.9 and
                                 cy > h * 0.05 and cy < h * 0.85)

                    if eye_count >= 2 and aspect_ok and in_bounds:
                        all_detections.append((fx, fy, fw, fh, 1.0))
                        detected = True
                    elif eye_count >= 1 and aspect_ok and in_bounds and fw * fh > (w * h * 0.015):
                        all_detections.append((fx, fy, fw, fh, 0.6))
                        detected = True
                    elif in_bounds and aspect_ok and fw * fh > (w * h * 0.03):
                        # Large, well-proportioned detection even without eyes
                        all_detections.append((fx, fy, fw, fh, 0.3))
                        detected = True

                if detected:
                    break

        # Profile face cascade
        if not detected and "profile" in self._cascades:
            profile_cascade = self._cascades["profile"]
            for img_to_check, flip_label in [(gray_eq, "normal"), (cv2.flip(gray_eq, 1), "flipped")]:
                faces = profile_cascade.detectMultiScale(
                    img_to_check,
                    scaleFactor=1.05,
                    minNeighbors=10,
                    minSize=(40, 40),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                if len(faces) > 0:
                    for (fx, fy, fw, fh) in faces:
                        if flip_label == "flipped":
                            fx = w - fx - fw
                        cx = fx + fw / 2.0
                        cy = fy + fh / 2.0
                        in_bounds = (cx > w * 0.1 and cx < w * 0.9 and
                                     cy > h * 0.05 and cy < h * 0.85)
                        aspect_ok = 0.6 < (fw / fh) < 1.5
                        if in_bounds and aspect_ok and fw * fh > (w * h * 0.01):
                            all_detections.append((fx, fy, fw, fh, 0.5))
                            detected = True
                    if detected:
                        break

        # Group overlapping detections
        if len(all_detections) > 1:
            all_detections = self._group_detections(all_detections)

        # Build face map
        centroid = None
        if all_detections:
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

    def _count_eyes(self, face_roi):
        """Count eyes in a face ROI. Returns count (0, 1, or 2+)."""
        if "eye" not in self._cascades:
            return 2  # Can't validate, assume valid

        if face_roi.shape[0] < 20 or face_roi.shape[1] < 20:
            return 2  # Too small to validate

        # Apply CLAHE to the ROI for better eye detection
        eye_cascade = self._cascades["eye"]
        eyes = eye_cascade.detectMultiScale(
            face_roi,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(8, 8),
            maxSize=(face_roi.shape[1] // 3, face_roi.shape[0] // 3)
        )
        return len(eyes)

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
                if self._rects_overlap((fx, fy, fw, fh), (jx, jy, jw, jh)):
                    group.append((jx, jy, jw, jh, jconf))
                    used.add(j)
            avg_x = sum(g[0] for g in group) / len(group)
            avg_y = sum(g[1] for g in group) / len(group)
            avg_w = sum(g[2] for g in group) / len(group)
            avg_h = sum(g[3] for g in group) / len(group)
            max_conf = max(g[4] for g in group)
            groups.append((avg_x, avg_y, avg_w, avg_h, max_conf))

        return groups

    @staticmethod
    def _rects_overlap(r1, r2):
        x1, y1, w1, h1 = r1
        x2, y2, w2, h2 = r2
        return not (x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1)

    def _upperbody_detection(self, image):
        """Upper body detection as fallback."""
        h, w = image.shape[:2]
        body_map = np.zeros((h, w), dtype=np.float32)
        detected = False

        if "upperbody" not in self._cascades:
            return {"map": body_map, "detected": detected}

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_eq = self._clahe.apply(gray)

        bodies = self._cascades["upperbody"].detectMultiScale(
            gray_eq,
            scaleFactor=1.05,
            minNeighbors=5,
            minSize=(50, 50),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        if len(bodies) > 0:
            best = max(bodies, key=lambda b: b[2] * b[3])
            bx, by, bw, bh = best
            cx = bx + bw / 2.0
            cy = by + bh / 2.0
            sigma = max(bw, bh) / 1.5
            y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
            body_map = np.exp(-((x - cx)**2 / (2 * sigma**2) + (y - cy)**2 / (2 * sigma**2)))
            detected = True

        return {"map": body_map, "detected": detected}

    # ─── Skin color with connected components ───

    def _skin_color_map_cc(self, image):
        """HSV skin color detection with connected component analysis."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, w = image.shape[:2]

        lower1 = np.array([0, 20, 40])
        upper1 = np.array([25, 200, 255])
        lower2 = np.array([160, 20, 40])
        upper2 = np.array([180, 200, 255])

        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        skin_mask = mask1 + mask2

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)

        skin_ratio = np.sum(skin_mask > 0) / (h * w)
        detected = 0.005 < skin_ratio < 0.5

        skin_map = np.zeros((h, w), dtype=np.float32)
        centroid = None

        if detected:
            labeled, num_features = label(skin_mask > 0)
            if num_features > 0:
                sizes = ndimage.sum(skin_mask, labeled, range(1, num_features + 1))
                if len(sizes) > 0:
                    largest_label = np.argmax(sizes) + 1
                    largest_blob = (labeled == largest_label).astype(np.float32)
                    cy, cx = center_of_mass(largest_blob)
                    centroid = (cx / w * 100, cy / h * 100)
                    sigma = max(w, h) / 8
                    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
                    skin_map = np.exp(-((x - cx)**2 / (2 * sigma**2) + (y - cy)**2 / (2 * sigma**2)))
                    blob_smoothed = gaussian_filter(largest_blob, sigma=5)
                    skin_map = 0.6 * skin_map + 0.4 * blob_smoothed

        return {"map": skin_map, "detected": detected, "ratio": skin_ratio, "centroid": centroid}

    # ─── Saliency methods ───

    def _multi_scale_spectral_residual(self, image):
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
        """Tighter center bias — focal points in art are usually central."""
        y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        # Upper-center bias (portraits have faces in upper-center)
        cx, cy = w / 2.0, h * 0.40
        # Tighter sigma — central 50% gets most weight
        sigma_x, sigma_y = w / 3.0, h / 3.0
        center_gauss = np.exp(-((x - cx)**2 / (2 * sigma_x**2) + (y - cy)**2 / (2 * sigma_y**2)))
        # Rule of thirds intersections (weaker)
        third_x = [w / 3.0, 2 * w / 3.0]
        third_y = [h / 3.0, 2 * h / 3.0]
        thirds_map = np.zeros((h, w))
        for tx in third_x:
            for ty in third_y:
                sigma_t = min(w, h) / 6.0
                thirds_map += 0.15 * np.exp(-((x - tx)**2 / (2 * sigma_t**2) + (y - ty)**2 / (2 * sigma_t**2)))
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

    def _border_suppression_mask(self, w, h, border_frac=0.15):
        """Border suppression — 15% by default (more aggressive than v3's 10%)."""
        y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        border = int(min(w, h) * border_frac)
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
    output = os.path.join(os.path.dirname(os.path.abspath(__file__)), "detection_results_v4.json")
    print(f"Running focus detection v4 on images in {image_dir}...")
    results = batch_detect(image_dir, output)
    print(f"\nResults saved to {output}")
    print(f"Processed {len(results)} images")
