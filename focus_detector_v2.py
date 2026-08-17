"""
Center-of-Focus Detector v2 — Classical Computer Vision (no AI/ML services).

Key improvements over v1:
1. Skin color detection (HSV) — finds face regions in paintings where Haar cascades fail
2. Multi-scale spectral residual — captures both fine and coarse saliency
3. Local contrast (unsharp masking) — better than global contrast for art
4. Region distinctiveness — watershed-based segmentation for large distinct objects
5. Corner/border suppression — prevents false peaks at image edges
6. Upper-third bias — portraits typically have subjects in upper-center
7. Max-based fusion — when a strong signal (face/skin) exists, it dominates
8. Adaptive weighting — adjusts weights based on image characteristics

All maps are normalized to [0,1], combined with adaptive weights, and the
peak of the final heatmap is the predicted focus point.

Output: (x_pct, y_pct) — focus point as percentage of image dimensions.
"""

import cv2
import numpy as np
import os
import json
from scipy import ndimage
from scipy.ndimage import gaussian_filter, uniform_filter, maximum_filter


class FocusDetector:
    """Self-contained center-of-focus detector using classical CV."""

    def __init__(self, weights=None, face_cascade_path=None):
        self._face_cascade = None
        self._init_face_cascade(face_cascade_path)

    def _init_face_cascade(self, custom_path=None):
        """Initialize Haar cascade for face detection (local, pre-trained)."""
        if custom_path:
            self._face_cascade = cv2.CascadeClassifier(custom_path)
            if not self._face_cascade.empty():
                return
            self._face_cascade = None
            return

        possible_paths = [
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml",
            cv2.data.haarcascades + "haarcascade_frontalface_alt.xml",
            cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml",
        ]
        for path in possible_paths:
            try:
                cascade = cv2.CascadeClassifier(path)
                if not cascade.empty():
                    self._face_cascade = cascade
                    return
            except Exception:
                continue
        self._face_cascade = None

    def detect(self, image_path_or_array):
        """
        Detect the center of focus in an image.

        Returns dict with x_pct, y_pct, x, y, heatmap, width, height, face_detected, method.
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

        # Multi-scale spectral residual
        maps["spectral_residual"] = self._multi_scale_spectral_residual(proc_image)

        # Frequency-tuned saliency
        maps["frequency_tuned"] = self._frequency_tuned_saliency(proc_image)

        # Edge density (with border suppression)
        maps["edge_density"] = self._edge_density_map(proc_image)

        # Color contrast (local)
        maps["color_contrast"] = self._local_color_contrast(proc_image)

        # Brightness
        maps["brightness"] = self._brightness_map(proc_image)

        # Local contrast (unsharp masking)
        maps["local_contrast"] = self._local_contrast_map(proc_image)

        # Symmetry
        maps["symmetry"] = self._symmetry_map(proc_image)

        # Skin color detection
        skin_result = self._skin_color_map(proc_image)
        maps["skin"] = skin_result["map"]
        skin_detected = skin_result["detected"]
        skin_ratio = skin_result["ratio"]

        # Face detection (Haar cascade)
        face_result = self._face_detection_map(proc_image)
        maps["face_detection"] = face_result["map"]
        face_detected = face_result["detected"]

        # Center bias
        maps["center_bias"] = self._center_bias_map(pw, ph)

        # Region distinctiveness
        maps["region_distinct"] = self._region_distinctiveness(proc_image)

        # ─── Normalize all maps ───
        for key in maps:
            maps[key] = self._normalize(maps[key])

        # ─── Apply border suppression to all maps ───
        border_mask = self._border_suppression_mask(pw, ph)
        for key in maps:
            if key not in ("center_bias",):
                maps[key] = maps[key] * border_mask

        # ─── Adaptive weight combination ───
        weights = self._compute_adaptive_weights(
            face_detected, skin_detected, skin_ratio, ph, pw
        )

        combined = np.zeros((ph, pw), dtype=np.float64)
        total_weight = 0.0
        for key, weight in weights.items():
            if key in maps and weight > 0:
                combined += weight * maps[key]
                total_weight += weight

        if total_weight > 0:
            combined /= total_weight

        # ─── Max-fusion for strong signals ───
        # If face or skin detection is strong, boost their contribution
        if face_detected or skin_detected:
            strong_signal = np.zeros((ph, pw), dtype=np.float64)
            if face_detected:
                strong_signal = np.maximum(strong_signal, maps["face_detection"])
            if skin_detected:
                strong_signal = np.maximum(strong_signal, maps["skin"])
            # Blend: 60% weighted average, 40% strong signal
            combined = 0.6 * combined + 0.4 * strong_signal

        # Apply Gaussian smoothing
        combined = gaussian_filter(combined, sigma=4)
        combined = self._normalize(combined)

        # Find peak
        peak_idx = np.unravel_index(np.argmax(combined), combined.shape)
        peak_y_proc, peak_x_proc = peak_idx

        # Scale back to original coordinates
        inv_scale = 1.0 / scale if scale < 1.0 else 1.0
        peak_x = int(peak_x_proc * inv_scale)
        peak_y = int(peak_y_proc * inv_scale)

        peak_x = max(0, min(w - 1, peak_x))
        peak_y = max(0, min(h - 1, peak_y))

        heatmap_full = cv2.resize(combined, (w, h))

        # Determine primary method
        if face_detected:
            method = "face_haar"
        elif skin_detected:
            method = "skin_color"
        else:
            method = "saliency"

        return {
            "x_pct": (peak_x / w) * 100,
            "y_pct": (peak_y / h) * 100,
            "x": peak_x,
            "y": peak_y,
            "heatmap": heatmap_full,
            "individual_maps": {k: cv2.resize(v, (w, h)) for k, v in maps.items()},
            "width": w,
            "height": h,
            "face_detected": face_detected,
            "skin_detected": skin_detected,
            "method": method,
        }

    def _compute_adaptive_weights(self, face_detected, skin_detected, skin_ratio, h, w):
        """Compute weights based on image characteristics."""
        weights = {
            "spectral_residual": 1.0,
            "frequency_tuned": 1.0,
            "edge_density": 0.5,
            "color_contrast": 0.8,
            "brightness": 0.3,
            "local_contrast": 1.2,
            "symmetry": 0.2,
            "skin": 0.0,
            "face_detection": 0.0,
            "center_bias": 0.8,
            "region_distinct": 0.8,
        }

        # Boost face/skin when detected
        if face_detected:
            weights["face_detection"] = 5.0
            weights["center_bias"] = 0.3  # reduce center bias, trust face detection
            weights["spectral_residual"] = 0.5
            weights["frequency_tuned"] = 0.5

        if skin_detected and skin_ratio < 0.3:  # skin present but not dominant
            weights["skin"] = 3.0
            if not face_detected:
                weights["center_bias"] = 0.5
                weights["spectral_residual"] = 0.7

        # If no face and no skin, rely more on saliency + center
        if not face_detected and not skin_detected:
            weights["center_bias"] = 1.0
            weights["local_contrast"] = 1.5
            weights["region_distinct"] = 1.0

        return weights

    # ─── Saliency methods ───

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
            # Upsample back
            if downsample > 1:
                sr = cv2.resize(sr, (gray.shape[1], gray.shape[0]))
            results.append(sr)

        # Combine scales
        combined = np.mean(results, axis=0)
        combined = gaussian_filter(combined, sigma=5)
        return combined

    def _spectral_residual_single(self, gray):
        """Single-scale spectral residual."""
        h, w = gray.shape
        if h < 8 or w < 8:
            return np.zeros_like(gray)

        fft = np.fft.fft2(gray)
        fft_shifted = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shifted)
        phase = np.angle(fft_shifted)

        log_magnitude = np.log(magnitude + 1e-8)

        # Use a larger averaging filter for better smoothing
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
        """Frequency-Tuned saliency (Achanta et al., 2009)."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        L, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]

        L_mean = np.mean(L)
        a_mean = np.mean(a)
        b_mean = np.mean(b)

        # Use larger sigma for better frequency tuning on art
        L_blur = gaussian_filter(L, sigma=5)
        a_blur = gaussian_filter(a, sigma=5)
        b_blur = gaussian_filter(b, sigma=5)

        saliency = (L_blur - L_mean) ** 2 + (a_blur - a_mean) ** 2 + (b_blur - b_mean) ** 2
        return saliency

    def _edge_density_map(self, image):
        """Edge density with border suppression."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # Suppress border edges (common in framed paintings)
        h, w = edges.shape
        border = max(5, min(h, w) // 20)
        edges[:border, :] = 0
        edges[-border:, :] = 0
        edges[:, :border] = 0
        edges[:, -border:] = 0

        edge_density = gaussian_filter(edges.astype(np.float32), sigma=12)
        return edge_density

    def _local_color_contrast(self, image):
        """Local color contrast using sliding window."""
        img_float = image.astype(np.float32)
        h, w = img_float.shape[:2]

        # Compute local mean with a large window
        local_mean = np.zeros_like(img_float)
        for c in range(3):
            local_mean[:, :, c] = gaussian_filter(img_float[:, :, c], sigma=20)

        # Contrast = distance from local mean
        diff = img_float - local_mean
        contrast = np.sqrt(np.sum(diff ** 2, axis=2))
        contrast = gaussian_filter(contrast, sigma=5)
        return contrast

    def _brightness_map(self, image):
        """Luminance hotspot map."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if gray.max() > 0:
            gray_norm = gray / gray.max()
        else:
            gray_norm = gray
        brightness = gaussian_filter(gray_norm, sigma=12)
        return brightness

    def _local_contrast_map(self, image):
        """Local contrast via unsharp masking."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        blurred = gaussian_filter(gray, sigma=8)
        high_freq = gray - blurred
        # Take absolute value and smooth
        contrast = np.abs(high_freq)
        contrast = gaussian_filter(contrast, sigma=6)
        return contrast

    def _symmetry_map(self, image):
        """Symmetry contribution."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        h, w = gray.shape

        # Vertical axis symmetry (left-right)
        h_flip = np.fliplr(gray)
        h_diff = np.abs(gray - h_flip)
        h_sym = 1.0 / (1.0 + h_diff / 50.0)  # scale the difference
        h_sym = gaussian_filter(h_sym, sigma=8)

        # Horizontal axis symmetry (top-bottom) — less common in art, lower weight
        v_flip = np.flipud(gray)
        v_diff = np.abs(gray - v_flip)
        v_sym = 1.0 / (1.0 + v_diff / 50.0)
        v_sym = gaussian_filter(v_sym, sigma=8)

        combined = h_sym * 0.7 + v_sym * 0.3
        return combined

    def _skin_color_map(self, image):
        """HSV-based skin color detection for portrait identification."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, w = image.shape[:2]

        # Skin color ranges in HSV (broader for paintings)
        lower1 = np.array([0, 20, 40])
        upper1 = np.array([25, 200, 255])
        lower2 = np.array([160, 20, 40])
        upper2 = np.array([180, 200, 255])

        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        skin_mask = mask1 + mask2

        # Clean up with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        skin_mask = cv2.GaussianBlur(skin_mask, (21, 21), 0)

        skin_ratio = np.sum(skin_mask > 0) / (h * w)
        detected = skin_ratio > 0.005 and skin_ratio < 0.5  # reasonable skin area

        skin_map = skin_mask.astype(np.float32) / 255.0
        skin_map = gaussian_filter(skin_map, sigma=10)

        return {"map": skin_map, "detected": detected, "ratio": skin_ratio}

    def _face_detection_map(self, image):
        """Face detection using Haar cascades."""
        h, w = image.shape[:2]
        face_map = np.zeros((h, w), dtype=np.float32)
        detected = False

        if self._face_cascade is not None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            for scale in [1.05, 1.1, 1.15, 1.2, 1.3]:
                faces = self._face_cascade.detectMultiScale(
                    gray, scaleFactor=scale, minNeighbors=5, minSize=(25, 25)
                )
                if len(faces) > 0:
                    detected = True
                    break

            if detected:
                for (fx, fy, fw, fh) in faces:
                    cx = fx + fw / 2.0
                    cy = fy + fh / 2.0
                    sigma = max(fw, fh) / 1.5

                    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
                    blob = np.exp(-((x - cx) ** 2 / (2 * sigma ** 2) + (y - cy) ** 2 / (2 * sigma ** 2)))
                    face_map += blob

        return {"map": face_map, "detected": detected}

    def _center_bias_map(self, w, h):
        """Center bias with upper-third preference for portraits."""
        y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

        # Center Gaussian
        cx, cy = w / 2.0, h * 0.45  # slightly upper-center
        sigma_x = w / 2.5
        sigma_y = h / 2.5
        center_gauss = np.exp(-((x - cx) ** 2 / (2 * sigma_x ** 2) + (y - cy) ** 2 / (2 * sigma_y ** 2)))

        # Rule of thirds
        third_x = [w / 3.0, 2 * w / 3.0]
        third_y = [h / 3.0, 2 * h / 3.0]
        thirds_map = np.zeros((h, w))
        for tx in third_x:
            for ty in third_y:
                sigma_t = min(w, h) / 5.0
                thirds_map += 0.2 * np.exp(
                    -((x - tx) ** 2 / (2 * sigma_t ** 2) + (y - ty) ** 2 / (2 * sigma_t ** 2))
                )

        return center_gauss + thirds_map

    def _region_distinctiveness(self, image):
        """Region distinctiveness using simple segmentation."""
        h, w = image.shape[:2]

        # Downsample for speed
        small_w, small_h = max(50, w // 2), max(50, h // 2)
        small = cv2.resize(image, (small_w, small_h))

        # K-means segmentation (simple color quantization)
        data = small.reshape((-1, 3)).astype(np.float32)
        k = 5
        _, labels, centers = cv2.kmeans(
            data, k, None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0),
            3, cv2.KMEANS_PP_CENTERS
        )

        # Find the most distinctive cluster (most different from mean color)
        mean_color = np.mean(data, axis=0)
        cluster_distances = []
        for i in range(k):
            cluster_pixels = data[labels.flatten() == i]
            if len(cluster_pixels) > 0:
                cluster_mean = np.mean(cluster_pixels, axis=0)
                dist = np.linalg.norm(cluster_mean - mean_color)
                # Weight by cluster size (prefer medium-sized clusters)
                size_ratio = len(cluster_pixels) / len(data)
                size_weight = 1.0 / (1.0 + abs(size_ratio - 0.15) * 5)  # peak at ~15% of image
                cluster_distances.append((i, dist * size_weight))
            else:
                cluster_distances.append((i, 0))

        # Create distinctiveness map
        distinct_map = np.zeros((small_h, small_w), dtype=np.float32)
        for i, score in cluster_distances:
            mask = (labels.reshape((small_h, small_w)) == i).astype(np.float32)
            distinct_map += score * mask

        # Smooth and upsample
        distinct_map = gaussian_filter(distinct_map, sigma=5)
        distinct_map = cv2.resize(distinct_map, (w, h))

        return distinct_map

    def _border_suppression_mask(self, w, h):
        """Create a mask that suppresses values near image borders."""
        y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        border = min(w, h) // 10

        # Distance from nearest border
        dist_x = np.minimum(x, w - 1 - x)
        dist_y = np.minimum(y, h - 1 - y)
        dist = np.minimum(dist_x, dist_y)

        mask = np.clip(dist / border, 0, 1)
        return mask.astype(np.float32)

    # ─── Utilities ───

    @staticmethod
    def _normalize(arr):
        """Normalize array to [0, 1]."""
        min_val = np.min(arr)
        max_val = np.max(arr)
        if max_val - min_val > 1e-8:
            return (arr - min_val) / (max_val - min_val)
        return np.zeros_like(arr)

    def visualize(self, image_path, result, output_path=None):
        """Create a visualization with the focus point marked."""
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
    """Run focus detection on all images in a directory."""
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
                "filepath": fpath,
                "x_pct": result["x_pct"],
                "y_pct": result["y_pct"],
                "x": result["x"],
                "y": result["y"],
                "width": result["width"],
                "height": result["height"],
                "face_detected": result["face_detected"],
                "skin_detected": result["skin_detected"],
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

    image_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
    if len(sys.argv) > 1:
        image_dir = sys.argv[1]

    output = os.path.join(os.path.dirname(os.path.abspath(__file__)), "detection_results_v2.json")
    print(f"Running focus detection v2 on images in {image_dir}...")
    results = batch_detect(image_dir, output)
    print(f"\nResults saved to {output}")
    print(f"Processed {len(results)} images")
