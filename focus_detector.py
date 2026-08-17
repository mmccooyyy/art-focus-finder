"""
Center-of-Focus Detector — Classical Computer Vision (no AI/ML services).

Combines multiple classical CV techniques to identify the primary focal point
of an image:

1. Spectral Residual Saliency — Fourier-based saliency detection
2. Frequency-Tuned Saliency — Lab color space, luminance/contrast
3. Edge Density — local edge concentration (Canny + Gaussian blur)
4. Color Contrast — local color distinctiveness from global mean
5. Center Bias — rule-of-thirds + center preference weighting
6. Face Detection — Haar cascade (local, pre-trained, not an AI service)
7. Brightness Hotspots — luminance peaks
8. Symmetry — horizontal/vertical symmetry contribution

All maps are normalized to [0,1], weighted, and combined into a final
saliency heatmap. The peak of the heatmap is the predicted focus point.

Output: (x_pct, y_pct) — focus point as percentage of image dimensions.
"""

import cv2
import numpy as np
import os
import json
from scipy import ndimage
from scipy.ndimage import gaussian_filter, uniform_filter


class FocusDetector:
    """Self-contained center-of-focus detector using classical CV."""

    # Weights for combining saliency maps
    DEFAULT_WEIGHTS = {
        "spectral_residual": 1.0,
        "frequency_tuned": 1.0,
        "edge_density": 0.8,
        "color_contrast": 0.8,
        "center_bias": 0.5,
        "face_detection": 3.0,  # high weight — faces are strong focal points
        "brightness": 0.5,
        "symmetry": 0.3,
    }

    def __init__(self, weights=None, face_cascade_path=None):
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        self._face_cascade = None
        self._init_face_cascade(face_cascade_path)

    def _init_face_cascade(self, custom_path=None):
        """Initialize Haar cascade for face detection (local, pre-trained)."""
        if custom_path:
            self._face_cascade = cv2.CascadeClassifier(custom_path)
            return

        # Try common OpenCV data paths
        possible_paths = [
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml",
            cv2.data.haarcascades + "haarcascade_frontalface_alt.xml",
            cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml",
            "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
            "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
        ]
        for path in possible_paths:
            try:
                cascade = cv2.CascadeClassifier(path)
                if not cascade.empty():
                    self._face_cascade = cascade
                    return
            except Exception:
                continue
        # No cascade found — face detection will be skipped
        self._face_cascade = None

    def detect(self, image_path_or_array):
        """
        Detect the center of focus in an image.

        Args:
            image_path_or_array: path to image file or numpy array (BGR)

        Returns:
            dict with keys:
                - x_pct, y_pct: focus point as percentage of image dimensions (0-100)
                - x, y: focus point in pixels
                - heatmap: combined saliency map (normalized 0-1)
                - individual_maps: dict of individual saliency maps
                - width, height: image dimensions
                - face_detected: bool
        """
        # Load image
        if isinstance(image_path_or_array, str):
            image = cv2.imread(image_path_or_array)
            if image is None:
                raise ValueError(f"Could not load image: {image_path_or_array}")
        else:
            image = image_path_or_array.copy()

        h, w = image.shape[:2]
        # Work at a standardized size for consistent processing
        max_dim = 400
        scale = max_dim / max(w, h)
        if scale < 1.0:
            proc_w, proc_h = int(w * scale), int(h * scale)
            proc_image = cv2.resize(image, (proc_w, proc_h))
        else:
            proc_image = image.copy()
            proc_w, proc_h = w, h

        # Compute individual saliency maps
        maps = {}
        maps["spectral_residual"] = self._spectral_residual_saliency(proc_image)
        maps["frequency_tuned"] = self._frequency_tuned_saliency(proc_image)
        maps["edge_density"] = self._edge_density_map(proc_image)
        maps["color_contrast"] = self._color_contrast_map(proc_image)
        maps["center_bias"] = self._center_bias_map(proc_w, proc_h)
        maps["brightness"] = self._brightness_map(proc_image)
        maps["symmetry"] = self._symmetry_map(proc_image)

        face_result = self._face_detection_map(proc_image)
        maps["face_detection"] = face_result["map"]
        face_detected = face_result["detected"]

        # Normalize all maps to [0, 1]
        for key in maps:
            maps[key] = self._normalize(maps[key])

        # Combine maps with weights
        combined = np.zeros_like(maps["spectral_residual"])
        total_weight = 0.0
        for key, weight in self.weights.items():
            if key in maps and weight > 0:
                combined += weight * maps[key]
                total_weight += weight

        if total_weight > 0:
            combined /= total_weight

        # Apply Gaussian smoothing to the combined map
        combined = gaussian_filter(combined, sigma=3)
        combined = self._normalize(combined)

        # Find the peak
        peak_idx = np.unravel_index(np.argmax(combined), combined.shape)
        peak_y_proc, peak_x_proc = peak_idx

        # Scale back to original image coordinates
        inv_scale = 1.0 / scale if scale < 1.0 else 1.0
        peak_x = int(peak_x_proc * inv_scale)
        peak_y = int(peak_y_proc * inv_scale)

        # Clamp to image bounds
        peak_x = max(0, min(w - 1, peak_x))
        peak_y = max(0, min(h - 1, peak_y))

        # Scale heatmap back to original size for visualization
        heatmap_full = cv2.resize(combined, (w, h))

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
        }

    # ─── Individual saliency methods ───

    def _spectral_residual_saliency(self, image):
        """Spectral Residual saliency (Hou & Zhang, 2007)."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = gray.astype(np.float32)

        # FFT
        fft = np.fft.fft2(gray)
        fft_shifted = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shifted)
        phase = np.angle(fft_shifted)

        # Log spectrum
        log_magnitude = np.log(magnitude + 1e-8)

        # Average with a local filter
        avg_filter = np.ones((3, 3)) / 9.0
        avg_log = ndimage.convolve(log_magnitude, avg_filter, mode="reflect")

        # Spectral residual
        spectral_residual = log_magnitude - avg_log

        # Reconstruct
        new_magnitude = np.exp(spectral_residual)
        new_fft = new_magnitude * np.exp(1j * phase)
        new_fft_shifted = np.fft.ifftshift(new_fft)
        saliency = np.abs(np.fft.ifft2(new_fft_shifted))

        # Square and smooth
        saliency = saliency ** 2
        saliency = gaussian_filter(saliency, sigma=8)

        return saliency

    def _frequency_tuned_saliency(self, image):
        """Frequency-Tuned saliency (Achanta et al., 2009)."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lab = lab.astype(np.float32)

        L = lab[:, :, 0]
        a = lab[:, :, 1]
        b = lab[:, :, 2]

        # Global mean
        L_mean = np.mean(L)
        a_mean = np.mean(a)
        b_mean = np.mean(b)

        # Gaussian blur for frequency tuning
        L_blur = gaussian_filter(L, sigma=3)
        a_blur = gaussian_filter(a, sigma=3)
        b_blur = gaussian_filter(b, sigma=3)

        # Saliency = squared distance from mean
        saliency = (L_blur - L_mean) ** 2 + (a_blur - a_mean) ** 2 + (b_blur - b_mean) ** 2

        return saliency

    def _edge_density_map(self, image):
        """Local edge density using Canny + Gaussian accumulation."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Canny edges
        edges = cv2.Canny(gray, 50, 150)

        # Accumulate edges with a larger Gaussian
        edge_density = gaussian_filter(edges.astype(np.float32), sigma=15)

        return edge_density

    def _color_contrast_map(self, image):
        """Local color distinctiveness from global mean color."""
        # Convert to float
        img_float = image.astype(np.float32)

        # Global mean color
        mean_color = np.mean(img_float, axis=(0, 1))

        # Distance from mean for each pixel
        diff = img_float - mean_color
        contrast = np.sqrt(np.sum(diff ** 2, axis=2))

        # Smooth
        contrast = gaussian_filter(contrast, sigma=10)

        return contrast

    def _center_bias_map(self, w, h):
        """2D Gaussian centered on image center with rule-of-thirds modulation."""
        y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

        # Center Gaussian
        cx, cy = w / 2.0, h / 2.0
        sigma_x = w / 3.0
        sigma_y = h / 3.0
        center_gauss = np.exp(-((x - cx) ** 2 / (2 * sigma_x ** 2) + (y - cy) ** 2 / (2 * sigma_y ** 2)))

        # Rule of thirds points (slightly elevated)
        third_x = [w / 3.0, 2 * w / 3.0]
        third_y = [h / 3.0, 2 * h / 3.0]
        thirds_map = np.zeros((h, w))
        for tx in third_x:
            for ty in third_y:
                sigma_t = min(w, h) / 6.0
                thirds_map += 0.3 * np.exp(
                    -((x - tx) ** 2 / (2 * sigma_t ** 2) + (y - ty) ** 2 / (2 * sigma_t ** 2))
                )

        combined = center_gauss + thirds_map
        return combined

    def _brightness_map(self, image):
        """Luminance hotspot map."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Normalize to 0-1
        if gray.max() > 0:
            gray_norm = gray / gray.max()
        else:
            gray_norm = gray

        # Smooth
        brightness = gaussian_filter(gray_norm, sigma=10)

        return brightness

    def _symmetry_map(self, image):
        """Horizontal and vertical symmetry contribution."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        h, w = gray.shape

        # Horizontal flip difference (vertical symmetry)
        h_flip = np.fliplr(gray)
        h_diff = np.abs(gray - h_flip)
        h_sym = 1.0 / (1.0 + h_diff)
        h_sym = gaussian_filter(h_sym, sigma=10)

        # Vertical flip difference (horizontal symmetry)
        v_flip = np.flipud(gray)
        v_diff = np.abs(gray - v_flip)
        v_sym = 1.0 / (1.0 + v_diff)
        v_sym = gaussian_filter(v_sym, sigma=10)

        # Combine — symmetry is strongest near the axis of symmetry
        combined = h_sym * 0.5 + v_sym * 0.5

        return combined

    def _face_detection_map(self, image):
        """Face detection using Haar cascades (local, pre-trained)."""
        h, w = image.shape[:2]
        face_map = np.zeros((h, w), dtype=np.float32)
        detected = False

        if self._face_cascade is not None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Detect faces at multiple scales
            for scale in [1.1, 1.2, 1.3]:
                faces = self._face_cascade.detectMultiScale(
                    gray, scaleFactor=scale, minNeighbors=3, minSize=(20, 20)
                )
                if len(faces) > 0:
                    detected = True
                    break

            if detected:
                for (fx, fy, fw, fh) in faces:
                    # Create a Gaussian blob at the face center
                    cx = fx + fw / 2.0
                    cy = fy + fh / 2.0
                    sigma = max(fw, fh) / 2.0

                    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
                    blob = np.exp(-((x - cx) ** 2 / (2 * sigma ** 2) + (y - cy) ** 2 / (2 * sigma ** 2)))
                    face_map += blob

        return {"map": face_map, "detected": detected}

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
        """Create a visualization with the focus point marked on the image."""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        h, w = image.shape[:2]
        x, y = result["x"], result["y"]

        # Draw heatmap overlay
        heatmap = result["heatmap"]
        heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(image, 0.6, heatmap_color, 0.4, 0)

        # Draw crosshair at focus point
        cv2.drawMarker(overlay, (x, y), (0, 255, 0), cv2.MARKER_CROSS, 30, 2)
        cv2.circle(overlay, (x, y), 10, (0, 255, 0), 2)

        # Add text
        label = f"({result['x_pct']:.1f}%, {result['y_pct']:.1f}%)"
        if result["face_detected"]:
            label += " [face]"
        cv2.putText(overlay, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if output_path:
            cv2.imwrite(output_path, overlay)

        return overlay


def batch_detect(image_dir, output_path=None):
    """
    Run focus detection on all images in a directory.

    Returns list of dicts with detection results.
    """
    import os
    import json

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
            })
            print(f"  {fname}: ({result['x_pct']:.1f}%, {result['y_pct']:.1f}%)"
                  f"  face={result['face_detected']}")
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

    output = os.path.join(os.path.dirname(os.path.abspath(__file__)), "detection_results.json")
    print(f"Running focus detection on images in {image_dir}...")
    results = batch_detect(image_dir, output)
    print(f"\nResults saved to {output}")
    print(f"Processed {len(results)} images")
