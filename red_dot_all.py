#!/usr/bin/env python3
"""
Run v4 focus detector on all 100 images and stamp a large red dot
at the detected focal point. Saves to red_dot_output/.
"""
import os
import cv2
import json
import numpy as np
from focus_detector_v4 import FocusDetector

IMAGES_DIR = os.path.join(os.path.dirname(__file__), "images")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "red_dot_output")
RESULTS_FILE = os.path.join(os.path.dirname(__file__), "red_dot_results.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)

detector = FocusDetector()
results = []

image_files = sorted(f for f in os.listdir(IMAGES_DIR) if f.endswith(".jpg"))
print(f"Processing {len(image_files)} images...\n")

for i, fname in enumerate(image_files):
    fpath = os.path.join(IMAGES_DIR, fname)
    result = detector.detect(fpath)

    # Load original image
    img = cv2.imread(fpath)
    if img is None:
        print(f"  {fname}: FAILED to load")
        continue

    h, w = img.shape[:2]

    # Convert percentage to pixel coordinates
    px = int(result["x_pct"] / 100.0 * w)
    py = int(result["y_pct"] / 100.0 * h)

    # Draw a large red dot — radius is 3.5% of the smaller dimension, min 18px
    radius = max(18, int(min(w, h) * 0.035))

    # White outline ring for visibility on dark backgrounds
    cv2.circle(img, (px, py), radius + 3, (255, 255, 255), -1)
    # Red dot
    cv2.circle(img, (px, py), radius, (0, 0, 255), -1)
    # Thin white center crosshair for precise location
    cv2.line(img, (px - radius, py), (px + radius, py), (255, 255, 255), 1)
    cv2.line(img, (px, py - radius), (px, py + radius), (255, 255, 255), 1)

    out_path = os.path.join(OUTPUT_DIR, fname)
    cv2.imwrite(out_path, img)

    entry = {
        "image": fname,
        "x_pct": round(result["x_pct"], 1),
        "y_pct": round(result["y_pct"], 1),
        "method": result["method"],
        "px": px,
        "py": py,
    }
    results.append(entry)

    if (i + 1) % 10 == 0 or i == 0:
        print(f"  [{i+1:3d}/{len(image_files)}] {fname}: ({result['x_pct']:.1f}%, {result['y_pct']:.1f}%) method={result['method']}")

# Save results JSON
with open(RESULTS_FILE, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nDone! {len(results)} images processed.")
print(f"Red-dot images saved to: {OUTPUT_DIR}/")
print(f"Results JSON saved to: {RESULTS_FILE}")
