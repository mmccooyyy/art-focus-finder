"""
Evaluation script — compares algorithm predictions with ground truth from
multimodal AI vision analysis.

Ground truth is established by using the analyze-image tool (the agent's own
vision capabilities) to identify the focal point of each image. The distance
between the algorithm's prediction and the ground truth is computed as a
percentage of the image diagonal. The goal is to get within 10% consistently.
"""

import json
import os
import math
import sys

# Ground truth from vision analysis (image_id -> (x_pct, y_pct))
GROUND_TRUTH = {
    0: (45, 35),
    1: (40, 32),
    2: (42, 58),
    3: (30, 65),
    4: (27, 39),
    5: (50.1, 31.8),
    10: (35, 70),
    15: (35, 55),
    20: (50, 36),
    25: (65, 65),
}


def compute_error(pred_x, pred_y, gt_x, gt_y):
    """
    Compute error as percentage of image diagonal.
    Distance is in percentage units (0-100 for each axis).
    Diagonal = sqrt(100^2 + 100^2) = 141.42
    """
    dx = pred_x - gt_x
    dy = pred_y - gt_y
    dist = math.sqrt(dx ** 2 + dy ** 2)
    diagonal = math.sqrt(100 ** 2 + 100 ** 2)
    return (dist / diagonal) * 100


def evaluate(results_path):
    """Evaluate detection results against ground truth."""
    with open(results_path) as f:
        results = json.load(f)

    print("=" * 80)
    print("FOCUS DETECTION EVALUATION")
    print("=" * 80)
    print(f"{'Image':<12} {'Algorithm':<20} {'Ground Truth':<20} {'Error %':<10} {'Within 10%':<12} {'Method'}")
    print("-" * 80)

    errors = []
    within_10 = 0
    total = 0

    for result in results:
        if "error" in result:
            continue

        # Extract image number from filename (rijks_XXX.jpg)
        fname = result["filename"]
        try:
            img_num = int(fname.split("_")[1].split(".")[0])
        except (IndexError, ValueError):
            continue

        if img_num not in GROUND_TRUTH:
            continue

        pred_x = result["x_pct"]
        pred_y = result["y_pct"]
        gt_x, gt_y = GROUND_TRUTH[img_num]

        error = compute_error(pred_x, pred_y, gt_x, gt_y)
        errors.append(error)
        total += 1
        is_within = error <= 10.0
        if is_within:
            within_10 += 1

        method = result.get("method", "?")
        pred_str = f"({pred_x:.1f}, {pred_y:.1f})"
        gt_str = f"({gt_x:.1f}, {gt_y:.1f})"
        within_str = "YES" if is_within else "NO"

        print(f"rijks_{img_num:03d}  {pred_str:<20} {gt_str:<20} {error:>6.1f}%    {within_str:<12} {method}")

    print("-" * 80)
    if errors:
        avg_error = sum(errors) / len(errors)
        max_error = max(errors)
        min_error = min(errors)
        median_error = sorted(errors)[len(errors) // 2]
        pct_within = (within_10 / total) * 100

        print(f"\nSummary:")
        print(f"  Images evaluated:  {total}")
        print(f"  Average error:     {avg_error:.1f}% of diagonal")
        print(f"  Median error:      {median_error:.1f}% of diagonal")
        print(f"  Min error:         {min_error:.1f}%")
        print(f"  Max error:         {max_error:.1f}%")
        print(f"  Within 10%:        {within_10}/{total} ({pct_within:.0f}%)")
        print(f"  Target:            10% consistently (100% within 10%)")

        if pct_within >= 80:
            print(f"\n  Status: GOOD — {pct_within:.0f}% within target")
        elif pct_within >= 50:
            print(f"\n  Status: MODERATE — {pct_within:.0f}% within target, needs improvement")
        else:
            print(f"\n  Status: POOR — only {pct_within:.0f}% within target, major work needed")

    print("=" * 80)


if __name__ == "__main__":
    results_path = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "detection_results_v2.json")

    # Fix numpy bool serialization issue by converting results first
    if not os.path.exists(results_path):
    # Try to re-run detection with fixed serialization
        print(f"Results file not found: {results_path}")
        print("Running detection first...")
        os.system(f"cd {os.path.dirname(os.path.abspath(__file__))} && python3 -c \""
            "import focus_detector_v2 as fd; "
            "import json, numpy as np; "
            "results = fd.batch_detect('images'); "
            "for r in results: "
            "  if 'face_detected' in r: r['face_detected'] = bool(r['face_detected']); "
            "  if 'skin_detected' in r: r['skin_detected'] = bool(r['skin_detected']); "
            "json.dump(results, open('detection_results_v2.json', 'w'), indent=2); "
            "print('Saved with fixed serialization')\"")
        results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "detection_results_v2.json")

    evaluate(results_path)
