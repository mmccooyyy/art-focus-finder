"""
Fetch 100 random images from the Rijksmuseum collection API.

API chain (no API key needed for the new Search API):
1. Search API → LOD identifiers (HumanMadeObject IDs)
2. Resolve HumanMadeObject → VisualItem ID (shows[0].id)
3. Resolve VisualItem → DigitalObject ID (digitally_shown_by[0].id)
4. Resolve DigitalObject → IIIF image URL (access_point[0].id)

Images are downloaded to /home/user/work/focus_finder/images/
Metadata is saved to /home/user/work/focus_finder/image_metadata.json
"""

import json
import os
import random
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_SEARCH = "https://data.rijksmuseum.nl/search/collection"
RESOLVE_BASE = "https://id.rijksmuseum.nl"
IMAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
META_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "image_metadata.json")

# Search across multiple types to get variety
SEARCH_TYPES = ["painting", "photograph", "print", "drawing", "sculpture", "decorative arts"]


def fetch_json(url, timeout=30):
    """Fetch JSON from a URL with Accept header."""
    req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "FocusFinder/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def download_image(url, filepath, timeout=60):
    """Download an image to a file."""
    req = urllib.request.Request(url, headers={"User-Agent": "FocusFinder/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
        with open(filepath, "wb") as f:
            f.write(data)
    return len(data)


def collect_lod_ids(target_count=120):
    """Collect LOD identifiers from multiple search pages and types."""
    all_ids = []
    seen = set()

    for stype in SEARCH_TYPES:
        if len(all_ids) >= target_count * 2:
            break
        url = f"{BASE_SEARCH}?type={stype}&imageAvailable=true"
        try:
            data = fetch_json(url)
            items = data.get("orderedItems", [])
            for item in items:
                lid = item["id"]
                if lid not in seen:
                    seen.add(lid)
                    all_ids.append(lid)
            # Also fetch a second page for more variety
            next_url = data.get("next", {}).get("id")
            if next_url and len(all_ids) < target_count * 2:
                try:
                    data2 = fetch_json(next_url)
                    for item in data2.get("orderedItems", []):
                        lid = item["id"]
                        if lid not in seen:
                            seen.add(lid)
                            all_ids.append(lid)
                except Exception:
                    pass
        except Exception as e:
            print(f"  Search type '{stype}' failed: {e}")

    # Shuffle for randomness
    random.seed(42)
    random.shuffle(all_ids)
    return all_ids[:target_count]


def resolve_to_image_url(lod_id):
    """
    Resolve a LOD identifier through the chain to get the IIIF image URL.
    Returns (image_url, title) or (None, None) on failure.
    """
    try:
        # Step 1: Resolve HumanMadeObject → get VisualItem + title
        obj_data = fetch_json(lod_id)
        shows = obj_data.get("shows", [])
        if not shows:
            return None, None
        visual_item_id = shows[0]["id"]

        # Extract title if available
        title = None
        for ref in obj_data.get("subject_of", []):
            if isinstance(ref, dict) and "part" in ref:
                for part in ref["part"]:
                    if isinstance(part, dict) and "part" in part:
                        for subpart in part["part"]:
                            if isinstance(subpart, dict) and subpart.get("classified_as"):
                                is_title = any(
                                    c.get("id") == "http://vocab.getty.edu/aat/300404670"
                                    for c in subpart.get("classified_as", [])
                                )
                                if is_title and "content" in subpart:
                                    title = subpart["content"]
                                    break

        # Step 2: Resolve VisualItem → get DigitalObject
        vi_data = fetch_json(visual_item_id)
        digitally_shown = vi_data.get("digitally_shown_by", [])
        if not digitally_shown:
            return None, title
        digital_obj_id = digitally_shown[0]["id"]

        # Step 3: Resolve DigitalObject → get image URL
        do_data = fetch_json(digital_obj_id)
        access_points = do_data.get("access_point", [])
        if not access_points:
            return None, title
        image_url = access_points[0]["id"]

        return image_url, title
    except Exception as e:
        print(f"  Resolve failed for {lod_id}: {e}")
        return None, None


def main():
    os.makedirs(IMAGE_DIR, exist_ok=True)

    # Check if we already have enough images
    existing = [f for f in os.listdir(IMAGE_DIR) if f.endswith(".jpg")]
    if len(existing) >= 100:
        print(f"Already have {len(existing)} images. Skipping fetch.")
        return

    print("Collecting LOD identifiers from Rijksmuseum...")
    lod_ids = collect_lod_ids(target_count=150)  # extra for failures
    print(f"Collected {len(lod_ids)} LOD identifiers")

    print("Resolving to image URLs (parallel)...")
    results = {}
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_id = {executor.submit(resolve_to_image_url, lid): lid for lid in lod_ids}
        for future in as_completed(future_to_id):
            lid = future_to_id[future]
            try:
                img_url, title = future.result()
                if img_url:
                    results[lid] = {"url": img_url, "title": title}
                    print(f"  Resolved {lid}: {img_url[:60]}...")
            except Exception as e:
                print(f"  Error resolving {lid}: {e}")

    print(f"Resolved {len(results)} image URLs")

    # Download images
    print("Downloading images...")
    metadata = []
    idx = 0
    for lod_id, info in results.items():
        if idx >= 100:
            break
        # Use a smaller size for efficiency: /full/800,/0/default.jpg
        img_url = info["url"].replace("/full/max/", "/full/800,/")
        filepath = os.path.join(IMAGE_DIR, f"rijks_{idx:03d}.jpg")
        try:
            size = download_image(img_url, filepath)
            # Get image dimensions
            from PIL import Image
            with Image.open(filepath) as img:
                w, h = img.size
            metadata.append({
                "id": idx,
                "lod_id": lod_id,
                "url": info["url"],
                "title": info["title"],
                "filepath": filepath,
                "width": w,
                "height": h,
                "size_bytes": size
            })
            print(f"  [{idx}] Downloaded {filepath} ({w}x{h})")
            idx += 1
        except Exception as e:
            print(f"  Download failed for {lod_id}: {e}")
            # Try original URL as fallback
            try:
                size = download_image(info["url"], filepath)
                from PIL import Image
                with Image.open(filepath) as img:
                    w, h = img.size
                metadata.append({
                    "id": idx,
                    "lod_id": lod_id,
                    "url": info["url"],
                    "title": info["title"],
                    "filepath": filepath,
                    "width": w,
                    "height": h,
                    "size_bytes": size
                })
                print(f"  [{idx}] Downloaded (fallback) {filepath} ({w}x{h})")
                idx += 1
            except Exception as e2:
                print(f"  Fallback also failed: {e2}")

    # Save metadata
    with open(META_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDone! Downloaded {len(metadata)} images to {IMAGE_DIR}")
    print(f"Metadata saved to {META_PATH}")


if __name__ == "__main__":
    main()
