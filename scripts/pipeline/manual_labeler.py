import cv2
import json
import random
import os
from pathlib import Path
import glob

# Config
RAW_DATA_DIR = Path("data/01_raw/wikimedia_collection")
OUTPUT_DIR = Path("data/quality_audit/yolo_validation")
OUTPUT_FILE = OUTPUT_DIR / "ground_truth_50_bboxes.json"
SAMPLE_SIZE = 50

def load_existing_annotations():
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, "r") as f:
            return json.load(f)
    return []

def save_annotations(annotations):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(annotations, f, indent=4)
    print(f"Saved {len(annotations)} annotations to {OUTPUT_FILE}")

def main():
    print("=== MANUAL LABELING TOOL FOR YOLO VALIDATION ===")
    print(f"Source: {RAW_DATA_DIR}")
    print(f"Target: {OUTPUT_FILE}")
    
    # 1. Get List of Images
    all_images = list(RAW_DATA_DIR.glob("*.jpg")) + list(RAW_DATA_DIR.glob("*.png"))
    if not all_images:
        print("ERROR: No images found in source directory.")
        return

    # 2. Load Progress
    existing = load_existing_annotations()
    annotated_filenames = {item['filename'] for item in existing}
    
    print(f"Found {len(existing)} existing annotations.")
    
    # 3. Select Samples
    # If we haven't started, pick 50 randoms. If we have, we need to finish the set.
    # Actually, we should define the *set* of 50 first, then iterate.
    # But for simplicity, let's just pick 50 randoms *total* that we want to label.
    # If we already have some, we keep them and add more to reach 50.
    
    needed = SAMPLE_SIZE - len(existing)
    if needed <= 0:
        print("You have already labeled 50 images! Task complete.")
        return

    # Filter out already done
    available = [img for img in all_images if img.name not in annotated_filenames]
    
    if len(available) < needed:
        print(f"WARNING: Only {len(available)} unlabeled images available. Labeling all of them.")
        to_label = available
    else:
        to_label = random.sample(available, needed)

    print(f"Starting labeling session for {len(to_label)} images...")
    print("INSTRUCTIONS:")
    print("  - Draw a box around the MAIN pottery object.")
    print("  - Press SPACE or ENTER to confirm the box.")
    print("  - Press 'c' to cancel selection and redraw.")
    print("  - If no pottery is visible, press ESC (will record as empty/background).")
    print("  - To Quit, close the window or Ctrl+C.")

    new_annotations = []
    
    for i, img_path in enumerate(to_label):
        print(f"[{i+1}/{len(to_label)}] Processing {img_path.name}...")
        
        img = cv2.imread(str(img_path))
        if img is None:
            print("  Could not read image. Skipping.")
            continue

        # Resize for screen if too huge
        height, width = img.shape[:2]
        scale = 1.0
        max_dim = 1000
        if height > max_dim or width > max_dim:
            scale = max_dim / max(height, width)
            img_display = cv2.resize(img, (0, 0), fx=scale, fy=scale)
        else:
            img_display = img.copy()

        # ROI Selector
        try:
            # showCrosshair=True, fromCenter=False
            bbox = cv2.selectROI("Labeling Tool", img_display, showCrosshair=True, fromCenter=False)
            cv2.destroyWindow("Labeling Tool")
        except KeyboardInterrupt:
            print("\nExiting...")
            break
            
        # bbox is (x, y, w, h)
        # If user pressed ESC, bbox is usually (0,0,0,0)
        has_object = bbox[2] > 0 and bbox[3] > 0
        
        real_bbox = []
        if has_object:
            # Scale back to original size
            x = int(bbox[0] / scale)
            y = int(bbox[1] / scale)
            w = int(bbox[2] / scale)
            h = int(bbox[3] / scale)
            real_bbox = [x, y, w, h]
            print(f"  Recorded Box: {real_bbox}")
        else:
            print("  Recorded as No Object.")

        record = {
            "filename": img_path.name,
            "bbox": real_bbox, # COCO format: [x,y,w,h]
            "label": "pottery" if has_object else "background",
            "image_height": height,
            "image_width": width
        }
        
        existing.append(record)
        
        # Save every 5 images to be safe
        if (i + 1) % 5 == 0:
            save_annotations(existing)

    # Final save
    save_annotations(existing)
    print("Done!")

if __name__ == "__main__":
    main()
