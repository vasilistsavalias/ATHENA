import cv2
import json
import argparse
from pathlib import Path
import shutil

def manual_labeler(input_dir, output_file, count=50):
    input_path = Path(input_dir)
    images = list(input_path.glob("*.jpg")) + list(input_path.glob("*.png"))
    
    if not images:
        print("No images found.")
        return

    # Select random 50
    import random
    random.shuffle(images)
    selection = images[:count]
    
    labels = {}
    
    print(f"Starting labeling session for {len(selection)} images.")
    print("Keys: [y] Yes (Pottery), [n] No (Trash), [q] Quit")
    
    cv2.namedWindow("Labeler", cv2.WINDOW_NORMAL)
    
    for img_path in selection:
        img = cv2.imread(str(img_path))
        if img is None: continue
        
        cv2.imshow("Labeler", img)
        key = cv2.waitKey(0)
        
        if key == ord('q'):
            break
        elif key == ord('y'):
            labels[img_path.name] = "pottery"
            print(f"{img_path.name}: POTTERY")
        elif key == ord('n'):
            labels[img_path.name] = "trash"
            print(f"{img_path.name}: TRASH")
        else:
            print(f"Skipping {img_path.name}")
            
    cv2.destroyAllWindows()
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(labels, f, indent=4)
    print(f"Saved {len(labels)} labels to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="data/labels/ground_truth_50.json")
    args = parser.parse_args()
    manual_labeler(args.input, args.output)
