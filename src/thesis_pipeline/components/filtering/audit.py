from pathlib import Path
import json
import cv2
import pandas as pd
from ultralytics import YOLO
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class YOLOAuditor:
    def __init__(self, model_path: str, data_dir: str):
        self.model = YOLO(model_path)
        self.data_dir = Path(data_dir)
        self.results = []

    def audit(self, ground_truth_file: str):
        """
        Runs YOLO on the images listed in ground_truth_file and compares results.
        Supports both classification labels ('pottery'/'trash') and bounding boxes.
        """
        gt_path = Path(ground_truth_file)
        if not gt_path.exists():
            raise FileNotFoundError(f"Ground truth file not found: {gt_path}")

        with open(gt_path, 'r') as f:
            ground_truth = json.load(f)

        print(f"Starting audit on {len(ground_truth)} images...")

        for filename, gt_data in ground_truth.items():
            img_path = self.data_dir / filename
            if not img_path.exists():
                print(f"Warning: Image {filename} not found in {self.data_dir}")
                continue

            # Ground Truth Processing
            is_pottery_gt = False
            gt_bbox = None
            
            if isinstance(gt_data, str):
                # Format: "filename": "pottery" or "trash"
                is_pottery_gt = (gt_data.lower() == "pottery")
            elif isinstance(gt_data, dict):
                # Format: "filename": {"label": "pottery", "bbox": [...]}
                is_pottery_gt = (gt_data.get("label", "").lower() == "pottery")
                gt_bbox = gt_data.get("bbox")
            elif isinstance(gt_data, list):
                 # Format: "filename": [x, y, w, h] (Implicitly pottery if box exists)
                 is_pottery_gt = True
                 gt_bbox = gt_data

            # YOLO Inference
            # We use a low confidence threshold to see everything, then filter
            results = self.model.predict(img_path, conf=0.25, verbose=False)
            
            # Prediction Processing
            # Assume class 0 is the target object (e.g. 'vase'/'pottery') or check all classes if generic model
            # For this thesis, we check if *any* object of relevant class is detected.
            # Assuming standard COCO pretrained: classes 39 (bottle), 40 (wine glass), 41 (cup), 42 (fork), 43 (knife), 44 (spoon), 45 (bowl)
            # OR if trained custom model, check class 0.
            # Let's assume ANY detection is a positive for now, or refine if needed.
            detected_boxes = results[0].boxes
            is_pottery_pred = len(detected_boxes) > 0
            
            pred_bbox = None
            iou = 0.0
            
            if is_pottery_pred:
                # Take the detection with highest confidence
                best_box = sorted(detected_boxes, key=lambda x: x.conf[0], reverse=True)[0]
                # xywh format
                x, y, w, h = best_box.xywh[0].tolist()
                # Convert to top-left x, y for consistency if needed, but IoU usually handles format
                # Let's verify format. YOLO xywh is center-x, center-y, width, height.
                # OpenCV/ManualLabeler usually outputs top-left x, top-left y, width, height.
                # Conversion:
                pred_bbox = [x - w/2, y - h/2, w, h] 

            # IoU Calculation (if GT box exists and Pred box exists)
            if gt_bbox and pred_bbox:
                iou = self._calculate_iou(gt_bbox, pred_bbox)

            self.results.append({
                "filename": filename,
                "gt_class": "pottery" if is_pottery_gt else "trash",
                "pred_class": "pottery" if is_pottery_pred else "trash",
                "correct_class": is_pottery_gt == is_pottery_pred,
                "iou": iou if is_pottery_gt else None # Only relevant if it WAS pottery
            })

        return self._generate_report()

    def _calculate_iou(self, boxA, boxB):
        # boxA/B = [x, y, w, h] (top-left)
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]

        if float(boxAArea + boxBArea - interArea) == 0:
            return 0.0
            
        return interArea / float(boxAArea + boxBArea - interArea)

    def _generate_report(self):
        df = pd.DataFrame(self.results)
        
        # Classification Metrics
        y_true = df["gt_class"]
        y_pred = df["pred_class"]
        
        metrics = {
            "total_images": int(len(df)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, pos_label="pottery")),
            "recall": float(recall_score(y_true, y_pred, pos_label="pottery")),
            "f1_score": float(f1_score(y_true, y_pred, pos_label="pottery")),
            "pottery_count_gt": int((df["gt_class"] == "pottery").sum()),
            "trash_count_gt": int((df["gt_class"] == "trash").sum())
        }

        # IoU Metrics (if boxes exist)
        if "iou" in df.columns and df["iou"].notna().any():
            metrics["mean_iou"] = float(df["iou"].mean())
            metrics["median_iou"] = float(df["iou"].median())
        else:
            metrics["mean_iou"] = "N/A (No BBox GT)"

        return metrics, df