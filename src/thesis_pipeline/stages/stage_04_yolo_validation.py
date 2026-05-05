from thesis_pipeline.components.filtering.audit import YOLOAuditor
from thesis_pipeline.config_manager import ConfigManager
from pathlib import Path
import json
import logging
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

class YOLOValidationStage:
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.config = config_manager.config
        self.paths = config_manager.get_paths()
        self.logger = logging.getLogger(__name__)

    def _resolve_ground_truth_path(self) -> Path:
        """Returns the configured ground-truth path or a default."""
        gt_path = self.config.intelligent_filtering.get(
            "ground_truth",
            "data/labels/ground_truth_50.json",
        )
        return Path(gt_path)
        
    def run(self):
        # Paths
        data_dir = self.paths.data.raw
        gt_file = self._resolve_ground_truth_path()
        
        model_path = self.config.intelligent_filtering.get(
            "model_name",
            "data/models/weights/yolov8n.pt",
        )
        
        output_dir = self.config_manager.get_stage_artifact_path("S04", "validation")
        output_dir.mkdir(parents=True, exist_ok=True)

        # --- Early check: ground-truth file must exist ---
        if not gt_file.exists():
            msg = (
                f"Ground-truth file not found: {gt_file}. "
                "Stage 03b cannot run without labelled data. "
                "Create ground_truth_50.json or update config.intelligent_filtering.ground_truth."
            )
            self.logger.warning(msg)
            (output_dir / "SKIPPED_REASON.txt").write_text(msg)
            return

        print("=== Stage 03b: YOLO Validation Audit ===")
        auditor = YOLOAuditor(model_path, data_dir)
        
        try:
            metrics, df = auditor.audit(gt_file)
            
            # Save Results
            df.to_csv(output_dir / "audit_details.csv", index=False)
            with open(output_dir / "metrics.json", "w") as f:
                json.dump(metrics, f, indent=4)

            # Detection metrics CSV
            det_rows = [{
                "metric": k,
                "value": v
            } for k, v in metrics.items()]
            det_df = pd.DataFrame(det_rows)
            det_df.to_csv(output_dir / "detection_metrics.csv", index=False)

            # V6: confusion_matrix.png
            if "gt_class" in df.columns and "pred_class" in df.columns:
                labels = ["pottery", "trash"]
                cm = confusion_matrix(df["gt_class"], df["pred_class"], labels=labels)
                plt.figure(figsize=(4, 4))
                plt.imshow(cm, cmap="Blues")
                plt.xticks([0, 1], labels, rotation=45)
                plt.yticks([0, 1], labels)
                plt.title("YOLO Validation Confusion Matrix")
                for i in range(2):
                    for j in range(2):
                        plt.text(j, i, cm[i, j], ha="center", va="center")
                plt.tight_layout()
                plt.savefig(output_dir / "confusion_matrix.png")
                plt.close()

            print("\nAudit Complete.")
            print(f"Accuracy: {metrics['accuracy']:.2f}")
            print(f"Precision: {metrics['precision']:.2f}")
            print(f"Recall: {metrics['recall']:.2f}")
            print(f"Mean IoU: {metrics['mean_iou']}")
            print(f"Report saved to {output_dir}")
            
        except Exception as e:
            print(f"Audit failed: {e}")

if __name__ == "__main__":
    cm = ConfigManager()
    stage = YOLOValidationStage(cm)
    stage.run()
