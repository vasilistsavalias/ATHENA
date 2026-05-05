import csv
from pathlib import Path
import logging
import threading
from datetime import datetime

class AuditLogger:
    """
    Thread-safe logger for data auditing (rejections, biases).
    """
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.rejection_file = output_dir / "rejection_log.csv"
        self.lock = threading.Lock()
        self.header = [
            "filename",
            "reason",
            "status",
            "yolo_confidence",
            "focus_metric",
            "focus_score",
            "focus_threshold_used",
            "bbox_area_ratio",
            "crop_index",
            "blur_variance",
        ]
        self._init_csv()

    def _init_csv(self):
        """Initialize CSV with headers if it doesn't exist."""
        if not self.rejection_file.exists():
            with open(self.rejection_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(self.header)
            return

        # Upgrade legacy schema in-place by rotating old file.
        try:
            with open(self.rejection_file, "r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                existing_header = next(reader, [])
        except Exception:
            existing_header = []

        if existing_header != self.header:
            backup = self.rejection_file.with_name(
                f"rejection_log_legacy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
            try:
                self.rejection_file.replace(backup)
            except Exception:
                pass
            with open(self.rejection_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(self.header)

    @staticmethod
    def _fmt_float(value):
        if value is None:
            return "N/A"
        try:
            return f"{float(value):.4f}"
        except Exception:
            return "N/A"

    def log_decision(
        self,
        filename,
        yolo_conf,
        blur_var,
        reason,
        status,
        *,
        focus_metric="tenengrad",
        focus_score=None,
        focus_threshold_used=None,
        bbox_area_ratio=None,
        crop_index=None,
    ):
        """
        Log a filtering decision.
        """
        with self.lock:
            with open(self.rejection_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    filename,
                    reason,
                    status,
                    self._fmt_float(yolo_conf),
                    focus_metric if focus_metric else "N/A",
                    self._fmt_float(focus_score),
                    self._fmt_float(focus_threshold_used),
                    self._fmt_float(bbox_area_ratio),
                    str(crop_index) if crop_index is not None else "N/A",
                    self._fmt_float(blur_var),
                ])
