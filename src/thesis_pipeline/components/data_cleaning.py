import logging
import shutil
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import csv
import json
from collections import Counter

class DataCleaner:
    """
    Identifies and segregates low-quality or irrelevant images (e.g., sketches, book scans)
    from the raw dataset. Instead of deleting, it moves them to a 'filtered' directory.
    """
    def __init__(self, input_dir: Path, filtered_dir: Path, extensions: list = ['.jpg', '.png', '.jpeg']):
        self.input_dir = input_dir
        self.filtered_dir = filtered_dir
        self.extensions = extensions
        self.logger = logging.getLogger(__name__)

    def _get_image_files(self) -> list:
        """Find all image files with given extensions (recursive).

        Stage 02 (data acquisition) stores images under per-source subfolders:
        `<raw>/wikimedia`, `<raw>/met`, `<raw>/europeana`.
        Data cleaning should therefore scan recursively.
        """
        image_files = []
        for ext in self.extensions:
            image_files.extend(self.input_dir.rglob(f'*{ext}'))
        return image_files

    @staticmethod
    def evaluate_image_quality(
        img: Image.Image,
        min_width: int = 0,
        min_height: int = 0,
        color_check: bool = True,
        saturation_threshold: float = 25.0,
        whiteness_threshold: float = 0.60,
    ) -> dict:
        """
        Evaluate whether an image should pass first-pass cleaning.

        Soft-gate policy:
        - Reject grayscale mode.
        - Reject too-small dimensions.
        - Reject only when BOTH low saturation and high whiteness are true.
        """
        mode = img.mode
        width, height = img.size

        result = {
            "mode": mode,
            "width": width,
            "height": height,
            "mean_saturation": None,
            "white_ratio": None,
            "accepted": True,
            "reason": "Valid",
        }

        if min_width and width < int(min_width):
            result["accepted"] = False
            result["reason"] = f"Too Small (width<{min_width})"
            return result
        if min_height and height < int(min_height):
            result["accepted"] = False
            result["reason"] = f"Too Small (height<{min_height})"
            return result

        if mode == "L":
            result["accepted"] = False
            result["reason"] = "Grayscale Mode"
            return result

        if not color_check:
            return result

        img_hsv = img.convert("HSV")
        np_img = np.array(img_hsv)
        saturation = np_img[:, :, 1]
        brightness = np_img[:, :, 2]
        mean_saturation = float(np.mean(saturation))

        white_pixels = np.sum((brightness > 240) & (saturation < 30))
        total_pixels = brightness.size
        white_ratio = float(white_pixels / total_pixels) if total_pixels else 0.0

        result["mean_saturation"] = mean_saturation
        result["white_ratio"] = white_ratio

        # Soft gate: reject only if both conditions hold.
        if mean_saturation < float(saturation_threshold) and white_ratio > float(whiteness_threshold):
            result["accepted"] = False
            result["reason"] = (
                f"Low Saturation + High Whiteness "
                f"(sat={mean_saturation:.2f}, white={white_ratio:.2f})"
            )

        return result

    def filter_grayscale_images(
        self,
        report_dir: Path | None = None,
        sample_limit: int = 50,
        min_width: int = 0,
        min_height: int = 0,
        color_check: bool = True,
        saturation_threshold: float = 25.0,
        whiteness_threshold: float = 0.60,
        seed: int = 42,
    ):
        """
        Copies colored (valid) images to the filtered directory.
        Keeps the raw directory intact.
        """
        self.logger.info(f"Scanning {self.input_dir} for colored pottery images...")
        
        # Ensure filtered directory exists
        self.filtered_dir.mkdir(parents=True, exist_ok=True)

        # Optional reporting artifacts
        report_path = Path(report_dir) if report_dir else None
        log_csv_path = None
        samples_accepted_dir = None
        samples_rejected_dir = None
        if report_path:
            report_path.mkdir(parents=True, exist_ok=True)
            (report_path / "samples").mkdir(parents=True, exist_ok=True)
            samples_accepted_dir = report_path / "samples" / "accepted"
            samples_rejected_dir = report_path / "samples" / "rejected"
            if samples_accepted_dir.exists():
                shutil.rmtree(samples_accepted_dir, ignore_errors=True)
            if samples_rejected_dir.exists():
                shutil.rmtree(samples_rejected_dir, ignore_errors=True)
            samples_accepted_dir.mkdir(parents=True, exist_ok=True)
            samples_rejected_dir.mkdir(parents=True, exist_ok=True)
            log_csv_path = report_path / "cleaning_log.csv"
        
        image_files = self._get_image_files()
        copied_count = 0
        skipped_count = 0
        
        # Heuristics (tuned for Greek Pottery)
        SATURATION_THRESHOLD = float(saturation_threshold)  # 0-255 (Low saturation = Grayscale/Sketch)
        WHITENESS_THRESHOLD = float(whiteness_threshold)    # >60% pixels are white

        accepted_for_sampling: list[Path] = []
        rejected_for_sampling: list[Path] = []
        reason_counter: Counter[str] = Counter()

        csv_writer = None
        csv_fh = None
        if log_csv_path:
            csv_fh = open(log_csv_path, "w", newline="", encoding="utf-8")
            csv_writer = csv.writer(csv_fh)
            csv_writer.writerow(
                [
                    "filename",
                    "mode",
                    "width",
                    "height",
                    "mean_saturation",
                    "white_ratio",
                    "decision",
                    "reason",
                ]
            )
        
        for img_path in tqdm(image_files, desc="Filtering Images"):
            try:
                with Image.open(img_path) as img:
                    quality = self.evaluate_image_quality(
                        img=img,
                        min_width=min_width,
                        min_height=min_height,
                        color_check=color_check,
                        saturation_threshold=SATURATION_THRESHOLD,
                        whiteness_threshold=WHITENESS_THRESHOLD,
                    )

                is_valid_pottery = bool(quality["accepted"])
                reason = str(quality["reason"])
                mode = quality["mode"]
                width = quality["width"]
                height = quality["height"]
                mean_saturation = quality["mean_saturation"]
                white_ratio = quality["white_ratio"]
                
                if is_valid_pottery:
                    # Construct destination path
                    dest_path = self.filtered_dir / img_path.name
                    
                    # Handle duplicates
                    if dest_path.exists():
                        stem = dest_path.stem
                        suffix = dest_path.suffix
                        dest_path = self.filtered_dir / f"{stem}_dup{suffix}"
                    
                    # self.logger.info(f"Copying {img_path.name} -> {self.filtered_dir}")
                    shutil.copy2(str(img_path), str(dest_path))
                    copied_count += 1
                    reason_counter[reason] += 1
                    if report_path:
                        accepted_for_sampling.append(img_path)
                else:
                    # self.logger.debug(f"Skipping {img_path.name}: {reason}")
                    skipped_count += 1
                    reason_counter[reason] += 1
                    if report_path:
                        rejected_for_sampling.append(img_path)

                if csv_writer:
                    csv_writer.writerow(
                        [
                            img_path.name,
                            mode or "N/A",
                            width or "N/A",
                            height or "N/A",
                            f"{mean_saturation:.4f}" if mean_saturation is not None else "N/A",
                            f"{white_ratio:.4f}" if white_ratio is not None else "N/A",
                            "Accepted" if is_valid_pottery else "Rejected",
                            reason,
                        ]
                    )
                    
            except Exception as e:
                self.logger.warning(f"Could not process {img_path}: {e}")

        if csv_fh:
            csv_fh.close()

        # Deterministic sample export
        if report_path and sample_limit and sample_limit > 0:
            accepted_for_sampling = sorted(accepted_for_sampling, key=lambda p: p.name)
            rejected_for_sampling = sorted(rejected_for_sampling, key=lambda p: p.name)

            for p in accepted_for_sampling[: int(sample_limit)]:
                try:
                    shutil.copy2(str(p), str(samples_accepted_dir / p.name))
                except Exception:
                    continue
            for p in rejected_for_sampling[: int(sample_limit)]:
                try:
                    shutil.copy2(str(p), str(samples_rejected_dir / p.name))
                except Exception:
                    continue

            stats = {
                "stage": "02b_data_cleaning",
                "input_dir": str(self.input_dir),
                "output_dir_filtered": str(self.filtered_dir),
                "total_images_scanned": len(image_files),
                "accepted_copied": copied_count,
                "rejected_skipped": skipped_count,
                "min_width": int(min_width) if min_width else 0,
                "min_height": int(min_height) if min_height else 0,
                "color_check": bool(color_check),
                "saturation_threshold": SATURATION_THRESHOLD,
                "whiteness_threshold": WHITENESS_THRESHOLD,
                "sample_limit": int(sample_limit),
                "reasons": dict(reason_counter),
            }
            with open(report_path / "cleaning_stats.json", "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=4)

            # Optional plots (best-effort)
            try:
                import matplotlib.pyplot as plt

                # Parse numeric columns from log if present
                if log_csv_path and log_csv_path.exists():
                    means = []
                    whites = []
                    with open(log_csv_path, "r", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            if row.get("mean_saturation") not in (None, "", "N/A"):
                                means.append(float(row["mean_saturation"]))
                            if row.get("white_ratio") not in (None, "", "N/A"):
                                whites.append(float(row["white_ratio"]))

                    if means:
                        plt.figure(figsize=(8, 4))
                        plt.hist(means, bins=40, color="#3B82F6", alpha=0.85)
                        plt.axvline(SATURATION_THRESHOLD, color="#EF4444", linewidth=2, linestyle="--")
                        plt.title("Mean Saturation Distribution")
                        plt.xlabel("mean_saturation (0-255)")
                        plt.ylabel("count")
                        plt.tight_layout()
                        plt.savefig(report_path / "saturation_hist.png", dpi=180)
                        plt.close()

                    if whites:
                        plt.figure(figsize=(8, 4))
                        plt.hist(whites, bins=40, color="#10B981", alpha=0.85)
                        plt.axvline(WHITENESS_THRESHOLD, color="#EF4444", linewidth=2, linestyle="--")
                        plt.title("White Ratio Distribution")
                        plt.xlabel("white_ratio (0-1)")
                        plt.ylabel("count")
                        plt.tight_layout()
                        plt.savefig(report_path / "white_ratio_hist.png", dpi=180)
                        plt.close()
            except Exception as e:
                self.logger.info(f"Plotting skipped (matplotlib unavailable or error): {e}")

        self.logger.info(f"Filtering Complete. Copied {copied_count} colored images to {self.filtered_dir}. Skipped {skipped_count} grayscale/irrelevant images.")
