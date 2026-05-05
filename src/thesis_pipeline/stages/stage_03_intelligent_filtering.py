from thesis_pipeline.logging_config import logger
from thesis_pipeline.config_manager import ConfigManager
from thesis_pipeline.utils.parallel_executor import ParallelExecutor
from thesis_pipeline.components.quality import QualityFilter
from thesis_pipeline.utils.hero_tracker import HeroTracker
from thesis_pipeline.components.audit import AuditLogger
from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np
import shutil
import json
import csv
import logging
from tqdm import tqdm
from thesis_pipeline.analysis.bias_analyzer import BiasAnalyzer

_worker_logger = logging.getLogger(__name__)

def _resolve_dataset_limit(config, stage_params):
    try:
        global_params = config.get("global_params", {}) if config else {}
        global_limit = global_params.get("dataset_limit")
    except Exception:
        global_limit = None

    try:
        stage_limit = stage_params.get("limit") if stage_params else None
    except Exception:
        stage_limit = None

    limits = [v for v in (global_limit, stage_limit) if isinstance(v, int) and v > 0]
    return min(limits) if limits else None


def _safe_cv2_imwrite(out_path: Path, image, *, context: str = "") -> bool:
    """
    Best-effort cv2.imwrite wrapper.

    OpenCV's Python bindings can throw if `image` isn't a real numeric ndarray.
    In the pipeline, we prefer to skip debug writes rather than abort the worker.
    """
    try:
        if image is None:
            _worker_logger.warning(f"cv2.imwrite skipped ({context}): image is None -> {out_path}")
            return False
        if not isinstance(image, np.ndarray):
            image = np.asarray(image)
        if not isinstance(image, np.ndarray) or image.dtype == object:
            _worker_logger.warning(
                f"cv2.imwrite skipped ({context}): non-numeric image type {type(image)} dtype={getattr(image, 'dtype', None)} -> {out_path}"
            )
            return False
        if image.size == 0:
            _worker_logger.warning(f"cv2.imwrite skipped ({context}): empty image -> {out_path}")
            return False
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ok = bool(cv2.imwrite(str(out_path), image))
        if not ok:
            _worker_logger.warning(f"cv2.imwrite returned False ({context}) -> {out_path}")
        return ok
    except Exception as e:
        _worker_logger.warning(f"cv2.imwrite failed ({context}) -> {out_path}: {e}")
        return False

def _worker_process(device, image_files, config_dict, worker_id=0):
    output_dir = Path(config_dict["filtered_path"])
    report_dir = Path(config_dict["report_path"])

    accepted_dir = output_dir / "accepted"
    rejected_dir = output_dir / "rejected"
    debug_yolo_dir = report_dir / "debug_yolo"

    accepted_dir.mkdir(parents=True, exist_ok=True)
    rejected_dir.mkdir(parents=True, exist_ok=True)
    debug_yolo_dir.mkdir(parents=True, exist_ok=True)

    audit_logger = AuditLogger(report_dir)

    model_name = config_dict.get("model_name", "yolov8x.pt")
    conf_threshold = float(config_dict.get("confidence_threshold", 0.4))
    yolo_accept_conf = float(config_dict.get("yolo_accept_confidence", 0.0))
    padding = float(config_dict.get("padding_ratio", 0.1))
    target_class = int(config_dict.get("target_class_id", 75))
    blur_threshold_main = float(config_dict.get("blur_threshold_main", config_dict.get("blur_threshold", 100.0)))
    blur_threshold_relaxed = float(config_dict.get("blur_threshold_relaxed", blur_threshold_main))
    high_confidence_gate = float(config_dict.get("high_confidence_gate", 0.85))
    min_area_ratio = float(config_dict.get("min_area_ratio", 0.04))
    blur_metric = config_dict.get("blur_metric", "tenengrad")
    resize_max_dim = int(config_dict.get("blur_resize_max_dim", 512))

    hero_config = config_dict.get("hero_config")
    hero_tracker = None
    if hero_config and hero_config.get("enabled"):
        hero_tracker = HeroTracker(Path(hero_config["output_dir"]), hero_config["hero_filenames"])

    quality_filter = QualityFilter(
        blur_threshold=blur_threshold_main,
        blur_metric=blur_metric,
        resize_max_dim=resize_max_dim,
    )

    print(f"[{device}] Loading Model: {model_name}")
    # If a local weights path is configured but missing (e.g., after cleanup),
    # fall back to the Ultralytics model name so it can be auto-downloaded.
    try:
        if ("/" in str(model_name) or "\\" in str(model_name)) and not Path(str(model_name)).exists():
            _worker_logger.warning(
                f"[{device}] YOLO weights not found at '{model_name}'. Falling back to 'yolov8x.pt'."
            )
            model_name = "yolov8x.pt"
    except Exception:
        pass
    model_device = device if device != "cpu" else "cpu"
    model = YOLO(model_name)

    stats = {
        "total": 0,
        "kept": 0,
        "rejected_yolo": 0,
        "rejected_low_conf": 0,
        "rejected_blur": 0,
        "crops": 0,
        "read_failures": 0,
        "total_detections": 0,
        "crops_passed_main": 0,
        "crops_passed_relaxed": 0,
        "rejected_all_crops": 0,
        "no_detection": 0,
    }
    process_counter = 0

    for img_path in tqdm(image_files, desc=f"[{device}] Filtering", position=worker_id, leave=True):
        process_counter += 1
        stats["total"] += 1

        if (accepted_dir / img_path.name).exists() or (rejected_dir / img_path.name).exists():
            continue

        try:
            results = model(img_path, verbose=False, device=model_device, classes=[target_class])
            max_conf = 0.0
            detections = []
            for r in results:
                for box in r.boxes:
                    conf = float(box.conf[0])
                    if conf > max_conf:
                        max_conf = conf
                    if conf > conf_threshold:
                        detections.append({"bbox": box.xyxy[0].cpu().numpy(), "conf": conf})

            # Debug Visuals (V7: First 10 + every 100th)
            if process_counter <= 10 or process_counter % 100 == 0:
                res_plotted = results[0].plot()
                _safe_cv2_imwrite(debug_yolo_dir / f"debug_{img_path.name}", res_plotted, context="debug_yolo_plot")

            if not detections:
                shutil.copy(img_path, rejected_dir / img_path.name)
                stats["rejected_yolo"] += 1
                stats["no_detection"] += 1
                audit_logger.log_decision(
                    img_path.name,
                    max_conf,
                    None,
                    "No Object Detected",
                    "Rejected",
                    focus_metric=blur_metric,
                )
            else:
                # Strict acceptance: if configured, require a high-confidence vase
                # detection before we commit to creating crops. This avoids low-confidence
                # "junk" crops (e.g., conf=0.49) entering the dataset.
                if yolo_accept_conf > 0.0:
                    detections = [d for d in detections if float(d.get("conf", 0.0)) >= yolo_accept_conf]

                if not detections:
                    shutil.copy(img_path, rejected_dir / img_path.name)
                    stats["rejected_yolo"] += 1
                    stats["rejected_low_conf"] += 1
                    audit_logger.log_decision(
                        img_path.name,
                        max_conf,
                        None,
                        "No High-Confidence Detection",
                        "Rejected",
                        focus_metric=blur_metric,
                        focus_threshold_used=yolo_accept_conf,
                    )
                    continue

                img = cv2.imread(str(img_path))
                if img is None:
                    stats["read_failures"] += 1
                    _worker_logger.warning(
                        f"[{device}] cv2.imread returned None for {img_path.name} — "
                        f"file may be corrupt or unreadable."
                    )
                    audit_logger.log_decision(
                        img_path.name,
                        max_conf,
                        None,
                        "Read Failure",
                        "Rejected",
                        focus_metric=blur_metric,
                    )
                    continue
                h, w, _ = img.shape
                valid_crops = 0
                stats["total_detections"] += len(detections)
                for i, det in enumerate(detections):
                    x1, y1, x2, y2 = det["bbox"]
                    det_conf = float(det["conf"])
                    bw, bh = x2 - x1, y2 - y1
                    pad_x, pad_y = bw * padding, bh * padding
                    nx1, ny1 = max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y))
                    nx2, ny2 = min(w, int(x2 + pad_x)), min(h, int(y2 + pad_y))
                    crop = img[ny1:ny2, nx1:nx2]
                    crop_w = max(0, nx2 - nx1)
                    crop_h = max(0, ny2 - ny1)
                    bbox_area_ratio = float((crop_w * crop_h) / float(w * h)) if (w * h) else 0.0

                    score = quality_filter.compute_focus_score(crop)
                    pass_main = score >= blur_threshold_main
                    pass_relaxed = (
                        score >= blur_threshold_relaxed
                        and det_conf >= high_confidence_gate
                        and bbox_area_ratio >= min_area_ratio
                    )
                    if not (pass_main or pass_relaxed):
                        stats["rejected_blur"] += 1
                        audit_logger.log_decision(
                            f"{img_path.stem}_crop{i}.jpg",
                            det_conf,
                            score,
                            "Failed Quality Gate",
                            "Rejected",
                            focus_metric=blur_metric,
                            focus_score=score,
                            focus_threshold_used=blur_threshold_main,
                            bbox_area_ratio=bbox_area_ratio,
                            crop_index=i,
                        )
                        continue
                    suffix = f"_crop{i}" if len(detections) > 1 else ""
                    _safe_cv2_imwrite(accepted_dir / f"{img_path.stem}{suffix}.jpg", crop, context="accepted_crop")
                    valid_crops += 1
                    stats["crops"] += 1
                    if pass_main:
                        stats["crops_passed_main"] += 1
                        threshold_used = blur_threshold_main
                        decision_reason = "Passed (Main Quality Gate)"
                    else:
                        stats["crops_passed_relaxed"] += 1
                        threshold_used = blur_threshold_relaxed
                        decision_reason = "Passed (Relaxed Quality Gate)"
                    audit_logger.log_decision(
                        f"{img_path.stem}{suffix}.jpg",
                        det_conf,
                        score,
                        decision_reason,
                        "Accepted",
                        focus_metric=blur_metric,
                        focus_score=score,
                        focus_threshold_used=threshold_used,
                        bbox_area_ratio=bbox_area_ratio,
                        crop_index=i,
                    )
                    if hero_tracker:
                        hero_tracker.log_image(crop, "03_filtered", f"{img_path.stem}{suffix}")
                if valid_crops > 0:
                    stats["kept"] += 1
                else:
                    shutil.copy(img_path, rejected_dir / img_path.name)
                    stats["rejected_all_crops"] += 1
                    audit_logger.log_decision(
                        img_path.name,
                        max_conf,
                        None,
                        "All Crops Failed Quality",
                        "Rejected",
                        focus_metric=blur_metric,
                        focus_threshold_used=blur_threshold_main,
                    )
        except Exception as e:
            _worker_logger.error(f"[{device}] Error processing {img_path.name}: {e}")

    # Stats to report_dir
    stats_file = report_dir / f"worker_{worker_id}_stats.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f)

class IntelligentFilteringStage:
    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.config = config_manager.config
        # BUG-FIX: Read from Stage 02b cleaned output (filtered root),
        # not raw. Previously Stage 02b's grayscale/sketch filtering was
        # entirely bypassed because Stage 03 re-read raw images directly.
        self.input_dir = Path(self.config.paths.data.filtered)
        self.output_dir = Path(self.config.paths.data.filtered)
        self.report_dir = self.config_manager.get_stage_artifact_dir("S03")
        self.params = self.config.intelligent_filtering

    def run(self):
        logger.info(">>> Starting Stage 03: Intelligent Filtering")
        self.report_dir.mkdir(parents=True, exist_ok=True)

        # --- Mock Mode ---
        if self.params.get("mock", False):
            logger.info("MOCK MODE ENABLED.")
            accepted_dir = self.output_dir / "accepted"
            accepted_dir.mkdir(parents=True, exist_ok=True)
            audit_logger = AuditLogger(self.report_dir)
            images = sorted(self.input_dir.glob("*.jpg"), key=lambda p: p.name)
            images += sorted(self.input_dir.glob("*.jpeg"), key=lambda p: p.name)
            images += sorted(self.input_dir.glob("*.png"), key=lambda p: p.name)
            limit = _resolve_dataset_limit(self.config, self.params)
            if limit:
                images = images[:limit]
                logger.info(f"Applying dataset limit: {limit} images (mock mode).")
            for img in images[:10]:
                shutil.copy(img, accepted_dir / img.name)
                audit_logger.log_decision(img.name, 0.99, 1000.0, "Mock Pass", "Accepted")
            return

        images = sorted(self.input_dir.glob("*.jpg"), key=lambda p: p.name)
        images += sorted(self.input_dir.glob("*.jpeg"), key=lambda p: p.name)
        images += sorted(self.input_dir.glob("*.png"), key=lambda p: p.name)
        limit = _resolve_dataset_limit(self.config, self.params)
        if limit:
            images = images[:limit]
            logger.info(f"Applying dataset limit: {limit} images.")
        worker_config = {
            'filtered_path': str(self.output_dir),
            'report_path': str(self.report_dir),
            'model_name': self.params.get("model_name", "yolov8x.pt"),
            'confidence_threshold': self.params.get("confidence_threshold", 0.25),
            'yolo_accept_confidence': self.params.get("yolo_accept_confidence", 0.0),
            'padding_ratio': self.params.get("padding_ratio", 0.1),
            'target_class_id': self.params.get("target_class_id", 75),
            'blur_threshold_main': self.params.get("blur_threshold_main", self.params.get("blur_threshold", 4500.0)),
            'blur_threshold_relaxed': self.params.get("blur_threshold_relaxed", 3000.0),
            'high_confidence_gate': self.params.get("high_confidence_gate", 0.85),
            'min_area_ratio': self.params.get("min_area_ratio", 0.04),
            'blur_metric': self.params.get("blur_metric", "tenengrad"),
            'blur_resize_max_dim': self.params.get("blur_resize_max_dim", 512),
            'hero_config': self.config.get("hero_tracking", {}).to_dict() if self.config.get("hero_tracking") else None
        }

        ParallelExecutor.run_gpu_parallel(_worker_process, images, config_dict=worker_config)

        total_stats = {
            "total": 0,
            "kept": 0,
            "rejected_yolo": 0,
            "rejected_low_conf": 0,
            "rejected_blur": 0,
            "crops": 0,
            "read_failures": 0,
            "total_detections": 0,
            "crops_passed_main": 0,
            "crops_passed_relaxed": 0,
            "rejected_all_crops": 0,
            "no_detection": 0,
        }
        for stat_file in self.report_dir.glob("worker_*_stats.json"):
            with open(stat_file, "r") as f:
                s = json.load(f)
                for k in total_stats:
                    total_stats[k] += s.get(k, 0)
            stat_file.unlink()
        with open(self.report_dir / "filtering_stats.json", "w") as f:
            json.dump(total_stats, f, indent=4)

        quality_summary = self._build_quality_summary(total_stats)
        with open(self.report_dir / "quality_summary.json", "w") as f:
            json.dump(quality_summary, f, indent=4)

        # --- Rich Verification Artifacts ---
        self._generate_verification_artifacts(total_stats, len(images))
        self._run_bias_disparity_audit()

        logger.info(f"Stage 03 Complete. Stats: {total_stats}")

    def _run_bias_disparity_audit(self):
        log_path = self.report_dir / "rejection_log.csv"
        if not log_path.exists():
            return

        analyzer = BiasAnalyzer(log_path, self.report_dir)
        analyzer.analyze()

        disparity_path = self.report_dir / "filtering_bias_disparity.json"
        if not disparity_path.exists():
            return

        try:
            disparity = json.loads(disparity_path.read_text(encoding="utf-8"))
        except Exception:
            return

        cfg = self.params.get("bias_audit", {}) if hasattr(self.params, "get") else {}
        threshold = float(cfg.get("max_acceptance_rate_gap", 0.15))
        strict_enabled = bool(cfg.get("strict_fail_on_disparity", False)) and bool(
            self.config.get("pipeline", {}).get("strict_fail_policy", False)
        )

        available = bool(disparity.get("available", False))
        gap = disparity.get("absolute_gap")
        if not available or gap is None:
            return

        if float(gap) > threshold:
            msg = (
                f"Filtering source disparity gap {float(gap):.4f} exceeds threshold {threshold:.4f}"
            )
            if strict_enabled:
                raise RuntimeError(msg)
            logger.warning(msg)

    def _build_quality_summary(self, total_stats):
        """Build quality summary from audit log to remain robust across reruns."""
        summary = {
            "focus_metric": self.params.get("blur_metric", "tenengrad"),
            "total_detections": int(total_stats.get("total_detections", 0)),
            "crops_passed_main": int(total_stats.get("crops_passed_main", 0)),
            "crops_passed_relaxed": int(total_stats.get("crops_passed_relaxed", 0)),
            "rejected_all_crops": int(total_stats.get("rejected_all_crops", 0)),
            "no_detection": int(total_stats.get("no_detection", 0)),
            "rejected_low_conf": int(total_stats.get("rejected_low_conf", 0)),
        }

        log_path = self.report_dir / "rejection_log.csv"
        if not log_path.exists():
            return summary

        try:
            total_detections = 0
            passed_main = 0
            passed_relaxed = 0
            rejected_all_crops = 0
            no_detection = 0
            rejected_low_conf = 0
            with open(log_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    reason = (row.get("reason") or "").strip()
                    crop_index = (row.get("crop_index") or "").strip()
                    if reason == "No Object Detected":
                        no_detection += 1
                    if reason == "No High-Confidence Detection":
                        rejected_low_conf += 1
                    if reason == "All Crops Failed Quality":
                        rejected_all_crops += 1
                    if reason.startswith("Passed (Main Quality Gate)"):
                        passed_main += 1
                    if reason.startswith("Passed (Relaxed Quality Gate)"):
                        passed_relaxed += 1
                    if crop_index not in ("", "N/A"):
                        total_detections += 1

            summary["total_detections"] = total_detections
            summary["crops_passed_main"] = passed_main
            summary["crops_passed_relaxed"] = passed_relaxed
            summary["rejected_all_crops"] = rejected_all_crops
            summary["no_detection"] = no_detection
            summary["rejected_low_conf"] = rejected_low_conf
        except Exception as exc:
            _worker_logger.warning(f"Failed to derive quality summary from audit log: {exc}")
        return summary

    def _generate_verification_artifacts(self, stats, total_input_images):
        """Generate human-verifiable artifacts: data flow summary + sample images."""
        import random

        # 1. Data flow summary
        accepted_dir = self.output_dir / "accepted"
        rejected_dir = self.output_dir / "rejected"
        n_accepted = len(list(accepted_dir.glob("*.jpg"))) if accepted_dir.exists() else 0
        n_rejected = len(list(rejected_dir.glob("*.jpg"))) if rejected_dir.exists() else 0

        data_flow = {
            "stage": "03_intelligent_filtering",
            "input_images": total_input_images,
            "images_processed": stats.get("total", 0),
            "images_with_valid_crops": stats.get("kept", 0),
            "total_crops_accepted": stats.get("crops", 0),
            "rejected_no_detection": stats.get("no_detection", 0),
            "rejected_low_conf": stats.get("rejected_low_conf", 0),
            "rejected_blur": stats.get("rejected_blur", 0),
            "rejected_all_crops": stats.get("rejected_all_crops", 0),
            "crops_passed_main": stats.get("crops_passed_main", 0),
            "crops_passed_relaxed": stats.get("crops_passed_relaxed", 0),
            "accepted_files_on_disk": n_accepted,
            "rejected_files_on_disk": n_rejected,
            "read_failures": stats.get("read_failures", 0),
        }
        with open(self.report_dir / "data_flow_summary.json", "w") as f:
            json.dump(data_flow, f, indent=4)
        with open(self.report_dir / "input_manifest.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "count": total_input_images,
                    "files": [p.name for p in sorted(self.input_dir.glob("*")) if p.is_file()],
                },
                f,
                indent=4,
            )
        with open(self.report_dir / "accepted_manifest.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "count": n_accepted,
                    "files": [p.name for p in sorted(accepted_dir.glob("*.jpg"))],
                },
                f,
                indent=4,
            )
        with open(self.report_dir / "rejected_manifest.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "count": n_rejected,
                    "files": [p.name for p in sorted(rejected_dir.glob("*.jpg"))],
                },
                f,
                indent=4,
            )

        # 2. Sample images for human verification (up to 50 per category)
        samples_dir = self.report_dir / "samples"
        sample_limit = 50

        for category, src_dir in [("accepted", accepted_dir), ("rejected", rejected_dir)]:
            cat_dir = samples_dir / category
            if cat_dir.exists():
                shutil.rmtree(cat_dir, ignore_errors=True)
            cat_dir.mkdir(parents=True, exist_ok=True)
            if src_dir.exists():
                files = sorted(src_dir.glob("*.jpg"))
                selected = files[:sample_limit] if len(files) <= sample_limit else random.sample(files, sample_limit)
                for f in selected:
                    shutil.copy(f, cat_dir / f.name)

if __name__ == "__main__":
    cm = ConfigManager()
    stage = IntelligentFilteringStage(cm)
    stage.run()
