import json
import logging
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from thesis_pipeline.components.captioning.local_vlm import (
    CaptionGenerationError,
    LocalVLM,
)
from thesis_pipeline.utils.parallel_executor import ParallelExecutor


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


def _normalize_caption_text(value: Any) -> str:
    if value is None:
        return ""
    text = " ".join(str(value).split()).strip()
    if not text:
        return ""
    if text.replace(".", "").replace(":", "").strip() == "":
        return ""
    return text


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


def _contains_any(text: str, keywords: list[str]) -> bool:
    lowered = str(text or "").lower()
    return any(k in lowered for k in keywords)


def _extract_spatial_caption(caption: str) -> str:
    text = _normalize_caption_text(caption)
    if not text:
        return ""
    spatial_keywords = [
        "left", "right", "center", "upper", "lower", "foreground", "background",
        "rim", "neck", "handle", "base", "body", "surface", "edge",
        "crack", "chip", "fracture", "missing", "damaged", "broken",
    ]
    chunks = re.split(r"(?<=[.!?])\s+", text)
    focused = [chunk for chunk in chunks if _contains_any(chunk, spatial_keywords)]
    if focused:
        return _normalize_caption_text(" ".join(focused))
    return text


def _vlm_worker(device, image_files, config_dict, worker_id=0):
    output_path = Path(config_dict["output_path"])
    model_name = config_dict.get("model_name", "Salesforce/blip2-opt-2.7b")
    oom_retry_limit = int(config_dict.get("oom_retry_limit", 1))
    oom_backoff_max_new_tokens = int(config_dict.get("oom_backoff_max_new_tokens", 24))
    cleanup_cuda_cache = bool(config_dict.get("cleanup_cuda_cache", True))
    max_worker_fail_streak = int(config_dict.get("max_worker_fail_streak", 10))
    max_new_tokens = int(config_dict.get("max_new_tokens", 50))

    results: dict[str, str] = {}
    report: dict[str, Any] = {
        "worker_id": worker_id,
        "device": device,
        "model_name": model_name,
        "total_images": len(image_files),
        "success_count": 0,
        "failure_count": 0,
        "oom_failure_count": 0,
        "retry_count": 0,
        "max_fail_streak_observed": 0,
        "max_worker_fail_streak": max_worker_fail_streak,
        "model_load_error": None,
        "aborted_due_to_fail_streak": False,
        "failures": [],
    }

    try:
        vlm = LocalVLM(model_name=model_name, device=device)
    except Exception as e:
        report["model_load_error"] = str(e)
        (output_path / f"worker_{worker_id}_report.json").write_text(
            json.dumps(_json_safe(report), indent=2),
            encoding="utf-8",
        )
        return

    current_fail_streak = 0
    for img_path in tqdm(image_files, desc=f"[{device}] Enriching", position=worker_id, leave=True):
        try:
            caption_report = vlm.dense_caption_with_report(
                str(img_path),
                max_new_tokens=max_new_tokens,
                oom_retry_limit=oom_retry_limit,
                oom_backoff_max_new_tokens=oom_backoff_max_new_tokens,
                cleanup_cuda_cache=cleanup_cuda_cache,
            )
            caption = _normalize_caption_text(caption_report["caption"])
            if not caption:
                raise CaptionGenerationError(
                    image_path=str(img_path),
                    prompt_label="dense_caption",
                    kind="invalid_caption",
                    attempts=int(caption_report.get("attempts", 1)),
                    retry_count=int(caption_report.get("retry_count", 0)),
                    message="Generated caption normalized to empty text.",
                )

            results[img_path.name] = caption
            report["success_count"] += 1
            report["retry_count"] += int(caption_report.get("retry_count", 0))
            current_fail_streak = 0
        except CaptionGenerationError as e:
            report["failure_count"] += 1
            report["retry_count"] += int(getattr(e, "retry_count", 0))
            if getattr(e, "kind", "") == "cuda_oom":
                report["oom_failure_count"] += 1
            current_fail_streak += 1
            report["max_fail_streak_observed"] = max(
                int(report["max_fail_streak_observed"]),
                current_fail_streak,
            )
            report["failures"].append(
                {
                    "image": img_path.name,
                    "kind": getattr(e, "kind", "caption_error"),
                    "message": str(e),
                    "attempts": int(getattr(e, "attempts", 1)),
                    "retry_count": int(getattr(e, "retry_count", 0)),
                    "prompt_label": getattr(e, "prompt_label", "dense_caption"),
                }
            )
            if current_fail_streak >= max_worker_fail_streak:
                report["aborted_due_to_fail_streak"] = True
                break
        except Exception as e:
            report["failure_count"] += 1
            current_fail_streak += 1
            report["max_fail_streak_observed"] = max(
                int(report["max_fail_streak_observed"]),
                current_fail_streak,
            )
            report["failures"].append(
                {
                    "image": img_path.name,
                    "kind": "unexpected_error",
                    "message": str(e),
                    "attempts": 1,
                    "retry_count": 0,
                    "prompt_label": "dense_caption",
                }
            )
            if current_fail_streak >= max_worker_fail_streak:
                report["aborted_due_to_fail_streak"] = True
                break

    (output_path / f"worker_{worker_id}_enriched.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )
    (output_path / f"worker_{worker_id}_report.json").write_text(
        json.dumps(_json_safe(report), indent=2),
        encoding="utf-8",
    )


class CaptionGenerationStage:
    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.config = config_manager.config
        self.logger = logging.getLogger(__name__)
        self.filtered_dir = Path(self.config.paths.data.filtered) / "accepted"
        self.raw_metadata_dir = Path(self.config.paths.data.raw) / "metadata"
        self.output_dir = self.config_manager.get_stage_artifact_dir("S07")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.caption_cfg = self.config.get("caption_generation", {}) if self.config else {}

    def _build_spatial_captions(self, enriched_captions: dict[str, str]) -> dict[str, str]:
        return {
            key: spatial
            for key, value in enriched_captions.items()
            for spatial in [_extract_spatial_caption(value)]
            if spatial
        }

    def _run_spatial_purity_audit(self, spatial_captions: dict[str, str], total_images: int) -> dict[str, Any]:
        cfg = self.caption_cfg.get("spatial_purity", {}) if hasattr(self.caption_cfg, "get") else {}
        min_rate = float(cfg.get("min_spatial_keyword_rate", 0.80) or 0.80)
        strict_fail = bool(cfg.get("strict_fail", True)) and bool(
            self.config.get("pipeline", {}).get("strict_fail_policy", False)
        )

        spatial_terms = [
            "left", "right", "center", "upper", "lower", "foreground", "background",
            "rim", "neck", "handle", "base", "body", "surface", "edge",
            "crack", "chip", "fracture", "missing", "damaged", "broken",
        ]
        style_leak_terms = ["beautiful", "masterpiece", "high quality", "award-winning", "gorgeous"]

        with_spatial = sum(1 for v in spatial_captions.values() if _contains_any(v, spatial_terms))
        with_style_leak = sum(1 for v in spatial_captions.values() if _contains_any(v, style_leak_terms))
        coverage_rate = (with_spatial / total_images) if total_images else 0.0
        style_leak_rate = (with_style_leak / max(1, len(spatial_captions)))

        report = {
            "total_images": int(total_images),
            "spatial_caption_count": int(len(spatial_captions)),
            "captions_with_spatial_terms": int(with_spatial),
            "spatial_keyword_rate": float(coverage_rate),
            "style_leak_count": int(with_style_leak),
            "style_leak_rate": float(style_leak_rate),
            "min_spatial_keyword_rate": float(min_rate),
            "strict_fail": bool(strict_fail),
            "pass": bool(coverage_rate >= min_rate and style_leak_rate <= 0.10),
        }
        (self.output_dir / "caption_spatial_purity_report.json").write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )

        if strict_fail and not report["pass"]:
            raise RuntimeError(
                "Spatial caption purity gate failed: "
                f"spatial_keyword_rate={coverage_rate:.3f} (min={min_rate:.3f}), "
                f"style_leak_rate={style_leak_rate:.3f}"
            )
        return report

    @staticmethod
    def _spatial_issue_reason(
        caption_text: str,
        *,
        spatial_terms: list[str],
        semantic_terms: list[str],
        max_contamination_ratio: float,
    ) -> str | None:
        text = _normalize_caption_text(caption_text)
        if not text:
            return "empty"

        lowered = text.lower()
        spatial_hits = [term for term in spatial_terms if term in lowered]
        semantic_hits = [term for term in semantic_terms if term in lowered]
        if not spatial_hits:
            return "missing_spatial_terms"
        contamination_ratio = float(len(semantic_hits) / max(1, len(spatial_hits)))
        if contamination_ratio > max_contamination_ratio:
            return "high_contamination"
        return None

    @staticmethod
    def _spatial_contamination_metrics(
        caption_text: str,
        *,
        spatial_terms: list[str],
        semantic_terms: list[str],
    ) -> dict[str, Any]:
        text = _normalize_caption_text(caption_text)
        lowered = text.lower()
        spatial_hits = [term for term in spatial_terms if term in lowered]
        semantic_hits = [term for term in semantic_terms if term in lowered]
        ratio = float(len(semantic_hits) / max(1, len(spatial_hits))) if text else float("inf")
        return {
            "spatial_hit_count": int(len(spatial_hits)),
            "semantic_hit_count": int(len(semantic_hits)),
            "spatial_hits": sorted(set(spatial_hits)),
            "semantic_hits": sorted(set(semantic_hits)),
            "contamination_ratio": float(ratio),
        }

    def _run_spatial_regeneration_loop(
        self,
        images: list[Path],
        spatial_captions: dict[str, str],
    ) -> dict[str, str]:
        cfg = self.caption_cfg.get("spatial_regeneration", {}) if hasattr(self.caption_cfg, "get") else {}
        enabled = bool(cfg.get("enabled", True))
        max_attempts = int(cfg.get("max_attempts", 2) or 2)
        fallback_prompt = str(
            cfg.get(
                "fallback_prompt",
                "Question: Describe only the exact spatial location and physical damage area on this ancient Greek pottery (rim/neck/handle/body/base, left/right/upper/lower/center), without stylistic adjectives. Answer:",
            )
        ).strip()
        max_new_tokens = int(cfg.get("max_new_tokens", self.caption_cfg.get("max_new_tokens", 50)) or 50)
        strict_fail = bool(cfg.get("strict_fail", True)) and bool(
            self.config.get("pipeline", {}).get("strict_fail_policy", False)
        )
        max_contamination_ratio = float(cfg.get("max_contamination_ratio", 0.50) or 0.50)

        whitelist_terms = cfg.get("whitelist_terms") if isinstance(cfg, dict) else None
        blacklist_terms = cfg.get("blacklist_terms") if isinstance(cfg, dict) else None

        spatial_terms = list(whitelist_terms) if isinstance(whitelist_terms, list) and whitelist_terms else [
            "left", "right", "center", "upper", "lower", "foreground", "background",
            "rim", "neck", "handle", "base", "body", "surface", "edge",
            "crack", "chip", "fracture", "missing", "damaged", "broken",
        ]
        semantic_terms = list(blacklist_terms) if isinstance(blacklist_terms, list) and blacklist_terms else [
            "beautiful", "masterpiece", "high quality", "award-winning", "gorgeous",
            "myth", "deity", "warrior", "hero", "athena", "dionysus", "iconography",
            "style", "motif", "pattern", "ornament", "decoration",
        ]

        cleaned = {k: _normalize_caption_text(v) for k, v in spatial_captions.items() if _normalize_caption_text(v)}
        report_events: list[dict[str, Any]] = []
        unresolved: list[dict[str, Any]] = []

        if not enabled:
            (self.output_dir / "captions_spatial_clean.json").write_text(
                json.dumps(cleaned, indent=2),
                encoding="utf-8",
            )
            return cleaned

        image_lookup = {p.name: p for p in images}
        use_mock = bool(self.caption_cfg.get("mock", False))
        vlm = None

        for image_name, image_path in image_lookup.items():
            current = _normalize_caption_text(cleaned.get(image_name, ""))
            issue = self._spatial_issue_reason(
                current,
                spatial_terms=spatial_terms,
                semantic_terms=semantic_terms,
                max_contamination_ratio=max_contamination_ratio,
            )
            attempts = 0

            while issue and attempts < max_attempts and not use_mock:
                attempts += 1
                if vlm is None:
                    vlm = LocalVLM(model_name=self.caption_cfg.get("model_name"))
                regenerated = vlm.generate_caption(
                    str(image_path),
                    prompt=fallback_prompt,
                    prompt_label=f"spatial_fallback_{attempts}",
                    max_new_tokens=max_new_tokens,
                    oom_retry_limit=int(self.caption_cfg.get("oom_retry_limit", 1) or 1),
                    oom_backoff_max_new_tokens=int(self.caption_cfg.get("oom_backoff_max_new_tokens", 24) or 24),
                    cleanup_cuda_cache=bool(self.caption_cfg.get("cleanup_cuda_cache", True)),
                )
                current = _extract_spatial_caption(regenerated)
                issue = self._spatial_issue_reason(
                    current,
                    spatial_terms=spatial_terms,
                    semantic_terms=semantic_terms,
                    max_contamination_ratio=max_contamination_ratio,
                )

            contamination = self._spatial_contamination_metrics(
                current,
                spatial_terms=spatial_terms,
                semantic_terms=semantic_terms,
            )

            event = {
                "image": image_name,
                "attempts": attempts,
                "resolved": issue is None,
                "final_issue": issue,
                **contamination,
            }
            report_events.append(event)

            if issue is None and current:
                cleaned[image_name] = current
            else:
                cleaned.pop(image_name, None)
                unresolved.append(event)

        report = {
            "enabled": True,
            "strict_fail": strict_fail,
            "max_attempts": max_attempts,
            "max_contamination_ratio": max_contamination_ratio,
            "whitelist_terms": sorted(set(spatial_terms)),
            "blacklist_terms": sorted(set(semantic_terms)),
            "fallback_prompt": fallback_prompt,
            "input_count": len(image_lookup),
            "clean_count": len(cleaned),
            "unresolved_count": len(unresolved),
            "events": report_events,
        }
        (self.output_dir / "caption_spatial_regeneration_report.json").write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )
        (self.output_dir / "captions_spatial_clean.json").write_text(
            json.dumps(cleaned, indent=2),
            encoding="utf-8",
        )

        if strict_fail and unresolved:
            raise RuntimeError(
                "Spatial regeneration failed for "
                f"{len(unresolved)} sample(s); unresolved examples="
                f"{[x['image'] for x in unresolved[:5]]}"
            )
        return cleaned

    @staticmethod
    def _token_set(text: str) -> set[str]:
        tokens = re.findall(r"[a-zA-Z]+", str(text or "").lower())
        return {t for t in tokens if len(t) >= 3}

    @staticmethod
    def _quadrant_presence(mask_np: np.ndarray) -> dict[str, int]:
        h, w = mask_np.shape[:2]
        if h == 0 or w == 0:
            return {"upper_left": 0, "upper_right": 0, "lower_left": 0, "lower_right": 0}
        mid_h, mid_w = h // 2, w // 2
        quads = {
            "upper_left": mask_np[:mid_h, :mid_w],
            "upper_right": mask_np[:mid_h, mid_w:],
            "lower_left": mask_np[mid_h:, :mid_w],
            "lower_right": mask_np[mid_h:, mid_w:],
        }
        return {k: int(np.any(v > 0)) for k, v in quads.items()}

    @staticmethod
    def _predicted_quadrants(caption: str) -> dict[str, int]:
        text = str(caption or "").lower()
        has_upper = any(tok in text for tok in ["upper", "top"])
        has_lower = any(tok in text for tok in ["lower", "bottom", "base"])
        has_left = "left" in text
        has_right = "right" in text
        has_center = any(tok in text for tok in ["center", "central", "middle"])

        preds = {"upper_left": 0, "upper_right": 0, "lower_left": 0, "lower_right": 0}
        if has_center:
            for k in preds:
                preds[k] = 1
            return preds

        if has_upper and has_left:
            preds["upper_left"] = 1
        if has_upper and has_right:
            preds["upper_right"] = 1
        if has_lower and has_left:
            preds["lower_left"] = 1
        if has_lower and has_right:
            preds["lower_right"] = 1

        if has_upper and not (has_left or has_right):
            preds["upper_left"] = 1
            preds["upper_right"] = 1
        if has_lower and not (has_left or has_right):
            preds["lower_left"] = 1
            preds["lower_right"] = 1
        if has_left and not (has_upper or has_lower):
            preds["upper_left"] = 1
            preds["lower_left"] = 1
        if has_right and not (has_upper or has_lower):
            preds["upper_right"] = 1
            preds["lower_right"] = 1

        return preds

    @staticmethod
    def _macro_f1_quadrants(rows: list[dict[str, Any]]) -> float:
        if not rows:
            return 0.0
        labels = ["upper_left", "upper_right", "lower_left", "lower_right"]
        f1s: list[float] = []
        for label in labels:
            tp = sum(1 for r in rows if int(r["pred_quadrants"].get(label, 0)) == 1 and int(r["gt_quadrants"].get(label, 0)) == 1)
            fp = sum(1 for r in rows if int(r["pred_quadrants"].get(label, 0)) == 1 and int(r["gt_quadrants"].get(label, 0)) == 0)
            fn = sum(1 for r in rows if int(r["pred_quadrants"].get(label, 0)) == 0 and int(r["gt_quadrants"].get(label, 0)) == 1)
            denom = (2 * tp + fp + fn)
            f1s.append(float((2 * tp) / denom) if denom > 0 else 0.0)
        return float(np.mean(f1s)) if f1s else 0.0

    @staticmethod
    def _mask_border_touch(mask_np: np.ndarray) -> int:
        if mask_np.size == 0:
            return 0
        top = np.any(mask_np[0, :] > 0)
        bottom = np.any(mask_np[-1, :] > 0)
        left = np.any(mask_np[:, 0] > 0)
        right = np.any(mask_np[:, -1] > 0)
        return int(top or bottom or left or right)

    @staticmethod
    def _predicted_border_touch(caption: str) -> int:
        text = str(caption or "").lower()
        return int(any(tok in text for tok in ["rim", "edge", "border", "boundary"]))

    @staticmethod
    def _predicted_area_score(caption: str) -> float:
        text = str(caption or "").lower()
        if any(tok in text for tok in ["tiny", "small", "minor", "slight"]):
            return 0.20
        if any(tok in text for tok in ["large", "major", "extensive", "wide"]):
            return 0.80
        return 0.50

    @staticmethod
    def _safe_correlation(a: list[float], b: list[float]) -> float:
        if len(a) < 2 or len(b) < 2:
            return 0.0
        arr_a = np.asarray(a, dtype=float)
        arr_b = np.asarray(b, dtype=float)
        if np.std(arr_a) == 0 or np.std(arr_b) == 0:
            return 0.0
        corr = float(np.corrcoef(arr_a, arr_b)[0, 1])
        if np.isnan(corr):
            return 0.0
        return corr

    def _resolve_mask_for_caption_key(self, key: str) -> Path | None:
        stem = Path(key).stem
        candidates = []
        try:
            candidates.append(Path(self.config.paths.data.inpainting) / "test" / "masks" / f"{stem}.png")
        except Exception:
            pass
        candidates.append((self.filtered_dir / "masks" / f"{stem}.png"))
        for path in candidates:
            if path.exists():
                return path
        return None

    def _run_grounding_validation(
        self,
        raw_captions: dict[str, str],
        enriched_captions: dict[str, str],
        spatial_captions: dict[str, str],
    ) -> dict[str, Any]:
        cfg = self.caption_cfg.get("grounding_validation", {}) if hasattr(self.caption_cfg, "get") else {}
        min_quadrant_macro_f1 = float(cfg.get("min_quadrant_macro_f1", 0.60) or 0.60)
        min_border_touch_accuracy = float(cfg.get("min_border_touch_accuracy", 0.75) or 0.75)
        min_area_correlation = float(cfg.get("min_area_correlation", 0.40) or 0.40)
        strict_fail = bool(cfg.get("strict_fail", True)) and bool(
            self.config.get("pipeline", {}).get("strict_fail_policy", False)
        )

        paired_rows: list[dict[str, Any]] = []
        for key, spa in spatial_captions.items():
            mask_path = self._resolve_mask_for_caption_key(key)
            if mask_path is None:
                continue
            try:
                mask_np = np.array(plt.imread(mask_path))
                if mask_np.ndim == 3:
                    mask_np = mask_np[..., 0]
                if mask_np.dtype != np.uint8:
                    mask_np = (mask_np > 0).astype(np.uint8) * 255
                gt_quads = self._quadrant_presence(mask_np)
                gt_border = self._mask_border_touch(mask_np)
                gt_area = float(np.sum(mask_np > 0) / max(1, mask_np.size))
                pred_quads = self._predicted_quadrants(spa)
                pred_border = self._predicted_border_touch(spa)
                pred_area = self._predicted_area_score(spa)
                paired_rows.append(
                    {
                        "sample_id": key,
                        "pred_quadrants": pred_quads,
                        "gt_quadrants": gt_quads,
                        "pred_border_touch": int(pred_border),
                        "gt_border_touch": int(gt_border),
                        "pred_area": float(pred_area),
                        "gt_area": float(gt_area),
                    }
                )
            except Exception:
                continue

        quadrant_macro_f1 = self._macro_f1_quadrants(paired_rows)
        border_touch_accuracy = float(
            np.mean([int(x["pred_border_touch"] == x["gt_border_touch"]) for x in paired_rows])
        ) if paired_rows else 0.0
        area_correlation = self._safe_correlation(
            [float(x["pred_area"]) for x in paired_rows],
            [float(x["gt_area"]) for x in paired_rows],
        )

        swapped_rows: list[dict[str, Any]] = []
        if len(paired_rows) >= 2:
            for idx, row in enumerate(paired_rows):
                swapped_gt = paired_rows[(idx + 1) % len(paired_rows)]
                swapped_rows.append(
                    {
                        "sample_id": row["sample_id"],
                        "pred_quadrants": row["pred_quadrants"],
                        "gt_quadrants": swapped_gt["gt_quadrants"],
                        "pred_border_touch": row["pred_border_touch"],
                        "gt_border_touch": swapped_gt["gt_border_touch"],
                        "pred_area": row["pred_area"],
                        "gt_area": swapped_gt["gt_area"],
                    }
                )

        swapped_quadrant_macro_f1 = self._macro_f1_quadrants(swapped_rows)
        swapped_border_touch_accuracy = float(
            np.mean([int(x["pred_border_touch"] == x["gt_border_touch"]) for x in swapped_rows])
        ) if swapped_rows else 0.0
        swapped_area_correlation = self._safe_correlation(
            [float(x["pred_area"]) for x in swapped_rows],
            [float(x["gt_area"]) for x in swapped_rows],
        )

        report = {
            "sample_count": int(len(paired_rows)),
            "quadrant_macro_f1": float(quadrant_macro_f1),
            "border_touch_accuracy": float(border_touch_accuracy),
            "area_correlation": float(area_correlation),
            "mask_swap_quadrant_macro_f1": float(swapped_quadrant_macro_f1),
            "mask_swap_border_touch_accuracy": float(swapped_border_touch_accuracy),
            "mask_swap_area_correlation": float(swapped_area_correlation),
            "min_quadrant_macro_f1": float(min_quadrant_macro_f1),
            "min_border_touch_accuracy": float(min_border_touch_accuracy),
            "min_area_correlation": float(min_area_correlation),
            "strict_fail": strict_fail,
            "pass": bool(
                quadrant_macro_f1 >= min_quadrant_macro_f1
                and border_touch_accuracy >= min_border_touch_accuracy
                and area_correlation >= min_area_correlation
            ),
            "mask_swap_delta": {
                "quadrant_macro_f1": float(quadrant_macro_f1 - swapped_quadrant_macro_f1),
                "border_touch_accuracy": float(border_touch_accuracy - swapped_border_touch_accuracy),
                "area_correlation": float(area_correlation - swapped_area_correlation),
            },
            "examples": paired_rows[:20],
        }
        (self.output_dir / "stage_06b_grounding_validation.json").write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )

        if strict_fail and not report["pass"]:
            raise RuntimeError(
                "Stage06b grounding validation failed: "
                f"quadrant_macro_f1={quadrant_macro_f1:.3f}, "
                f"border_touch_accuracy={border_touch_accuracy:.3f}, "
                f"area_correlation={area_correlation:.3f}"
            )
        return report

    def extract_raw_captions(self, image_files):
        raw_captions = {}
        for img_path in tqdm(image_files, desc="Extracting Raw"):
            base_stem = re.sub(r"_crop\d+$", "", img_path.stem)
            json_path = self.raw_metadata_dir / f"{base_stem}.json"
            desc = "No metadata found."
            if json_path.exists():
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        val = data.get("metadata", {}).get("ImageDescription", {}).get("value", "")
                        val = re.sub(r"<[^>]*>", "", str(val)).strip()

                        if val:
                            desc = val
                        else:
                            title = str(data.get("title") or "").strip()
                            source = str(data.get("source") or "").strip().lower()
                            raw = data.get("raw_metadata") or {}

                            if not title and isinstance(raw, dict):
                                t = raw.get("title")
                                if isinstance(t, list) and t:
                                    title = str(t[0]).strip()

                            extra_bits = []
                            if source == "met" and isinstance(raw, dict):
                                for key in ("objectName", "culture", "period", "medium", "artistDisplayName"):
                                    value = str(raw.get(key) or "").strip()
                                    if value:
                                        extra_bits.append(value)

                            if source == "europeana" and isinstance(raw, dict):
                                for key in ("dcDescription", "dcCreator", "year", "type", "what"):
                                    value = raw.get(key)
                                    if isinstance(value, list) and value:
                                        extra_bits.append(str(value[0]).strip())

                            desc = " — ".join([part for part in [title, *extra_bits] if part]) or "No description in metadata."
                except Exception:
                    desc = "Metadata error."
            raw_captions[img_path.name] = desc
        return raw_captions

    @staticmethod
    def _caption_lengths(captions: dict[str, Any]) -> list[int]:
        return [len(text.split()) for text in (_normalize_caption_text(v) for v in captions.values()) if text]

    def generate_stats(self, raw, enriched):
        raw_lengths = self._caption_lengths(raw)
        enriched_lengths = self._caption_lengths(enriched)
        if raw_lengths and enriched_lengths:
            stats = {
                "metric": ["count", "mean", "std", "min", "max"],
                "raw": [len(raw_lengths), np.mean(raw_lengths), np.std(raw_lengths), np.min(raw_lengths), np.max(raw_lengths)],
                "enriched": [
                    len(enriched_lengths),
                    np.mean(enriched_lengths),
                    np.std(enriched_lengths),
                    np.min(enriched_lengths),
                    np.max(enriched_lengths),
                ],
            }
            pd.DataFrame(stats).to_csv(self.output_dir / "caption_stats.csv", index=False)

        plt.figure(figsize=(10, 6))
        if raw_lengths:
            plt.hist(raw_lengths, bins=30, alpha=0.5, label="Raw", color="blue")
        if enriched_lengths:
            plt.hist(enriched_lengths, bins=30, alpha=0.5, label="Enriched", color="green")
        plt.title("Caption Length Distribution")
        plt.legend()
        plt.savefig(self.output_dir / "caption_length_comparison.png")
        plt.close()

    def _load_worker_reports(self) -> list[dict[str, Any]]:
        reports = []
        for report_file in sorted(self.output_dir.glob("worker_*_report.json"), key=lambda p: p.name):
            with open(report_file, "r", encoding="utf-8") as f:
                reports.append(json.load(f))
            report_file.unlink()
        return reports

    def _load_worker_results(self) -> dict[str, str]:
        enriched_captions: dict[str, str] = {}
        for temp_file in sorted(self.output_dir.glob("worker_*_enriched.json"), key=lambda p: p.name):
            with open(temp_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            for key, value in data.items():
                normalized = _normalize_caption_text(value)
                if normalized:
                    enriched_captions[key] = normalized
            temp_file.unlink()
        return enriched_captions

    def _build_generation_report(
        self,
        image_files: list[Path],
        raw_captions: dict[str, str],
        enriched_captions: dict[str, str],
        worker_reports: list[dict[str, Any]],
    ) -> dict[str, Any]:
        input_count = len(image_files)
        success_count = len(enriched_captions)
        failed_images = []
        retry_count = 0
        oom_failure_count = 0
        model_load_errors = []
        aborted_workers = []
        max_fail_streak = 0

        for worker_report in worker_reports:
            retry_count += int(worker_report.get("retry_count", 0) or 0)
            oom_failure_count += int(worker_report.get("oom_failure_count", 0) or 0)
            max_fail_streak = max(max_fail_streak, int(worker_report.get("max_fail_streak_observed", 0) or 0))
            if worker_report.get("model_load_error"):
                model_load_errors.append(
                    {
                        "worker_id": worker_report.get("worker_id"),
                        "device": worker_report.get("device"),
                        "message": worker_report.get("model_load_error"),
                    }
                )
            if worker_report.get("aborted_due_to_fail_streak"):
                aborted_workers.append(worker_report.get("worker_id"))
            for failure in worker_report.get("failures", []):
                failed_images.append(failure)

        success_rate = (success_count / input_count) if input_count else 0.0
        report = {
            "input_image_count": input_count,
            "raw_caption_count": len(raw_captions),
            "enriched_success_count": success_count,
            "success_rate": success_rate,
            "failure_count": input_count - success_count,
            "oom_failure_count": oom_failure_count,
            "retry_count": retry_count,
            "worker_count": len(worker_reports),
            "model_load_errors": model_load_errors,
            "aborted_workers": aborted_workers,
            "max_fail_streak_observed": max_fail_streak,
            "failed_images": failed_images,
            "worker_reports": worker_reports,
        }
        return report

    def _enforce_failure_policy(self, report: dict[str, Any]) -> None:
        failure_policy = str(self.caption_cfg.get("failure_policy", "hard_fail")).lower()
        success_rate_min = float(self.caption_cfg.get("success_rate_min", 0.95) or 0.95)
        max_fail_streak = int(self.caption_cfg.get("max_worker_fail_streak", 10) or 10)

        reasons = []
        if report["model_load_errors"]:
            reasons.append("worker model load failure detected")
        if report["success_rate"] < success_rate_min:
            reasons.append(
                f"caption success rate {report['success_rate']:.3f} below threshold {success_rate_min:.3f}"
            )
        if report["max_fail_streak_observed"] >= max_fail_streak:
            reasons.append(
                f"worker fail streak {report['max_fail_streak_observed']} reached threshold {max_fail_streak}"
            )

        report_path = self.output_dir / "caption_generation_report.json"
        report_path.write_text(json.dumps(_json_safe(report), indent=2), encoding="utf-8")

        if reasons and failure_policy == "hard_fail":
            raise RuntimeError("Caption generation failed strict policy: " + " | ".join(reasons))

    def run(self):
        self.logger.info("=" * 20 + " STAGE 06: Caption Generation " + "=" * 20)
        images = sorted(self.filtered_dir.glob("*.jpg"), key=lambda p: p.name)
        if not images:
            return

        limit = _resolve_dataset_limit(self.config, self.caption_cfg)
        if limit:
            images = images[:limit]
            self.logger.info(f"Applying dataset limit: {limit} images.")

        raw_captions = self.extract_raw_captions(images)
        with open(self.output_dir / "captions_raw.json", "w", encoding="utf-8") as f:
            json.dump(raw_captions, f, indent=2)

        enriched_file = self.output_dir / "captions_enriched.json"
        if self.caption_cfg.get("mock", False):
            enriched_captions = {img.name: "Mock pottery description." for img in images}
            worker_reports = [
                {
                    "worker_id": 0,
                    "device": "mock",
                    "model_name": "mock",
                    "total_images": len(images),
                    "success_count": len(images),
                    "failure_count": 0,
                    "oom_failure_count": 0,
                    "retry_count": 0,
                    "max_fail_streak_observed": 0,
                    "max_worker_fail_streak": 0,
                    "model_load_error": None,
                    "aborted_due_to_fail_streak": False,
                    "failures": [],
                }
            ]
        else:
            worker_config = {
                "output_path": str(self.output_dir),
                "model_name": self.caption_cfg.get("model_name"),
                "oom_retry_limit": int(self.caption_cfg.get("oom_retry_limit", 1) or 1),
                "oom_backoff_max_new_tokens": int(
                    self.caption_cfg.get("oom_backoff_max_new_tokens", 24) or 24
                ),
                "cleanup_cuda_cache": bool(self.caption_cfg.get("cleanup_cuda_cache", True)),
                "max_worker_fail_streak": int(self.caption_cfg.get("max_worker_fail_streak", 10) or 10),
                "max_new_tokens": int(self.caption_cfg.get("max_new_tokens", 50) or 50),
            }
            ParallelExecutor.run_gpu_parallel(_vlm_worker, images, config_dict=worker_config)
            worker_reports = self._load_worker_reports()
            enriched_captions = self._load_worker_results()

        with open(enriched_file, "w", encoding="utf-8") as f:
            json.dump(enriched_captions, f, indent=2)

        spatial_captions = self._build_spatial_captions(enriched_captions)
        with open(self.output_dir / "captions_spatial.json", "w", encoding="utf-8") as f:
            json.dump(spatial_captions, f, indent=2)

        spatial_clean_captions = self._run_spatial_regeneration_loop(images, spatial_captions)

        self.generate_stats(raw_captions, enriched_captions)
        self._generate_quality_report(raw_captions, enriched_captions, len(images))
        self._generate_quality_charts(raw_captions, enriched_captions)
        self._run_spatial_purity_audit(spatial_clean_captions, len(images))
        self._run_grounding_validation(raw_captions, enriched_captions, spatial_clean_captions)

        captions_dir = self.filtered_dir / "captions"
        captions_dir.mkdir(parents=True, exist_ok=True)
        for img_name, caption in enriched_captions.items():
            normalized = _normalize_caption_text(caption)
            if normalized:
                stem = Path(img_name).stem
                (captions_dir / f"{stem}.txt").write_text(normalized, encoding="utf-8")

        report = self._build_generation_report(images, raw_captions, enriched_captions, worker_reports)
        self._enforce_failure_policy(report)

        self.logger.info("=" * 20 + " STAGE 06 COMPLETED " + "=" * 20 + "\n")

    def _generate_quality_report(self, raw_captions, enriched_captions, total_images: int):
        import random

        all_keys = sorted(raw_captions.keys())
        sample_keys = all_keys[:50] if len(all_keys) <= 50 else random.sample(all_keys, 50)

        samples = []
        empty_count = 0
        short_count = 0
        for key in sample_keys:
            raw = _normalize_caption_text(raw_captions.get(key, ""))
            enriched = _normalize_caption_text(enriched_captions.get(key, ""))
            if not enriched:
                empty_count += 1
            elif len(enriched.split()) < 5:
                short_count += 1
            samples.append(
                {
                    "image": key,
                    "raw_caption": raw[:200],
                    "enriched_caption": enriched[:300],
                    "enriched_word_count": len(enriched.split()) if enriched else 0,
                }
            )

        report = {
            "total_images": total_images,
            "successful_enriched_captions": len(enriched_captions),
            "empty_captions": total_images - len(enriched_captions),
            "short_captions_lt5_words": sum(
                1 for v in enriched_captions.values() if len(_normalize_caption_text(v).split()) < 5
            ),
            "sample_count": len(samples),
            "samples": samples,
        }

        with open(self.output_dir / "caption_quality_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    def _generate_quality_charts(self, raw_captions, enriched_captions):
        try:
            pottery_terms = [
                "amphora",
                "krater",
                "kylix",
                "lekythos",
                "hydria",
                "oinochoe",
                "pelike",
                "skyphos",
                "kantharos",
                "vessel",
                "vase",
                "pottery",
                "jar",
            ]
            period_terms = ["geometric", "archaic", "classical", "hellenistic", "red-figure", "black-figure", "white-ground"]
            iconography_terms = [
                "warrior",
                "deity",
                "god",
                "goddess",
                "symposium",
                "banquet",
                "athlete",
                "hero",
                "dionysus",
                "athena",
                "heracles",
                "scene",
                "figure",
                "motif",
                "decoration",
            ]

            normalized_caps = [_normalize_caption_text(cap).lower() for cap in enriched_captions.values()]
            normalized_caps = [cap for cap in normalized_caps if cap]
            total = max(1, len(normalized_caps))
            pottery_pct = (sum(1 for cap in normalized_caps if any(term in cap for term in pottery_terms)) / total) * 100
            period_pct = (sum(1 for cap in normalized_caps if any(term in cap for term in period_terms)) / total) * 100
            icon_pct = (sum(1 for cap in normalized_caps if any(term in cap for term in iconography_terms)) / total) * 100

            fig1, ax1 = plt.subplots(figsize=(8, 6))
            categories = ["Pottery\nTerminology", "Period/Style\nMentions", "Iconography\nDescriptions"]
            percentages = [pottery_pct, period_pct, icon_pct]
            colors = ["#2ca02c", "#1f77b4", "#ff7f0e"]

            bars = ax1.bar(categories, percentages, color=colors, alpha=0.8, edgecolor="black", linewidth=1.5)
            for bar, pct in zip(bars, percentages):
                height = bar.get_height()
                ax1.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 1,
                    f"{pct:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=12,
                    fontweight="bold",
                )

            ax1.set_ylabel("Coverage (%)", fontsize=13, fontweight="bold")
            ax1.set_title("Domain Vocabulary in Enriched Captions", fontsize=14, fontweight="bold", pad=20)
            ax1.set_ylim(0, 105)
            ax1.axhline(y=90, color="red", linestyle="--", linewidth=1, alpha=0.5, label="90% threshold")
            ax1.legend(loc="lower right")
            ax1.grid(axis="y", alpha=0.3)

            textstr = f"n = {len(normalized_caps):,} captions\nBLIP2 3-pass template"
            props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
            ax1.text(0.03, 0.97, textstr, transform=ax1.transAxes, fontsize=10, verticalalignment="top", bbox=props)

            plt.tight_layout()
            plt.savefig(self.output_dir / "caption_quality_vocabulary.png", dpi=300, bbox_inches="tight")
            plt.close()

            raw_lengths = self._caption_lengths(raw_captions)
            enr_lengths = self._caption_lengths(enriched_captions)
            if not raw_lengths or not enr_lengths:
                return

            raw_mean, raw_std = np.mean(raw_lengths), np.std(raw_lengths)
            enr_mean, enr_std = np.mean(enr_lengths), np.std(enr_lengths)
            raw_min, raw_max = min(raw_lengths), max(raw_lengths)
            enr_min, enr_max = min(enr_lengths), max(enr_lengths)

            fig2, ax2 = plt.subplots(figsize=(8, 6))
            x = ["Raw\n(Metadata)", "Enriched\n(BLIP2)"]
            means = [raw_mean, enr_mean]
            stds = [raw_std, enr_std]
            colors2 = ["#1f77b4", "#2ca02c"]

            bars2 = ax2.bar(
                x,
                means,
                yerr=stds,
                capsize=10,
                color=colors2,
                alpha=0.8,
                edgecolor="black",
                linewidth=1.5,
                error_kw={"linewidth": 2.5, "ecolor": "black"},
            )

            for bar, mean, std in zip(bars2, means, stds):
                height = bar.get_height()
                ax2.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + std + 5,
                    f"{mean:.1f} ± {std:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=12,
                    fontweight="bold",
                )

            ax2.set_ylabel("Caption Length (words)", fontsize=13, fontweight="bold")
            ax2.set_title("Caption Length Stabilization", fontsize=14, fontweight="bold", pad=20)
            ax2.set_ylim(0, max(means) + max(stds) + 20)
            ax2.grid(axis="y", alpha=0.3)

            reduction = ((1 - enr_std / raw_std) * 100) if raw_std else 0.0
            textstr = (
                f"σ reduction: {reduction:.1f}%\n({raw_std:.1f} → {enr_std:.1f} words)\n\n"
                f"Range:\nRaw: {int(raw_min)}–{int(raw_max)}\nEnriched: {int(enr_min)}–{int(enr_max)}"
            )
            props = dict(boxstyle="round", facecolor="lightgreen", alpha=0.6, edgecolor="black", linewidth=1.5)
            ax2.text(
                0.97,
                0.97,
                textstr,
                transform=ax2.transAxes,
                fontsize=10,
                verticalalignment="top",
                horizontalalignment="right",
                bbox=props,
                fontweight="bold",
            )

            plt.tight_layout()
            plt.savefig(self.output_dir / "caption_length_stability.png", dpi=300, bbox_inches="tight")
            plt.close()

            self.logger.info("Caption quality charts generated successfully.")
        except Exception as e:
            self.logger.warning(f"Failed to generate quality charts: {e}")


if __name__ == "__main__":
    from thesis_pipeline.config_manager import ConfigManager

    cm = ConfigManager()
    stage = CaptionGenerationStage(cm)
    stage.run()
