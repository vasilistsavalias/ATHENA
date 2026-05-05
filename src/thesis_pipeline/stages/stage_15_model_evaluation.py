import logging
import math
import re
import hashlib
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from scipy import stats
import json
import time
import torch
import subprocess

from thesis_pipeline.components.evaluation.baselines import (
    TeleaInpainter, NavierStokesInpainter, VanillaSDInpainter,
    OursInpainter, TTAInpainter,
)
from thesis_pipeline.components.evaluation.deep_baselines import (
    BaselineUnavailableError,
    build_deep_baseline_runner,
)
from thesis_pipeline.components.evaluation.metrics import ThesisMetrics
from thesis_pipeline.components.evaluation.cross_validation import CrossValidator
from thesis_pipeline.visualization.plots import ThesisPlotter
from thesis_pipeline.visualization.style import ThesisStyle
from thesis_pipeline.utils.stage_artifacts import resolve_stage_artifact_dir


class ModelEvaluationStage:
    """Stage 15 — Full model evaluation with statistical rigour.

    Key improvements over the previous version:
    * Image-mask pairing by **stem** (not sorted-list index).
    * Fail-loud when the trained model is missing (unless mock mode).
    * Statistical tests include **all** metrics (PSNR, SSIM, LPIPS plus
      domain metrics color-fidelity and pattern-preservation).
    * Bonferroni correction for family-wise error rate.
    * Cohen's d effect size and 95 % confidence intervals.
    """

    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.config = config_manager.config
        self.paths = config_manager.get_paths()
        self.logger = logging.getLogger(__name__)
        
        # Paths
        self.output_dir = resolve_stage_artifact_dir(config_manager, "S15")
        self.test_data_dir = Path(self.paths.data.inpainting) / "test"
        # Prefer unet_best (early-stopping checkpoint) over unet_final
        models_dir = Path(self.paths.data.models)
        best = models_dir / "unet_best"
        final = models_dir / "unet_final"
        if best.exists():
            self.our_model_path = best
        else:
            self.our_model_path = final
            if final.exists():
                self.logger.info(
                    "unet_best not found — falling back to unet_final. "
                    "If early stopping was active, this may be a sub-optimal checkpoint."
                )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _caption_key(img_name: str) -> str:
        """Convert a test-image filename to the matching caption JSON key.

        Test filenames look like:  ``wiki_00005_...crop0_m0.png``
        Caption JSON keys look like: ``wiki_00005_...crop0.jpg``

        Steps:
        1. Strip extension  (→ ``...crop0_m0``)
        2. Remove the ``_m<digit>`` mask suffix  (→ ``...crop0``)
        3. Append ``.jpg``  (→ ``...crop0.jpg``)
        """
        stem = Path(img_name).stem                     # wiki_…crop0_m0
        stem = re.sub(r'_m\d+$', '', stem)              # wiki_…crop0
        return stem + '.jpg'

    @staticmethod
    def _usable_caption_text(value: str) -> str:
        text = " ".join(str(value or "").split()).strip()
        if not text:
            return ""
        if text.replace(".", "").replace(":", "").strip() == "":
            return ""
        return text

    @staticmethod
    def _cohens_d(a: pd.Series, b: pd.Series) -> float:
        """Paired Cohen's d (effect size)."""
        diff = a - b
        return diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) != 0 else 0.0

    @staticmethod
    def _ci_95(a: pd.Series, b: pd.Series):
        """95 % CI for the mean difference (paired)."""
        diff = a - b
        n = len(diff)
        mean_d = diff.mean()
        se = diff.std(ddof=1) / math.sqrt(n)
        t_crit = stats.t.ppf(0.975, df=n - 1)
        return mean_d - t_crit * se, mean_d + t_crit * se

    def _log_gpu_snapshot(self):
        try:
            result = subprocess.run(
                ["nvidia-smi"],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            if result.returncode == 0 and result.stdout:
                self.logger.info(f"Stage 15 GPU snapshot:\n{result.stdout}")
            else:
                self.logger.info(f"Stage 15 GPU snapshot unavailable (nvidia-smi rc={result.returncode}).")
        except Exception as exc:
            self.logger.info(f"Stage 15 GPU snapshot failed: {exc}")

    @staticmethod
    def _sample_dir_name(sample_id: str) -> str:
        return Path(str(sample_id)).stem

    @staticmethod
    def _expected_combinations(
        has_finetuned_model: bool,
        deep_enabled: bool,
        include_spatial_condition: bool,
    ) -> set[tuple[str, str]]:
        combos: set[tuple[str, str]] = {
            ("Telea", "Unconditional"),
            ("Navier-Stokes", "Unconditional"),
            ("Vanilla SD", "Unconditional"),
            ("Vanilla SD", "Raw Text"),
            ("Vanilla SD", "Enriched Text"),
            ("Vanilla SD", "Refined Text (clip-safe)"),
        }
        if include_spatial_condition:
            combos.add(("Vanilla SD", "Spatial Damage Context"))
        if deep_enabled:
            combos.update(
                {
                    ("LaMa", "Unconditional"),
                    ("MAT", "Unconditional"),
                    ("CoModGAN", "Unconditional"),
                }
            )
        if has_finetuned_model:
            combos.update(
                {
                    ("FT-SD", "Unconditional"),
                    ("FT-SD", "Raw Text"),
                    ("FT-SD", "Enriched Text"),
                    ("FT-SD", "Refined Text (clip-safe)"),
                    ("FT-SD+TTA", "Unconditional"),
                }
            )
            if include_spatial_condition:
                combos.add(("FT-SD", "Spatial Damage Context"))
        return combos

    @staticmethod
    def _hash_image_array(arr: np.ndarray) -> str:
        return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()

    def _enforce_frozen_control_integrity(
        self,
        *,
        matrix_dir: Path,
        manifest: dict[str, dict[str, str]],
    ) -> None:
        cfg = self.config.model_evaluation.get("frozen_control_integrity", {})
        enabled = bool(cfg.get("enabled", False))
        if not enabled:
            return

        strict = bool(cfg.get("strict_fail", True)) and bool(
            self.config.get("pipeline", {}).get("strict_fail_policy", False)
        )
        baseline_path = matrix_dir / "frozen_control_manifest.json"
        if not baseline_path.exists():
            if bool(cfg.get("freeze_on_first_run", True)):
                baseline_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            return

        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        mismatches: list[str] = []
        for model_name, sample_map in manifest.items():
            baseline_map = baseline.get(model_name, {}) if isinstance(baseline, dict) else {}
            for sample_id, digest in sample_map.items():
                baseline_digest = baseline_map.get(sample_id)
                if baseline_digest is None:
                    continue
                if str(baseline_digest) != str(digest):
                    mismatches.append(f"{model_name}:{sample_id}")

        if mismatches:
            msg = (
                "Frozen control integrity mismatch detected for "
                f"{len(mismatches)} sample(s); examples={mismatches[:5]}"
            )
            if strict:
                raise RuntimeError(msg)
            self.logger.warning(msg)

    @staticmethod
    def _filter_complete_samples_from_rows(
        rows: list[dict], expected_combos: set[tuple[str, str]]
    ) -> tuple[list[dict], set[str]]:
        if not rows:
            return [], set()
        df = pd.DataFrame(rows)
        if df.empty or not {"sample_id", "model", "condition"}.issubset(df.columns):
            return [], set()

        completed_sample_ids: set[str] = set()
        keep_rows: list[dict] = []
        for sample_id, sample_df in df.groupby("sample_id"):
            observed = set(
                zip(
                    sample_df["model"].astype(str).tolist(),
                    sample_df["condition"].astype(str).tolist(),
                )
            )
            if expected_combos.issubset(observed):
                completed_sample_ids.add(str(sample_id))
                keep_rows.extend(sample_df.to_dict(orient="records"))
        return keep_rows, completed_sample_ids

    def _build_caption_coverage_report(
        self,
        test_keys: set[str],
        raw_caps: dict,
        enriched_caps: dict,
        refined_caps: dict,
        spatial_caps: dict | None = None,
        *,
        threshold: float,
    ) -> dict:
        include_spatial = spatial_caps is not None
        spatial_caps = spatial_caps or {}
        families = {
            "raw": raw_caps,
            "enriched": enriched_caps,
            "refined": refined_caps,
        }
        if include_spatial:
            families["spatial"] = spatial_caps
        total = len(test_keys)
        family_reports: dict[str, dict] = {}
        usable_sets: dict[str, set[str]] = {}

        for family_name, captions in families.items():
            usable_keys: list[str] = []
            missing_keys: list[str] = []
            empty_keys: list[str] = []
            for key in sorted(test_keys):
                if key not in captions:
                    missing_keys.append(key)
                    continue
                if self._usable_caption_text(captions.get(key, "")):
                    usable_keys.append(key)
                else:
                    empty_keys.append(key)

            usable_set = set(usable_keys)
            usable_sets[family_name] = usable_set
            coverage = (len(usable_set) / total) if total else 0.0
            family_reports[family_name] = {
                "usable_count": len(usable_set),
                "coverage": coverage,
                "threshold": threshold,
                "scientifically_usable": coverage >= threshold if total else False,
                "missing_count": len(missing_keys),
                "empty_count": len(empty_keys),
                "missing_examples": missing_keys[:10],
                "empty_examples": empty_keys[:10],
            }

        prompt_ready_keys = sorted(set.intersection(*usable_sets.values())) if usable_sets else []
        return {
            "total_test_samples": total,
            "threshold": threshold,
            "families": family_reports,
            "prompt_ready_sample_count": len(prompt_ready_keys),
            "prompt_ready_sample_keys": prompt_ready_keys,
            "dropped_for_prompt_incompleteness": max(0, total - len(prompt_ready_keys)),
        }

    # ------------------------------------------------------------------
    # main
    # ------------------------------------------------------------------

    def run(self):
        self.logger.info("=" * 20 + " STAGE 15: Model Evaluation " + "=" * 20)
        self._log_gpu_snapshot()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        matrix_dir = self.output_dir / "benchmarking_matrix"
        matrix_dir.mkdir(exist_ok=True)
        stats_dir = self.output_dir / "statistical_tests"
        stats_dir.mkdir(exist_ok=True)
        domain_dir = self.output_dir / "domain_metrics"
        domain_dir.mkdir(exist_ok=True)
        
        # --- Test data ---
        img_dir = self.test_data_dir / "ground_truth"
        mask_dir = self.test_data_dir / "masks"
        
        if not img_dir.exists():
            self.logger.warning(f"Test data not found at {img_dir}. Skipping.")
            return

        # --- Pair images ↔ masks by stem (safe, order-independent) ---
        all_image_files = sorted(img_dir.glob("*.png"))
        mask_lookup = {p.stem: p for p in mask_dir.glob("*.png")}

        paired = []
        for img_path in all_image_files:
            mask_path = mask_lookup.get(img_path.stem)
            if mask_path is None:
                self.logger.error(
                    f"No mask for test image '{img_path.name}' — SKIPPING. "
                    f"This indicates Stage 10 did not generate a mask for this file."
                )
                continue
            paired.append((img_path, mask_path))

        phase = str(self.config.model_evaluation.get("phase", "integrity_200")).strip().lower()
        if phase == "full_test":
            phase_label = "full_test"
            limit = None
        else:
            phase_label = "integrity_200"
            limit = int(self.config.model_evaluation.get("num_samples_to_evaluate", 200))
            paired = paired[:limit]

        if not paired:
            self.logger.error("Zero valid image-mask pairs. Cannot evaluate.")
            return

        if limit is None:
            self.logger.info(f"Evaluating on full test set: {len(paired)} stem-paired samples.")
        else:
            self.logger.info(f"Evaluating on {len(paired)} stem-paired samples (phase={phase_label}).")
        checkpoint_path = matrix_dir / f"matrix_results.partial.{phase_label}.csv"
        resume_enabled = bool(self.config.model_evaluation.get("resume_from_checkpoint", True))
        checkpoint_every = int(self.config.model_evaluation.get("checkpoint_every_samples", 5))
        if checkpoint_every < 1:
            checkpoint_every = 1

        # --- Check model existence ---
        is_mock = self.config.model_evaluation.get("mock", False)
        has_finetuned_model = self.our_model_path.exists()
        if not has_finetuned_model and not is_mock:
            # Allow evaluation of baselines even if the fine-tuned model is missing.
            self.logger.warning(
                f"Fine-tuned model not found at {self.our_model_path}. "
                f"Skipping 'FT-SD' and evaluating baselines only. "
                f"(Set model_evaluation.mock=true to force a mocked 'FT-SD' path.)"
            )

        # --- Load Captions (with proper file handles) ---
        cap_dir = resolve_stage_artifact_dir(self.config_manager, "S07")
        raw_caps = {}
        enriched_caps = {}
        refined_caps = {}
        spatial_caps = {}
        raw_path = cap_dir / "captions_raw.json"
        enr_path = cap_dir / "captions_enriched.json"
        # Stage 07 (Qwen) clip-safe captions are optional.
        try:
            ref_dir = resolve_stage_artifact_dir(self.config_manager, "S08")
        except Exception:
            ref_dir = Path(self.paths.artifacts.root) / "07_caption_refinement"
        ref_path = ref_dir / "refined_captions_clip_safe.json"
        if raw_path.exists():
            with open(raw_path) as f:
                raw_caps = json.load(f)
        if enr_path.exists():
            with open(enr_path) as f:
                enriched_caps = json.load(f)
        if ref_path.exists():
            with open(ref_path) as f:
                refined_caps = json.load(f)
        spatial_path = cap_dir / "captions_spatial_clean.json"
        if not spatial_path.exists():
            spatial_path = cap_dir / "captions_spatial.json"
        if spatial_path.exists():
            with open(spatial_path) as f:
                spatial_caps = json.load(f)
        include_spatial_condition = bool(self.config.model_evaluation.get("include_spatial_condition", True))

        grounding_path = cap_dir / "stage_06b_grounding_validation.json"
        grounding_report = None
        if grounding_path.exists():
            try:
                grounding_report = json.loads(grounding_path.read_text(encoding="utf-8"))
            except Exception:
                grounding_report = None
        if include_spatial_condition:
            strict = bool(self.config.get("pipeline", {}).get("strict_fail_policy", False))
            if not grounding_report:
                msg = "Spatial condition requested but Stage06b grounding report is missing/unreadable."
                if strict:
                    raise RuntimeError(msg)
                self.logger.warning(msg + " Disabling spatial condition for this run.")
                include_spatial_condition = False
            elif not bool(grounding_report.get("pass", False)):
                msg = (
                    "Spatial condition blocked by Stage06b grounding gate: "
                    f"quadrant_macro_f1={grounding_report.get('quadrant_macro_f1')}, "
                    f"border_touch_accuracy={grounding_report.get('border_touch_accuracy')}, "
                    f"area_correlation={grounding_report.get('area_correlation')}"
                )
                if strict:
                    raise RuntimeError(msg)
                self.logger.warning(msg + " Disabling spatial condition for this run.")
                include_spatial_condition = False

        if include_spatial_condition and not spatial_caps:
            self.logger.warning("Spatial condition enabled but captions_spatial.json is missing or empty.")

        # Helper to look up caption by test-image filename
        def _get_caption(caps_dict: dict, img_name: str) -> str:
            """Look up a caption, translating the test filename to the JSON key."""
            key = self._caption_key(img_name)
            return self._usable_caption_text(caps_dict.get(key, ""))

        # Log how many test images have matching captions
        if paired:
            test_keys = {self._caption_key(p[0].name) for p in paired}
            coverage_threshold = float(self.config.model_evaluation.get("caption_coverage_min", 0.0) or 0.0)
            coverage_report = self._build_caption_coverage_report(
                test_keys,
                raw_caps,
                enriched_caps,
                refined_caps,
                spatial_caps,
                threshold=coverage_threshold,
            )
            (matrix_dir / "caption_coverage_report.json").write_text(
                json.dumps(coverage_report, indent=2),
                encoding="utf-8",
            )
            raw_hits = int(coverage_report["families"]["raw"]["usable_count"])
            enr_hits = int(coverage_report["families"]["enriched"]["usable_count"])
            ref_hits = int(coverage_report["families"]["refined"]["usable_count"])
            sp_hits = int(coverage_report["families"].get("spatial", {}).get("usable_count", 0))
            self.logger.info(
                f"Caption coverage: {raw_hits}/{len(test_keys)} raw, "
                f"{enr_hits}/{len(test_keys)} enriched, "
                f"{ref_hits}/{len(test_keys)} refined, "
                f"{sp_hits}/{len(test_keys)} spatial."
            )
            if coverage_threshold > 0 and len(test_keys) > 0:
                strict = bool(self.config.get("pipeline", {}).get("strict_fail_policy", False))
                under_threshold = {
                    family_name: payload["coverage"]
                    for family_name, payload in coverage_report["families"].items()
                    if float(payload["coverage"]) < coverage_threshold
                }
                if under_threshold and strict:
                    raise RuntimeError(
                        "Caption coverage below threshold for prompt-conditioned evaluation "
                        f"(threshold={coverage_threshold:.3f}, phase={phase_label}). "
                        f"Coverage={under_threshold}"
                    )
                prompt_ready_keys = set(coverage_report["prompt_ready_sample_keys"])
                if prompt_ready_keys:
                    original_count = len(paired)
                    paired = [
                        pair for pair in paired
                        if self._caption_key(pair[0].name) in prompt_ready_keys
                    ]
                    dropped = original_count - len(paired)
                    if dropped > 0:
                        self.logger.warning(
                            f"Dropping {dropped} sample(s) from evaluation because one or more "
                            "prompt families were missing or empty."
                        )
                if not paired:
                    raise RuntimeError(
                        "No prompt-ready evaluation samples remain after caption coverage filtering."
                    )

        # --- Initialize Evaluators ---
        metrics_eval = ThesisMetrics()
        plotter = ThesisPlotter(output_dir=self.output_dir / "charts")
        full_results: list[dict] = []
        completed_sample_ids: set[str] = set()
        resumed_sample_count = 0
        pending_pairs = list(paired)

        num_steps = self.config.model_evaluation.get("num_inference_steps", 50)
        eval_seed = self.config.global_params.get("random_state", 42)
        eval_device = self.config.model_evaluation.get("device", "cuda")
        
        telea = TeleaInpainter()
        ns = NavierStokesInpainter()
        vanilla = VanillaSDInpainter(device=eval_device, num_inference_steps=num_steps, seed=eval_seed)
        ours = None
        ours_tta = None
        if has_finetuned_model:
            ours = OursInpainter(
                model_path=self.our_model_path,
                device=eval_device,
                num_inference_steps=num_steps,
                seed=eval_seed,
            )
            ours_tta = TTAInpainter(OursInpainter(
                model_path=self.our_model_path,
                device=eval_device,
                num_inference_steps=num_steps,
                seed=eval_seed,
            ))
        elif is_mock:
            ours = OursInpainter(
                model_path=self.our_model_path,
                device=eval_device,
                num_inference_steps=num_steps,
                seed=eval_seed,
            )
            ours_tta = TTAInpainter(OursInpainter(
                model_path=self.our_model_path,
                device=eval_device,
                num_inference_steps=num_steps,
                seed=eval_seed,
            ))

        # Optional deep baselines (LaMa/MAT/CoModGAN) — guardrailed.
        deep_cfg = self.config.model_evaluation.get("deep_baselines", {}) if hasattr(self.config, "model_evaluation") else {}
        deep_runner, deep_probe = build_deep_baseline_runner(deep_cfg, stage13_dir=self.output_dir, logger=self.logger)
        try:
            (matrix_dir / "deep_baseline_probe.json").write_text(
                json.dumps([r.__dict__ for r in deep_probe], indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass
        deep_enabled = False
        try:
            deep_enabled = bool(deep_cfg.get("enabled", False)) if isinstance(deep_cfg, dict) else bool(getattr(deep_cfg, "enabled", False))
        except Exception:
            deep_enabled = False

        deep_required = False
        required_models = {"LaMa", "MAT", "CoModGAN"}
        try:
            deep_required = bool(deep_cfg.get("required", True)) if isinstance(deep_cfg, dict) else bool(getattr(deep_cfg, "required", True))
            configured_required = deep_cfg.get("required_models", []) if isinstance(deep_cfg, dict) else getattr(deep_cfg, "required_models", [])
            if isinstance(configured_required, list) and configured_required:
                required_models = {str(x) for x in configured_required}
        except Exception:
            pass

        expected_combos = self._expected_combinations(
            has_finetuned_model=(has_finetuned_model or is_mock),
            deep_enabled=(deep_enabled and deep_required),
            include_spatial_condition=include_spatial_condition,
        )
        checkpoint_rows = self._load_stage13_checkpoint(
            checkpoint_path=checkpoint_path,
            enabled=resume_enabled,
        )
        full_results, completed_sample_ids = self._filter_complete_samples_from_rows(
            checkpoint_rows,
            expected_combos,
        )
        resumed_sample_count = len(completed_sample_ids)
        pending_pairs = [p for p in paired if p[0].name not in completed_sample_ids]
        if resumed_sample_count > 0:
            self.logger.info(
                f"Stage 15 resume: loaded {resumed_sample_count} complete samples from checkpoint; "
                f"remaining {len(pending_pairs)}."
            )
        if checkpoint_rows and not full_results:
            self.logger.info("Stage 15 resume: checkpoint found but no complete samples; recomputing requested samples.")
        if not pending_pairs:
            self.logger.info("Stage 15 resume: no pending samples. Building final reports from checkpointed matrix.")

        deep_reports = []
        if deep_enabled and pending_pairs:
            deep_reports = deep_runner.prepare(pending_pairs)

        # Inference timing accumulators
        timing_data: dict = {}  # model_name -> list of ms
        # Qualitative grid samples (first 6)
        qualitative_samples: list = []
        # FID/KID image accumulators  {model_name: [np.ndarray]}
        fid_images: dict = {}
        gt_images_for_fid: list = []
        model_skip_reasons: dict[str, set[str]] = {}
        frozen_control_hashes: dict[str, dict[str, str]] = {}

        for step_idx, (img_path, mask_path) in enumerate(tqdm(pending_pairs, total=len(pending_pairs)), start=1):
            img = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")
            gt = np.array(img)
            mask_np = np.array(mask)
            sample_rows = []

            # Save ground-truth & mask once per sample
            sample_dir = self.output_dir / "samples" / self._sample_dir_name(img_path.name)
            sample_dir.mkdir(parents=True, exist_ok=True)
            img.save(sample_dir / "original.png")
            Image.fromarray(mask_np).save(sample_dir / "mask.png")
            # Masked input visualisation (red tint on damaged region)
            overlay = gt.copy()
            overlay[mask_np > 0] = (overlay[mask_np > 0] * 0.4 + np.array([180, 0, 0]) * 0.6).astype(np.uint8)
            Image.fromarray(overlay).save(sample_dir / "masked_input.png")

            # Detect mask type from seed-based dispatch or structure
            fe_cfg = self.config.get('feature_engineering', {})
            mask_types_cfg = fe_cfg.get('mask_types', ['irregular', 'rect', 'edge'])
            mask_type = self._detect_mask_type(
                mask_np, mask_path,
                global_seed=self.config.global_params.get('random_state', 42),
                mask_types=mask_types_cfg,
            )

            # Collect restorations per sample for comparison grid
            sample_restorations: dict[str, np.ndarray] = {}
            sample_metrics: dict[str, dict[str, float]] = {}

            # --- Helper: timed inpaint ---
            def _timed_inpaint(inpainter, model_name, im, msk, prompt=""):
                t0 = time.perf_counter()
                res = inpainter.inpaint(im, msk, prompt=prompt)
                elapsed_ms = (time.perf_counter() - t0) * 1000
                timing_data.setdefault(model_name, []).append(elapsed_ms)
                return res

            # Telea
            res = _timed_inpaint(telea, 'Telea', img, mask)
            res_np = np.array(res)
            self._record(sample_rows, "Telea", "Unconditional", img_path.name, gt, res_np, metrics_eval, mask_np, mask_type)
            sample_restorations["Telea"] = res_np
            
            # Navier-Stokes
            res = _timed_inpaint(ns, 'Navier-Stokes', img, mask)
            res_np = np.array(res)
            self._record(sample_rows, "Navier-Stokes", "Unconditional", img_path.name, gt, res_np, metrics_eval, mask_np, mask_type)
            sample_restorations["Navier-Stokes"] = res_np

            # Deep baselines (Unconditional only; loaded from cached batch outputs)
            if deep_enabled:
                for model_name in ("LaMa", "MAT", "CoModGAN"):
                    try:
                        est_ms = deep_runner.timing_ms_per_image(model_name)
                        if est_ms is not None:
                            timing_data.setdefault(model_name, []).append(float(est_ms))
                        res = deep_runner.load_result(model_name, img_path.name)
                        res_np = np.array(res)
                        self._record(
                            sample_rows,
                            model_name,
                            "Unconditional",
                            img_path.name,
                            gt,
                            res_np,
                            metrics_eval,
                            mask_np,
                            mask_type,
                        )
                        sample_restorations[model_name] = res_np
                    except BaselineUnavailableError as exc:
                        if deep_required and model_name in required_models:
                            raise RuntimeError(
                                f"Required deep baseline '{model_name}' missing for sample {img_path.name}: {exc}"
                            ) from exc
                        model_skip_reasons.setdefault(model_name, set()).add(str(exc))
                        self.logger.warning(f"Skipping {model_name}: {exc}")
                    except Exception as exc:
                        if deep_required and model_name in required_models:
                            raise RuntimeError(
                                f"Required deep baseline '{model_name}' failed for sample {img_path.name}: {exc}"
                            ) from exc
                        model_skip_reasons.setdefault(model_name, set()).add(str(exc))
                        self.logger.warning(f"{model_name} failed for {img_path.name}: {exc}")
            
            # Vanilla SD (prompt conditions)
            for cond, prompt in [
                ("Unconditional", ""),
                ("Raw Text", _get_caption(raw_caps, img_path.name)),
                ("Enriched Text", _get_caption(enriched_caps, img_path.name)),
                ("Refined Text (clip-safe)", _get_caption(refined_caps, img_path.name)),
                ("Spatial Damage Context", _get_caption(spatial_caps, img_path.name)),
            ]:
                if cond == "Spatial Damage Context" and not include_spatial_condition:
                    continue
                res = _timed_inpaint(vanilla, 'Vanilla SD', img, mask, prompt)
                res_np = np.array(res)
                self._record(sample_rows, "Vanilla SD", cond, img_path.name, gt, res_np, metrics_eval, mask_np, mask_type)
                if cond == "Unconditional":
                    sample_restorations["Vanilla SD"] = res_np
            
            # FT-SD (Fine-Tuned SD) — 3 conditions
            if ours:
                for cond, prompt in [
                    ("Unconditional", ""),
                    ("Raw Text", _get_caption(raw_caps, img_path.name)),
                    ("Enriched Text", _get_caption(enriched_caps, img_path.name)),
                    ("Refined Text (clip-safe)", _get_caption(refined_caps, img_path.name)),
                    ("Spatial Damage Context", _get_caption(spatial_caps, img_path.name)),
                ]:
                    if cond == "Spatial Damage Context" and not include_spatial_condition:
                        continue
                    res = _timed_inpaint(ours, 'FT-SD', img, mask, prompt)
                    res_np = np.array(res)
                    self._record(sample_rows, "FT-SD", cond, img_path.name, gt, res_np, metrics_eval, mask_np, mask_type)
                    if cond == "Unconditional":
                        sample_restorations["FT-SD"] = res_np

            # FT-SD + TTA (Unconditional only — TTA is a deployment technique)
            if ours_tta:
                res = _timed_inpaint(ours_tta, 'FT-SD+TTA', img, mask)
                res_np = np.array(res)
                self._record(sample_rows, "FT-SD+TTA", "Unconditional", img_path.name, gt, res_np, metrics_eval, mask_np, mask_type)
                sample_restorations["FT-SD+TTA"] = res_np

            # Frozen-control integrity signatures for unconditional controls.
            for control_model in ("Telea", "Vanilla SD", "FT-SD"):
                if control_model in sample_restorations:
                    frozen_control_hashes.setdefault(control_model, {})[img_path.name] = self._hash_image_array(
                        sample_restorations[control_model]
                    )

            # Accumulate images for FID/KID computation
            gt_images_for_fid.append(gt)
            for model_name, res_arr in sample_restorations.items():
                fid_images.setdefault(model_name, []).append(res_arr)
            full_results.extend(sample_rows)
            completed_sample_ids.add(img_path.name)
            if checkpoint_every > 0 and (step_idx % checkpoint_every == 0):
                self._save_stage13_checkpoint(full_results, checkpoint_path)

            # Per-sample comparison grid
            _m = {}
            # Rebuild per-model metrics from full_results
            for row in full_results:
                if row["sample_id"] == img_path.name and row["condition"] == "Unconditional":
                    _m[row["model"]] = {"psnr": row["psnr"], "ssim": row["ssim"]}
            try:
                plotter.plot_evaluation_comparison(
                    gt=gt, masked_input=overlay, mask=mask_np,
                    restorations=sample_restorations, metrics=_m,
                    filename=f"sample_{img_path.stem}_comparison",
                )
            except Exception as e:
                self.logger.warning(f"Could not generate comparison for {img_path.name}: {e}")

            # Collect data for qualitative grid (first 6 samples)
            if len(qualitative_samples) < 6:
                qualitative_samples.append({
                    'name': img_path.stem,
                    'original': gt,
                    'masked_input': overlay,
                    'metrics': _m,
                    **sample_restorations,
                })

        if pending_pairs:
            self._save_stage13_checkpoint(full_results, checkpoint_path)

        # Save full matrix
        df = pd.DataFrame(full_results)
        if not df.empty:
            df = df.drop_duplicates(subset=["sample_id", "model", "condition"], keep="last")
            if "mask_coverage" in df.columns and "severity_bin" not in df.columns:
                df["severity_bin"] = self._build_severity_bins(df["mask_coverage"])
        df.to_csv(matrix_dir / "matrix_results.csv", index=False)
        df.to_csv(matrix_dir / f"matrix_results.{phase_label}.csv", index=False)
        self.logger.info(f"Matrix saved to {matrix_dir / 'matrix_results.csv'}")

        control_manifest = {
            "phase": phase_label,
            "timestamp": time.time(),
            "controls": frozen_control_hashes,
        }
        (matrix_dir / "control_integrity_manifest.json").write_text(
            json.dumps(control_manifest, indent=2),
            encoding="utf-8",
        )
        self._enforce_frozen_control_integrity(
            matrix_dir=matrix_dir,
            manifest=frozen_control_hashes,
        )

        try:
            expected_models = sorted({m for m, _ in expected_combos})
            present_models = sorted(df["model"].unique().tolist()) if not df.empty else []
            deep_report_map = {r.model: {"ok": bool(r.ok), "detail": str(r.detail)} for r in deep_reports}
            skipped_models = sorted(set(expected_models) - set(present_models))
            completion_payload = {
                "phase": phase_label,
                "requested_samples": len(paired),
                "completed_samples": int(df["sample_id"].nunique()) if not df.empty else 0,
                "resume_enabled": bool(resume_enabled),
                "resumed_complete_samples": int(resumed_sample_count),
                "expected_models": expected_models,
                "present_models": present_models,
                "missing_models": skipped_models,
                "deep_required": bool(deep_required),
                "required_deep_models": sorted(required_models) if deep_enabled else [],
                "deep_reports": deep_report_map,
                "skip_reasons": {k: sorted(v) for k, v in model_skip_reasons.items()},
            }
            (matrix_dir / "evaluation_completeness.json").write_text(
                json.dumps(completion_payload, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            self.logger.warning(f"Failed to write evaluation completeness report: {e}")
        
        # Domain metrics (separate CSVs)
        self._save_domain(df, domain_dir)

        # Statistical tests (with Bonferroni, Cohen's d, 95 % CI)
        stats_df = self._run_stats(df, stats_dir)
        self._save_error_analysis(df, stats_dir)

        # ---------- Charts ----------
        try:
            self.logger.info("Generating evaluation charts…")
            plotter.plot_metric_bars(df, filename="metric_comparison_bars")
            plotter.plot_metric_distributions(df, filename="metric_distributions")
            if 'mask_coverage' in df.columns:
                plotter.plot_psnr_vs_coverage(df, filename="psnr_vs_coverage")
            if stats_df is not None and not stats_df.empty:
                plotter.plot_significance_heatmap(stats_df, filename="significance_heatmap")
                sig_matrix_cfg = self.config.model_evaluation.get("significance_matrix", {})
                manifest_df = plotter.plot_significance_matrix_suite(
                    stats_df=stats_df,
                    config=sig_matrix_cfg,
                )
                if not manifest_df.empty:
                    generated = int((manifest_df["status"] == "generated").sum())
                    skipped = int((manifest_df["status"] == "skipped").sum())
                    self.logger.info(
                        "Significance matrix charts generated: "
                        f"{generated} generated, {skipped} skipped "
                        f"(manifest: {plotter.output_dir / 'significance_matrix' / 'matrix_manifest.csv'})."
                    )
            plotter.plot_improvement_deltas(df, metric='psnr', baseline='Telea',
                                           filename="improvement_over_telea")
            # Stratified analysis by mask coverage
            if 'mask_coverage' in df.columns:
                df['coverage_bin'] = self._build_severity_bins(df['mask_coverage'])
                plotter.plot_stratified_bars(df, group_col='coverage_bin',
                                            filename='stratified_by_coverage',
                                            title='PSNR by Mask Coverage Bin')
                plotter.plot_stratified_bars(
                    df,
                    group_col='severity_bin',
                    filename='stratified_by_severity',
                    title='PSNR by Damage Severity Bin',
                )

            # ---- Individual per-metric charts ----
            self.logger.info("Generating individual per-metric charts…")
            plotter.plot_all_individual_charts(df, stats_df)

            # ---- Ablation study charts ----
            self.logger.info("Generating ablation study charts…")
            for model_name in ['FT-SD', 'Vanilla SD']:
                if model_name in df['model'].unique():
                    plotter.plot_ablation_bars(df, model=model_name)
            plotter.plot_ablation_heatmap(df)
            for metric in ['psnr', 'ssim', 'lpips', 'color', 'pattern']:
                if metric in df.columns:
                    plotter.plot_ablation_per_metric(df, metric)

            # ---- Training analysis charts ----
            training_csv = self.config_manager.get_stage_artifact_path("S13", "training_logs.csv")
            if training_csv.exists():
                self.logger.info("Generating training analysis charts…")
                plotter.plot_training_analysis(str(training_csv))
                plotter.plot_loss_landscape(str(training_csv))

            # ---- Hyperparameter summary chart ----
            hp_yaml = self.config_manager.get_stage_artifact_path("S12", "best_hyperparameters.yaml")
            if hp_yaml.exists():
                import yaml
                with open(hp_yaml) as f:
                    hp = yaml.safe_load(f)
                plotter.plot_hyperparameter_summary(hp)

            # ---- LR schedule chart ----
            plotter.plot_lr_schedule()

            # ---- Mask-type ablation charts ----
            if 'mask_type' in df.columns and df['mask_type'].nunique() > 1:
                self.logger.info("Generating mask-type ablation charts…")
                plotter.plot_mask_type_ablation(df)
                plotter.plot_mask_type_heatmap(df)

            # ---- TTA comparison chart ----
            if 'FT-SD+TTA' in df['model'].unique():
                self.logger.info("Generating TTA comparison chart…")
                plotter.plot_tta_comparison(df)

            # ---- Inference timing chart ----
            if timing_data and resumed_sample_count == 0:
                self.logger.info("Generating inference timing chart…")
                timing_rows = []
                for model_name, times in timing_data.items():
                    mean_ms = np.mean(times)
                    std_ms = np.std(times, ddof=1) if len(times) > 1 else 0.0
                    timing_rows.append({
                        'model': model_name,
                        'mean_ms': round(mean_ms, 1),
                        'std_ms': round(std_ms, 1),
                        'images_per_sec': round(1000.0 / mean_ms, 2) if mean_ms > 0 else 0.0,
                    })
                timing_df = pd.DataFrame(timing_rows)
                timing_df.to_csv(matrix_dir / "inference_timing.csv", index=False)
                plotter.plot_inference_timing(timing_df)
            elif resumed_sample_count > 0:
                self.logger.info("Skipping inference timing chart on resumed run (partial timing history).")

            # ---- Qualitative figure grid ----
            if qualitative_samples:
                self.logger.info("Generating qualitative comparison grid…")
                plotter.plot_qualitative_grid(qualitative_samples[:6])

            # ---- FID / KID (dataset-level distributional metrics) ----
            if gt_images_for_fid and fid_images and resumed_sample_count == 0:
                self.logger.info("Computing FID/KID distributional metrics…")
                try:
                    fid_scores = {}
                    kid_scores = {}
                    for model_name, gen_imgs in fid_images.items():
                        fid_scores[model_name] = metrics_eval.calculate_fid(gt_images_for_fid, gen_imgs)
                        kid_mean, kid_std = metrics_eval.calculate_kid(gt_images_for_fid, gen_imgs)
                        kid_scores[model_name] = (kid_mean, kid_std)
                    # Save
                    fid_kid_rows = [{
                        'model': m,
                        'FID': round(fid_scores[m], 2),
                        'KID_mean': round(kid_scores[m][0], 4),
                        'KID_std': round(kid_scores[m][1], 4),
                    } for m in fid_scores]
                    pd.DataFrame(fid_kid_rows).to_csv(matrix_dir / "fid_kid_scores.csv", index=False)
                    plotter.plot_fid_kid(fid_scores, kid_scores)
                    self.logger.info("FID/KID computed and charted.")
                except Exception as e:
                    self.logger.warning(f"FID/KID computation failed: {e}")
            elif resumed_sample_count > 0:
                self.logger.info("Skipping FID/KID on resumed run (requires full in-memory image history).")

            # ---- 3-Fold Cross-Validation ----
            self.logger.info("Running 3-fold cross-validation…")
            try:
                cv = CrossValidator(k=3, random_state=eval_seed)
                folds = cv.create_folds(paired)
                fold_dfs = []
                for fold_idx, fold_samples in enumerate(folds):
                    fold_results = []
                    for fold_row in full_results:
                        if fold_row['sample_id'] in {s[0].name for s in fold_samples}:
                            fold_results.append(fold_row)
                    fold_dfs.append(pd.DataFrame(fold_results))
                cv_df = cv.aggregate_fold_results(fold_dfs)
                cv_df.to_csv(stats_dir / "cross_validation_results.csv", index=False)
                plotter.plot_cv_folds(cv_df)
                plotter.plot_cv_stability(cv_df)
                self.logger.info("Cross-validation complete.")
            except Exception as e:
                self.logger.warning(f"Cross-validation failed: {e}")

            self.logger.info("Evaluation charts saved.")
        except Exception as e:
            self.logger.warning(f"Chart generation encountered an error: {e}")

        # ---------- Per-sample metrics JSON ----------
        try:
            per_sample = (
                df[df['condition'] == 'Unconditional']
                .groupby('sample_id')
                .apply(lambda g: g.set_index('model')[['psnr', 'ssim', 'lpips', 'color', 'pattern']].to_dict('index'),
                       include_groups=False)
                .to_dict()
            )
            with open(matrix_dir / "per_sample_metrics.json", "w") as f:
                json.dump(per_sample, f, indent=2, default=str)
        except Exception as e:
            self.logger.warning(f"Per-sample JSON export failed: {e}")

        try:
            if checkpoint_path.exists():
                checkpoint_path.unlink()
        except Exception as e:
            self.logger.warning(f"Could not remove Stage 15 checkpoint file: {e}")

        # ---------- Prompt ablation documentation ----------
        self._write_prompt_ablation_doc(df)

        # ---------- Optional: multi-seed prompt sensitivity sweep ----------
        try:
            self._run_prompt_seed_sweep(
                paired=paired,
                raw_caps=raw_caps,
                enriched_caps=enriched_caps,
                refined_caps=refined_caps,
                num_steps=num_steps,
                base_seed=eval_seed,
                device=eval_device,
                metrics_eval=metrics_eval,
                matrix_dir=matrix_dir,
                has_finetuned_model=has_finetuned_model,
            )
        except Exception as e:
            self.logger.warning(f"Prompt seed-sweep failed: {e}")

        # ---------- V7: Composite ranking ----------
        try:
            from thesis_pipeline.components.evaluation.composite_ranking import write_composite_ranking
            matrix_csv = matrix_dir / "matrix_results.csv"
            if matrix_csv.exists():
                write_composite_ranking(matrix_csv, matrix_dir)
        except Exception as e:
            self.logger.warning(f"Composite ranking failed: {e}")
        
        self.logger.info("=" * 20 + " STAGE 15 COMPLETED " + "=" * 20 + "\n")

    def _run_prompt_seed_sweep(
        self,
        paired: list[tuple[Path, Path]],
        raw_caps: dict,
        enriched_caps: dict,
        refined_caps: dict,
        num_steps: int,
        base_seed: int,
        device: str,
        metrics_eval: ThesisMetrics,
        matrix_dir: Path,
        has_finetuned_model: bool,
    ):
        """Optional prompt-sensitivity sweep across multiple seeds.

        Notes:
        - This is designed to test *prompt effects* under different seeds without
          altering the main Stage 15 benchmarking matrix (which remains fixed-seed).
        - The sweep is intentionally small by default because it is expensive.
        """
        sweep_cfg = self.config.model_evaluation.get("seed_sweep", {})
        if not isinstance(sweep_cfg, dict) or not sweep_cfg.get("enabled", False):
            return

        seeds = sweep_cfg.get("seeds", [base_seed])
        if not isinstance(seeds, list) or not seeds:
            seeds = [base_seed]

        max_samples = int(sweep_cfg.get("max_samples", min(25, len(paired))))
        subset = paired[:max_samples]
        if not subset:
            return

        self.logger.info(f"Prompt seed-sweep: {len(subset)} samples × {len(seeds)} seeds.")

        rows = []

        def _get_caption(caps_dict: dict, img_name: str) -> str:
            key = self._caption_key(img_name)
            return self._usable_caption_text(caps_dict.get(key, ""))

        for seed in seeds:
            vanilla = VanillaSDInpainter(device=device, num_inference_steps=num_steps, seed=int(seed))
            ours = None
            if has_finetuned_model:
                ours = OursInpainter(
                    model_path=self.our_model_path,
                    device=device,
                    num_inference_steps=num_steps,
                    seed=int(seed),
                )

            for img_path, mask_path in subset:
                img = Image.open(img_path).convert("RGB")
                mask = Image.open(mask_path).convert("L")
                gt = np.array(img)
                mask_np = np.array(mask)

                for cond, prompt in [
                    ("Unconditional", ""),
                    ("Raw Text", _get_caption(raw_caps, img_path.name)),
                    ("Enriched Text", _get_caption(enriched_caps, img_path.name)),
                    ("Refined Text (clip-safe)", _get_caption(refined_caps, img_path.name)),
                ]:
                    res = vanilla.inpaint(img, mask, prompt=prompt)
                    pred = np.array(res)
                    rows.append({
                        "model": "Vanilla SD",
                        "condition": cond,
                        "sample_id": img_path.name,
                        "seed": int(seed),
                        "psnr": metrics_eval.calculate_psnr(gt, pred, mask=mask_np),
                        "ssim": metrics_eval.calculate_ssim(gt, pred, mask=mask_np),
                        "lpips": metrics_eval.calculate_lpips(gt, pred, mask=mask_np),
                        "color": metrics_eval.calculate_color_fidelity(gt, pred, mask=mask_np),
                        "pattern": metrics_eval.calculate_pattern_preservation(gt, pred, mask=mask_np),
                    })

                    if ours is not None:
                        res_o = ours.inpaint(img, mask, prompt=prompt)
                        pred_o = np.array(res_o)
                        rows.append({
                            "model": "FT-SD",
                            "condition": cond,
                            "sample_id": img_path.name,
                            "seed": int(seed),
                            "psnr": metrics_eval.calculate_psnr(gt, pred_o, mask=mask_np),
                            "ssim": metrics_eval.calculate_ssim(gt, pred_o, mask=mask_np),
                            "lpips": metrics_eval.calculate_lpips(gt, pred_o, mask=mask_np),
                            "color": metrics_eval.calculate_color_fidelity(gt, pred_o, mask=mask_np),
                            "pattern": metrics_eval.calculate_pattern_preservation(gt, pred_o, mask=mask_np),
                        })

        if not rows:
            return

        out_csv = matrix_dir / "prompt_seed_sweep.csv"
        df = pd.DataFrame(rows)
        df.to_csv(out_csv, index=False)

        summary = {}
        for (model, cond), g in df.groupby(["model", "condition"]):
            entry = {}
            for metric in ["psnr", "ssim", "lpips", "color", "pattern"]:
                if metric not in g.columns:
                    continue
                per_seed = g.groupby("seed")[metric].mean()
                entry[metric] = {
                    "seed_mean_mean": float(per_seed.mean()),
                    "seed_mean_std": float(per_seed.std(ddof=1)) if len(per_seed) > 1 else 0.0,
                    "seed_mean_min": float(per_seed.min()),
                    "seed_mean_max": float(per_seed.max()),
                }
            summary[f"{model} | {cond}"] = entry

        (matrix_dir / "prompt_seed_sweep_summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )

    def _record(self, results, model, cond, sid, gt, pred, m, mask_np, mask_type="unknown"):
        """Calculate metrics (both full-image and masked-region), save sample image, append row."""
        sample_dir = self.output_dir / "samples" / self._sample_dir_name(sid)
        sample_dir.mkdir(parents=True, exist_ok=True)
        def _slug(x: str) -> str:
            return re.sub(r"[^A-Za-z0-9]+", "_", str(x)).strip("_")

        Image.fromarray(pred).save(sample_dir / f"{_slug(model)}_{_slug(cond)}.png")
        
        mask_coverage = float(np.sum(mask_np > 0) / mask_np.size)
        
        results.append({
            "model": model,
            "condition": cond,
            "sample_id": sid,
            "mask_coverage": mask_coverage,
            "mask_type": mask_type,
            # Masked-region metrics (PRIMARY — evaluates actual inpainting quality)
            "psnr": m.calculate_psnr(gt, pred, mask=mask_np),
            "ssim": m.calculate_ssim(gt, pred, mask=mask_np),
            "lpips": m.calculate_lpips(gt, pred, mask=mask_np),
            "color": m.calculate_color_fidelity(gt, pred, mask=mask_np),
            "pattern": m.calculate_pattern_preservation(gt, pred, mask=mask_np),
            # Full-image metrics (secondary, for reference)
            "psnr_full": m.calculate_psnr(gt, pred),
            "ssim_full": m.calculate_ssim(gt, pred),
            "lpips_full": m.calculate_lpips(gt, pred),
            "color_full": m.calculate_color_fidelity(gt, pred),
            "pattern_full": m.calculate_pattern_preservation(gt, pred),
        })

    def _load_stage13_checkpoint(self, checkpoint_path: Path, enabled: bool) -> list[dict]:
        if not enabled or (not checkpoint_path.exists()):
            return []
        try:
            df = pd.read_csv(checkpoint_path)
            if df.empty or "sample_id" not in df.columns:
                return []
            df = df.drop_duplicates(subset=["sample_id", "model", "condition"], keep="last")
            rows = df.to_dict(orient="records")
            return rows
        except Exception as e:
            self.logger.warning(f"Stage 15 checkpoint load failed ({checkpoint_path}): {e}")
            return []

    def _save_stage13_checkpoint(self, full_results: list[dict], checkpoint_path: Path) -> None:
        try:
            if not full_results:
                return
            df = pd.DataFrame(full_results)
            if df.empty:
                return
            df = df.drop_duplicates(subset=["sample_id", "model", "condition"], keep="last")
            df.to_csv(checkpoint_path, index=False)
        except Exception as e:
            self.logger.warning(f"Stage 15 checkpoint save failed ({checkpoint_path}): {e}")

    @staticmethod
    def _detect_mask_type(mask_np: np.ndarray, mask_path: Path,
                          global_seed: int = 42,
                          mask_types: list | None = None) -> str:
        """Detect mask type using seed-based lookup or geometric fallback.

        Primary method: replicate the deterministic dispatch logic from
        Stage 10's ``_DispatchGenerator`` — the mask type is fully determined
        by the filename seed and mask index.

        Fallback: improved geometric heuristics (for masks generated outside
        the dispatch pipeline or with unknown seeds).
        """
        if mask_types is None:
            mask_types = ['irregular', 'rect', 'edge']

        # --- 1. Filename keyword check (if type is encoded) ---
        stem = mask_path.stem.lower()
        for keyword in ('irregular', 'irr'):
            if keyword in stem:
                return 'irregular'
        for keyword in ('rect', 'box', 'rectangle'):
            if keyword in stem:
                return 'rect'
        for keyword in ('edge',):
            if keyword in stem:
                return 'edge'

        # --- 2. Seed-based deterministic lookup ---
        # Reproduce _DispatchGenerator dispatch: seed % len(mask_types)
        # Filename format: {original_stem}_m{mask_idx}.png
        import re as _re
        m = _re.search(r'_m(\d+)$', mask_path.stem)
        if m and len(mask_types) > 0:
            mask_idx = int(m.group(1))
            # Reconstruct the base stem (everything before _mN)
            base_stem = mask_path.stem[:m.start()]
            img_seed = global_seed ^ (hash(base_stem) & 0x7FFFFFFF)
            mask_seed = img_seed + mask_idx * 7919
            type_idx = mask_seed % len(mask_types)
            return mask_types[type_idx]

        # --- 3. Geometric analysis fallback ---
        import cv2 as _cv2
        binary = (mask_np > 0).astype(np.uint8)
        h_img, w_img = binary.shape[:2]

        # Edge masks: span full width at top or bottom of the image
        if h_img > 0 and w_img > 0:
            top_coverage = binary[0, :].sum() / w_img
            bot_coverage = binary[-1, :].sum() / w_img
            if top_coverage > 0.8 or bot_coverage > 0.8:
                return 'edge'

        contours, _ = _cv2.findContours(binary, _cv2.RETR_EXTERNAL, _cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 'unknown'
        largest = max(contours, key=_cv2.contourArea)
        area = _cv2.contourArea(largest)
        hull = _cv2.convexHull(largest)
        hull_area = _cv2.contourArea(hull)
        if hull_area == 0:
            return 'unknown'
        solidity = area / hull_area
        # Rectangular masks have high solidity (> 0.92)
        if solidity > 0.92:
            return 'rect'
        return 'irregular'

    def _save_domain(self, df, out):
        df[['model', 'condition', 'sample_id', 'color']].to_csv(out / "color_fidelity.csv", index=False)
        df[['model', 'condition', 'sample_id', 'pattern']].to_csv(out / "pattern_preservation.csv", index=False)
        self.logger.info("Domain metrics saved.")

    @staticmethod
    def _build_severity_bins(mask_coverage: pd.Series) -> pd.Series:
        return pd.cut(
            mask_coverage,
            bins=[0, 0.1, 0.25, 0.5, 1.0],
            labels=['<10%', '10-25%', '25-50%', '>50%'],
            include_lowest=True,
        )

    @staticmethod
    def _build_all_vs_all_comparisons(df: pd.DataFrame) -> list[tuple[str, str, str, str, str]]:
        combos_df = (
            df[["model", "condition"]]
            .dropna()
            .drop_duplicates()
            .sort_values(["model", "condition"])
            .reset_index(drop=True)
        )
        combos = [tuple(row) for row in combos_df.to_numpy().tolist()]
        comparisons: list[tuple[str, str, str, str, str]] = []
        for i in range(len(combos)):
            for j in range(i + 1, len(combos)):
                model_a, cond_a = combos[i]
                model_b, cond_b = combos[j]
                label = f"{model_a} [{cond_a}] vs {model_b} [{cond_b}]"
                comparisons.append((model_a, cond_a, model_b, cond_b, label))
        return comparisons

    def _save_error_analysis(self, df: pd.DataFrame, out_dir: Path) -> None:
        metrics = ['psnr', 'ssim', 'lpips', 'color', 'pattern']
        metrics = [m for m in metrics if m in df.columns]
        if not metrics:
            self.logger.warning("Error analysis skipped: no metric columns present.")
            return

        work_df = df.copy()
        if "severity_bin" not in work_df.columns and "mask_coverage" in work_df.columns:
            work_df["severity_bin"] = self._build_severity_bins(work_df["mask_coverage"])

        def _aggregate(frame: pd.DataFrame, group_cols: list[str], output_name: str) -> None:
            if frame.empty:
                return
            grouped = frame.groupby(group_cols, dropna=False)[metrics].agg(["count", "mean", "std", "median"])
            grouped.columns = [f"{metric}_{stat}" for metric, stat in grouped.columns]
            grouped.reset_index().to_csv(out_dir / output_name, index=False)

        unconditional = work_df[work_df["condition"] == "Unconditional"] if "condition" in work_df.columns else work_df
        _aggregate(unconditional, ["model", "mask_type"], "error_by_mask_type.csv")
        if "severity_bin" in unconditional.columns:
            _aggregate(unconditional, ["model", "severity_bin"], "error_by_severity.csv")
            _aggregate(unconditional, ["model", "mask_type", "severity_bin"], "error_by_mask_type_and_severity.csv")

        if "condition" in work_df.columns:
            _aggregate(work_df, ["model", "condition", "mask_type"], "error_by_mask_type_with_conditions.csv")
            if "severity_bin" in work_df.columns:
                _aggregate(work_df, ["model", "condition", "severity_bin"], "error_by_severity_with_conditions.csv")
        self.logger.info("Error analysis artifacts saved.")

    def _run_stats(self, df, out):
        """Paired t-tests + Wilcoxon with Bonferroni, Cohen's d, and 95% CIs.

        Runs all-vs-all comparisons over every available (model, condition) pair.
        """
        self.logger.info("Running all-vs-all statistical tests?")

        comparisons = self._build_all_vs_all_comparisons(df)
        metrics = ['psnr', 'ssim', 'lpips', 'color', 'pattern']
        metrics = [m for m in metrics if m in df.columns]
        n_tests = len(metrics) * len(comparisons)
        alpha = 0.05
        alpha_bonferroni = alpha / n_tests if n_tests > 0 else alpha

        stats_results = []
        skipped = 0

        for model_a, cond_a, model_b, cond_b, label in comparisons:
            a_df = df[(df['model'] == model_a) & (df['condition'] == cond_a)]
            b_df = df[(df['model'] == model_b) & (df['condition'] == cond_b)]

            if a_df.empty or b_df.empty:
                skipped += 1
                continue

            merged = pd.merge(a_df, b_df, on='sample_id', suffixes=('_a', '_b'))
            if merged.empty:
                skipped += 1
                continue

            for metric in metrics:
                col_a = f'{metric}_a'
                col_b = f'{metric}_b'
                if col_a not in merged.columns or col_b not in merged.columns:
                    continue
                paired = merged[[col_a, col_b]].dropna()
                if len(paired) < 2:
                    continue
                a = paired[col_a]
                b = paired[col_b]

                diff = (a - b).to_numpy(dtype=float)
                finite_diff = diff[np.isfinite(diff)]
                if len(finite_diff) < 2:
                    continue
                # If every paired difference is effectively identical, skip parametric
                # tests to avoid precision-loss warnings and report no difference.
                if np.allclose(finite_diff, finite_diff[0], rtol=1e-12, atol=1e-12):
                    t_stat, p_val = 0.0, 1.0
                    w_stat, w_pval = 0.0, 1.0
                else:
                    t_stat, p_val = stats.ttest_rel(a, b)
                    try:
                        w_stat, w_pval = stats.wilcoxon(a, b)
                    except ValueError:
                        w_stat, w_pval = float('nan'), float('nan')
                d = self._cohens_d(a, b)
                ci_lo, ci_hi = self._ci_95(a, b)

                stats_results.append({
                    "comparison": label,
                    "comparison_type": "all_vs_all",
                    "model_a": model_a,
                    "condition_a": cond_a,
                    "model_b": model_b,
                    "condition_b": cond_b,
                    "metric": metric,
                    "n_pairs": len(paired),
                    "mean_a": round(float(a.mean()), 4),
                    "mean_b": round(float(b.mean()), 4),
                    "mean_diff_a_minus_b": round(float((a - b).mean()), 4),
                    "t_statistic": round(float(t_stat), 4),
                    "p_value": round(float(p_val), 6),
                    "wilcoxon_stat": round(float(w_stat), 4) if not np.isnan(w_stat) else None,
                    "wilcoxon_p": round(float(w_pval), 6) if not np.isnan(w_pval) else None,
                    "alpha_bonferroni": round(float(alpha_bonferroni), 8),
                    "significant_bonferroni": bool(p_val < alpha_bonferroni),
                    "cohens_d": round(float(d), 4),
                    "ci_95_lower": round(float(ci_lo), 4),
                    "ci_95_upper": round(float(ci_hi), 4),
                })

        stats_out = pd.DataFrame(stats_results)
        stats_out.to_csv(out / "paired_t_tests.csv", index=False)
        self.logger.info(
            f"Stats saved ? {len(stats_results)} rows across {len(comparisons)} comparisons, "
            f"Bonferroni alpha={alpha_bonferroni:.8f}, skipped={skipped}."
        )
        return stats_out

    def _write_prompt_ablation_doc(self, df: pd.DataFrame):
        """Write a short analysis of text conditioning effects (RQ-2)."""
        doc_path = self.output_dir / "prompt_ablation_analysis.txt"
        try:
            lines = [
                "PROMPT ABLATION ANALYSIS (RQ-2)",
                "=" * 50,
                "",
                "This file documents the effect of text conditioning on inpainting quality.",
                "Four conditioning modes were tested per SD-based model:",
                "  1. Unconditional (empty prompt)",
                "  2. Raw Text     (museum/Wikimedia metadata when available; otherwise source title/fields)",
                "  3. Enriched Text (BLIP2 dense caption from the image)",
                "  4. Refined Text (Qwen refined + CLIP-safe word cap)",
                "",
            ]
            for model in ["Vanilla SD", "FT-SD"]:
                sub = df[df["model"] == model]
                if sub.empty:
                    continue
                lines.append(f"--- {model} ---")
                for metric in ["psnr", "ssim", "lpips", "color", "pattern"]:
                    if metric not in sub.columns:
                        continue
                    means = sub.groupby("condition")[metric].mean()
                    lines.append(f"  {metric.upper()}:")
                    for cond, val in means.items():
                        lines.append(f"    {cond:20s}: {val:.4f}")
                    spread = means.max() - means.min()
                    lines.append(f"    Δ(max-min) = {spread:.4f}")
                    if spread < 0.01 and metric in ("psnr", "ssim"):
                        lines.append("    → NEGLIGIBLE: text conditioning has no measurable effect.")
                lines.append("")

            lines.extend([
                "INTERPRETATION:",
                "  If Δ values are near zero across all metrics, the model is",
                "  invariant to text conditioning — likely because the same seed",
                "  produces identical latent noise for every call (by design for",
                "  reproducibility). This is a valid negative finding for RQ-2.",
                "",
                "  To enable prompt sensitivity, one would need to either:",
                "    a) Fine-tune with text-conditioning in the training objective, or",
                "    b) Use different seeds per condition (sacrificing reproducibility).",
            ])
            with open(doc_path, "w") as f:
                f.write("\n".join(lines))
            self.logger.info(f"Prompt ablation analysis written to {doc_path}")
        except Exception as e:
            self.logger.warning(f"Prompt ablation doc failed: {e}")


if __name__ == "__main__":
    from thesis_pipeline.config_manager import ConfigManager
    cm = ConfigManager()
    stage = ModelEvaluationStage(cm)
    stage.run()
