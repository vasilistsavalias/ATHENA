import argparse
import os
import logging
import time
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any
from thesis_pipeline.core.config import ConfigManager
from thesis_pipeline.core.logging import LoggingConfig

# V7 stage registry
from thesis_pipeline.stage_registry import (
    StageID, resolve_stage_id, execution_order,
)

# Canonical stage classes (S00-S18)
from thesis_pipeline.pipeline.stage_groups import (
    S00_ReproducibilityStage,
    S01_ResearchDesignStage,
    S02_DataAcquisitionStage,
    S03_IntelligentFilteringStage,
    S04_YoloValidationStage,
    S05_BaselinesStage,
    S06_ExploratoryDataAnalysisStage,
    S07_CaptionGenerationStage,
    S08_CaptionRefinementStage,
    S09_DataProcessingStage,
    S10_DataSplittingStage,
    S11_FeatureEngineeringStage,
    S12_HyperparameterTuningStage,
    S13_ModelTrainingStage,
    S14_BaselineFineTuningStage,
    S15_ModelEvaluationStage,
    S16_DeploymentPreparationStage,
    S17_ReportingStage,
    S18_ExpertValidationStage,
)
from thesis_pipeline.pipeline.governance import (
    get_pipeline_runtime_policy,
    run_preflight_checks,
)


def _as_int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _is_primary_process() -> bool:
    # In non-distributed runs these variables are unset; treat as primary.
    rank = _as_int_env("RANK", 0)
    local_rank = _as_int_env("LOCAL_RANK", 0)
    return rank == 0 and local_rank in (0, -1)


class RunStateLedger:
    """Persistent per-stage run-state ledger for strict/resume execution."""

    VALID_STATUSES = {"pending", "running", "success", "failed", "skipped"}

    def __init__(self, path: Path, stages: list[str]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, Any] = self._load()
        self._data.setdefault("updated_at", datetime.now().isoformat())
        self._data.setdefault("stages", {})
        self._ensure_stages(stages)
        self._save()

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, dict):
                return payload
        except Exception:
            pass
        return {}

    def _save(self):
        self._data["updated_at"] = datetime.now().isoformat()
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2)

    @staticmethod
    def _default_entry() -> dict[str, Any]:
        return {
            "status": "pending",
            "started_at": None,
            "ended_at": None,
            "duration": None,
            "error": None,
            "artifacts": [],
            "previous_status": None,
        }

    def _ensure_stages(self, stages: list[str]):
        stage_map = self._data.setdefault("stages", {})
        for stage in stages:
            if stage not in stage_map or not isinstance(stage_map.get(stage), dict):
                stage_map[stage] = self._default_entry()
            else:
                entry = stage_map[stage]
                for key, value in self._default_entry().items():
                    entry.setdefault(key, value)
                if entry.get("status") not in self.VALID_STATUSES:
                    entry["status"] = "pending"

    def get_status(self, stage: str) -> str:
        return self._data.get("stages", {}).get(stage, {}).get("status", "pending")

    def mark_running(self, stage: str):
        entry = self._data["stages"][stage]
        entry["previous_status"] = entry.get("status")
        entry["status"] = "running"
        entry["started_at"] = datetime.now().isoformat()
        entry["ended_at"] = None
        entry["duration"] = None
        entry["error"] = None
        self._save()

    def mark_success(self, stage: str, *, artifacts: list[str] | None = None):
        entry = self._data["stages"][stage]
        entry["previous_status"] = entry.get("status")
        entry["status"] = "success"
        end = datetime.now().isoformat()
        entry["ended_at"] = end
        started = entry.get("started_at")
        if started:
            try:
                entry["duration"] = (
                    datetime.fromisoformat(end) - datetime.fromisoformat(started)
                ).total_seconds()
            except Exception:
                entry["duration"] = None
        entry["error"] = None
        entry["artifacts"] = artifacts or []
        self._save()

    def mark_failed(self, stage: str, error: str):
        entry = self._data["stages"][stage]
        entry["previous_status"] = entry.get("status")
        entry["status"] = "failed"
        end = datetime.now().isoformat()
        entry["ended_at"] = end
        started = entry.get("started_at")
        if started:
            try:
                entry["duration"] = (
                    datetime.fromisoformat(end) - datetime.fromisoformat(started)
                ).total_seconds()
            except Exception:
                entry["duration"] = None
        entry["error"] = str(error)
        self._save()

    def mark_skipped(self, stage: str, *, reason: str, preserve_success: bool = False):
        entry = self._data["stages"][stage]
        entry["previous_status"] = entry.get("status")
        if preserve_success and entry.get("status") == "success":
            entry["status"] = "success"
        else:
            entry["status"] = "skipped"
        entry["started_at"] = entry.get("started_at") or datetime.now().isoformat()
        entry["ended_at"] = datetime.now().isoformat()
        entry["error"] = reason
        self._save()


def _infer_stage_artifacts(stage_num: str, config_manager: ConfigManager, logs_dir: Path) -> list[str]:
    artifacts: list[str] = []
    paths = config_manager.get_paths()

    try:
        stage_dir = config_manager.get_stage_artifact_dir(stage_num)
    except Exception:
        stage_dir = None
    if stage_dir and stage_dir.exists():
        artifacts.append(str(stage_dir))
    log_path = logs_dir / f"stage_{stage_num}.log"
    if log_path.exists():
        artifacts.append(str(log_path))

    if stage_num in ("S13", "12"):
        models_root = Path(paths.data.models)
        for model_subdir in ("unet_best", "unet_final", "sweep"):
            model_path = models_root / model_subdir
            if model_path.exists():
                artifacts.append(str(model_path))
    elif stage_num in ("S15", "13"):
        eval_dir = config_manager.get_stage_artifact_dir("S15")
        matrix_csv = eval_dir / "benchmarking_matrix" / "matrix_results.csv"
        if matrix_csv.exists():
            artifacts.append(str(matrix_csv))
    return artifacts


def _check_stage_prerequisites(stage_num: str, config_manager: ConfigManager) -> list[str]:
    """Return a list of missing prerequisite descriptions for a stage."""
    missing: list[str] = []
    paths = config_manager.get_paths()
    config = config_manager.config

    if stage_num in ("S13", "12"):
        hp_dir = config_manager.get_stage_artifact_dir("S12")
        best_yaml = hp_dir / "best_hyperparameters.yaml"
        if not best_yaml.exists():
            missing.append(f"S12 output missing: {best_yaml}")
        mode = str(config.get("hyperparameter_tuning", {}).get("mode", "literature_informed")).lower()
        if mode == "mini_sweep":
            sweep_yaml = hp_dir / "sweep_plan.yaml"
            if not sweep_yaml.exists():
                missing.append(f"S12 sweep plan missing: {sweep_yaml}")

    if stage_num in ("S15", "13"):
        eval_mock = bool(config.get("model_evaluation", {}).get("mock", False))
        if not eval_mock:
            models_root = Path(paths.data.models)
            unet_best = models_root / "unet_best"
            unet_final = models_root / "unet_final"
            if not unet_best.exists() and not unet_final.exists():
                missing.append(f"S13 model missing: expected {unet_best} or {unet_final}")

    if stage_num in ("S18", "16"):
        eval_dir = config_manager.get_stage_artifact_dir("S15")
        samples_dir = eval_dir / "samples"
        matrix_csv = eval_dir / "benchmarking_matrix" / "matrix_results.csv"
        if not samples_dir.exists():
            missing.append(f"S15 samples missing: {samples_dir}")
        if not matrix_csv.exists():
            missing.append(f"S15 benchmarking matrix missing: {matrix_csv}")

    if stage_num in ("S14",):
        # Baseline fine-tuning requires masks from S11 feature engineering.
        inpainting_dir = Path(paths.data.inpainting)
        if not inpainting_dir.exists() or not list(inpainting_dir.glob("train/masks/*.png")):
            missing.append(f"S11 masks missing: {inpainting_dir / 'train/masks/'}")

    return missing


# ==============================================================================
# Pipeline Timer
# ==============================================================================
class PipelineTimer:
    def __init__(self, stage_name, stage_num, log_file=None):
        self.stage_name = stage_name
        self.stage_num = stage_num
        self.log_file = Path(log_file) if log_file else Path("outputs/00_logs/execution_times.csv")
        self.start_time = None
        self._enabled = _is_primary_process()

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self._enabled:
            return
        duration = time.time() - self.start_time
        status = "FAILED" if exc_type else "SUCCESS"
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        file_exists = self.log_file.exists()
        try:
            with open(self.log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(['timestamp', 'stage_num', 'stage_name', 'status', 'duration_seconds'])
                writer.writerow([datetime.now().isoformat(), self.stage_num, self.stage_name, status, f"{duration:.2f}"])
        except Exception as e:
            logging.getLogger(__name__).error(f"Failed to log execution time: {e}")

# ==============================================================================
# Main Pipeline Orchestrator
# ==============================================================================

def main(
    stages_to_run=None,
    smoke_test=False,
    resume=False,
    force_rerun_successful=False,
    *,
    config_override: str | None = None,
    stage02_limit: int | None = None,
    raw_dir_override: str | None = None,
    artifacts_root_override: str | None = None,
    eval_phase: str | None = None,
):
    try:
        project_root = Path(__file__).resolve().parents[3]
        if config_override:
            config_path = Path(config_override)
            if not config_path.is_absolute():
                config_path = project_root / config_path
        elif smoke_test:
            config_path = project_root / "config/pipeline/smoke_test_config.yaml"
        else:
            config_path = project_root / "config/pipeline/main_config.yaml"
            
        config_manager = ConfigManager(config_filepath=config_path)

        # Optional runtime overrides (useful for local quick runs of Stage 02).
        if raw_dir_override:
            config_manager.config.paths.data.raw = str(raw_dir_override)
        if artifacts_root_override:
            config_manager.rewrite_artifacts_root(artifacts_root_override)

        if stage02_limit is not None and stage02_limit > 0:
            try:
                da = config_manager.config.data_acquisition
                da.max_per_source = int(stage02_limit)
                da.wikimedia_limit = int(stage02_limit)
                da.download_limit = int(stage02_limit)
                da.europeana_limit = int(stage02_limit)
            except Exception:
                pass

        if eval_phase:
            config_manager.config.model_evaluation.phase = str(eval_phase)

        runtime_policy = get_pipeline_runtime_policy(config_manager.config)
        strict = bool(runtime_policy["strict_fail_policy"])
        resume = bool(resume or runtime_policy["resume_enabled"])

        logging_config = LoggingConfig(config_manager)
        logging_config.setup_logging()
        
        logger = logging.getLogger(__name__)
        is_primary = _is_primary_process()

        if is_primary:
            logger.info("="*40)
            logger.info(" Thesis Pipeline V8 Finalization Started ".center(40, "="))
            logger.info("="*40)
            logger.info(
                f"Runtime policy: strict={strict}, resume={resume}, "
                f"force_rerun_successful={force_rerun_successful}, "
                "clean=False (flag removed)"
            )
        
    except Exception as e:
        logging.basicConfig(level=logging.INFO)
        logging.error(f"FATAL: Pipeline setup failed: {e}", exc_info=True)
        return

    # Canonical stage mapping - linear S00-S18
    all_stages = {
        StageID.S00: S00_ReproducibilityStage,
        StageID.S01: S01_ResearchDesignStage,
        StageID.S02: S02_DataAcquisitionStage,
        StageID.S03: S03_IntelligentFilteringStage,
        StageID.S04: S04_YoloValidationStage,
        StageID.S05: S05_BaselinesStage,
        StageID.S06: S06_ExploratoryDataAnalysisStage,
        StageID.S07: S07_CaptionGenerationStage,
        StageID.S08: S08_CaptionRefinementStage,
        StageID.S09: S09_DataProcessingStage,
        StageID.S10: S10_DataSplittingStage,
        StageID.S11: S11_FeatureEngineeringStage,
        StageID.S12: S12_HyperparameterTuningStage,
        StageID.S13: S13_ModelTrainingStage,
        StageID.S14: S14_BaselineFineTuningStage,
        StageID.S15: S15_ModelEvaluationStage,
        StageID.S16: S16_DeploymentPreparationStage,
        StageID.S17: S17_ReportingStage,
        StageID.S18: S18_ExpertValidationStage,
    }

    # Resolve requested stages (supports both S-prefixed and legacy IDs)
    if stages_to_run:
        stages: list[StageID] = []
        for raw in stages_to_run:
            try:
                sid = resolve_stage_id(raw)
                stages.append(sid)
            except ValueError:
                logger.warning(f"Unknown stage identifier: {raw!r}, skipping.")
    else:
        stages = execution_order()

    if is_primary:
        logger.info(f"Executing stages: {', '.join(s.value for s in stages)}")

    preflight_errors = run_preflight_checks(
        config_manager.config,
        project_root=project_root,
        stages=stages,
    )
    if preflight_errors:
        joined = " | ".join(preflight_errors)
        if strict:
            logger.error(f"Global preflight failed (strict=true): {joined}")
            return
        logger.warning(f"Global preflight warnings (strict=false): {joined}")

    from loguru import logger as loguru_logger
    logs_dir = Path(config_manager.get_paths().artifacts.logs)
    timer_log = str(logs_dir / "execution_times.csv")
    run_state_path = logs_dir / "run_state.json"
    stage_ids_str = [s.value for s in stages]
    state_ledger = RunStateLedger(run_state_path, stage_ids_str) if _is_primary_process() else None

    for stage_id in stages:
        stage_num = stage_id.value  # e.g. "S00"
        if stage_id in all_stages:
            if is_primary and state_ledger and resume and not force_rerun_successful:
                previous_status = state_ledger.get_status(stage_num)
                if previous_status == "success":
                    logger.info(
                        f"Stage {stage_num}: skipping due to resume (already marked success in run_state)."
                    )
                    state_ledger.mark_skipped(
                        stage_num,
                        reason="resume: previously successful",
                        preserve_success=True,
                    )
                    continue

            toggle_map = runtime_policy.get("stage_toggles", {})
            stage_enabled = bool(toggle_map.get(stage_num, True))
            if not stage_enabled:
                reason = f"stage toggle disabled for {stage_num}"
                if strict and bool(runtime_policy.get("strict_mode_enabled", strict)):
                    logger.error(f"Stage {stage_num}: strict-mode toggle gate failed: {reason}")
                    if is_primary and state_ledger:
                        state_ledger.mark_failed(stage_num, reason)
                    return
                logger.warning(f"Stage {stage_num}: skipped ({reason})")
                if is_primary and state_ledger:
                    state_ledger.mark_skipped(stage_num, reason=reason)
                continue

            missing = _check_stage_prerequisites(stage_num, config_manager)
            if missing:
                msg = " | ".join(missing)
                if strict:
                    logger.error(f"Stage {stage_num}: strict prerequisite gate failed: {msg}")
                    if is_primary and state_ledger:
                        state_ledger.mark_failed(stage_num, msg)
                    return
                logger.warning(f"Stage {stage_num}: prerequisite warning (strict=false): {msg}")

            StageClass = all_stages[stage_id]
            log_file_path = logs_dir / f"stage_{stage_num}.log"
            log_handler_id = loguru_logger.add(log_file_path, level=logging_config.log_level)

            try:
                if is_primary:
                    logger.info(f"--- Starting Stage {stage_num} ---")
                    if state_ledger:
                        state_ledger.mark_running(stage_num)
                stage_instance = StageClass(config_manager)
                with PipelineTimer(StageClass.__name__, stage_num, log_file=timer_log):
                    stage_instance.run()
                if is_primary:
                    logger.info(f"--- Completed Stage {stage_num} ---")
                    if state_ledger:
                        artifacts = _infer_stage_artifacts(stage_num, config_manager, logs_dir)
                        state_ledger.mark_success(stage_num, artifacts=artifacts)
            except Exception as e:
                logger.error(f"FATAL: Error in Stage {stage_num}.", exc_info=True)
                if is_primary and state_ledger:
                    state_ledger.mark_failed(stage_num, str(e))
                loguru_logger.remove(log_handler_id)
                return 
            loguru_logger.remove(log_handler_id)
        else:
            logger.warning(f"Stage '{stage_num}' not found in all_stages.")
            if is_primary and state_ledger:
                state_ledger.mark_skipped(stage_num, reason="stage_not_found")

    # --- Write run manifest ---
    try:
        if not is_primary:
            return
        artifacts_root = Path(config_manager.get_paths().artifacts.root)
        manifest = {
            "timestamp": datetime.now().isoformat(),
            "config_file": str(config_path),
            "config_profile": Path(config_path).stem,
            "smoke_test": smoke_test,
            "stages_executed": [s.value if hasattr(s, 'value') else s for s in stages],
            "strict_run": bool(strict),
            "resume_run": bool(resume),
            "pipeline_version": str(config_manager.config.get("pipeline", {}).get("version_label", "V8")),
            "run_state_file": str(run_state_path),
        }
        # Read git state if available
        git_state_file = config_manager.get_stage_artifact_path("S00", "git_state.json")
        if git_state_file.exists():
            with open(git_state_file) as f:
                manifest["git_state"] = json.load(f)
        # Read execution times if available
        exec_times_file = Path(config_manager.get_paths().artifacts.logs) / "execution_times.csv"
        if exec_times_file.exists():
            import csv as csv_mod
            with open(exec_times_file) as f:
                reader = csv_mod.DictReader(f)
                manifest["execution_times"] = list(reader)
        with open(artifacts_root / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=4)
        logger.info(f"Run manifest written to {artifacts_root / 'manifest.json'}")
    except Exception as e:
        logger.warning(f"Failed to write manifest: {e}")

def cli():
    parser = argparse.ArgumentParser(description="ATHENA thesis pipeline (V8)")
    parser.add_argument("--stages", nargs='+', help="Stage IDs to run, e.g. S00 S04 S06 (also accepts legacy IDs like 10 12)")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--full", action="store_true", help="Run all stages S00-S18")
    parser.add_argument("--resume", action="store_true", help="Skip stages already marked successful in run_state.json.")
    parser.add_argument(
        "--force-rerun-successful",
        action="store_true",
        help="When used with --resume, rerun stages even if they were previously successful.",
    )
    parser.add_argument(
        "--phase",
        choices=["integrity_200", "full_test"],
        help="Override model_evaluation.phase for S15.",
    )
    parser.add_argument("--config", help="Path to config YAML (defaults to main_config.yaml).")
    parser.add_argument(
        "--stage02-limit",
        type=int,
        help="Override S01 data-acquisition to download at most N images per source.",
    )
    parser.add_argument("--raw-dir", help="Override paths.data.raw (useful for temp local runs).")
    parser.add_argument("--artifacts-root", help="Override paths.artifacts.root (useful for temp local runs).")
    args = parser.parse_args()
    stages_to_run = None if args.full else args.stages
    main(
        stages_to_run=stages_to_run,
        smoke_test=args.smoke_test,
        resume=args.resume,
        force_rerun_successful=args.force_rerun_successful,
        config_override=args.config,
        stage02_limit=args.stage02_limit,
        raw_dir_override=args.raw_dir,
        artifacts_root_override=args.artifacts_root,
        eval_phase=args.phase,
    )


if __name__ == '__main__':
    cli()

