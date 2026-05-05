"""Canonical configuration access layer for ATHENA."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from box import ConfigBox
from box.exceptions import BoxValueError
from thesis_pipeline.stage_registry import StageID, execution_order, output_dir_for, resolve_stage_id


class ConfigManager:
    """Load the active YAML config and expose stable section accessors."""

    def __init__(self, config_filepath: Path = Path("config/pipeline/main_config.yaml")):
        self.config_filepath = Path(config_filepath)
        self.config = self._load_config()
        self.logger = logging.getLogger(__name__)
        self.logger.info("Configuration loaded successfully from: %s", self.config_filepath)

    def _load_config(self) -> ConfigBox:
        merged = self._load_config_file(self.config_filepath.resolve(), seen=set())
        return ConfigBox(merged)

    def _load_config_file(self, config_path: Path, seen: set[Path]) -> dict[str, Any]:
        config_path = Path(config_path).resolve()
        if config_path in seen:
            raise ValueError(f"Recursive config extends detected for {config_path}")
        seen = set(seen)
        seen.add(config_path)
        try:
            with open(config_path, "r", encoding="utf-8") as handle:
                config = yaml.safe_load(handle)
            if config is None:
                config = {}
            if not isinstance(config, dict):
                raise BoxValueError(f"Configuration root must be a mapping: {config_path}")

            extends = config.pop("extends", None)
            if extends:
                parent_path = Path(extends)
                if not parent_path.is_absolute():
                    parent_path = config_path.parent / parent_path
                parent_config = self._load_config_file(parent_path, seen=seen)
                return self._deep_merge_dicts(parent_config, config)
            return config
        except FileNotFoundError:
            logging.error("Configuration file not found at: %s", config_path)
            raise
        except (yaml.YAMLError, BoxValueError) as error:
            logging.error("Error parsing configuration file %s: %s", config_path, error)
            raise

    @staticmethod
    def _deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        merged = dict(base)
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = ConfigManager._deep_merge_dicts(merged[key], value)
            else:
                merged[key] = value
        return merged

    def get_global_params(self) -> ConfigBox:
        return self.config.global_params

    def get_paths(self) -> ConfigBox:
        paths = self.config.paths
        artifacts = getattr(paths, "artifacts", None)
        if artifacts is not None:
            # Keep compatibility across canonical (S00) and legacy (stage_00)
            # artifact keys so mixed-callers do not fail at runtime.
            for idx in range(19):
                canonical_key = f"S{idx:02d}"
                legacy_key = f"stage_{idx:02d}"
                canonical_value = artifacts.get(canonical_key)
                legacy_value = artifacts.get(legacy_key)

                if canonical_value is not None and legacy_value is None:
                    artifacts[legacy_key] = canonical_value
                elif legacy_value is not None and canonical_value is None:
                    artifacts[canonical_key] = legacy_value

        return paths

    @staticmethod
    def _normalize_stage_id(stage: str | int | StageID) -> StageID:
        if isinstance(stage, StageID):
            return stage
        return resolve_stage_id(str(stage))

    def get_stage_artifact_dir(self, stage: str | int | StageID) -> Path:
        sid = self._normalize_stage_id(stage)
        paths = self.get_paths()
        artifacts = getattr(paths, "artifacts", None)
        if artifacts is None:
            raise AttributeError("Config paths.artifacts is missing")

        canonical_value = artifacts.get(sid.value)
        if canonical_value is None:
            root = Path(getattr(artifacts, "root", "outputs"))
            canonical_value = str(root / output_dir_for(sid))
            artifacts[sid.value] = canonical_value
            artifacts[f"stage_{sid.value[1:]}"] = canonical_value
        return Path(canonical_value)

    def get_stage_artifact_path(self, stage: str | int | StageID, *parts: str) -> Path:
        return self.get_stage_artifact_dir(stage).joinpath(*parts)

    def rewrite_artifacts_root(self, artifacts_root: str | Path) -> None:
        root = Path(artifacts_root)
        paths = self.get_paths()
        artifacts = getattr(paths, "artifacts", None)
        if artifacts is None:
            raise AttributeError("Config paths.artifacts is missing")

        artifacts.root = str(root)
        for sid in execution_order():
            stage_dir = root / output_dir_for(sid)
            artifacts[sid.value] = str(stage_dir)
            artifacts[f"stage_{sid.value[1:]}"] = str(stage_dir)
        artifacts.logs = str(root / "00_logs")

    def get_logging_config(self) -> ConfigBox:
        return self.config.logging

    def get_data_processing_config(self) -> ConfigBox:
        return self.config.data_processing

    def get_data_splitting_config(self) -> ConfigBox:
        return self.config.data_splitting

    def get_feature_engineering_config(self) -> ConfigBox:
        return self.config.feature_engineering

    def get_data_acquisition_config(self) -> ConfigBox:
        return self.config.data_acquisition

    def get_exploratory_data_analysis_config(self) -> ConfigBox:
        return self.config.exploratory_data_analysis

    def get_caption_generation_config(self) -> ConfigBox:
        return self.config.caption_generation

    def get_hyperparameter_tuning_config(self) -> ConfigBox:
        return self.config.hyperparameter_tuning

    def get_training_config(self) -> ConfigBox:
        return self.config.training

    def get_model_evaluation_config(self) -> ConfigBox:
        return self.config.model_evaluation

    def get_deployment_preparation_config(self) -> ConfigBox:
        return self.config.deployment_preparation
