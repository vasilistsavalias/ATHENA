"""Canonical V7 stage identifiers and metadata."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class StageID(str, Enum):
    """Canonical stage identifiers."""

    S00 = "S00"
    S01 = "S01"
    S02 = "S02"
    S03 = "S03"
    S04 = "S04"
    S05 = "S05"
    S06 = "S06"
    S07 = "S07"
    S08 = "S08"
    S09 = "S09"
    S10 = "S10"
    S11 = "S11"
    S12 = "S12"
    S13 = "S13"
    S14 = "S14"
    S15 = "S15"
    S16 = "S16"
    S17 = "S17"
    S18 = "S18"


@dataclass(frozen=True)
class StageInfo:
    sid: StageID
    name: str
    output_dir: str
    legacy_ids: tuple[str, ...]


STAGE_CATALOG: tuple[StageInfo, ...] = (
    StageInfo(StageID.S00, "Reproducibility & Preflight", "S00_reproducibility", ("00",)),
    StageInfo(StageID.S01, "Research Design", "S01_research_design", ("01",)),
    StageInfo(StageID.S02, "Data Acquisition", "S02_data_acquisition", ("02", "02b")),
    StageInfo(StageID.S03, "Intelligent Filtering", "S03_intelligent_filtering", ("03",)),
    StageInfo(StageID.S04, "YOLO Validation", "S04_yolo_validation", ("03b",)),
    StageInfo(StageID.S05, "Baselines", "S05_baselines", ("04",)),
    StageInfo(StageID.S06, "Exploratory Data Analysis", "S06_exploratory_data_analysis", ("05",)),
    StageInfo(StageID.S07, "Caption Generation", "S07_caption_generation", ("06",)),
    StageInfo(StageID.S08, "Caption Refinement", "S08_caption_refinement", ("07",)),
    StageInfo(StageID.S09, "Data Processing", "S09_data_processing", ("08",)),
    StageInfo(StageID.S10, "Data Splitting", "S10_data_splitting", ("09",)),
    StageInfo(StageID.S11, "Feature Engineering", "S11_feature_engineering", ("10", "10b")),
    StageInfo(StageID.S12, "Hyperparameter Tuning", "S12_hyperparameter_tuning", ("11",)),
    StageInfo(StageID.S13, "Model Training", "S13_model_training", ("12",)),
    StageInfo(StageID.S14, "Baseline Fine-Tuning", "S14_baseline_finetuning", ("s07",)),
    StageInfo(StageID.S15, "Model Evaluation", "S15_model_evaluation", ("13",)),
    StageInfo(StageID.S16, "Deployment Preparation", "S16_deployment_preparation", ("14",)),
    StageInfo(StageID.S17, "Reporting", "S17_reporting", ("15",)),
    StageInfo(StageID.S18, "Expert Validation", "S18_expert_validation", ("16",)),
)

STAGE_ORDER = [info.sid for info in STAGE_CATALOG]
STAGE_BY_ID = {info.sid: info for info in STAGE_CATALOG}
LEGACY_TO_NEW: dict[str, StageID] = {
    legacy_id: info.sid for info in STAGE_CATALOG for legacy_id in info.legacy_ids
}


def resolve_stage_id(raw: str) -> StageID:
    """Resolve a CLI token to a canonical :class:`StageID`."""

    raw = raw.strip()
    try:
        return StageID(raw)
    except ValueError:
        pass
    if raw in LEGACY_TO_NEW:
        return LEGACY_TO_NEW[raw]
    raise ValueError(f"Unknown stage identifier: {raw!r}")


def output_dir_for(sid: StageID) -> str:
    return STAGE_BY_ID[sid].output_dir


def execution_order() -> list[StageID]:
    return list(STAGE_ORDER)
