"""Canonical sequential stages (S00-S18)."""

from __future__ import annotations

import logging

from thesis_pipeline.core.config import ConfigManager
from thesis_pipeline.stages.stage_00_reproducibility import ReproducibilityStage as S00_ReproducibilityStage
from thesis_pipeline.stages.stage_01_research_design import ResearchDesignStage as S01_ResearchDesignStage
from thesis_pipeline.stages.stage_02_data_acquisition import DataAcquisitionStage as S02_DataAcquisitionStage
from thesis_pipeline.stages.stage_03_intelligent_filtering import IntelligentFilteringStage as S03_IntelligentFilteringStage
from thesis_pipeline.stages.stage_04_yolo_validation import YOLOValidationStage as S04_YoloValidationStage
from thesis_pipeline.stages.stage_05_baselines import BaselinesStage as S05_BaselinesStage
from thesis_pipeline.stages.stage_06_exploratory_data_analysis import ExploratoryDataAnalysisStage as S06_ExploratoryDataAnalysisStage
from thesis_pipeline.stages.stage_07_caption_generation import CaptionGenerationStage as S07_CaptionGenerationStage
from thesis_pipeline.stages.stage_08_caption_refinement import CaptionRefinementStage as S08_CaptionRefinementStage
from thesis_pipeline.stages.stage_09_data_processing import DataProcessingStage as S09_DataProcessingStage
from thesis_pipeline.stages.stage_10_data_splitting import DataSplittingStage as S10_DataSplittingStage
from thesis_pipeline.stages.stage_11_feature_engineering import FeatureEngineeringStage as S11_FeatureEngineeringStage
from thesis_pipeline.stages.stage_12_hyperparameter_tuning import HyperparameterTuningStage as S12_HyperparameterTuningStage
from thesis_pipeline.stages.stage_13_model_training import ModelTrainingStage as S13_ModelTrainingStage
from thesis_pipeline.stages.stage_14_baseline_finetuning import BaselineFineTuningStage as S14_BaselineFineTuningStage
from thesis_pipeline.stages.stage_15_model_evaluation import ModelEvaluationStage as S15_ModelEvaluationStage
from thesis_pipeline.stages.stage_16_deployment_preparation import DeploymentPreparationStage as S16_DeploymentPreparationStage
from thesis_pipeline.stages.stage_17_reporting import ReportingStage as S17_ReportingStage
from thesis_pipeline.stages.stage_18_expert_validation import ExpertValidationStage as S18_ExpertValidationStage


class _GroupedStage:
    stage_label = ""

    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)

    def _run_stages(self, *stage_types):
        self.logger.info("%s STARTED", self.stage_label)
        for stage_type in stage_types:
            stage_type(self.config_manager).run()
        self.logger.info("%s COMPLETED", self.stage_label)

