import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
import shutil
import cv2
import numpy as np
import os
import sys

if os.environ.get("TORCH_IMPORT_OK", "1") != "1":
    pytest.skip(
        "Skipping torch-dependent tests because torch import failed/crashed in a subprocess probe. "
        "Fix your torch install or set TORCH_IMPORT_OK=1 to force-run.",
        allow_module_level=True,
    )

torch = pytest.importorskip("torch")
from thesis_pipeline.stages.stage_03_intelligent_filtering import IntelligentFilteringStage, _worker_process

@pytest.fixture
def test_env(tmp_path):
    # Setup directories
    raw_dir = tmp_path / "raw"
    filtered_dir = tmp_path / "filtered"
    raw_dir.mkdir()
    filtered_dir.mkdir()
    
    # Create dummy image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(raw_dir / "test_artifact.jpg"), img)
    
    # Mock ConfigManager
    config_manager = MagicMock()
    config_manager.config.paths.data.raw = str(raw_dir)
    config_manager.config.paths.data.filtered = str(filtered_dir)
    config_manager.config.intelligent_filtering.model_name = "mock.pt"
    config_manager.config.intelligent_filtering.confidence_threshold = 0.5
    
    return config_manager, raw_dir, filtered_dir

@patch('thesis_pipeline.stages.stage_03_intelligent_filtering.YOLO')
@patch('thesis_pipeline.stages.stage_03_intelligent_filtering.QualityFilter')
@patch('thesis_pipeline.stages.stage_03_intelligent_filtering.AuditLogger')
def test_filtering_and_cropping(mock_audit, mock_quality, mock_yolo, test_env):
    config_manager, raw_dir, filtered_dir = test_env
    
    # --- Mock YOLO Result ---
    mock_model = mock_yolo.return_value
    mock_result = MagicMock()
    mock_box = MagicMock()
    mock_box.xyxy = [torch.tensor([10, 10, 50, 50])] 
    mock_box.conf = [0.9]
    mock_result.boxes = [mock_box]
    mock_model.return_value = [mock_result]
    
    # --- Mock Quality Result ---
    mock_quality.return_value.compute_focus_score.return_value = 9000.0

    # --- Worker Config ---
    worker_config = {
        'filtered_path': str(filtered_dir),
        'report_path': str(filtered_dir / "report"), # Added for V7
        'model_name': 'mock.pt',
        'confidence_threshold': 0.5,
        'padding_ratio': 0.1,
        'target_class_id': 75,
        'blur_threshold_main': 4500.0,
        'blur_threshold_relaxed': 3000.0,
        'high_confidence_gate': 0.85,
        'min_area_ratio': 0.04,
        'blur_metric': 'tenengrad',
        'blur_resize_max_dim': 512,
        'hero_config': None
    }
    
    image_files = [raw_dir / "test_artifact.jpg"]
    
    # Execute Worker
    _worker_process("cpu", image_files, worker_config)

    # --- Assertions ---
    accepted_dir = filtered_dir / "accepted"
    assert accepted_dir.exists()
    
    crops = list(accepted_dir.glob("*.jpg"))
    assert len(crops) == 1, "Image should have been accepted and cropped"

