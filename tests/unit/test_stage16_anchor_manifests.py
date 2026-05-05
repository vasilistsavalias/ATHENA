from pathlib import Path

from thesis_pipeline.stages.stage_18_expert_validation import ExpertValidationStage


def test_anchor_selection_deterministic():
    stage = ExpertValidationStage.__new__(ExpertValidationStage)
    anchors = stage._anchor_sample_ids(["s3", "s1", "s2", "s4"], anchor_count=2)
    assert anchors == {"s1", "s2"}


def test_write_expert_manifests_creates_files(tmp_path):
    stage = ExpertValidationStage.__new__(ExpertValidationStage)
    stage.seed = 42

    pack_dir = Path(tmp_path) / "pack"
    public_items = [
        {"sample_id": "s1", "is_anchor": True},
        {"sample_id": "s2", "is_anchor": False},
    ]
    private_items = [
        {"sample_id": "s1", "is_anchor": True, "mapping": {"A": "real", "B": "restored"}},
        {"sample_id": "s2", "is_anchor": False, "mapping": {"A": "restored", "B": "real"}},
    ]

    created = stage._write_expert_manifests(
        pack_dir=pack_dir,
        pack_id_prefix="Expert_Pack_Top1vsReal",
        public_items=public_items,
        private_items=private_items,
        expert_count=2,
    )
    assert len(created) == 4
    assert (pack_dir / "expert_manifests" / "manifest_public_E01.json").exists()
    assert (pack_dir / "expert_manifests" / "manifest_private_E02.json").exists()


