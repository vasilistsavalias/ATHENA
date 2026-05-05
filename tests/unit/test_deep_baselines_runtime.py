from pathlib import Path
import subprocess

from PIL import Image

from thesis_pipeline.components.evaluation.deep_baselines import DeepBaselineRunner


def test_iopaint_run_omits_max_size_when_cli_does_not_support_it(tmp_path, monkeypatch):
    images_dir = tmp_path / "images"
    masks_dir = tmp_path / "masks"
    out_dir = tmp_path / "out"
    images_dir.mkdir()
    masks_dir.mkdir()
    Image.new("RGB", (8, 8), color="white").save(images_dir / "a.png")
    Image.new("L", (8, 8), color=255).save(masks_dir / "a.png")

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if "--help" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout="usage: iopaint run [OPTIONS]")
        if "run" in cmd:
            Image.new("RGB", (8, 8), color="white").save(out_dir / "a.png")
        return subprocess.CompletedProcess(cmd, 0, stdout="ok")

    monkeypatch.setattr(subprocess, "run", fake_run)
    runner = DeepBaselineRunner(cfg={"iopaint_cli": "iopaint"}, stage13_dir=tmp_path)
    report = runner._run_iopaint_model("LaMa", "lama", images_dir, masks_dir, out_dir)

    assert report.ok is True
    run_cmd = calls[-1]
    assert "--max-size" not in run_cmd


def test_comodgan_weights_are_discovered_recursively(tmp_path):
    nested = tmp_path / "downloads" / "subdir"
    nested.mkdir(parents=True)
    weight_file = nested / "comodgan_512_places2.pt"
    weight_file.write_bytes(b"x")

    runner = DeepBaselineRunner(
        cfg={
            "enabled": True,
            "comodgan": {
                "weights_dir": str(tmp_path),
            },
        },
        stage13_dir=tmp_path,
    )

    chosen = runner._ensure_comodgan_weights()
    assert Path(chosen) == weight_file


def test_comodgan_run_uses_absolute_paths(tmp_path, monkeypatch):
    images_dir = tmp_path / "images"
    masks_dir = tmp_path / "masks"
    out_dir = tmp_path / "out"
    images_dir.mkdir()
    masks_dir.mkdir()
    original_name = "a name with spaces.png"
    Image.new("RGB", (8, 8), color="white").save(images_dir / original_name)
    Image.new("L", (8, 8), color=255).save(masks_dir / original_name)

    runner = DeepBaselineRunner(
        cfg={"enabled": True},
        stage13_dir=tmp_path,
    )

    monkeypatch.setattr(
        runner,
        "_ensure_migan_repo",
        lambda: tmp_path,
    )
    monkeypatch.setattr(
        runner,
        "_ensure_comodgan_weights",
        lambda: tmp_path / "comodgan_512_places2.pt",
    )
    (tmp_path / "comodgan_512_places2.pt").write_bytes(b"x")

    recorded_cmd = {}

    def fake_run(cmd, **kwargs):
        recorded_cmd["cmd"] = cmd
        images_flag = cmd.index("--images-dir")
        out_flag = cmd.index("--out-dir")
        safe_images = Path(cmd[images_flag + 1])
        safe_out = Path(cmd[out_flag + 1])
        safe_out.mkdir(parents=True, exist_ok=True)
        for image_file in safe_images.glob("*.png"):
            Image.new("RGB", (8, 8), color="white").save(safe_out / image_file.name)
        return subprocess.CompletedProcess(cmd, 0, stdout="ok")

    monkeypatch.setattr(subprocess, "run", fake_run)

    report = runner._run_comodgan(images_dir, masks_dir, out_dir)
    assert report.ok is True
    assert (out_dir / original_name).exists()
    cmd = recorded_cmd["cmd"]
    mi_gan_dir = Path(cmd[cmd.index("--mi-gan-dir") + 1])
    weights = Path(cmd[cmd.index("--weights") + 1])
    images = Path(cmd[cmd.index("--images-dir") + 1])
    masks = Path(cmd[cmd.index("--masks-dir") + 1])
    out = Path(cmd[cmd.index("--out-dir") + 1])
    assert Path(cmd[1]).is_absolute()
    assert mi_gan_dir.is_absolute()
    assert weights.is_absolute()
    assert images.is_absolute()
    assert masks.is_absolute()
    assert out.is_absolute()
