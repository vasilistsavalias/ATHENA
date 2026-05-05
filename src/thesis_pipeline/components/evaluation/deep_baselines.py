from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


class BaselineUnavailableError(RuntimeError):
    """Raised when a deep baseline backend is unavailable or misconfigured."""


@dataclass
class BaselineLoadReport:
    name: str
    available: bool
    detail: str


@dataclass
class DeepBaselineRunReport:
    model: str
    ok: bool
    detail: str
    duration_seconds: float | None = None


def _has_module(mod: str) -> bool:
    try:
        import importlib.util

        return importlib.util.find_spec(mod) is not None
    except Exception:
        return False


def _resolve_iopaint_cmd(cfg: Any | None) -> list[str] | None:
    """
    Resolve an `iopaint` invocation in a way that doesn't require `iopaint` to be installed
    in the *main* pipeline venv.

    Priority:
    1) cfg['iopaint_cli'] (path to iopaint executable)
    2) env var IOPAINT_CLI
    3) repo-local `.venv_iopaint` convention
    4) `iopaint` in PATH
    5) fallback: `python -m iopaint` if module exists
    """
    try:
        if cfg is not None:
            if isinstance(cfg, dict):
                val = cfg.get("iopaint_cli")
            else:
                val = getattr(cfg, "iopaint_cli", None)
            if val:
                return [str(val)]
    except Exception:
        pass

    env_cli = (os.environ.get("IOPAINT_CLI") or "").strip()
    if env_cli:
        return [env_cli]

    repo_root = Path(__file__).resolve().parents[4]
    candidates = [
        repo_root / ".venv_iopaint" / "bin" / "iopaint",
        repo_root / ".venv_iopaint" / "Scripts" / "iopaint.exe",
    ]
    for c in candidates:
        if c.exists():
            return [str(c)]

    which = shutil.which("iopaint")
    if which:
        return [which]

    if _has_module("iopaint"):
        return [sys.executable, "-m", "iopaint"]

    return None


def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _find_weight_candidates(root_dir: Path) -> list[Path]:
    # gdown folder downloads may place files in nested subfolders.
    candidates: list[Path] = []
    for pattern in ("*.pt", "*.pth"):
        candidates.extend([p for p in root_dir.rglob(pattern) if p.is_file()])
    return candidates


def _index_image_outputs(root_dir: Path) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    if not root_dir.exists():
        return outputs
    for file_path in root_dir.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            continue
        outputs[file_path.name] = file_path
    return outputs


def _copy_pairs_to_dirs(
    pairs: Iterable[tuple[Path, Path]],
    images_dir: Path,
    masks_dir: Path,
) -> list[str]:
    _safe_mkdir(images_dir)
    _safe_mkdir(masks_dir)
    names: list[str] = []
    for img_path, mask_path in pairs:
        name = img_path.name
        names.append(name)
        shutil.copy2(img_path, images_dir / name)
        shutil.copy2(mask_path, masks_dir / name)
    return names


def _collect_image_filenames(images_dir: Path) -> list[str]:
    names: list[str] = []
    for p in images_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            continue
        names.append(p.name)
    names.sort()
    return names


class DeepBaselineRunner:
    """Batch runner for deep inpainting baselines (LaMa, MAT, CoModGAN).

    This intentionally runs in *batch mode*:
    - LaMa/MAT via `iopaint run` (auto-downloads weights).
    - CoModGAN via MI-GAN's public inference code + downloaded weights.

    In main pipeline runs you likely want `required=true` so failures are visible.
    In smoke tests you can disable or set `required=false`.
    """

    MODEL_NAMES = ("LaMa", "MAT", "CoModGAN")

    def __init__(self, cfg: Any | None, stage13_dir: Path, logger: logging.Logger | None = None):
        self.cfg = cfg or {}
        self.stage13_dir = Path(stage13_dir)
        self.logger = logger or logging.getLogger(__name__)

        self.enabled = bool(self._cfg_get("enabled", False))
        self.required = bool(self._cfg_get("required", True))
        self.device = str(self._cfg_get("device", "cuda"))
        self.max_size = int(self._cfg_get("max_size", 512))

        comod = self._cfg_get("comodgan", {}) or {}
        self.migan_git_url = str(comod.get("migan_git_url", "https://github.com/Picsart-AI-Research/MI-GAN.git"))
        self.migan_commit = str(comod.get("migan_commit", "2381ef9d322caa4f90550f4b7072a6f681efb8c2"))
        self.migan_cache_dir = Path(comod.get("migan_cache_dir", str(Path(".cache") / "migan")))
        self.comodgan_model_name = str(comod.get("model_name", "comodgan-512"))
        self.comodgan_weights_url = str(
            comod.get("weights_url", "https://drive.google.com/drive/folders/1VATyNQQJW2VpuHND02bc-3_4ukJMHQ44")
        )
        self.comodgan_weights_dir = Path(comod.get("weights_dir", str(Path(".cache") / "migan_weights")))

        self._timing_ms_per_image: dict[str, float] = {}
        self._iopaint_supports_max_size: bool | None = None

    def _cfg_get(self, key: str, default: Any) -> Any:
        if isinstance(self.cfg, dict):
            return self.cfg.get(key, default)
        try:
            return getattr(self.cfg, key, default)
        except Exception:
            return default

    def probe(self) -> list[BaselineLoadReport]:
        reports: list[BaselineLoadReport] = []

        iopaint_cmd = _resolve_iopaint_cmd(self.cfg)
        iopaint_ok = iopaint_cmd is not None
        reports.append(
            BaselineLoadReport(
                name="iopaint",
                available=iopaint_ok,
                detail=f"cmd={' '.join(iopaint_cmd)}" if iopaint_ok else "missing (no iopaint command found)",
            )
        )

        gdown_ok = _has_module("gdown")
        reports.append(
            BaselineLoadReport(
                name="gdown",
                available=gdown_ok,
                detail="found" if gdown_ok else "missing (pip install gdown)",
            )
        )

        git_ok = shutil.which("git") is not None
        reports.append(
            BaselineLoadReport(
                name="git",
                available=git_ok,
                detail="found" if git_ok else "missing in PATH (required to clone MI-GAN)",
            )
        )

        return reports

    def prepare(self, pairs: list[tuple[Path, Path]]) -> list[DeepBaselineRunReport]:
        if not self.enabled:
            return []

        probe_reports = self.probe()
        missing = [r for r in probe_reports if not r.available]
        if missing and self.required:
            detail = "; ".join(f"{m.name}: {m.detail}" for m in missing)
            raise BaselineUnavailableError(f"Deep baselines required but missing deps ({detail}).")
        if missing and not self.required:
            for rep in missing:
                self.logger.warning(f"Deep baseline dependency missing: {rep.name} ({rep.detail})")

        deep_root = self.stage13_dir / "deep_baselines"
        inputs_dir = deep_root / "_inputs"
        images_dir = inputs_dir / "images"
        masks_dir = inputs_dir / "masks"

        if inputs_dir.exists():
            shutil.rmtree(inputs_dir, ignore_errors=True)
        sample_names = _copy_pairs_to_dirs(pairs, images_dir, masks_dir)

        reports: list[DeepBaselineRunReport] = []
        reports.append(self._run_iopaint_model("LaMa", "lama", images_dir, masks_dir, deep_root / "LaMa"))
        reports.append(self._run_iopaint_model("MAT", "mat", images_dir, masks_dir, deep_root / "MAT"))
        reports.append(self._run_comodgan(images_dir, masks_dir, deep_root / "CoModGAN"))

        for rep in reports:
            if rep.ok and rep.duration_seconds is not None and len(sample_names) > 0:
                self._timing_ms_per_image[rep.model] = (rep.duration_seconds * 1000.0) / len(sample_names)

        try:
            (deep_root / "deep_baseline_run_report.json").write_text(
                json.dumps([r.__dict__ for r in reports], indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass

        if self.required:
            failed = [r for r in reports if not r.ok]
            if failed:
                detail = "; ".join(f"{r.model}: {r.detail}" for r in failed)
                raise BaselineUnavailableError(f"Deep baseline execution failed ({detail}).")

        return reports

    def load_result(self, model: str, sample_filename: str) -> Image.Image:
        out_path = self.stage13_dir / "deep_baselines" / model / sample_filename
        if out_path.exists():
            return Image.open(out_path).convert("RGB")

        out_dir = self.stage13_dir / "deep_baselines" / model
        stem = Path(sample_filename).stem
        ext_priority = {".png": 0, ".jpg": 1, ".jpeg": 2, ".webp": 3}

        candidates = [p for p in out_dir.rglob("*") if p.is_file() and (p.stem == stem or p.stem.startswith(stem))]
        if candidates:
            candidates.sort(key=lambda p: (ext_priority.get(p.suffix.lower(), 99), len(str(p))))
            return Image.open(candidates[0]).convert("RGB")

        raise BaselineUnavailableError(f"Deep baseline output missing for {model}: {out_path}")

    def timing_ms_per_image(self, model: str) -> float | None:
        return self._timing_ms_per_image.get(model)

    def _run_iopaint_model(
        self,
        label: str,
        iopaint_model: str,
        images_dir: Path,
        masks_dir: Path,
        out_dir: Path,
    ) -> DeepBaselineRunReport:
        _safe_mkdir(out_dir)
        iopaint_cmd = _resolve_iopaint_cmd(self.cfg)
        if not iopaint_cmd:
            return DeepBaselineRunReport(model=label, ok=False, detail="missing dependency: iopaint")

        expected = {p.name for p in images_dir.glob("*.png")}
        existing = set(_index_image_outputs(out_dir).keys())
        if expected and expected.issubset(existing):
            return DeepBaselineRunReport(model=label, ok=True, detail="cached outputs reused", duration_seconds=0.0)

        cmd = [
            *iopaint_cmd,
            "run",
            "--model",
            iopaint_model,
            "--device",
            self.device,
            "--image",
            str(images_dir),
            "--mask",
            str(masks_dir),
            "--output",
            str(out_dir),
        ]
        if self._supports_iopaint_max_size(iopaint_cmd):
            cmd.extend(["--max-size", str(self.max_size)])

        self.logger.info(f"Deep baseline {label}: running IOPaint batch '{iopaint_model}' ...")
        t0 = time.perf_counter()
        env = self._build_iopaint_env_with_hf_shim(out_dir)
        try:
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
            )
            dur = time.perf_counter() - t0
            produced = set(_index_image_outputs(out_dir).keys())
            missing = sorted(expected - produced)
            if missing:
                preview = ", ".join(missing[:3])
                suffix = "..." if len(missing) > 3 else ""
                return DeepBaselineRunReport(
                    model=label,
                    ok=False,
                    detail=f"iopaint incomplete: missing {len(missing)}/{len(expected)} outputs ({preview}{suffix})",
                    duration_seconds=dur,
                )
            return DeepBaselineRunReport(model=label, ok=True, detail="ok", duration_seconds=dur)
        except subprocess.CalledProcessError as exc:
            tail = (exc.stdout or "")[-1500:]
            return DeepBaselineRunReport(model=label, ok=False, detail=f"iopaint failed: {tail}", duration_seconds=None)

    def _build_iopaint_env_with_hf_shim(self, out_dir: Path) -> dict[str, str]:
        shim_dir = out_dir.parent / "_runtime_py_shim"
        _safe_mkdir(shim_dir)
        shim_file = shim_dir / "sitecustomize.py"
        shim_file.write_text(
            textwrap.dedent(
                """
                import huggingface_hub as _hh
                try:
                    from huggingface_hub.utils import is_offline_mode as _is_offline_mode
                except Exception:
                    def _is_offline_mode():
                        return False
                if not hasattr(_hh, "is_offline_mode"):
                    _hh.is_offline_mode = _is_offline_mode
                if not hasattr(_hh, "cached_download"):
                    try:
                        from huggingface_hub import hf_hub_download as _hf_hub_download
                        def cached_download(*args, **kwargs):
                            return _hf_hub_download(*args, **kwargs)
                        _hh.cached_download = cached_download
                    except Exception:
                        pass
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )
        env = os.environ.copy()
        current = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            f"{shim_dir}{os.pathsep}{current}" if current else str(shim_dir)
        )
        return env

    def _supports_iopaint_max_size(self, iopaint_cmd: list[str]) -> bool:
        if self._iopaint_supports_max_size is not None:
            return self._iopaint_supports_max_size
        try:
            probe = subprocess.run(
                [*iopaint_cmd, "run", "--help"],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            self._iopaint_supports_max_size = "--max-size" in (probe.stdout or "")
        except Exception:
            self._iopaint_supports_max_size = False
        return self._iopaint_supports_max_size

    def _ensure_migan_repo(self) -> Path:
        repo_dir = Path(self.migan_cache_dir) / f"MI-GAN_{self.migan_commit[:12]}"
        if repo_dir.exists() and (repo_dir / ".git").exists():
            return repo_dir

        _safe_mkdir(repo_dir.parent)
        if repo_dir.exists():
            shutil.rmtree(repo_dir, ignore_errors=True)
        self.logger.info(f"Cloning MI-GAN repo into {repo_dir} ...")
        subprocess.run(["git", "clone", self.migan_git_url, str(repo_dir)], check=True)
        subprocess.run(["git", "checkout", self.migan_commit], cwd=str(repo_dir), check=True)
        return repo_dir

    def _pick_comodgan_weight(self, candidates: list[Path]) -> Path | None:
        if not candidates:
            return None

        want_res = "512" if "512" in self.comodgan_model_name else ""
        scored: list[tuple[int, Path]] = []
        for p in candidates:
            name = p.name.lower()
            score = 0
            if "comod" in name or "co-mod" in name or "comodgan" in name:
                score += 10
            if want_res and want_res in name:
                score += 5
            scored.append((score, p))
        scored.sort(key=lambda t: (t[0], t[1].name), reverse=True)
        return scored[0][1]

    def _ensure_comodgan_weights(self) -> Path:
        _safe_mkdir(self.comodgan_weights_dir)
        candidates = _find_weight_candidates(self.comodgan_weights_dir)
        chosen = self._pick_comodgan_weight(candidates)
        if chosen is not None:
            return chosen.resolve()

        if not _has_module("gdown"):
            raise BaselineUnavailableError("CoModGAN weights missing and 'gdown' is not installed.")

        self.logger.info("Downloading CoModGAN weights via gdown (Google Drive folder) ...")
        try:
            import gdown  # type: ignore

            gdown.download_folder(url=self.comodgan_weights_url, output=str(self.comodgan_weights_dir), quiet=True)
        except Exception as exc:
            raise BaselineUnavailableError(f"Failed downloading CoModGAN weights: {exc}") from exc

        candidates = _find_weight_candidates(self.comodgan_weights_dir)
        chosen = self._pick_comodgan_weight(candidates)
        if chosen is None:
            raise BaselineUnavailableError(f"Downloaded weights but none found in {self.comodgan_weights_dir}")
        return chosen.resolve()

    def _run_comodgan(self, images_dir: Path, masks_dir: Path, out_dir: Path) -> DeepBaselineRunReport:
        _safe_mkdir(out_dir)

        try:
            repo_dir = self._ensure_migan_repo()
        except Exception as exc:
            return DeepBaselineRunReport(model="CoModGAN", ok=False, detail=f"MI-GAN clone failed: {exc}")

        try:
            weights_path = self._ensure_comodgan_weights()
        except Exception as exc:
            return DeepBaselineRunReport(model="CoModGAN", ok=False, detail=str(exc))

        expected_names = _collect_image_filenames(images_dir)
        expected = set(expected_names)
        existing = set(_index_image_outputs(out_dir).keys())
        if expected and expected.issubset(existing):
            return DeepBaselineRunReport(model="CoModGAN", ok=True, detail="cached outputs reused", duration_seconds=0.0)

        # deep_baselines.py lives at: src/thesis_pipeline/components/evaluation/deep_baselines.py
        # Repo root is 4 parents up from this file.
        script_path = (Path(__file__).resolve().parents[4] / "scripts" / "utilities" / "run_comodgan_batch.py").resolve()
        if not script_path.exists():
            return DeepBaselineRunReport(model="CoModGAN", ok=False, detail=f"missing runner script: {script_path}")

        repo_dir = repo_dir.resolve()
        weights_path = weights_path.resolve()
        images_dir = images_dir.resolve()
        masks_dir = masks_dir.resolve()
        out_dir = out_dir.resolve()

        # Use sanitized deterministic filenames for CoModGAN input/output to avoid
        # path/parser edge-cases from long names, spaces or special characters.
        safe_root = out_dir.parent / "_comodgan_safe_io"
        safe_images_dir = safe_root / "images"
        safe_masks_dir = safe_root / "masks"
        safe_out_dir = safe_root / "outputs"
        if safe_root.exists():
            shutil.rmtree(safe_root, ignore_errors=True)
        _safe_mkdir(safe_images_dir)
        _safe_mkdir(safe_masks_dir)
        _safe_mkdir(safe_out_dir)

        name_map: dict[str, str] = {}
        missing_masks: list[str] = []
        for idx, original_name in enumerate(expected_names):
            src_img = images_dir / original_name
            src_mask = masks_dir / original_name
            if not src_mask.exists():
                missing_masks.append(original_name)
                continue
            safe_name = f"s{idx:06d}.png"
            name_map[original_name] = safe_name
            Image.open(src_img).convert("RGB").save(safe_images_dir / safe_name, format="PNG")
            Image.open(src_mask).convert("L").save(safe_masks_dir / safe_name, format="PNG")

        if missing_masks:
            preview = ", ".join(missing_masks[:3])
            suffix = "..." if len(missing_masks) > 3 else ""
            return DeepBaselineRunReport(
                model="CoModGAN",
                ok=False,
                detail=f"missing input masks: {len(missing_masks)}/{len(expected_names)} ({preview}{suffix})",
            )

        cmd = [
            sys.executable,
            str(script_path),
            "--mi-gan-dir",
            str(repo_dir),
            "--model-name",
            self.comodgan_model_name,
            "--weights",
            str(weights_path),
            "--images-dir",
            str(safe_images_dir),
            "--masks-dir",
            str(safe_masks_dir),
            "--out-dir",
            str(safe_out_dir),
            "--device",
            self.device,
            "--invert-mask",
        ]

        self.logger.info("Deep baseline CoModGAN: running batch script ...")
        t0 = time.perf_counter()
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            dur = time.perf_counter() - t0

            for original_name, safe_name in name_map.items():
                generated = safe_out_dir / safe_name
                if generated.exists():
                    shutil.copy2(generated, out_dir / original_name)

            produced = set(_index_image_outputs(out_dir).keys())
            missing = sorted(expected - produced)
            if missing:
                preview = ", ".join(missing[:3])
                suffix = "..." if len(missing) > 3 else ""
                return DeepBaselineRunReport(
                    model="CoModGAN",
                    ok=False,
                    detail=f"incomplete outputs: missing {len(missing)}/{len(expected)} ({preview}{suffix})",
                    duration_seconds=dur,
                )
            return DeepBaselineRunReport(model="CoModGAN", ok=True, detail="ok", duration_seconds=dur)
        except subprocess.CalledProcessError as exc:
            tail = (exc.stdout or "")[-1500:]
            return DeepBaselineRunReport(model="CoModGAN", ok=False, detail=f"comodgan failed: {tail}")


def build_deep_baseline_runner(
    cfg: Any | None, stage13_dir: Path, logger: logging.Logger | None = None
) -> tuple[DeepBaselineRunner, list[BaselineLoadReport]]:
    runner = DeepBaselineRunner(cfg=cfg, stage13_dir=stage13_dir, logger=logger)
    return runner, runner.probe()
