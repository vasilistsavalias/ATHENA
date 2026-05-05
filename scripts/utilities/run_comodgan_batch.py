from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch CoModGAN inference via MI-GAN (for Stage 13 deep baselines).")
    parser.add_argument("--mi-gan-dir", required=True, type=Path, help="Path to a MI-GAN repo checkout.")
    parser.add_argument("--model-name", required=True, type=str, help="MI-GAN model name, e.g. 'comodgan-512'.")
    parser.add_argument("--weights", required=True, type=Path, help="Path to CoModGAN weights (.pt/.pth).")
    parser.add_argument("--images-dir", required=True, type=Path, help="Directory of PNG images.")
    parser.add_argument("--masks-dir", required=True, type=Path, help="Directory of PNG masks (same filenames).")
    parser.add_argument("--out-dir", required=True, type=Path, help="Directory to write inpainted PNG outputs.")
    parser.add_argument("--device", default="cuda", type=str, help="Device, e.g. 'cuda' or 'cpu'.")
    parser.add_argument("--invert-mask", action="store_true", help="Invert mask semantics (needed for our pipeline masks).")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    demo_script = args.mi_gan_dir / "scripts" / "demo.py"
    if not demo_script.exists():
        raise SystemExit(f"MI-GAN demo script not found: {demo_script}")

    # Delegate execution to MI-GAN's maintained demo entrypoint to avoid
    # depending on private module-level inference helpers.
    cmd = [
        sys.executable,
        "-m",
        "scripts.demo",
        "--model-name",
        args.model_name,
        "--model-path",
        str(args.weights),
        "--images-dir",
        str(args.images_dir),
        "--masks-dir",
        str(args.masks_dir),
        "--output-dir",
        str(args.out_dir),
        "--device",
        args.device,
    ]
    if args.invert_mask:
        cmd.append("--invert-mask")
    subprocess.run(cmd, cwd=str(args.mi_gan_dir), check=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

