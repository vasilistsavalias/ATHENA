"""Repack expert-validation zip so images are grouped by block folders.

Transforms manifest image paths from:
    images/<sample_id>/<file>.png
to:
    images/block_a/<sample_id>/<file>.png
    images/block_b/<sample_id>/<file>.png
    images/block_c/<sample_id>/<file>.png

Block assignment rules:
- `block_c_items` section -> block_c
- explicit `block_part == "C"` -> block_c
- sample_id prefix `BLOCKA__` or `blkA_` -> block_a
- otherwise -> block_b
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

PATH_KEYS = ("input", "A", "B", "C", "D")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repack expert zip into block_a/block_b/block_c image folders.")
    parser.add_argument(
        "--zip-path",
        default="website/services/api/bootstrap/final_expert_pack.zip",
        help="Input zip path.",
    )
    parser.add_argument(
        "--output-zip",
        default=None,
        help="Output zip path. If omitted, input zip is overwritten in place.",
    )
    return parser.parse_args()


def _infer_block_bucket(item: dict, section_name: str) -> str:
    part = str(item.get("block_part") or "").strip().upper()
    sample_id = str(item.get("sample_id") or "")
    if section_name == "block_c_items" or part == "C":
        return "block_c"
    if sample_id.startswith("BLOCKA__") or sample_id.startswith("blkA_"):
        return "block_a"
    return "block_b"


def _zip_dir(source_dir: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(source_dir).as_posix())


def _rewrite_manifest_paths(root: Path) -> tuple[int, dict[str, int]]:
    manifest_path = root / "manifest_public.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest_public.json in extracted zip: {root}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    moved_files = 0
    counters = {"block_a": 0, "block_b": 0, "block_c": 0}
    old_paths_to_delete: set[Path] = set()

    for section_name in ("practice_items", "items", "block_c_items"):
        section_items = manifest.get(section_name) or []
        for item in section_items:
            sample_id = str(item.get("sample_id") or "").strip()
            if not sample_id:
                continue
            bucket = _infer_block_bucket(item, section_name)
            counters[bucket] += 1
            for key in PATH_KEYS:
                relative_path = item.get(key)
                if not relative_path:
                    continue
                old_rel = str(relative_path).replace("\\", "/")
                old_path = root / old_rel
                if not old_path.exists():
                    continue
                filename = Path(old_rel).name
                new_rel = f"images/{bucket}/{sample_id}/{filename}"
                new_path = root / new_rel
                if new_path.resolve() != old_path.resolve():
                    new_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(old_path, new_path)
                    moved_files += 1
                    old_paths_to_delete.add(old_path)
                item[key] = new_rel

    for path in sorted(old_paths_to_delete):
        if path.exists():
            path.unlink()
    _purge_legacy_image_layout(root / "images")
    _remove_empty_dirs(root / "images")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return moved_files, counters


def _remove_empty_dirs(root: Path) -> None:
    if not root.exists():
        return
    for directory in sorted([d for d in root.rglob("*") if d.is_dir()], key=lambda d: len(d.parts), reverse=True):
        try:
            directory.rmdir()
        except OSError:
            pass


def _purge_legacy_image_layout(images_root: Path) -> None:
    if not images_root.exists():
        return
    for child in images_root.iterdir():
        if not child.is_dir():
            continue
        if child.name in {"block_a", "block_b", "block_c"}:
            continue
        shutil.rmtree(child, ignore_errors=True)


def main() -> int:
    args = parse_args()
    input_zip = Path(args.zip_path).resolve()
    if not input_zip.exists():
        raise FileNotFoundError(f"Zip file not found: {input_zip}")
    output_zip = Path(args.output_zip).resolve() if args.output_zip else input_zip

    with tempfile.TemporaryDirectory(prefix="expert_pack_repack_") as tmp:
        tmp_dir = Path(tmp)
        with zipfile.ZipFile(input_zip, "r") as archive:
            archive.extractall(tmp_dir)

        moved_files, counters = _rewrite_manifest_paths(tmp_dir)
        _zip_dir(tmp_dir, output_zip)

    print(
        json.dumps(
            {
                "input_zip": str(input_zip),
                "output_zip": str(output_zip),
                "moved_files": moved_files,
                "items_by_bucket": counters,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
