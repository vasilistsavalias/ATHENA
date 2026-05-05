from __future__ import annotations

import shutil
import tempfile
import zipfile
from pathlib import Path

from fastapi import UploadFile
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models import Campaign
from app.services.ingest_service import import_pack


def _safe_extract_zip(zip_path: Path, extract_dir: Path) -> None:
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            member_path = extract_dir / member.filename
            resolved = member_path.resolve()
            if extract_dir.resolve() not in resolved.parents and resolved != extract_dir.resolve():
                raise ValueError(f"Unsafe zip entry detected: {member.filename}")
        archive.extractall(extract_dir)


def _discover_pack_root(extract_dir: Path) -> Path:
    direct_manifest = extract_dir / "manifest_public.json"
    if direct_manifest.exists():
        return extract_dir

    manifests = list(extract_dir.rglob("manifest_public.json"))
    if not manifests:
        raise FileNotFoundError("Uploaded zip does not contain manifest_public.json.")
    if len(manifests) > 1:
        raise ValueError("Uploaded zip contains multiple manifest_public.json files; expected exactly one.")
    return manifests[0].parent


def import_pack_upload(
    db: Session,
    *,
    pack_file: UploadFile,
    campaign_name: str,
    seed: int,
    activate: bool = True,
    disjoint_blocks: bool = True,
) -> Campaign:
    if not pack_file.filename or not pack_file.filename.lower().endswith(".zip"):
        raise ValueError("Uploaded file must be a .zip pack.")

    settings = get_settings()
    temp_root = settings.storage_root / "tmp"
    temp_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=temp_root, prefix="pack_upload_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        archive_path = tmp_path / "upload.zip"
        with archive_path.open("wb") as handle:
            shutil.copyfileobj(pack_file.file, handle)
        return import_pack_zip(
            db,
            zip_path=archive_path,
            campaign_name=campaign_name,
            seed=seed,
            activate=activate,
            disjoint_blocks=disjoint_blocks,
        )


def import_pack_zip(
    db: Session,
    *,
    zip_path: Path,
    campaign_name: str,
    seed: int,
    activate: bool = True,
    disjoint_blocks: bool = True,
) -> Campaign:
    settings = get_settings()
    temp_root = settings.storage_root / "tmp"
    temp_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=temp_root, prefix="pack_zip_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        archive_copy = tmp_path / "pack.zip"
        shutil.copy2(zip_path, archive_copy)

        extract_dir = tmp_path / "extracted"
        extract_dir.mkdir(parents=True, exist_ok=True)
        _safe_extract_zip(archive_copy, extract_dir)
        pack_root = _discover_pack_root(extract_dir)

        return import_pack(
            db,
            pack_dir=pack_root,
            campaign_name=campaign_name,
            seed=seed,
            stage13_samples=None,
            activate=activate,
            disjoint_blocks=disjoint_blocks,
        )
