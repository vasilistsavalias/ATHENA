from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Response, UploadFile, status
from fastapi.responses import JSONResponse, PlainTextResponse
from sqlalchemy.orm import Session

from app.api.deps import require_admin_access
from app.core.config import get_settings
from app.core.security import create_admin_session_token, secure_compare
from app.db.session import get_db
from app.schemas.admin import (
    AdminDashboardResponse,
    AdminLoginRequest,
    AdminRuntimeResetRequest,
    AdminRuntimeResetResponse,
    ImportPackResponse,
    AdminSessionResponse,
    ImportPackRequest,
)
from app.services.admin_dashboard_service import build_admin_dashboard
from app.services.admin_reset_service import reset_study_runtime
from app.services.audit_service import log_event
from app.services.export_service import build_export_bundle, bundle_to_csv
from app.services.ingest_service import import_pack
from app.services.upload_import_service import import_pack_upload

router = APIRouter()


def _set_admin_cookie(response: Response):
    settings = get_settings()
    cookie_kwargs: dict = dict(
        key=settings.admin_cookie_name,
        value=create_admin_session_token(),
        httponly=True,
        secure=settings.effective_cookie_secure,
        samesite=settings.effective_cookie_samesite,
        max_age=settings.admin_session_ttl_hours * 3600,
        path="/",
    )
    if settings.effective_cookie_domain:
        cookie_kwargs["domain"] = settings.effective_cookie_domain
    response.set_cookie(**cookie_kwargs)


def _clear_admin_cookie(response: Response):
    settings = get_settings()
    delete_kwargs: dict = dict(
        key=settings.admin_cookie_name,
        secure=settings.effective_cookie_secure,
        samesite=settings.effective_cookie_samesite,
        httponly=True,
        path="/",
    )
    if settings.effective_cookie_domain:
        delete_kwargs["domain"] = settings.effective_cookie_domain
    response.delete_cookie(**delete_kwargs)


@router.post("/auth/login", response_model=AdminSessionResponse)
def admin_login(payload: AdminLoginRequest, response: Response, db: Session = Depends(get_db)):
    settings = get_settings()
    if not secure_compare(payload.password, settings.effective_admin_ui_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid admin password.")

    _set_admin_cookie(response)
    dashboard = build_admin_dashboard(db)
    log_event(db, action="admin.login", campaign_id=dashboard["campaign"]["id"], payload={"auth_mode": "cookie"})
    db.commit()
    return AdminSessionResponse(
        authenticated=True,
        auth_mode="cookie",
        campaign_id=dashboard["campaign"]["id"] if dashboard["campaign"] else None,
        campaign_name=dashboard["campaign"]["name"] if dashboard["campaign"] else None,
    )


@router.post("/auth/logout")
def admin_logout(response: Response):
    _clear_admin_cookie(response)
    return {"message": "Admin logged out"}


@router.get("/session", response_model=AdminSessionResponse)
def admin_session(
    access: dict = Depends(require_admin_access),
    db: Session = Depends(get_db),
):
    dashboard = build_admin_dashboard(db)
    return AdminSessionResponse(
        authenticated=True,
        auth_mode=access["auth_mode"],
        campaign_id=dashboard["campaign"]["id"] if dashboard["campaign"] else None,
        campaign_name=dashboard["campaign"]["name"] if dashboard["campaign"] else None,
    )


@router.get("/dashboard", response_model=AdminDashboardResponse)
def admin_dashboard(
    campaign_id: int | None = Query(default=None),
    _: dict = Depends(require_admin_access),
    db: Session = Depends(get_db),
):
    try:
        return build_admin_dashboard(db, campaign_id=campaign_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/import-pack", response_model=ImportPackResponse)
def admin_import_pack(
    payload: ImportPackRequest,
    _: dict = Depends(require_admin_access),
    db: Session = Depends(get_db),
):
    try:
        campaign = import_pack(
            db,
            pack_dir=Path(payload.pack_dir),
            campaign_name=payload.campaign_name,
            seed=payload.seed,
            stage13_samples=Path(payload.stage13_samples) if payload.stage13_samples else None,
            activate=payload.activate,
            disjoint_blocks=payload.disjoint_blocks,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    log_event(
        db,
        action="admin.import_pack",
        campaign_id=campaign.id,
        payload={"campaign_name": campaign.name, "seed": campaign.seed},
    )
    db.commit()
    return ImportPackResponse(campaign_id=campaign.id, campaign_name=campaign.name, is_active=campaign.is_active)


@router.post("/import-pack-upload", response_model=ImportPackResponse)
def admin_import_pack_upload(
    campaign_name: str = Form(...),
    seed: int = Form(42),
    activate: bool = Form(True),
    disjoint_blocks: bool = Form(True),
    pack: UploadFile = File(...),
    _: dict = Depends(require_admin_access),
    db: Session = Depends(get_db),
):
    try:
        campaign = import_pack_upload(
            db,
            pack_file=pack,
            campaign_name=campaign_name,
            seed=seed,
            activate=activate,
            disjoint_blocks=disjoint_blocks,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        pack.file.close()

    log_event(
        db,
        action="admin.import_pack_upload",
        campaign_id=campaign.id,
        payload={"campaign_name": campaign.name, "seed": campaign.seed, "filename": pack.filename},
    )
    db.commit()
    return ImportPackResponse(campaign_id=campaign.id, campaign_name=campaign.name, is_active=campaign.is_active)


@router.post("/runtime/reset", response_model=AdminRuntimeResetResponse)
def admin_runtime_reset(
    payload: AdminRuntimeResetRequest,
    access: dict = Depends(require_admin_access),
    db: Session = Depends(get_db),
):
    settings = get_settings()
    if not secure_compare(payload.confirm_phrase, settings.admin_reset_confirm_phrase):
        raise HTTPException(status_code=400, detail="Invalid confirm_phrase.")
    try:
        report = reset_study_runtime(
            db,
            campaign_id=payload.campaign_id,
            all_active_campaigns=payload.all_active_campaigns,
            all_campaigns=payload.all_campaigns,
            remove_assets=payload.remove_assets,
            storage_root=settings.storage_root,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    log_event(
        db,
        action="admin.runtime_reset",
        campaign_id=report["campaign_ids"][0] if report["campaign_ids"] else None,
        payload={
            "campaign_ids": report["campaign_ids"],
            "deleted_counts": report["deleted_counts"],
            "remove_assets": payload.remove_assets,
            "all_campaigns": payload.all_campaigns,
            "auth_mode": access.get("auth_mode"),
            "identity_reset": report["identity_reset"],
        },
    )
    db.commit()
    return AdminRuntimeResetResponse(**report)


@router.get("/export/responses.csv")
def export_csv(
    campaign_id: int | None = Query(default=None),
    _: dict = Depends(require_admin_access),
    db: Session = Depends(get_db),
):
    try:
        bundle = build_export_bundle(db, campaign_id=campaign_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    csv_payload = bundle_to_csv(bundle)
    return PlainTextResponse(csv_payload, media_type="text/csv")


@router.get("/export/responses.json")
def export_json(
    campaign_id: int | None = Query(default=None),
    _: dict = Depends(require_admin_access),
    db: Session = Depends(get_db),
):
    try:
        bundle = build_export_bundle(db, campaign_id=campaign_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return JSONResponse(bundle)


@router.get("/export/quality_report.json")
def export_quality_report(
    campaign_id: int | None = Query(default=None),
    _: dict = Depends(require_admin_access),
    db: Session = Depends(get_db),
):
    try:
        bundle = build_export_bundle(db, campaign_id=campaign_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return JSONResponse(bundle["quality_report"])


@router.get("/debug/asset-probe")
def admin_asset_probe(
    campaign_id: int,
    relative_path: str,
    _: dict = Depends(require_admin_access),
):
    settings = get_settings()
    target = (settings.storage_root / "campaigns" / str(campaign_id) / Path(relative_path)).resolve()
    root = (settings.storage_root / "campaigns" / str(campaign_id)).resolve()
    if root not in target.parents and target != root:
        raise HTTPException(status_code=400, detail="Invalid relative_path.")
    return {
        "campaign_id": campaign_id,
        "relative_path": relative_path,
        "absolute_path": str(target),
        "exists": target.exists(),
        "is_file": target.is_file(),
    }
