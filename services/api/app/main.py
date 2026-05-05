from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text

from app.api.endpoints import api_router
from app.core.config import get_settings
from app.db.session import SessionLocal, init_db
from app.services.bootstrap_service import ensure_bootstrap_campaign

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    settings = get_settings()
    application = FastAPI(
        title=settings.app_name,
        description="ATHENA expert-evaluation backend for invite-code sessions, optional expert profiles, and stage-end feedback.",
        version="0.1.0",
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.include_router(api_router, prefix="/api/v1")

    campaigns_path = settings.storage_root / "campaigns"
    campaigns_path.mkdir(parents=True, exist_ok=True)
    application.mount(settings.static_mount_path, StaticFiles(directory=campaigns_path), name="static")

    @application.on_event("startup")
    def _on_startup():
        init_db()
        with SessionLocal() as db:
            ensure_bootstrap_campaign(db, settings)
            db.commit()
        logger.info("APP_ENV=%s", settings.app_env)
        logger.info("ALLOWED_ORIGINS=%s", settings.allowed_origins_list)
        logger.info(
            "Cookie: samesite=%s  secure=%s  domain=%s",
            settings.session_cookie_samesite,
            settings.effective_cookie_secure,
            settings.effective_cookie_domain or "(browser default)",
        )

    @application.get("/health", tags=["health"])
    def health():
        return {"status": "ok"}

    @application.get("/health/db", tags=["health"])
    def health_db():
        with SessionLocal() as db:
            db.execute(text("SELECT 1"))
        return {"status": "ok", "database": "reachable"}

    @application.get("/", tags=["health"])
    def root():
        return {"service": settings.app_name, "docs": "/docs"}

    return application


app = create_app()
