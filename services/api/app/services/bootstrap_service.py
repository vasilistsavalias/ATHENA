from __future__ import annotations

import logging
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.models import BlockBItem, Campaign
from app.services.upload_import_service import import_pack_zip

logger = logging.getLogger(__name__)


def _extract_relative_static_path(url: str, campaign_id: int) -> str | None:
    prefix = f"/static/{campaign_id}/"
    if not url.startswith(prefix):
        return None
    return url[len(prefix) :]


def _campaign_assets_ready(db: Session, settings: Settings, campaign: Campaign) -> bool:
    sample_item = db.execute(
        select(BlockBItem).where(BlockBItem.campaign_id == campaign.id).order_by(BlockBItem.id.asc()).limit(1)
    ).scalar_one_or_none()
    if sample_item is None:
        return False

    root = settings.storage_root / "campaigns" / str(campaign.id)
    candidate_urls = [sample_item.input_url, sample_item.option_a_url, sample_item.option_b_url]
    for url in candidate_urls:
        relative = _extract_relative_static_path(url, campaign.id)
        if not relative:
            return False
        if not (root / relative).exists():
            return False
    return True


def ensure_bootstrap_campaign(db: Session, settings: Settings) -> None:
    if not settings.bootstrap_pack_on_startup:
        return

    zip_path = Path(settings.bootstrap_pack_zip_path)
    if not zip_path.exists():
        message = f"Bootstrap pack zip not found: {zip_path}"
        if settings.bootstrap_strict:
            raise FileNotFoundError(message)
        logger.warning(message)
        return

    active_campaign = db.execute(
        select(Campaign).where(Campaign.is_active.is_(True)).order_by(Campaign.id.desc()).limit(1)
    ).scalar_one_or_none()

    if active_campaign and _campaign_assets_ready(db, settings, active_campaign):
        logger.info(
            "Bootstrap skipped; active campaign id=%s already has accessible assets.",
            active_campaign.id,
        )
        return

    reason = "missing active campaign"
    if active_campaign is not None:
        reason = f"active campaign id={active_campaign.id} has missing assets"
    logger.warning("Bootstrap import triggered (%s). Importing %s", reason, zip_path)

    imported = import_pack_zip(
        db,
        zip_path=zip_path,
        campaign_name=settings.bootstrap_campaign_name,
        seed=settings.bootstrap_campaign_seed,
        activate=settings.bootstrap_activate,
        disjoint_blocks=settings.bootstrap_disjoint_blocks,
    )
    logger.info(
        "Bootstrap import completed: campaign id=%s name='%s' active=%s",
        imported.id,
        imported.name,
        imported.is_active,
    )
