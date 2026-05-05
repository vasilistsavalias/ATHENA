from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.session import get_db
from app.models import BlockBItem, Campaign
from app.schemas.session import CampaignInfo

router = APIRouter()


@router.get("/active", response_model=CampaignInfo)
def get_active_campaign(db: Session = Depends(get_db)):
    campaign = db.query(Campaign).filter(Campaign.is_active.is_(True)).order_by(Campaign.id.desc()).first()
    if campaign is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No active campaign.")
    settings = get_settings()
    block_c_target_count = 0
    for row in db.query(BlockBItem).filter(BlockBItem.campaign_id == campaign.id).all():
        part = str((row.metadata_json or {}).get("block_part") or "B").upper()
        if part == "C":
            block_c_target_count += 1
    return CampaignInfo(
        id=campaign.id,
        name=campaign.name,
        seed=campaign.seed,
        protocol_version=campaign.protocol_version or settings.expert_protocol_version,
        block_a_target_count=campaign.block_a_target_count,
        block_b_target_count=campaign.block_b_target_count,
        block_c_target_count=block_c_target_count,
    )
