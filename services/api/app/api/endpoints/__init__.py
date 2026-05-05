from fastapi import APIRouter

from app.api.endpoints import admin, auth, block_a, block_b, block_c, campaign, progress, session


api_router = APIRouter()
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(session.router, prefix="/session", tags=["session"])
api_router.include_router(campaign.router, prefix="/campaign", tags=["campaign"])
api_router.include_router(block_a.router, prefix="/block-a", tags=["block-a"])
api_router.include_router(block_b.router, prefix="/block-b", tags=["block-b"])
api_router.include_router(block_c.router, prefix="/block-c", tags=["block-c"])
api_router.include_router(progress.router, tags=["progress"])
api_router.include_router(admin.router, prefix="/admin", tags=["admin"])
