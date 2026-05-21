from fastapi import APIRouter

from app.api.v1 import agents, feedback, health, search, triage

router = APIRouter()
router.include_router(health.router, tags=["health"])
router.include_router(search.router, prefix="/v1", tags=["search"])
router.include_router(triage.router, prefix="/v1", tags=["triage"])
router.include_router(agents.router, prefix="/v1", tags=["agents"])
router.include_router(feedback.router, prefix="/v1", tags=["feedback"])
