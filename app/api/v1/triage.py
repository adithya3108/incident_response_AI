from fastapi import APIRouter

from app.dependencies import LLMDep
from app.models.requests import TriageRequest
from app.models.responses import TriageResponse

router = APIRouter()


@router.post("/triage", response_model=TriageResponse)
async def triage(request: TriageRequest, llm: LLMDep):
    return await llm.classify_priority(
        description=request.description,
        impact=request.impact or 2,
        urgency=request.urgency or 2,
    )
