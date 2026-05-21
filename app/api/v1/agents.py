from fastapi import APIRouter

from app.agents.graph import GraphCoordinator
from app.dependencies import LLMDep, RetrieverDep
from app.models.requests import AnalyzeRequest, EscalateRequest
from app.models.responses import AnalyzeResponse, EscalateResponse

router = APIRouter()


@router.post("/escalate", response_model=EscalateResponse)
async def escalate(request: EscalateRequest, retriever: RetrieverDep, llm: LLMDep):
    coordinator = GraphCoordinator(retriever, llm)
    result = await coordinator.route(
        incident_id=request.incident_id,
        description=request.description,
        resolution_attempts=request.resolution_attempts,
        start_tier=request.current_tier,
    )
    return EscalateResponse(
        escalated_to=result["tier"],
        agent_response=result["response"],
        rca_initiated=result["rca_initiated"],
        knowledge_shared=result["knowledge_ids"],
    )


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest, retriever: RetrieverDep, llm: LLMDep):
    coordinator = GraphCoordinator(retriever, llm)
    return await coordinator.analyze(request.incident_ids)
