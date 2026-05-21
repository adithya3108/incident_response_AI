from fastapi import APIRouter

from app.agents.base_agent import AgentContext
from app.agents.coordinator import AgentCoordinator
from app.dependencies import LLMDep, RetrieverDep
from app.models.requests import AnalyzeRequest, EscalateRequest
from app.models.responses import AnalyzeResponse, EscalateResponse

router = APIRouter()


def _get_coordinator(retriever, llm) -> AgentCoordinator:
    return AgentCoordinator(retriever, llm)


@router.post("/escalate", response_model=EscalateResponse)
async def escalate(request: EscalateRequest, retriever: RetrieverDep, llm: LLMDep):
    coordinator = _get_coordinator(retriever, llm)
    context = AgentContext(
        incident_id=request.incident_id,
        description=request.description,
        resolution_attempts=request.resolution_attempts,
    )
    response = await coordinator.route(context, start_tier=request.current_tier)
    return EscalateResponse(
        escalated_to=response.tier,
        agent_response=response.response,
        rca_initiated=response.tier == "RCA",
        knowledge_shared=response.knowledge_ids,
    )


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest, retriever: RetrieverDep, llm: LLMDep):
    coordinator = _get_coordinator(retriever, llm)
    return await coordinator.analyze(request.incident_ids)
