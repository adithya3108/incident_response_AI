from fastapi import APIRouter

from app.dependencies import RerankerDep
from app.models.requests import FeedbackRequest
from app.models.responses import FeedbackResponse

router = APIRouter()


@router.post("/feedback", response_model=FeedbackResponse)
async def feedback(request: FeedbackRequest, reranker: RerankerDep):
    new_rate = reranker.record_feedback(request.incident_id, request.helpful)
    return FeedbackResponse(accepted=True, new_success_rate=round(new_rate, 3))
