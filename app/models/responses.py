from typing import Optional
from pydantic import BaseModel
from app.models.incident import RetrievedIncident


class RoutingSuggestion(BaseModel):
    team: str
    tier: str
    reasoning: str


class SearchResponse(BaseModel):
    query: str
    retrieved_incidents: list[RetrievedIncident]
    resolution_suggestion: str
    routing_suggestion: Optional[RoutingSuggestion] = None
    confidence: str
    processing_time_ms: int


class BulkSearchResponse(BaseModel):
    results: list[SearchResponse]
    total_tokens_used: int = 0


class TriageResponse(BaseModel):
    suggested_priority: str
    confidence: float
    reasoning: str
    suggested_team: str
    escalation_path: list[str]


class EscalateResponse(BaseModel):
    escalated_to: str
    agent_response: str
    rca_initiated: bool = False
    knowledge_shared: list[str] = []


class AnalyzeResponse(BaseModel):
    root_cause: str
    contributing_factors: list[str]
    recommended_permanent_fix: str
    similar_past_incidents: list[str]


class FeedbackResponse(BaseModel):
    accepted: bool
    new_success_rate: float


class HealthResponse(BaseModel):
    status: str
    indexes_loaded: bool
    model: str
    uptime_seconds: float


class MetricsResponse(BaseModel):
    total_queries: int
    avg_latency_ms: float
    cache_hit_rate: float
    escalations: dict[str, int]
