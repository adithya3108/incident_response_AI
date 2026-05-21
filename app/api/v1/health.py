import time

from fastapi import APIRouter

from app.models.responses import HealthResponse, MetricsResponse

router = APIRouter()
_start_time = time.time()

# simple in-memory counters (shared via module state)
stats = {
    "total_queries": 0,
    "total_latency_ms": 0.0,
    "escalations": {"L1": 0, "L2": 0, "L3": 0},
}


@router.get("/health", response_model=HealthResponse)
async def health():
    from app.dependencies import _retriever, _llm_client

    return HealthResponse(
        status="healthy",
        indexes_loaded=_retriever is not None,
        model=_llm_client.model if _llm_client else "not loaded",
        uptime_seconds=round(time.time() - _start_time, 1),
    )


@router.get("/metrics", response_model=MetricsResponse)
async def metrics():
    from app.dependencies import _llm_client

    total = stats["total_queries"]
    avg_latency = stats["total_latency_ms"] / total if total > 0 else 0.0
    cache_hit_rate = _llm_client.cache_hit_rate if _llm_client else 0.0

    return MetricsResponse(
        total_queries=total,
        avg_latency_ms=round(avg_latency, 1),
        cache_hit_rate=round(cache_hit_rate, 3),
        escalations=stats["escalations"],
    )
