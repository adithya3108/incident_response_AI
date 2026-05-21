import asyncio
import time

from fastapi import APIRouter

from app.api.v1.health import stats
from app.core.confidence import assign_confidence
from app.dependencies import EmbedderDep, LLMDep, RerankerDep, RetrieverDep
from app.models.requests import BulkSearchRequest, SearchRequest
from app.models.responses import BulkSearchResponse, SearchResponse

router = APIRouter()


async def _run_search(
    query: str,
    top_k: int,
    request: SearchRequest | BulkSearchRequest,
    embedder,
    retriever,
    reranker,
    llm,
    settings_max_tokens: int = 4096,
) -> SearchResponse:
    t0 = time.perf_counter()

    query_vec = embedder.embed_query(query)
    filters = request.filters if hasattr(request, "filters") else None
    incidents = retriever.search(query_vec=query_vec, query=query, top_k=top_k, filters=filters)
    incidents = reranker.rerank(incidents)
    incidents = assign_confidence(incidents)

    resolution, routing = await llm.generate_resolution(query, incidents, settings_max_tokens)

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    stats["total_queries"] += 1
    stats["total_latency_ms"] += elapsed_ms

    top_confidence = incidents[0].confidence if incidents else "LOW"

    return SearchResponse(
        query=query,
        retrieved_incidents=incidents,
        resolution_suggestion=resolution,
        routing_suggestion=routing,
        confidence=top_confidence,
        processing_time_ms=elapsed_ms,
    )


@router.post("/search", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    embedder: EmbedderDep,
    retriever: RetrieverDep,
    reranker: RerankerDep,
    llm: LLMDep,
):
    return await _run_search(
        request.query, request.top_k, request, embedder, retriever, reranker, llm
    )


@router.post("/bulk-search", response_model=BulkSearchResponse)
async def bulk_search(
    request: BulkSearchRequest,
    embedder: EmbedderDep,
    retriever: RetrieverDep,
    reranker: RerankerDep,
    llm: LLMDep,
):
    tasks = [
        _run_search(q, request.top_k, request, embedder, retriever, reranker, llm)
        for q in request.queries
    ]
    results = await asyncio.gather(*tasks)
    return BulkSearchResponse(results=list(results))
