"""Shared singletons loaded at startup and injected via FastAPI dependencies."""
from typing import Annotated

from fastapi import Depends

from app.core.confidence import assign_confidence
from app.core.embedder import Embedder
from app.core.indexer import HybridIndexer
from app.core.reranker import Reranker
from app.core.retriever import HybridRetriever
from app.llm.client import ClaudeClient

# populated during lifespan startup
_embedder: Embedder | None = None
_retriever: HybridRetriever | None = None
_reranker: Reranker | None = None
_llm_client: ClaudeClient | None = None


def init_all(settings) -> None:
    global _embedder, _retriever, _reranker, _llm_client

    _embedder = Embedder(settings.embedding_model, settings.embeddings_cache_path)

    faiss_index, bm25, metadata = HybridIndexer.load(
        settings.faiss_index_path,
        settings.bm25_index_path,
        settings.metadata_store_path,
    )
    _retriever = HybridRetriever(faiss_index, bm25, metadata, rrf_k=settings.rrf_k)
    _reranker = Reranker()
    _llm_client = ClaudeClient(
        api_key=settings.openrouter_api_key,
        model=settings.llm_model,
        base_url=settings.openrouter_base_url,
        http_referer=settings.http_referer,
        x_title=settings.x_title,
    )


def get_embedder() -> Embedder:
    return _embedder


def get_retriever() -> HybridRetriever:
    return _retriever


def get_reranker() -> Reranker:
    return _reranker


def get_llm_client() -> ClaudeClient:
    return _llm_client


EmbedderDep = Annotated[Embedder, Depends(get_embedder)]
RetrieverDep = Annotated[HybridRetriever, Depends(get_retriever)]
RerankerDep = Annotated[Reranker, Depends(get_reranker)]
LLMDep = Annotated[ClaudeClient, Depends(get_llm_client)]
