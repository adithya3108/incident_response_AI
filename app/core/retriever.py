from datetime import datetime
from typing import Optional

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from app.models.incident import RetrievedIncident
from app.models.requests import MetadataFilter


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


class HybridRetriever:
    def __init__(
        self,
        faiss_index: faiss.Index,
        bm25: BM25Okapi,
        metadata: list[dict],
        rrf_k: int = 60,
    ):
        self.faiss_index = faiss_index
        self.bm25 = bm25
        self.metadata = metadata
        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        query_vec: np.ndarray,
        top_k: int = 10,
        filters: Optional[MetadataFilter] = None,
    ) -> list[RetrievedIncident]:
        fetch_k = min(top_k * 4, len(self.metadata))

        faiss_ids = self._faiss_search(query_vec, fetch_k)
        bm25_ids = self._bm25_search(query, fetch_k)

        if filters:
            allowed = self._filter_ids(filters)
            faiss_ids = [(i, s) for i, s in faiss_ids if i in allowed]
            bm25_ids = [(i, s) for i, s in bm25_ids if i in allowed]

        merged = self._rrf(bm25_ids, faiss_ids)
        top = merged[:top_k]

        results = []
        for idx, score in top:
            m = self.metadata[idx]
            resolved_at = None
            if m.get("resolved_at"):
                try:
                    resolved_at = datetime.fromisoformat(m["resolved_at"])
                except ValueError:
                    pass
            recency_days = None
            if resolved_at:
                recency_days = (datetime.utcnow() - resolved_at).days

            results.append(RetrievedIncident(
                incident_id=m["incident_id"],
                description=m["description"],
                resolution_notes=m["resolution_notes"],
                priority=m["priority"],
                impact=m["impact"],
                assigned_to=m["assigned_to"],
                incident_state=m["incident_state"],
                resolved_at=resolved_at,
                rrf_score=round(score, 6),
                combined_score=round(score, 6),
                recency_days=recency_days,
            ))
        return results

    def _faiss_search(self, query_vec: np.ndarray, k: int) -> list[tuple[int, float]]:
        scores, indices = self.faiss_index.search(query_vec.reshape(1, -1), k)
        return [(int(idx), float(score)) for idx, score in zip(indices[0], scores[0]) if idx >= 0]

    def _bm25_search(self, query: str, k: int) -> list[tuple[int, float]]:
        tokens = _tokenize(query)
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in top_indices if scores[i] > 0]

    def _rrf(
        self,
        list_a: list[tuple[int, float]],
        list_b: list[tuple[int, float]],
    ) -> list[tuple[int, float]]:
        scores: dict[int, float] = {}
        for rank, (idx, _) in enumerate(list_a):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (self.rrf_k + rank + 1)
        for rank, (idx, _) in enumerate(list_b):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (self.rrf_k + rank + 1)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def _filter_ids(self, filters: MetadataFilter) -> set[int]:
        allowed = set()
        for i, m in enumerate(self.metadata):
            if filters.priority and m["priority"] not in filters.priority:
                continue
            if filters.impact and m["impact"] not in filters.impact:
                continue
            if filters.incident_state and m["incident_state"] not in filters.incident_state:
                continue
            allowed.add(i)
        return allowed
