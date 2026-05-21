import math
from datetime import datetime

from app.models.incident import RetrievedIncident


class Reranker:
    RECENCY_LAMBDA = 0.01
    W_RRF = 0.6
    W_RECENCY = 0.2
    W_SUCCESS = 0.2

    def __init__(self, feedback_store: dict[str, tuple[int, int]] | None = None):
        # feedback_store: incident_id → (positive_count, total_count)
        self.feedback_store: dict[str, tuple[int, int]] = feedback_store or {}

    def rerank(self, results: list[RetrievedIncident]) -> list[RetrievedIncident]:
        if not results:
            return results

        rrf_scores = [r.rrf_score for r in results]
        min_rrf = min(rrf_scores)
        max_rrf = max(rrf_scores)
        rrf_range = max_rrf - min_rrf or 1.0

        now = datetime.utcnow()
        for r in results:
            norm_rrf = (r.rrf_score - min_rrf) / rrf_range

            recency = 0.5
            if r.resolved_at:
                days = (now - r.resolved_at).days
                recency = math.exp(-self.RECENCY_LAMBDA * days)

            pos, total = self.feedback_store.get(r.incident_id, (0, 0))
            success_rate = pos / total if total > 0 else 0.5

            combined = (
                self.W_RRF * norm_rrf
                + self.W_RECENCY * recency
                + self.W_SUCCESS * success_rate
            )
            r.combined_score = round(combined, 6)

        results.sort(key=lambda x: x.combined_score, reverse=True)
        return results

    def record_feedback(self, incident_id: str, helpful: bool) -> float:
        pos, total = self.feedback_store.get(incident_id, (0, 0))
        total += 1
        if helpful:
            pos += 1
        self.feedback_store[incident_id] = (pos, total)
        return pos / total
