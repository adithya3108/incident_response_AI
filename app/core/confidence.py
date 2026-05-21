from app.models.incident import RetrievedIncident

HIGH_THRESHOLD = 0.7
LOW_THRESHOLD = 0.4
RANK_PENALTY = 0.05


def assign_confidence(results: list[RetrievedIncident]) -> list[RetrievedIncident]:
    if not results:
        return results

    scores = [r.combined_score for r in results]
    min_s, max_s = min(scores), max(scores)
    rng = max_s - min_s or 1.0

    for rank, r in enumerate(results):
        normalized = (r.combined_score - min_s) / rng
        penalized = normalized * (1.0 - RANK_PENALTY * rank)
        penalized = max(0.0, min(1.0, penalized))

        if penalized >= HIGH_THRESHOLD:
            r.confidence = "HIGH"
        elif penalized >= LOW_THRESHOLD:
            r.confidence = "MEDIUM"
        else:
            r.confidence = "LOW"

    return results
