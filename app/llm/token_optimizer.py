from app.models.incident import RetrievedIncident

MAX_RESOLUTION_CHARS = 300
MAX_DESCRIPTION_CHARS = 400


def format_incidents_xml(incidents: list[RetrievedIncident], max_tokens: int = 4096) -> str:
    """Format retrieved incidents as XML context, truncating to fit token budget."""
    parts = []
    estimated_tokens = 0

    for i, inc in enumerate(incidents):
        desc = inc.description[:MAX_DESCRIPTION_CHARS]
        res = inc.resolution_notes[:MAX_RESOLUTION_CHARS] if inc.resolution_notes else "N/A"

        chunk = (
            f"<incident id='{inc.incident_id}' priority='{inc.priority}' "
            f"confidence='{inc.confidence}'>\n"
            f"  <description>{desc}</description>\n"
            f"  <resolution>{res}</resolution>\n"
            f"</incident>"
        )
        # rough token estimate: 1 token ≈ 4 chars
        chunk_tokens = len(chunk) // 4
        if estimated_tokens + chunk_tokens > max_tokens:
            break
        parts.append(chunk)
        estimated_tokens += chunk_tokens

    return "\n".join(parts)


def deduplicate_contexts(contexts: list[str]) -> list[str]:
    """Remove duplicate context blocks in bulk queries."""
    seen: set[str] = set()
    result = []
    for ctx in contexts:
        if ctx not in seen:
            seen.add(ctx)
            result.append(ctx)
    return result
