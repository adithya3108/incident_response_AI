from app.agents.base_agent import AgentContext, AgentResponse, BaseAgent
from app.models.responses import AnalyzeResponse


class RCAAgent(BaseAgent):
    tier = "RCA"

    async def handle(self, context: AgentContext) -> AgentResponse:
        incidents = []
        try:
            from app.dependencies import _embedder
            if _embedder:
                qv = _embedder.embed_query(context.description)
                incidents = self.retriever.search(query=context.description, query_vec=qv, top_k=10)
        except Exception:
            pass

        result = await self.llm.analyze_root_cause(incidents)
        response_text = (
            f"ROOT CAUSE: {result.get('root_cause', 'Unknown')}\n"
            f"CONTRIBUTING FACTORS: {', '.join(result.get('contributing_factors', []))}\n"
            f"RECOMMENDED FIX: {result.get('recommended_permanent_fix', 'N/A')}"
        )
        return AgentResponse(
            tier=self.tier,
            response=response_text,
            resolved=True,
            knowledge_ids=[i.incident_id for i in incidents[:5]],
        )

    async def analyze_incidents(self, incident_ids: list[str]) -> AnalyzeResponse:
        incidents = []
        try:
            from app.dependencies import _retriever, _embedder
            if _retriever and _embedder:
                query = " ".join(incident_ids)
                qv = _embedder.embed_query(query)
                all_results = _retriever.search(query=query, query_vec=qv, top_k=15)
                id_set = set(incident_ids)
                incidents = [r for r in all_results if r.incident_id in id_set] or all_results[:10]
        except Exception:
            pass

        result = await self.llm.analyze_root_cause(incidents)
        return AnalyzeResponse(
            root_cause=result.get("root_cause", "Unable to determine"),
            contributing_factors=result.get("contributing_factors", []),
            recommended_permanent_fix=result.get("recommended_permanent_fix", "Manual investigation required"),
            similar_past_incidents=result.get("similar_past_incidents", []),
        )
