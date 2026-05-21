from app.agents.base_agent import AgentContext, AgentResponse, BaseAgent
from app.llm.prompts import L2_AGENT_PROMPT
from app.llm.token_optimizer import format_incidents_xml


class L2Agent(BaseAgent):
    tier = "L2"

    async def handle(self, context: AgentContext) -> AgentResponse:
        incidents = []
        try:
            from app.dependencies import _embedder
            if _embedder:
                qv = _embedder.embed_query(context.description)
                incidents = self.retriever.search(query=context.description, query_vec=qv, top_k=8)
        except Exception:
            pass

        incidents_xml = format_incidents_xml(incidents)
        attempts_text = "\n".join(context.resolution_attempts[-5:]) or "None"

        prompt = L2_AGENT_PROMPT.format(
            description=context.description,
            incidents_xml=incidents_xml,
            attempts=attempts_text,
        )
        response = await self.llm.call_agent(
            "You are an L2 technical support engineer. Provide detailed technical analysis.",
            prompt,
        )

        if response.strip().upper().startswith("ESCALATE_TO_L3"):
            return AgentResponse(
                tier=self.tier,
                response=response,
                resolved=False,
                escalation_reason="L2 determined specialist (L3) required",
            )

        await self._publish_knowledge(context, AgentResponse(tier=self.tier, response=response, resolved=True))
        return AgentResponse(
            tier=self.tier,
            response=response,
            resolved=True,
            knowledge_ids=[i.incident_id for i in incidents[:3]],
        )
