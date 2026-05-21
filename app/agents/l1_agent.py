import numpy as np

from app.agents.base_agent import AgentContext, AgentResponse, BaseAgent
from app.llm.prompts import L1_AGENT_PROMPT
from app.llm.token_optimizer import format_incidents_xml


class L1Agent(BaseAgent):
    tier = "L1"

    async def handle(self, context: AgentContext) -> AgentResponse:
        query_vec = self.llm.client if False else None  # avoid circular; embed inline
        incidents = []
        try:
            from app.dependencies import _embedder
            if _embedder:
                qv = _embedder.embed_query(context.description)
                incidents = self.retriever.search(query=context.description, query_vec=qv, top_k=5)
        except Exception:
            pass

        incidents_xml = format_incidents_xml(incidents)
        attempts_text = "\n".join(context.resolution_attempts[-3:]) or "None"

        prompt = L1_AGENT_PROMPT.format(
            description=context.description,
            incidents_xml=incidents_xml,
            attempts=attempts_text,
        )
        response = await self.llm.call_agent(
            "You are an L1 IT support agent. Be concise and practical.",
            prompt,
        )

        if response.strip().upper().startswith("ESCALATE_TO_L2"):
            return AgentResponse(
                tier=self.tier,
                response=response,
                resolved=False,
                escalation_reason="L1 determined escalation to L2 required",
            )

        await self._publish_knowledge(context, AgentResponse(tier=self.tier, response=response, resolved=True))
        return AgentResponse(
            tier=self.tier,
            response=response,
            resolved=True,
            knowledge_ids=[i.incident_id for i in incidents[:3]],
        )
