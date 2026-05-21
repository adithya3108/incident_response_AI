from app.agents.base_agent import AgentContext, AgentResponse, BaseAgent
from app.llm.prompts import L3_AGENT_PROMPT
from app.llm.token_optimizer import format_incidents_xml


class L3Agent(BaseAgent):
    tier = "L3"

    async def handle(self, context: AgentContext) -> AgentResponse:
        incidents = []
        try:
            from app.dependencies import _embedder
            if _embedder:
                qv = _embedder.embed_query(context.description)
                incidents = self.retriever.search(query=context.description, query_vec=qv, top_k=10)
        except Exception:
            pass

        incidents_xml = format_incidents_xml(incidents)
        attempts_text = "\n".join(context.resolution_attempts) or "None"

        prompt = L3_AGENT_PROMPT.format(
            description=context.description,
            incidents_xml=incidents_xml,
            attempts=attempts_text,
        )
        response = await self.llm.call_agent(
            "You are an L3 specialist engineer. Provide expert root cause analysis and resolution.",
            prompt,
        )

        await self._publish_knowledge(context, AgentResponse(tier=self.tier, response=response, resolved=True))
        return AgentResponse(
            tier=self.tier,
            response=response,
            resolved=True,
            knowledge_ids=[i.incident_id for i in incidents[:5]],
        )
