from app.agents.base_agent import AgentContext, AgentResponse, BaseAgent
from app.llm.prompts import TRIAGE_PROMPT


class TriageAgent(BaseAgent):
    tier = "TRIAGE"

    async def handle(self, context: AgentContext) -> AgentResponse:
        result = await self.llm.classify_priority(context.description)

        tier = result.escalation_path[-1] if result.escalation_path else "L1"
        response_text = (
            f"Priority: {result.suggested_priority} | Team: {result.suggested_team} | "
            f"Start at: {tier}"
        )
        return AgentResponse(
            tier=self.tier,
            response=response_text,
            resolved=False,
            escalation_reason=f"Route to {tier}",
        )
