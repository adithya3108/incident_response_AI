from app.agents.a2a_bus import A2ABus, A2AEvent
from app.agents.base_agent import AgentContext, AgentResponse
from app.agents.l1_agent import L1Agent
from app.agents.l2_agent import L2Agent
from app.agents.l3_agent import L3Agent
from app.agents.rca_agent import RCAAgent
from app.agents.triage_agent import TriageAgent
from app.api.v1.health import stats
from app.core.retriever import HybridRetriever
from app.llm.client import ClaudeClient

TIER_ORDER = ["L1", "L2", "L3"]


class AgentCoordinator:
    def __init__(self, retriever: HybridRetriever, llm: ClaudeClient):
        self.bus = A2ABus()
        self.triage = TriageAgent(retriever, llm, self.bus)
        self.agents = {
            "L1": L1Agent(retriever, llm, self.bus),
            "L2": L2Agent(retriever, llm, self.bus),
            "L3": L3Agent(retriever, llm, self.bus),
        }
        self.rca = RCAAgent(retriever, llm, self.bus)

    async def route(self, context: AgentContext, start_tier: str = "L1") -> AgentResponse:
        triage_resp = await self.triage.handle(context)
        context.add_attempt("TRIAGE", triage_resp.response)

        start_idx = TIER_ORDER.index(start_tier) if start_tier in TIER_ORDER else 0

        for tier in TIER_ORDER[start_idx:]:
            agent = self.agents[tier]
            stats["escalations"][tier] = stats["escalations"].get(tier, 0) + 1
            response = await agent.handle(context)

            if response.resolved:
                await self.bus.publish(A2AEvent(
                    event_type="KNOWLEDGE_SHARE",
                    source_agent=tier,
                    target_agent="ALL",
                    payload={"incident_id": context.incident_id, "resolved_by": tier},
                ))
                return response

            context.add_attempt(tier, response.response)

        # all tiers exhausted → RCA
        return await self.rca.handle(context)

    async def analyze(self, incident_ids: list[str]) -> object:
        return await self.rca.analyze_incidents(incident_ids)
