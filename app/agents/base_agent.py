from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from app.agents.a2a_bus import A2ABus, A2AEvent
from app.core.retriever import HybridRetriever
from app.llm.client import ClaudeClient


@dataclass
class AgentContext:
    incident_id: str
    description: str
    resolution_attempts: list[str] = field(default_factory=list)
    tier_history: list[str] = field(default_factory=list)

    def add_attempt(self, tier: str, response: str) -> None:
        self.resolution_attempts.append(f"[{tier}]: {response[:300]}")
        self.tier_history.append(tier)


@dataclass
class AgentResponse:
    tier: str
    response: str
    resolved: bool
    escalation_reason: str = ""
    knowledge_ids: list[str] = field(default_factory=list)


class BaseAgent(ABC):
    tier: str = "BASE"

    def __init__(self, retriever: HybridRetriever, llm: ClaudeClient, bus: A2ABus):
        self.retriever = retriever
        self.llm = llm
        self.bus = bus

    @abstractmethod
    async def handle(self, context: AgentContext) -> AgentResponse: ...

    async def _publish_knowledge(self, context: AgentContext, response: AgentResponse) -> None:
        await self.bus.publish(A2AEvent(
            event_type="KNOWLEDGE_SHARE",
            source_agent=self.tier,
            target_agent="ALL",
            payload={"incident_id": context.incident_id, "resolution": response.response[:500]},
        ))
