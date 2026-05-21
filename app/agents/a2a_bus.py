import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class A2AEvent:
    event_type: str  # ESCALATE | KNOWLEDGE_SHARE | RCA_REQUEST | RCA_RESULT
    source_agent: str
    target_agent: str
    payload: Any
    timestamp: datetime = field(default_factory=datetime.utcnow)


class A2ABus:
    def __init__(self):
        self._queue: asyncio.Queue[A2AEvent] = asyncio.Queue()
        self._log: list[A2AEvent] = []

    async def publish(self, event: A2AEvent) -> None:
        self._log.append(event)
        await self._queue.put(event)

    async def consume(self, timeout: float = 0.1) -> A2AEvent | None:
        try:
            return await asyncio.wait_for(self._queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    def get_knowledge_shares(self) -> list[A2AEvent]:
        return [e for e in self._log if e.event_type == "KNOWLEDGE_SHARE"]
