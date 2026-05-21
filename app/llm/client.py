import json
from typing import Any

from langsmith import traceable
from openai import AsyncOpenAI

from app.llm.prompts import (
    SYSTEM_PROMPT,
    TRIAGE_PROMPT,
    RCA_PROMPT,
    JUDGE_PROMPT,
)
from app.llm.token_optimizer import format_incidents_xml
from app.models.incident import RetrievedIncident
from app.models.responses import TriageResponse, RoutingSuggestion


class ClaudeClient:
    def __init__(
        self,
        api_key: str,
        model: str = "anthropic/claude-sonnet-4-5",
        base_url: str = "https://openrouter.ai/api/v1",
        http_referer: str = "http://localhost:8000",
        x_title: str = "Incident KB Assistant",
    ):
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers={
                "HTTP-Referer": http_referer,
                "X-Title": x_title,
            },
        )
        self.model = model
        self._total_input_tokens = 0
        self._total_output_tokens = 0

    @traceable(name="openrouter_chat")
    async def _chat(self, system: str, user: str, max_tokens: int = 1024) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        usage = response.usage
        if usage:
            self._total_input_tokens += usage.prompt_tokens
            self._total_output_tokens += usage.completion_tokens
        return response.choices[0].message.content or ""

    @traceable(name="generate_resolution")
    async def generate_resolution(
        self,
        query: str,
        incidents: list[RetrievedIncident],
        max_context_tokens: int = 4096,
    ) -> tuple[str, RoutingSuggestion | None]:
        incidents_xml = format_incidents_xml(incidents, max_context_tokens)
        user_msg = (
            f"SIMILAR HISTORICAL INCIDENTS:\n{incidents_xml}\n\n"
            f"QUERY: {query}\n\n"
            "Provide resolution steps, estimated time, team recommendation, and any escalation warnings."
        )
        text = await self._chat(SYSTEM_PROMPT, user_msg, max_tokens=1024)
        routing = self._extract_routing(text, incidents)
        return text, routing

    @traceable(name="classify_priority")
    async def classify_priority(self, description: str, impact: int = 2, urgency: int = 2) -> TriageResponse:
        prompt = TRIAGE_PROMPT.format(description=description, impact=impact, urgency=urgency)
        text = await self._chat(SYSTEM_PROMPT, prompt, max_tokens=512)
        return self._parse_triage(text)

    @traceable(name="analyze_root_cause")
    async def analyze_root_cause(self, incidents: list[RetrievedIncident]) -> dict[str, Any]:
        incidents_xml = format_incidents_xml(incidents, 6000)
        prompt = RCA_PROMPT.format(incidents_xml=incidents_xml)
        text = await self._chat(SYSTEM_PROMPT, prompt, max_tokens=1024)
        return self._parse_json(text, {
            "root_cause": "Unable to determine",
            "contributing_factors": [],
            "recommended_permanent_fix": "Manual investigation required",
            "similar_past_incidents": [],
        })

    async def call_agent(self, system_prompt: str, user_prompt: str) -> str:
        return await self._chat(system_prompt, user_prompt, max_tokens=1024)

    async def judge_steps(self, description: str, suggested_steps: str, ground_truth: str) -> dict:
        prompt = JUDGE_PROMPT.format(
            description=description,
            suggested_steps=suggested_steps,
            ground_truth=ground_truth,
        )
        text = await self._chat(SYSTEM_PROMPT, prompt, max_tokens=512)
        return self._parse_json(text, {
            "safety": 3, "completeness": 3, "ordering": 3,
            "technical_accuracy": 3, "overall": 3, "reasoning": "",
        })

    @property
    def cache_hit_rate(self) -> float:
        # OpenRouter doesn't expose cache hit rate; return 0
        return 0.0

    def _extract_routing(self, text: str, incidents: list[RetrievedIncident]) -> RoutingSuggestion | None:
        tier = "L1"
        if any(w in text.lower() for w in ["critical", "p1", "specialist", "escalate"]):
            tier = "L2"
        if any(w in text.lower() for w in ["root cause", "infrastructure", "database failure", "l3"]):
            tier = "L3"
        team = incidents[0].assigned_to if incidents else "IT Support"
        if not team or team.lower() in ("nan", "none", ""):
            team = "IT Support"
        return RoutingSuggestion(team=team, tier=tier, reasoning="Based on incident similarity and resolution complexity.")

    def _parse_triage(self, text: str) -> TriageResponse:
        data = self._parse_json(text, {
            "suggested_priority": "P3",
            "confidence": 0.5,
            "reasoning": text[:200],
            "suggested_team": "IT Support",
            "escalation_path": ["L1"],
        })
        return TriageResponse(**data)

    @staticmethod
    def _parse_json(text: str, fallback: dict) -> dict:
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
        except (json.JSONDecodeError, ValueError):
            pass
        return fallback
