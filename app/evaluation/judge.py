from dataclasses import dataclass

from app.llm.client import ClaudeClient


@dataclass
class JudgeScore:
    safety: int
    completeness: int
    ordering: int
    technical_accuracy: int
    overall: int
    reasoning: str


class LLMJudge:
    def __init__(self, llm: ClaudeClient):
        self.llm = llm

    async def judge_steps(
        self,
        incident_description: str,
        suggested_steps: str,
        ground_truth_resolution: str,
    ) -> JudgeScore:
        result = await self.llm.judge_steps(
            description=incident_description,
            suggested_steps=suggested_steps,
            ground_truth=ground_truth_resolution,
        )
        return JudgeScore(
            safety=result.get("safety", 3),
            completeness=result.get("completeness", 3),
            ordering=result.get("ordering", 3),
            technical_accuracy=result.get("technical_accuracy", 3),
            overall=result.get("overall", 3),
            reasoning=result.get("reasoning", ""),
        )
