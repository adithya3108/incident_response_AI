import asyncio
from dataclasses import dataclass

from deepeval.test_case import LLMTestCase

from app.core.confidence import assign_confidence
from app.core.reranker import Reranker
from app.evaluation.metrics import FixAccuracyMetric, RetrievalRelevanceMetric, ResolutionTimePredictionMetric
from app.models.incident import IncidentRecord


@dataclass
class EvaluationReport:
    fix_accuracy: float
    retrieval_relevance: float
    resolution_time_accuracy: float
    total_cases: int
    passed: int


class IncidentEvaluator:
    def __init__(self, retriever, embedder, llm):
        self.retriever = retriever
        self.embedder = embedder
        self.llm = llm
        self.reranker = Reranker()

    def build_test_cases(self, test_incidents: list[IncidentRecord]) -> list[LLMTestCase]:
        cases = []
        for inc in test_incidents:
            if not inc.resolution_notes or len(inc.resolution_notes) < 20:
                continue
            qv = self.embedder.embed_query(inc.description)
            retrieved = self.retriever.search(query=inc.description, query_vec=qv, top_k=5)
            retrieved = self.reranker.rerank(retrieved)
            retrieved = assign_confidence(retrieved)
            ctx = [r.description + " " + r.resolution_notes for r in retrieved]
            cases.append(LLMTestCase(
                input=inc.description,
                actual_output="",  # filled during async run
                expected_output=inc.resolution_notes,
                retrieval_context=ctx,
            ))
        return cases

    async def run_suite(self, test_incidents: list[IncidentRecord]) -> EvaluationReport:
        cases = self.build_test_cases(test_incidents[:50])

        for case in cases:
            retrieved_text = "\n".join(case.retrieval_context[:3]) if case.retrieval_context else ""
            resolution, _ = await self.llm.generate_resolution(case.input, [])
            case.actual_output = resolution

        fix_metric = FixAccuracyMetric(threshold=0.6)
        rel_metric = RetrievalRelevanceMetric(self.embedder, threshold=0.75)
        time_metric = ResolutionTimePredictionMetric(threshold=0.7)

        fix_scores = [fix_metric.measure(c) for c in cases]
        rel_scores = [rel_metric.measure(c) for c in cases]
        time_scores = [time_metric.measure(c) for c in cases]

        passed = sum(
            1 for f, r, t in zip(fix_scores, rel_scores, time_scores)
            if f >= 0.6 and r >= 0.75 and t >= 0.7
        )

        return EvaluationReport(
            fix_accuracy=sum(fix_scores) / len(fix_scores) if fix_scores else 0.0,
            retrieval_relevance=sum(rel_scores) / len(rel_scores) if rel_scores else 0.0,
            resolution_time_accuracy=sum(time_scores) / len(time_scores) if time_scores else 0.0,
            total_cases=len(cases),
            passed=passed,
        )
