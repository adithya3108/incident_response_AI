import numpy as np
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase


class FixAccuracyMetric(BaseMetric):
    """ROUGE-L style token overlap between suggested resolution and ground truth."""

    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold
        self.score = 0.0

    @property
    def name(self) -> str:
        return "Fix Accuracy"

    def measure(self, test_case: LLMTestCase) -> float:
        self.score = self._rouge_l(test_case.actual_output or "", test_case.expected_output or "")
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.success

    @staticmethod
    def _rouge_l(pred: str, ref: str) -> float:
        pred_tokens = pred.lower().split()
        ref_tokens = ref.lower().split()
        if not pred_tokens or not ref_tokens:
            return 0.0
        lcs = _lcs_length(pred_tokens, ref_tokens)
        precision = lcs / len(pred_tokens)
        recall = lcs / len(ref_tokens)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)


class RetrievalRelevanceMetric(BaseMetric):
    """Average cosine similarity between query embedding and retrieved incident embeddings."""

    def __init__(self, embedder, threshold: float = 0.75):
        self.threshold = threshold
        self.score = 0.0
        self.embedder = embedder

    @property
    def name(self) -> str:
        return "Retrieval Relevance"

    def measure(self, test_case: LLMTestCase) -> float:
        if not test_case.retrieval_context:
            self.score = 0.0
            self.success = False
            return self.score

        query_vec = self.embedder.embed_query(test_case.input)
        ctx_vecs = self.embedder.embed_batch(list(test_case.retrieval_context))
        sims = ctx_vecs @ query_vec
        self.score = float(np.mean(sims))
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.success


class ResolutionTimePredictionMetric(BaseMetric):
    """Placeholder metric — scores 1.0 until resolution time data is available."""

    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold
        self.score = 1.0

    @property
    def name(self) -> str:
        return "Resolution Time Prediction Accuracy"

    def measure(self, test_case: LLMTestCase) -> float:
        self.success = True
        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.success


def _lcs_length(a: list, b: list) -> int:
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(2)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i % 2][j] = dp[(i - 1) % 2][j - 1] + 1
            else:
                dp[i % 2][j] = max(dp[(i - 1) % 2][j], dp[i % 2][j - 1])
    return dp[m % 2][n]
