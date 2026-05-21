"""Run DeepEval evaluation on held-out 20% of incidents."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings
from app.core.embedder import Embedder
from app.core.indexer import HybridIndexer
from app.core.ingestion import IncidentIngester
from app.core.retriever import HybridRetriever
from app.evaluation.evaluator import IncidentEvaluator
from app.llm.client import ClaudeClient


async def main():
    ingester = IncidentIngester()
    records = ingester.load_and_clean("data/raw/incidents.csv")

    split = int(len(records) * 0.8)
    test_records = records[split:]
    print(f"Evaluating on {len(test_records)} held-out incidents...")

    faiss_idx, bm25, metadata = HybridIndexer.load(
        settings.faiss_index_path, settings.bm25_index_path, settings.metadata_store_path
    )
    embedder = Embedder(settings.embedding_model, settings.embeddings_cache_path)
    retriever = HybridRetriever(faiss_idx, bm25, metadata, rrf_k=settings.rrf_k)
    llm = ClaudeClient(api_key=settings.anthropic_api_key, model=settings.claude_model)

    evaluator = IncidentEvaluator(retriever, embedder, llm)
    report = await evaluator.run_suite(test_records)

    print("\n===== EVALUATION REPORT =====")
    print(f"Total cases:               {report.total_cases}")
    print(f"Passed (all metrics):      {report.passed}")
    print(f"Fix Accuracy (≥0.6):       {report.fix_accuracy:.3f}")
    print(f"Retrieval Relevance (≥0.75): {report.retrieval_relevance:.3f}")
    print(f"Resolution Time Acc (≥0.7):  {report.resolution_time_accuracy:.3f}")
    print("==============================")


if __name__ == "__main__":
    asyncio.run(main())
