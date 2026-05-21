"""Run once: CSV → embeddings → FAISS + BM25 indexes."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings
from app.core.embedder import Embedder
from app.core.indexer import HybridIndexer
from app.core.ingestion import IncidentIngester


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/incidents.csv")
    args = parser.parse_args()

    print(f"Loading CSV: {args.input}")
    ingester = IncidentIngester()
    records = ingester.load_and_clean(args.input)
    print(f"Loaded {len(records)} clean incident records")

    print(f"Embedding with {settings.embedding_model} ...")
    embedder = Embedder(settings.embedding_model, settings.embeddings_cache_path)
    texts = [r.index_text for r in records]
    embeddings = embedder.embed_batch(texts)
    print(f"Embeddings shape: {embeddings.shape}")

    print("Building FAISS + BM25 indexes ...")
    indexer = HybridIndexer()
    indexer.build(
        records,
        embeddings,
        settings.faiss_index_path,
        settings.bm25_index_path,
        settings.metadata_store_path,
    )
    print("Done. Indexes saved to data/indexes/")


if __name__ == "__main__":
    main()
