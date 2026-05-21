import json
import pickle
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from app.models.incident import IncidentRecord


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


class HybridIndexer:
    def build(
        self,
        records: list[IncidentRecord],
        embeddings: np.ndarray,
        faiss_path: str,
        bm25_path: str,
        metadata_path: str,
    ) -> None:
        Path(faiss_path).parent.mkdir(parents=True, exist_ok=True)

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, faiss_path)

        corpus = [_tokenize(r.index_text) for r in records]
        bm25 = BM25Okapi(corpus)
        with open(bm25_path, "wb") as f:
            pickle.dump(bm25, f)

        metadata = []
        for r in records:
            metadata.append({
                "incident_id": r.incident_id,
                "incident_state": r.incident_state,
                "impact": r.impact,
                "urgency": r.urgency,
                "priority": r.priority,
                "description": r.description,
                "assigned_to": r.assigned_to,
                "resolution_notes": r.resolution_notes,
                "resolved_at": r.resolved_at.isoformat() if r.resolved_at else None,
                "index_text": r.index_text,
            })
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False)

        print(f"Indexed {len(records)} incidents. FAISS dim={embeddings.shape[1]}")

    @staticmethod
    def load(
        faiss_path: str,
        bm25_path: str,
        metadata_path: str,
    ) -> tuple[faiss.Index, BM25Okapi, list[dict]]:
        index = faiss.read_index(faiss_path)
        with open(bm25_path, "rb") as f:
            bm25 = pickle.load(f)
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return index, bm25, metadata
