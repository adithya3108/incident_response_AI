import hashlib
import pickle
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str, cache_path: str):
        self.model = SentenceTransformer(model_name)
        self.cache_path = Path(cache_path)
        self._cache: dict[str, np.ndarray] = self._load_cache()

    def _load_cache(self) -> dict[str, np.ndarray]:
        if self.cache_path.exists():
            with open(self.cache_path, "rb") as f:
                return pickle.load(f)
        return {}

    def _save_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "wb") as f:
            pickle.dump(self._cache, f)

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        if not texts:
            raise ValueError("embed_batch called with 0 texts — check that the CSV loaded records correctly")

        keys = [self._key(t) for t in texts]
        missing_indices = [i for i, k in enumerate(keys) if k not in self._cache]
        missing_texts = [texts[i] for i in missing_indices]

        if missing_texts:
            new_vecs = self.model.encode(missing_texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
            for i, vec in zip(missing_indices, new_vecs):
                self._cache[keys[i]] = vec
            self._save_cache()

        result = np.stack([self._cache[k] for k in keys])
        return result.astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        vec = self.model.encode([text], normalize_embeddings=True)
        return vec[0].astype(np.float32)
