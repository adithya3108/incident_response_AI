from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    llm_model: str = "anthropic/claude-sonnet-4-5"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    top_k: int = 10
    rrf_k: int = 60
    chunk_size: int = 512
    chunk_overlap: int = 64

    faiss_index_path: str = "data/indexes/faiss_index.bin"
    bm25_index_path: str = "data/indexes/bm25_index.pkl"
    metadata_store_path: str = "data/indexes/metadata_store.json"
    embeddings_cache_path: str = "data/indexes/embeddings_cache.pkl"

    max_context_tokens: int = 4096
    http_referer: str = "http://localhost:8000"
    x_title: str = "Incident KB Assistant"

    # LangSmith tracing
    langchain_tracing_v2: str = "false"
    langchain_api_key: str = ""
    langchain_project: str = "incident-kb-assistant"


settings = Settings()
