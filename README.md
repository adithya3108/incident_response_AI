# AI-Powered Incident Knowledge Base Assistant

Semantic retrieval system that allows IT support teams to enter natural-language queries and retrieve relevant historical incidents with resolution guidance.

## Architecture

```
CSV Dataset → Ingestion → Embedder (all-MiniLM-L6-v2) → FAISS Index
                                                        → BM25 Index
                                                        → Metadata Store

Query → Embed Query → FAISS Search ─┐
                   → BM25 Search  ──┤ RRF Fusion → Rerank → Confidence → Claude → Response
                                    └─ Metadata Filters
```

**Stack:** FastAPI · FAISS · BM25 · sentence-transformers · Claude via OpenRouter · DeepEval · Streamlit

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env and set OPENROUTER_API_KEY (get one at https://openrouter.ai/keys)

# 3. Download dataset
# From: https://github.com/AkshayDusad/ITSM-Incident-Management/blob/master/ITSM_data.csv
# Save to: data/raw/incidents.csv

# 4. Build indexes (run once)
python scripts/ingest_and_index.py --input data/raw/incidents.csv

# 5. Start API server
uvicorn app.main:app --reload

# 6. Start frontend (separate terminal)
streamlit run frontend/app.py
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/metrics` | Query stats and cache hit rate |
| POST | `/v1/search` | Semantic hybrid search with RAG resolution |
| POST | `/v1/bulk-search` | Batch queries (up to 50) |
| POST | `/v1/triage` | Priority classification and team routing |
| POST | `/v1/escalate` | Multi-tier agent escalation (L1→L2→L3) |
| POST | `/v1/analyze` | Root cause analysis across incidents |
| POST | `/v1/feedback` | Submit helpful/not-helpful feedback |

## Sample Usage

**Search:**
```bash
curl -X POST http://localhost:8000/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "Users cannot connect to VPN despite correct credentials", "top_k": 5}'
```

**Triage:**
```bash
curl -X POST http://localhost:8000/v1/triage \
  -H "Content-Type: application/json" \
  -d '{"description": "Database server unresponsive, all services down", "impact": 1, "urgency": 1}'
```

**Escalate:**
```bash
curl -X POST http://localhost:8000/v1/escalate \
  -H "Content-Type: application/json" \
  -d '{
    "incident_id": "INC0001234",
    "current_tier": "L1",
    "description": "SAP HANA memory exhaustion causing system crash",
    "resolution_attempts": ["Restarted service - recurred after 2 hours"]
  }'
```

## Running Evaluation

```bash
python scripts/run_evaluation.py
```

Evaluates on held-out 20% of the dataset. Metrics:
- **Fix Accuracy** (target ≥ 0.6): ROUGE-L overlap between suggested and actual resolution
- **Retrieval Relevance** (target ≥ 0.75): cosine similarity of retrieved incidents to query
- **LLM-as-Judge**: Safety / Completeness / Ordering / Technical Accuracy (1–5 scale)

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| FAISS `IndexFlatIP` + L2 normalization | Exact cosine similarity; swap to `IndexIVFFlat` for >500k incidents |
| Reciprocal Rank Fusion (RRF k=60) | Robust fusion without calibrating BM25/cosine score scales |
| Prompt caching (system prompt + context) | Reduces Claude API costs significantly on repeated queries |
| Exponential recency decay | More recent resolutions weighted higher; recent fixes more relevant |
| Async bulk-search with `asyncio.gather` | Parallel execution; token optimizer deduplicates context blocks |
| Feedback → reranking loop | Each `/feedback` submission updates success rate for future reranking |
