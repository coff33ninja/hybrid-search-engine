# 🔍 Hybrid Search Engine

A production-ready search engine combining **semantic understanding** with **lexical matching** for superior search results.

```
Query: "how to start a project"
├── Semantic → finds "A journey begins with a single step" (meaning)
├── Lexical  → finds docs containing "start", "project" (keywords)
└── Hybrid   → combines both for best results
```

## ✨ Features

| Category | Features |
|----------|----------|
| **Search** | Hybrid (semantic + lexical), BM25, FAISS ANN, cross-encoder reranking |
| **Pipelines** | Basic, Chunked, Reranked, RAG, Multi-stage, Diversity (MMR) |
| **Processing** | Document chunking, highlighting, metadata extraction |
| **Learning** | Query memory, feedback loop, weight optimization |
| **API** | REST API, batch search, WebSocket (coming soon) |
| **Caching** | In-memory, Redis, semantic similarity cache |
| **Auth** | API keys, rate limiting, scopes |
| **UI** | Streamlit web interface, CLI |
| **Deploy** | Docker, docker-compose |

## 🚀 Quick Start

```bash
# Install
pip install -r requirements.txt

# Run demo
python main.py

# Start API
uvicorn search_engine.api:app --reload

# Start UI
streamlit run ui.py
```

## 📦 Pipelines

Choose the right pipeline for your use case:

### Basic Pipeline
```python
from search_engine.pipelines import create_pipeline

pipeline = create_pipeline("basic", semantic_weight=0.7)
pipeline.index(["doc1", "doc2", "doc3"])
results = pipeline.search("my query", top_k=5)
```

### Chunked Pipeline
Best for long documents (articles, books, PDFs).

```python
pipeline = create_pipeline(
    "chunked",
    chunk_strategy="paragraph",  # or "sentence", "sliding", "semantic"
    chunk_size=500
)
pipeline.index(long_documents)
results = pipeline.search("query", return_parent=True)  # Get original doc
```

### Reranked Pipeline
Higher precision using cross-encoder reranking.

```python
pipeline = create_pipeline(
    "reranked",
    rerank_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    initial_k=20  # Retrieve 20, rerank to top 5
)
```

### BM25 Pipeline
Pure keyword search when exact terms matter.

```python
pipeline = create_pipeline("bm25", k1=1.5, b=0.75)
```

### Hybrid BM25 Pipeline
Combines semantic embeddings with BM25 scoring.

```python
pipeline = create_pipeline(
    "hybrid_bm25",
    semantic_weight=0.6,
    bm25_weight=0.4
)
```

### RAG Pipeline
Retrieval-Augmented Generation for Q&A.

```python
pipeline = create_pipeline(
    "rag",
    llm_provider="openai",  # or "anthropic", "local"
    model="gpt-4",
    use_reranking=True
)
pipeline.index(knowledge_base)
answer = pipeline.ask("What is the capital of France?")
```

### Multi-Stage Pipeline
Three-stage retrieval for large collections:
1. Fast semantic retrieval (100 candidates)
2. BM25 filtering (20 candidates)
3. Cross-encoder reranking (5 results)

```python
pipeline = create_pipeline(
    "multi_stage",
    stage1_k=100,
    stage2_k=20,
    final_k=5
)
```

### Diversity Pipeline
Avoids redundant results using Maximal Marginal Relevance (MMR).

```python
pipeline = create_pipeline(
    "diversity",
    lambda_param=0.5  # 0=max diversity, 1=max relevance
)
```

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/search` | POST | Hybrid search |
| `/search/batch` | POST | Multiple queries |
| `/search/highlighted` | POST | Search with snippets |
| `/ask` | POST | RAG question answering |
| `/rerank` | POST | Cross-encoder reranking |
| `/index` | POST | Index documents |
| `/index/directory` | POST | Index from folder |
| `/index/add` | POST | Incremental indexing |
| `/feedback` | POST | Submit relevance feedback |
| `/watcher/start` | POST | Auto-reindex on changes |
| `/stats` | GET | Statistics |
| `/documents` | GET | List documents |
| `/health` | GET | Health check |

### Example Requests

```bash
# Search
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "machine learning", "top_k": 5}'

# Batch search
curl -X POST http://localhost:8000/search/batch \
  -d '{"queries": ["query1", "query2"], "top_k": 3}'

# RAG
curl -X POST http://localhost:8000/ask \
  -d '{"question": "What is Python?", "llm_provider": "openai"}'

# Index directory
curl -X POST http://localhost:8000/index/directory \
  -d '{"directory": "data", "extensions": [".txt", ".md"]}'
```

## 💻 CLI Usage

```bash
# Search
python -m search_engine search "your query" -k 5 -w 0.8

# Index
python -m search_engine index data/
python -m search_engine add data/new_docs/

# Stats
python -m search_engine stats --json

# Export/Import
python -m search_engine export -o backup.json
python -m search_engine import -i backup.json

# Start server
python -m search_engine serve --port 8000 --reload
```

## 🐳 Docker

```bash
# Basic
docker-compose up -d

# With Redis caching
docker-compose --profile with-redis up -d

# With Streamlit UI
docker-compose --profile with-ui up -d

# Full stack
docker-compose --profile with-redis --profile with-ui up -d
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        API Layer                            │
│  FastAPI + Auth + Rate Limiting + Caching                   │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                      Pipeline Layer                         │
│  Basic │ Chunked │ Reranked │ RAG │ MultiStage │ Diversity  │
└─────────────────────────────────────────────────────────────┘
                              │
┌──────────────┬──────────────┬──────────────┬───────────────┐
│   Indexer    │   Searcher   │   Reranker   │     RAG       │
│  + Chunker   │  + BM25      │  CrossEnc    │  + LLM        │
└──────────────┴──────────────┴──────────────┴───────────────┘
                              │
┌──────────────┬──────────────┬──────────────┬───────────────┐
│   DuckDB     │    FAISS     │   Embeddings │    Cache      │
│  (storage)   │   (ANN)      │  (vectors)   │ (Redis/Mem)   │
└──────────────┴──────────────┴──────────────┴───────────────┘
```

## 📊 Evaluation

```python
from search_engine.evaluation import SearchEvaluator

evaluator = SearchEvaluator(searcher, docs_df, vectors)

# Load test queries
test_set = [
    {"query": "python programming", "relevant_doc_ids": [1, 5, 12]},
    {"query": "machine learning", "relevant_doc_ids": [3, 7, 8]},
]

# Evaluate
report = evaluator.evaluate(test_set, top_k=10)
print(report.summary())

# Compare configurations
configs = [
    {"name": "semantic_heavy", "semantic_weight": 0.9, "lexical_weight": 0.1},
    {"name": "balanced", "semantic_weight": 0.5, "lexical_weight": 0.5},
]
comparison = evaluator.compare_configs(test_set, configs)
```

Metrics: Precision, Recall, F1, MRR, NDCG, Latency (p50/p95/p99)

## 🔧 Configuration

### Embedding Models

| Model | Quality | Speed | Size |
|-------|---------|-------|------|
| `all-MiniLM-L6-v2` | Good | Fast | 80MB |
| `all-mpnet-base-v2` | Better | Medium | 420MB |
| `bge-large-en-v1.5` | Best | Slow | 1.3GB |

### Reranking Models

| Model | Quality | Speed |
|-------|---------|-------|
| `cross-encoder/ms-marco-TinyBERT-L-2-v2` | Good | Fastest |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Better | Fast |
| `BAAI/bge-reranker-large` | Best | Slow |

## 🛠️ Tech Stack

- **Python 3.x**
- **sentence-transformers** - Embeddings
- **DuckDB** - Document storage
- **FAISS** - Vector indexing
- **FastAPI** - REST API
- **Polars** - DataFrames
- **numba** - JIT compilation
- **rapidfuzz** - Fuzzy matching
- **watchdog** - File watching
- **Streamlit** - Web UI
- **Redis** - Caching
- **OpenAI/Anthropic** - RAG

## 📁 Project Structure

```
├── main.py                 # Demo script
├── ui.py                   # Streamlit UI
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── data/                   # Document files
└── search_engine/
    ├── core.py             # Searcher class
    ├── indexer.py          # Indexer class
    ├── pipelines.py        # Pre-built pipelines
    ├── bm25.py             # BM25 scoring
    ├── chunker.py          # Document chunking
    ├── highlighter.py      # Result highlighting
    ├── reranker.py         # Cross-encoder reranking
    ├── rag.py              # RAG/LLM integration
    ├── evaluation.py       # Metrics & evaluation
    ├── cache.py            # Caching layer
    ├── auth.py             # Auth & rate limiting
    ├── watcher.py          # File watcher
    ├── api.py              # FastAPI app
    ├── cli.py              # CLI interface
    └── utils.py            # Utilities
```

## 📈 Roadmap

- [ ] WebSocket streaming for real-time search
- [ ] Multi-modal search (images, audio)
- [ ] Distributed indexing with Ray
- [ ] GraphRAG integration
- [ ] Fine-tuning pipeline for domain adaptation
- [ ] A/B testing framework
- [ ] Query autocomplete
- [ ] Personalized ranking

## 📄 License

MIT
