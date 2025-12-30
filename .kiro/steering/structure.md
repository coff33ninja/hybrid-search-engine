# Project Structure

```
├── main.py                 # Entry point and demo script
├── requirements.txt        # Python dependencies
├── index.duckdb           # DuckDB database file (generated)
├── data/                  # Directory for document data files
└── search_engine/         # Core search engine package
    ├── __init__.py        # Package init
    ├── core.py            # Searcher class - hybrid search logic
    ├── indexer.py         # Indexer class - document vectorization and DB storage
    ├── extractor.py       # Text tokenization utilities
    └── utils.py           # Math utilities (cosine similarity with numba)
```

## Module Responsibilities

| Module | Purpose |
|--------|---------|
| `core.py` | `Searcher` class handles query embedding and hybrid scoring |
| `indexer.py` | `Indexer` class manages DB connections, document storage, and embedding generation |
| `extractor.py` | Text preprocessing and token extraction |
| `utils.py` | Performance-optimized math functions |

## Patterns

- **Context Manager**: `Indexer` uses `__enter__`/`__exit__` for DB connection lifecycle
- **Dependency Injection**: Model names are configurable via constructor parameters
- **Separation of Concerns**: Indexing and searching are separate classes
