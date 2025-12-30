# Tech Stack

## Language
- Python 3.x

## Core Libraries
- **polars**: DataFrame operations (preferred over pandas)
- **numpy**: Numerical computations and vector operations
- **numba**: JIT compilation for performance-critical functions (e.g., cosine similarity)
- **duckdb**: Embedded analytical database for document storage
- **sentence-transformers**: Text embedding generation (default model: `all-MiniLM-L6-v2`)
- **rapidfuzz**: Fuzzy string matching for lexical search
- **torch/transformers**: Backend for sentence-transformers

## Optional/API
- **fastapi**: REST API framework
- **uvicorn**: ASGI server

## Utilities
- **loguru**: Logging (configured to stderr at INFO level)

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the demo
python main.py
```

## Performance Notes
- Uses `@njit` decorator from numba for optimized cosine similarity calculations
- Embeddings are stored as `np.float32` for memory efficiency
