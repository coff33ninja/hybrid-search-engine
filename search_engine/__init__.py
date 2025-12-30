"""
Hybrid Search Engine Package

Combines semantic (vector-based) and lexical (fuzzy text matching) search.
"""

from .core import Searcher, QueryMemory
from .indexer import Indexer, FAISSIndex, FAISS_AVAILABLE
from .extractor import (
    extract_tokens, 
    preprocess_text, 
    extract_metadata, 
    load_file_content,
    discover_documents
)
from .utils import cosine_sim, batch_cosine_sim, normalize_scores, top_k_indices, pairwise_cosine_sim
from .bm25 import BM25, BM25Plus
from .chunker import (
    Chunk,
    SentenceChunker,
    ParagraphChunker,
    SlidingWindowChunker,
    SemanticChunker,
    get_chunker
)
from .highlighter import (
    Highlighter,
    HighlightedResult,
    TerminalHighlighter,
    HTMLHighlighter,
    MarkdownHighlighter
)

# Optional imports
try:
    from .watcher import FileWatcher, create_auto_indexer, WATCHDOG_AVAILABLE
except ImportError:
    WATCHDOG_AVAILABLE = False

try:
    from .reranker import CrossEncoderReranker, create_reranker
except ImportError:
    pass

try:
    from .rag import RAGPipeline, HyDEPipeline, AgenticSearch, RAGResponse
except ImportError:
    pass

try:
    from .cache import InMemoryCache, RedisCache, SemanticCache, cached_search
except ImportError:
    pass

try:
    from .evaluation import SearchEvaluator, EvaluationReport, QueryResult
except ImportError:
    pass

try:
    from .pipelines import (
        create_pipeline,
        BasicPipeline,
        ChunkedPipeline,
        RerankedPipeline,
        BM25Pipeline,
        HybridBM25Pipeline,
        RAGPipeline,
        MultiStagePipeline,
        DiversityPipeline,
        PipelineResult
    )
except ImportError:
    pass

try:
    from .api import app as api_app
except ImportError:
    api_app = None

__all__ = [
    # Core
    "Searcher",
    "Indexer",
    "QueryMemory",
    # FAISS
    "FAISSIndex",
    "FAISS_AVAILABLE",
    # Extractor
    "extract_tokens",
    "preprocess_text",
    "extract_metadata",
    "load_file_content",
    "discover_documents",
    # Utils
    "cosine_sim",
    "batch_cosine_sim",
    "normalize_scores",
    "top_k_indices",
    "pairwise_cosine_sim",
    # BM25
    "BM25",
    "BM25Plus",
    # Chunker
    "Chunk",
    "SentenceChunker",
    "ParagraphChunker",
    "SlidingWindowChunker",
    "SemanticChunker",
    "get_chunker",
    # Highlighter
    "Highlighter",
    "HighlightedResult",
    "TerminalHighlighter",
    "HTMLHighlighter",
    "MarkdownHighlighter",
    # Watcher
    "FileWatcher",
    "create_auto_indexer",
    "WATCHDOG_AVAILABLE",
    # Reranker
    "CrossEncoderReranker",
    "create_reranker",
    # RAG
    "RAGPipeline",
    "HyDEPipeline",
    "AgenticSearch",
    "RAGResponse",
    # Cache
    "InMemoryCache",
    "RedisCache",
    "SemanticCache",
    "cached_search",
    # Evaluation
    "SearchEvaluator",
    "EvaluationReport",
    "QueryResult",
    # Pipelines
    "create_pipeline",
    "BasicPipeline",
    "ChunkedPipeline",
    "RerankedPipeline",
    "BM25Pipeline",
    "HybridBM25Pipeline",
    "RAGPipeline",
    "MultiStagePipeline",
    "DiversityPipeline",
    "PipelineResult",
    # API
    "api_app",
]

__version__ = "2.0.0"
