"""
FastAPI REST API for the search engine.
"""
import numpy as np
from pathlib import Path
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from loguru import logger

from .indexer import Indexer
from .core import Searcher
from .watcher import FileWatcher, WATCHDOG_AVAILABLE
from .autocomplete import AutocompleteEngine, Suggestion
from .semantic_cache import SemanticCache, SearchResult as CacheSearchResult
from .deduplication import DeduplicationEngine, DedupeAction, DATASKETCH_AVAILABLE
from .language import LanguageDetector, LANGDETECT_AVAILABLE
from .jobs import JobQueue, JobState, IndexingTask
from .metadata import MetadataFilter, MetadataStore


# --- Pydantic Models ---

class Document(BaseModel):
    content: str = Field(..., description="Document text content")
    source_path: Optional[str] = Field(None, description="Optional source file path")


class IndexRequest(BaseModel):
    documents: List[Document] = Field(..., description="Documents to index")


class IndexFromDirRequest(BaseModel):
    directory: str = Field("data", description="Directory to index from")
    extensions: Optional[List[str]] = Field(None, description="File extensions to include")


class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    top_k: int = Field(5, ge=1, le=100, description="Number of results")
    semantic_weight: Optional[float] = Field(None, ge=0, le=1, description="Semantic weight")
    lexical_weight: Optional[float] = Field(None, ge=0, le=1, description="Lexical weight")
    use_learned_weights: bool = Field(False, description="Use weights learned from feedback")


class FeedbackRequest(BaseModel):
    query_id: int = Field(..., description="Query ID from search results")
    doc_id: int = Field(..., description="Document ID")
    relevance_score: int = Field(0, ge=0, le=5, description="Relevance rating (0-5)")
    clicked: bool = Field(False, description="Whether result was clicked")


class SearchResult(BaseModel):
    score: float
    content: str
    doc_id: int


class SearchResponse(BaseModel):
    query: str
    query_id: Optional[int] = None
    results: List[SearchResult]
    weights_used: dict


class StatsResponse(BaseModel):
    total_documents: int
    total_queries: int
    unique_queries: int
    total_feedback: int
    avg_relevance: Optional[float]


# --- Global State ---

class SearchEngineState:
    """Holds the search engine state."""
    def __init__(self):
        self.docs_df = None
        self.vectors = None
        self.searcher: Optional[Searcher] = None
        self.watcher: Optional[FileWatcher] = None
        self.db_path = "index.duckdb"
        self.use_faiss = False
        self._last_query_id: Optional[int] = None
        
        # High-impact features
        self.autocomplete: Optional[AutocompleteEngine] = None
        self.semantic_cache: Optional[SemanticCache] = None
        self.deduplication: Optional[DeduplicationEngine] = None
        self.language_detector: Optional[LanguageDetector] = None
        self.job_queue: Optional[JobQueue] = None
        self.metadata_filter: Optional[MetadataFilter] = None
        self.metadata_store: Optional[MetadataStore] = None

state = SearchEngineState()


# --- Lifespan ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup resources."""
    logger.info("Starting search engine API...")
    
    # Initialize searcher
    state.searcher = Searcher(
        db_path=state.db_path,
        use_faiss=state.use_faiss,
        enable_query_memory=True
    )
    
    # Initialize high-impact features
    state.autocomplete = AutocompleteEngine(max_suggestions=10)
    state.semantic_cache = SemanticCache(similarity_threshold=0.95, ttl_seconds=3600)
    state.metadata_filter = MetadataFilter()
    state.metadata_store = MetadataStore(db_path=state.db_path)
    state.job_queue = JobQueue(max_concurrent=3, max_retries=3)
    
    # Initialize optional features
    if DATASKETCH_AVAILABLE:
        state.deduplication = DeduplicationEngine(threshold=0.9, action=DedupeAction.FLAG)
    
    if LANGDETECT_AVAILABLE:
        state.language_detector = LanguageDetector()
    
    # Try to load existing index
    try:
        with Indexer(db_path=state.db_path, use_faiss=state.use_faiss) as indexer:
            state.docs_df, docs = indexer.get_all_documents()
            if len(docs) > 0:
                state.vectors = indexer.embed(docs)
                logger.info(f"Loaded {len(docs)} documents from existing index.")
    except Exception as e:
        logger.warning(f"Could not load existing index: {e}")
    
    yield
    
    # Cleanup
    if state.watcher and state.watcher.is_running():
        state.watcher.stop()
    if state.job_queue:
        state.job_queue.shutdown(wait=False)
    logger.info("Search engine API stopped.")


# --- FastAPI App ---

app = FastAPI(
    title="Hybrid Search Engine API",
    description="REST API for semantic + lexical hybrid search",
    version="1.0.0",
    lifespan=lifespan
)


# --- Endpoints ---

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "indexed_documents": len(state.docs_df) if state.docs_df is not None else 0,
        "watcher_active": state.watcher.is_running() if state.watcher else False
    }


@app.post("/index", response_model=dict)
async def index_documents(request: IndexRequest):
    """Index a list of documents."""
    if not request.documents:
        raise HTTPException(status_code=400, detail="No documents provided")
    
    docs = [d.content for d in request.documents]
    paths = [d.source_path or "" for d in request.documents]
    
    try:
        with Indexer(db_path=state.db_path, use_faiss=state.use_faiss) as indexer:
            state.docs_df, state.vectors = indexer.index_documents(docs, paths)
        
        return {
            "status": "success",
            "documents_indexed": len(docs)
        }
    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/index/directory", response_model=dict)
async def index_from_directory(request: IndexFromDirRequest):
    """Index documents from a directory."""
    dir_path = Path(request.directory)
    if not dir_path.exists():
        raise HTTPException(status_code=404, detail=f"Directory not found: {request.directory}")
    
    try:
        with Indexer(db_path=state.db_path, use_faiss=state.use_faiss) as indexer:
            state.docs_df, state.vectors = indexer.index_from_directory(
                request.directory, 
                request.extensions
            )
        
        doc_count = len(state.docs_df) if state.docs_df is not None else 0
        return {
            "status": "success",
            "documents_indexed": doc_count,
            "directory": request.directory
        }
    except Exception as e:
        logger.error(f"Directory indexing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/index/add", response_model=dict)
async def add_documents(request: IndexRequest):
    """Add documents to existing index (incremental)."""
    if not request.documents:
        raise HTTPException(status_code=400, detail="No documents provided")
    
    docs = [d.content for d in request.documents]
    paths = [d.source_path or "" for d in request.documents]
    
    try:
        with Indexer(db_path=state.db_path, use_faiss=state.use_faiss) as indexer:
            new_df, new_vectors = indexer.add_documents(docs, paths)
            
            # Merge with existing
            if state.docs_df is not None and state.vectors is not None:
                import polars as pl
                state.docs_df = pl.concat([state.docs_df, new_df])
                state.vectors = np.vstack([state.vectors, new_vectors])
            else:
                state.docs_df = new_df
                state.vectors = new_vectors
        
        return {
            "status": "success",
            "documents_added": len(docs),
            "total_documents": len(state.docs_df)
        }
    except Exception as e:
        logger.error(f"Adding documents failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """Perform hybrid search."""
    if state.docs_df is None or state.vectors is None:
        raise HTTPException(status_code=400, detail="No documents indexed. Index documents first.")
    
    if not state.searcher:
        raise HTTPException(status_code=500, detail="Searcher not initialized")
    
    # Validate weights
    sem_w = request.semantic_weight
    lex_w = request.lexical_weight
    
    if sem_w is not None and lex_w is not None:
        if not np.isclose(sem_w + lex_w, 1.0):
            raise HTTPException(status_code=400, detail="Weights must sum to 1.0")
    elif sem_w is not None:
        lex_w = 1.0 - sem_w
    elif lex_w is not None:
        sem_w = 1.0 - lex_w
    
    try:
        results = state.searcher.search(
            query=request.query,
            docs_df=state.docs_df,
            vectors=state.vectors,
            top_k=request.top_k,
            semantic_weight=sem_w,
            lexical_weight=lex_w,
            use_learned_weights=request.use_learned_weights
        )
        
        # Get the query_id from the last logged query
        stats = state.searcher.get_stats()
        query_id = stats.get('total_queries')
        
        return SearchResponse(
            query=request.query,
            query_id=query_id,
            results=[
                SearchResult(score=score, content=content, doc_id=doc_id)
                for score, content, doc_id in results
            ],
            weights_used={
                "semantic": sem_w or 0.7,
                "lexical": lex_w or 0.3,
                "learned": request.use_learned_weights
            }
        )
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    """Submit feedback for a search result."""
    if not state.searcher:
        raise HTTPException(status_code=500, detail="Searcher not initialized")
    
    try:
        state.searcher.record_feedback(
            query_id=request.query_id,
            doc_id=request.doc_id,
            relevance_score=request.relevance_score,
            clicked=request.clicked
        )
        return {"status": "success", "message": "Feedback recorded"}
    except Exception as e:
        logger.error(f"Feedback submission failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get search engine statistics."""
    doc_count = len(state.docs_df) if state.docs_df is not None else 0
    
    query_stats = {}
    if state.searcher:
        query_stats = state.searcher.get_stats()
    
    return StatsResponse(
        total_documents=doc_count,
        total_queries=query_stats.get('total_queries', 0),
        unique_queries=query_stats.get('unique_queries', 0),
        total_feedback=query_stats.get('total_feedback', 0),
        avg_relevance=query_stats.get('avg_relevance')
    )


@app.post("/watcher/start")
async def start_watcher(
    directory: str = Query("data", description="Directory to watch")
):
    """Start file watcher for auto-reindexing."""
    if not WATCHDOG_AVAILABLE:
        raise HTTPException(status_code=501, detail="watchdog not installed")
    
    if state.watcher and state.watcher.is_running():
        return {"status": "already_running", "directory": directory}
    
    def on_change(event_type: str, file_path: str):
        logger.info(f"File {event_type}: {file_path}, triggering reindex...")
        # Reindex in background
        try:
            with Indexer(db_path=state.db_path, use_faiss=state.use_faiss) as indexer:
                state.docs_df, state.vectors = indexer.index_from_directory(directory)
        except Exception as e:
            logger.error(f"Auto-reindex failed: {e}")
    
    state.watcher = FileWatcher(
        watch_dir=directory,
        on_change=on_change
    )
    state.watcher.start()
    
    return {"status": "started", "directory": directory}


@app.post("/watcher/stop")
async def stop_watcher():
    """Stop file watcher."""
    if state.watcher and state.watcher.is_running():
        state.watcher.stop()
        return {"status": "stopped"}
    return {"status": "not_running"}


@app.get("/documents")
async def list_documents(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0)
):
    """List indexed documents."""
    if state.docs_df is None:
        return {"documents": [], "total": 0}
    
    total = len(state.docs_df)
    docs = state.docs_df.slice(offset, limit).to_dicts()
    
    return {
        "documents": docs,
        "total": total,
        "limit": limit,
        "offset": offset
    }


# --- Additional Endpoints ---

class BatchSearchRequest(BaseModel):
    queries: List[str] = Field(..., description="List of search queries")
    top_k: int = Field(5, ge=1, le=100, description="Results per query")
    semantic_weight: Optional[float] = Field(None, ge=0, le=1)
    lexical_weight: Optional[float] = Field(None, ge=0, le=1)


class HighlightedSearchRequest(SearchRequest):
    highlight: bool = Field(True, description="Include highlighted snippets")
    snippet_length: int = Field(150, ge=50, le=500, description="Snippet length")


class AskRequest(BaseModel):
    question: str = Field(..., description="Question to answer")
    top_k: int = Field(5, ge=1, le=10, description="Documents to retrieve")
    llm_provider: str = Field("openai", description="LLM provider: openai, anthropic, local")
    model: Optional[str] = Field(None, description="Model name override")


@app.post("/search/batch")
async def batch_search(request: BatchSearchRequest):
    """Perform multiple searches in one request."""
    if state.docs_df is None or state.vectors is None:
        raise HTTPException(status_code=400, detail="No documents indexed")
    
    sem_w = request.semantic_weight or 0.7
    lex_w = request.lexical_weight or 0.3
    
    results = {}
    for query in request.queries:
        try:
            search_results = state.searcher.search(
                query=query,
                docs_df=state.docs_df,
                vectors=state.vectors,
                top_k=request.top_k,
                semantic_weight=sem_w,
                lexical_weight=lex_w
            )
            results[query] = [
                {"score": s, "content": c, "doc_id": d}
                for s, c, d in search_results
            ]
        except Exception as e:
            results[query] = {"error": str(e)}
    
    return {"results": results, "queries_processed": len(request.queries)}


@app.post("/search/highlighted")
async def search_with_highlights(request: HighlightedSearchRequest):
    """Search with highlighted snippets."""
    if state.docs_df is None or state.vectors is None:
        raise HTTPException(status_code=400, detail="No documents indexed")
    
    from .highlighter import Highlighter
    
    sem_w = request.semantic_weight or 0.7
    lex_w = request.lexical_weight or 0.3
    
    results = state.searcher.search(
        query=request.query,
        docs_df=state.docs_df,
        vectors=state.vectors,
        top_k=request.top_k,
        semantic_weight=sem_w,
        lexical_weight=lex_w
    )
    
    highlighter = Highlighter(snippet_length=request.snippet_length)
    highlighted_results = []
    
    for score, content, doc_id in results:
        snippets = highlighter.extract_snippets(content, request.query)
        highlighted_results.append({
            "score": score,
            "content": content,
            "doc_id": doc_id,
            "snippets": snippets
        })
    
    return {
        "query": request.query,
        "results": highlighted_results
    }


@app.post("/ask")
async def ask_question(request: AskRequest):
    """RAG endpoint - answer questions using retrieved documents."""
    if state.docs_df is None or state.vectors is None:
        raise HTTPException(status_code=400, detail="No documents indexed")
    
    try:
        from .rag import RAGPipeline
        
        rag = RAGPipeline(
            searcher=state.searcher,
            docs_df=state.docs_df,
            vectors=state.vectors,
            llm_provider=request.llm_provider,
            model=request.model or ("gpt-3.5-turbo" if request.llm_provider == "openai" else "claude-3-haiku-20240307")
        )
        
        response = rag.ask(request.question, top_k=request.top_k)
        
        return {
            "question": request.question,
            "answer": response.answer,
            "sources": response.sources,
            "model": response.model,
            "tokens_used": response.tokens_used
        }
    except ImportError as e:
        raise HTTPException(status_code=501, detail=f"RAG dependencies not installed: {e}")
    except Exception as e:
        logger.error(f"RAG failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rerank")
async def rerank_results(
    query: str = Query(..., description="Search query"),
    doc_ids: List[int] = Query(..., description="Document IDs to rerank"),
    top_k: int = Query(5, ge=1, le=20)
):
    """Rerank documents using cross-encoder."""
    if state.docs_df is None:
        raise HTTPException(status_code=400, detail="No documents indexed")
    
    try:
        from .reranker import CrossEncoderReranker
        
        # Get documents by ID
        docs_dict = {row['doc_id']: row['content'] for row in state.docs_df.to_dicts()}
        results = [(0.0, docs_dict[did], did) for did in doc_ids if did in docs_dict]
        
        if not results:
            raise HTTPException(status_code=404, detail="No matching documents found")
        
        reranker = CrossEncoderReranker()
        reranked = reranker.rerank(query, results, top_k=top_k)
        
        return {
            "query": query,
            "results": [
                {"score": s, "content": c, "doc_id": d}
                for s, c, d in reranked
            ]
        }
    except ImportError as e:
        raise HTTPException(status_code=501, detail=f"Reranker not available: {e}")
    except Exception as e:
        logger.error(f"Reranking failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# --- Run with: uvicorn search_engine.api:app --reload ---


# =============================================================================
# HIGH-IMPACT FEATURE ENDPOINTS
# =============================================================================

# --- Autocomplete Endpoints ---

class AutocompleteRequest(BaseModel):
    partial_query: str = Field(..., min_length=1, description="Partial query to complete")
    limit: int = Field(10, ge=1, le=50, description="Maximum suggestions")


class AutocompleteSuggestion(BaseModel):
    text: str
    score: float
    source: str
    frequency: int


@app.post("/autocomplete", response_model=List[AutocompleteSuggestion])
async def get_autocomplete(request: AutocompleteRequest):
    """Get autocomplete suggestions for partial query."""
    if not state.autocomplete:
        raise HTTPException(status_code=500, detail="Autocomplete not initialized")
    
    suggestions: List[Suggestion] = state.autocomplete.suggest(request.partial_query, limit=request.limit)
    
    return [
        AutocompleteSuggestion(
            text=s.text,
            score=s.score,
            source=s.source,
            frequency=s.frequency
        )
        for s in suggestions
    ]


@app.post("/autocomplete/record")
async def record_autocomplete_selection(
    partial_query: str = Query(..., description="Partial query typed"),
    selected: str = Query(..., description="Suggestion selected")
):
    """Record user selection for autocomplete learning."""
    if not state.autocomplete:
        raise HTTPException(status_code=500, detail="Autocomplete not initialized")
    
    state.autocomplete.record_selection(partial_query, selected)
    return {"status": "recorded"}


# --- Semantic Cache Endpoints ---

@app.get("/cache/stats")
async def get_cache_stats():
    """Get semantic cache statistics."""
    if not state.semantic_cache:
        raise HTTPException(status_code=500, detail="Cache not initialized")
    
    stats = state.semantic_cache.stats
    
    # Include CacheSearchResult info in response
    return {
        **stats,
        "result_type": CacheSearchResult.__name__,
        "cache_enabled": True
    }


@app.post("/cache/invalidate")
async def invalidate_cache():
    """Invalidate all cached results."""
    if not state.semantic_cache:
        raise HTTPException(status_code=500, detail="Cache not initialized")
    
    count = state.semantic_cache.invalidate()
    return {"status": "invalidated", "entries_cleared": count}


# --- Deduplication Endpoints ---

class DuplicateInfo(BaseModel):
    doc_id: int
    similarity: float
    is_canonical: bool


@app.get("/documents/duplicates", response_model=List[dict])
async def list_duplicates():
    """List detected duplicate documents."""
    if not state.deduplication:
        raise HTTPException(status_code=501, detail="Deduplication not available (datasketch not installed)")
    
    if state.docs_df is None:
        return []
    
    # Get all documents and check for duplicates
    duplicates = []
    seen_content: dict[int, str] = {}  # Track content hashes to find duplicates
    
    for row in state.docs_df.to_dicts():
        doc_id = row.get('doc_id')
        content = row.get('content', '')
        
        if not content:
            continue
        
        # Store content hash for tracking
        content_hash = state.deduplication.compute_content_hash(content)
        seen_content[doc_id] = content_hash
        
        fp = state.deduplication.compute_fingerprint(content)
        matches = state.deduplication.find_duplicates(fp)
        
        if matches:
            duplicates.append({
                "doc_id": doc_id,
                "content_hash": content_hash,
                "matches": [
                    {"doc_id": m.doc_id, "similarity": m.similarity}
                    for m in matches[:5]
                ]
            })
    
    return duplicates


@app.post("/documents/{doc_id}/mark-duplicate")
async def mark_as_duplicate(
    doc_id: int,
    canonical_id: int = Query(..., description="ID of canonical document")
):
    """Manually mark a document as duplicate of another."""
    if not state.deduplication:
        raise HTTPException(status_code=501, detail="Deduplication not available")
    
    # This would update the database to mark the relationship
    # For now, just acknowledge the request
    logger.info(f"Marked doc_id={doc_id} as duplicate of canonical_id={canonical_id}")
    
    return {
        "status": "marked",
        "doc_id": doc_id,
        "canonical_id": canonical_id
    }


# --- Language Detection Endpoints ---

class LanguageDetectionRequest(BaseModel):
    texts: List[str] = Field(..., description="Texts to detect language for")


class LanguageDetectionResult(BaseModel):
    text_preview: str
    language: str
    confidence: float
    script: Optional[str] = None


@app.post("/language/detect", response_model=List[LanguageDetectionResult])
async def detect_language(request: LanguageDetectionRequest):
    """Detect language of provided texts."""
    if not state.language_detector:
        raise HTTPException(status_code=501, detail="Language detection not available (langdetect not installed)")
    
    results = state.language_detector.detect_batch(request.texts)
    
    return [
        LanguageDetectionResult(
            text_preview=text[:100] + "..." if len(text) > 100 else text,
            language=result.language,
            confidence=result.confidence,
            script=result.script
        )
        for text, result in zip(request.texts, results)
    ]


# --- Enhanced Search with Language Filter ---

class FilteredSearchRequest(SearchRequest):
    language: Optional[str] = Field(None, description="Filter by language (ISO 639-1 code)")
    metadata_filter: Optional[str] = Field(None, description="Metadata filter expression")


@app.post("/search/filtered", response_model=SearchResponse)
async def search_with_filters(request: FilteredSearchRequest):
    """Search with language and metadata filters."""
    if state.docs_df is None or state.vectors is None:
        raise HTTPException(status_code=400, detail="No documents indexed")
    
    if not state.searcher:
        raise HTTPException(status_code=500, detail="Searcher not initialized")
    
    sem_w = request.semantic_weight or 0.7
    lex_w = request.lexical_weight or 0.3
    
    # Perform base search
    results = state.searcher.search(
        query=request.query,
        docs_df=state.docs_df,
        vectors=state.vectors,
        top_k=request.top_k * 3,  # Get more results for filtering
        semantic_weight=sem_w,
        lexical_weight=lex_w
    )
    
    # Apply language filter
    if request.language and state.language_detector:
        filtered_results = []
        for score, content, doc_id in results:
            detected = state.language_detector.detect_simple(content)
            if detected == request.language:
                filtered_results.append((score, content, doc_id))
        results = filtered_results
    
    # Apply metadata filter
    if request.metadata_filter and state.metadata_filter and state.metadata_store:
        try:
            ast = state.metadata_filter.parse(request.metadata_filter)
            filtered_results = []
            for score, content, doc_id in results:
                metadata = state.metadata_store.get(doc_id)
                if state.metadata_filter._evaluate(ast, metadata):
                    filtered_results.append((score, content, doc_id))
            results = filtered_results
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid filter: {e}")
    
    # Limit to top_k
    results = results[:request.top_k]
    
    return SearchResponse(
        query=request.query,
        query_id=None,
        results=[
            SearchResult(score=score, content=content, doc_id=doc_id)
            for score, content, doc_id in results
        ],
        weights_used={"semantic": sem_w, "lexical": lex_w, "learned": False}
    )


# --- Async Job Endpoints ---

class AsyncIndexRequest(BaseModel):
    documents: List[Document] = Field(..., description="Documents to index")
    webhook_url: Optional[str] = Field(None, description="URL to notify on completion")


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress_total: int
    progress_processed: int
    progress_percentage: float
    error: Optional[str] = None
    created_at: str
    completed_at: Optional[str] = None


@app.post("/jobs/index", response_model=JobStatusResponse)
async def create_async_index_job(request: AsyncIndexRequest):
    """Create async indexing job for large document batches."""
    if not state.job_queue:
        raise HTTPException(status_code=500, detail="Job queue not initialized")
    
    if not request.documents:
        raise HTTPException(status_code=400, detail="No documents provided")
    
    # Convert to task format
    task_docs = [
        {"content": d.content, "source_path": d.source_path or ""}
        for d in request.documents
    ]
    
    task = IndexingTask(documents=task_docs)
    
    # Set up task handler if not already set
    def index_handler(task: IndexingTask, progress_cb):
        docs = [d["content"] for d in task.documents]
        paths = [d.get("source_path", "") for d in task.documents]
        
        with Indexer(db_path=state.db_path, use_faiss=state.use_faiss) as indexer:
            for i, (doc, path) in enumerate(zip(docs, paths)):
                indexer.add_documents([doc], [path])
                progress_cb(i + 1)
        
        return {"documents_indexed": len(docs)}
    
    state.job_queue.set_task_handler(index_handler)
    
    job = state.job_queue.enqueue(task, webhook_url=request.webhook_url)
    
    return JobStatusResponse(
        job_id=job.id,
        status=job.status.value,
        progress_total=job.progress.total,
        progress_processed=job.progress.processed,
        progress_percentage=job.progress.percentage,
        error=job.error,
        created_at=job.created_at.isoformat(),
        completed_at=job.completed_at.isoformat() if job.completed_at else None
    )


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get status of async indexing job."""
    if not state.job_queue:
        raise HTTPException(status_code=500, detail="Job queue not initialized")
    
    job = state.job_queue.get_status(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    
    return JobStatusResponse(
        job_id=job.id,
        status=job.status.value,
        progress_total=job.progress.total,
        progress_processed=job.progress.processed,
        progress_percentage=job.progress.percentage,
        error=job.error,
        created_at=job.created_at.isoformat(),
        completed_at=job.completed_at.isoformat() if job.completed_at else None
    )


@app.delete("/jobs/{job_id}")
async def cancel_job(job_id: str):
    """Cancel pending async indexing job."""
    if not state.job_queue:
        raise HTTPException(status_code=500, detail="Job queue not initialized")
    
    cancelled = state.job_queue.cancel(job_id)
    
    if cancelled:
        return {"status": "cancelled", "job_id": job_id}
    else:
        raise HTTPException(status_code=400, detail="Job cannot be cancelled (not pending or not found)")


@app.get("/jobs", response_model=List[JobStatusResponse])
async def list_jobs(
    status: Optional[str] = Query(None, description="Filter by status")
):
    """List all async indexing jobs."""
    if not state.job_queue:
        raise HTTPException(status_code=500, detail="Job queue not initialized")
    
    status_filter = None
    if status:
        try:
            status_filter = JobState(status)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid status: {status}")
    
    jobs = state.job_queue.backend.list_jobs(status_filter)
    
    return [
        JobStatusResponse(
            job_id=job.id,
            status=job.status.value,
            progress_total=job.progress.total,
            progress_processed=job.progress.processed,
            progress_percentage=job.progress.percentage,
            error=job.error,
            created_at=job.created_at.isoformat(),
            completed_at=job.completed_at.isoformat() if job.completed_at else None
        )
        for job in jobs
    ]


# --- Metadata Endpoints ---

class MetadataUpdateRequest(BaseModel):
    doc_id: int = Field(..., description="Document ID")
    metadata: dict = Field(..., description="Metadata to set")


@app.post("/documents/{doc_id}/metadata")
async def set_document_metadata(doc_id: int, request: MetadataUpdateRequest):
    """Set metadata for a document."""
    if not state.metadata_store:
        raise HTTPException(status_code=500, detail="Metadata store not initialized")
    
    state.metadata_store.set(doc_id, request.metadata)
    return {"status": "updated", "doc_id": doc_id}


@app.get("/documents/{doc_id}/metadata")
async def get_document_metadata(doc_id: int):
    """Get metadata for a document."""
    if not state.metadata_store:
        raise HTTPException(status_code=500, detail="Metadata store not initialized")
    
    metadata = state.metadata_store.get(doc_id)
    return {"doc_id": doc_id, "metadata": metadata}
