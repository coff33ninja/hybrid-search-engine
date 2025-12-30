# Design Document: High-Impact Features

## Overview

This design covers six high-impact features for the hybrid search engine: Query Autocomplete, Semantic Caching, Multi-Language Support, Document Deduplication, Metadata Filtering, and Async Indexing. These features are designed to integrate with the existing architecture while maintaining backward compatibility.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         API Layer                               │
│  FastAPI + New Endpoints (autocomplete, jobs, filters)          │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                      Feature Layer                              │
│  Autocomplete │ SemanticCache │ MetadataFilter │ JobQueue       │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                      Core Layer                                 │
│  Indexer + Dedup │ Searcher │ LanguageDetector │ Embedder       │
└─────────────────────────────────────────────────────────────────┘
                              │
┌──────────────┬──────────────┬──────────────┬───────────────────┐
│   DuckDB     │    FAISS     │    Redis     │   Job Backend     │
│  (storage)   │   (vectors)  │   (cache)    │  (Celery/RQ)      │
└──────────────┴──────────────┴──────────────┴───────────────────┘
```

## Components and Interfaces

### 1. Autocomplete Engine

```python
class AutocompleteEngine:
    def __init__(
        self,
        trie: PrefixTrie,
        embedder: SentenceTransformer,
        query_history: QueryHistory,
        max_suggestions: int = 10
    ):
        pass
    
    def suggest(
        self,
        partial_query: str,
        limit: int = 10
    ) -> List[Suggestion]:
        """
        Returns ranked suggestions combining:
        1. Prefix matches from trie (query history)
        2. Fuzzy matches for typo tolerance
        3. Semantic similarity to indexed content
        """
        pass
    
    def record_selection(self, partial: str, selected: str) -> None:
        """Log selection to improve future rankings."""
        pass


class PrefixTrie:
    def insert(self, query: str, frequency: int = 1) -> None:
        pass
    
    def search_prefix(self, prefix: str, limit: int) -> List[Tuple[str, int]]:
        """Returns (query, frequency) pairs matching prefix."""
        pass
    
    def fuzzy_search(self, query: str, max_distance: int = 2) -> List[str]:
        """Returns queries within edit distance."""
        pass


@dataclass
class Suggestion:
    text: str
    score: float
    source: str  # "history", "semantic", "fuzzy"
```

### 2. Semantic Cache

```python
class SemanticCache:
    def __init__(
        self,
        backend: CacheBackend,  # InMemory or Redis
        similarity_threshold: float = 0.95,
        ttl_seconds: int = 3600,
        max_size: int = 10000
    ):
        pass
    
    def get(self, query: str, query_embedding: np.ndarray) -> Optional[CachedResult]:
        """
        Check cache using LSH for fast similarity lookup.
        Returns cached results if similarity > threshold.
        """
        pass
    
    def set(
        self,
        query: str,
        query_embedding: np.ndarray,
        results: List[SearchResult]
    ) -> None:
        """Store query and results in cache."""
        pass
    
    def invalidate(self, pattern: Optional[str] = None) -> int:
        """Invalidate cache entries. Returns count invalidated."""
        pass


class LSHIndex:
    def __init__(self, num_tables: int = 10, hash_size: int = 8):
        pass
    
    def add(self, key: str, embedding: np.ndarray) -> None:
        pass
    
    def query(self, embedding: np.ndarray, threshold: float) -> List[Tuple[str, float]]:
        """Returns (key, similarity) pairs above threshold."""
        pass


@dataclass
class CachedResult:
    query: str
    results: List[SearchResult]
    similarity: float
    cached_at: datetime
```

### 3. Language Detector

```python
class LanguageDetector:
    def __init__(self, model: str = "fasttext"):
        pass
    
    def detect(self, text: str) -> LanguageResult:
        """Detect language with confidence score."""
        pass
    
    def detect_batch(self, texts: List[str]) -> List[LanguageResult]:
        """Batch detection for efficiency."""
        pass


@dataclass
class LanguageResult:
    language: str  # ISO 639-1 code (e.g., "en", "fr", "de")
    confidence: float
    script: Optional[str]  # e.g., "Latin", "Cyrillic"


class MultilingualEmbedder:
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        pass
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts in any supported language."""
        pass
    
    @property
    def supported_languages(self) -> List[str]:
        pass
```

### 4. Deduplication Engine

```python
class DeduplicationEngine:
    def __init__(
        self,
        num_perm: int = 128,
        threshold: float = 0.9,
        action: DedupeAction = DedupeAction.FLAG
    ):
        pass
    
    def compute_fingerprint(self, content: str) -> MinHashSignature:
        """Compute MinHash fingerprint for document."""
        pass
    
    def find_duplicates(self, fingerprint: MinHashSignature) -> List[DuplicateMatch]:
        """Find existing documents similar to this fingerprint."""
        pass
    
    def add_to_index(self, doc_id: int, fingerprint: MinHashSignature) -> None:
        """Add fingerprint to LSH index for future lookups."""
        pass


class DedupeAction(Enum):
    SKIP = "skip"      # Don't index duplicate
    MERGE = "merge"    # Combine metadata with canonical
    FLAG = "flag"      # Index but mark as duplicate


@dataclass
class MinHashSignature:
    values: np.ndarray  # uint64 array of hash values
    
    def jaccard_similarity(self, other: "MinHashSignature") -> float:
        pass


@dataclass
class DuplicateMatch:
    doc_id: int
    similarity: float
    is_canonical: bool
```

### 5. Metadata Filter

```python
class MetadataFilter:
    def __init__(self, schema: Optional[MetadataSchema] = None):
        pass
    
    def parse(self, filter_expr: str) -> FilterAST:
        """
        Parse filter expression into AST.
        Syntax: field:value, field:>value, field:[v1,v2], NOT field:value
        Combined with AND, OR, parentheses
        """
        pass
    
    def apply(
        self,
        filter_ast: FilterAST,
        doc_ids: List[int],
        metadata_store: MetadataStore
    ) -> List[int]:
        """Apply filter to document IDs, return matching IDs."""
        pass
    
    def to_sql(self, filter_ast: FilterAST) -> str:
        """Convert filter to SQL WHERE clause for DuckDB."""
        pass


@dataclass
class MetadataSchema:
    fields: Dict[str, FieldType]  # field_name -> type


class FieldType(Enum):
    TEXT = "text"
    DATE = "date"
    NUMBER = "number"
    ARRAY = "array"
    BOOLEAN = "boolean"


class FilterAST:
    """Abstract syntax tree for filter expressions."""
    pass


class MetadataStore:
    def __init__(self, db_path: str):
        pass
    
    def set(self, doc_id: int, metadata: Dict[str, Any]) -> None:
        pass
    
    def get(self, doc_id: int) -> Dict[str, Any]:
        pass
    
    def query(self, sql_where: str) -> List[int]:
        """Execute SQL query, return matching doc_ids."""
        pass
```

### 6. Job Queue

```python
class JobQueue:
    def __init__(
        self,
        backend: JobBackend,  # Celery, RQ, or InMemory
        max_concurrent: int = 3,
        max_retries: int = 3
    ):
        pass
    
    def enqueue(
        self,
        task: IndexingTask,
        webhook_url: Optional[str] = None
    ) -> Job:
        """Enqueue task, return job with ID for tracking."""
        pass
    
    def get_status(self, job_id: str) -> JobStatus:
        """Get current job status and progress."""
        pass
    
    def cancel(self, job_id: str) -> bool:
        """Cancel pending job. Returns True if cancelled."""
        pass


@dataclass
class Job:
    id: str
    status: JobState
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    progress: JobProgress
    error: Optional[str]


class JobState(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class JobProgress:
    total: int
    processed: int
    
    @property
    def percentage(self) -> float:
        return (self.processed / self.total * 100) if self.total > 0 else 0


@dataclass
class IndexingTask:
    documents: List[Document]
    source_paths: Optional[List[str]]
    options: IndexingOptions
```

## Data Models

### Extended Document Schema (DuckDB)

```sql
CREATE TABLE docs (
    doc_id INTEGER PRIMARY KEY,
    content TEXT,
    source_path TEXT,
    char_count INTEGER,
    word_count INTEGER,
    indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    -- New fields
    language VARCHAR(10),
    fingerprint BLOB,
    is_duplicate BOOLEAN DEFAULT FALSE,
    canonical_doc_id INTEGER,
    metadata JSON
);

CREATE INDEX idx_docs_language ON docs(language);
CREATE INDEX idx_docs_duplicate ON docs(is_duplicate);
```

### Query History Schema

```sql
CREATE TABLE query_suggestions (
    query_text TEXT PRIMARY KEY,
    frequency INTEGER DEFAULT 1,
    last_used TIMESTAMP,
    avg_result_count FLOAT
);
```

### Job Schema

```sql
CREATE TABLE indexing_jobs (
    job_id VARCHAR(36) PRIMARY KEY,
    status VARCHAR(20),
    total_docs INTEGER,
    processed_docs INTEGER,
    created_at TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    error_message TEXT,
    webhook_url TEXT,
    retry_count INTEGER DEFAULT 0
);
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Autocomplete Response Constraints

*For any* partial query string of 2+ characters, the Autocomplete_Engine SHALL return a list of at most `max_suggestions` items, where each suggestion has a non-negative score, and suggestions are sorted by score in descending order.

**Validates: Requirements 1.1, 1.5**

### Property 2: Autocomplete Fuzzy Tolerance

*For any* valid query term and any string within edit distance 2 of that term, the Autocomplete_Engine SHALL include the original term (or semantically similar terms) in its suggestions.

**Validates: Requirements 1.3**

### Property 3: Semantic Cache Similarity Threshold

*For any* two queries Q1 and Q2 with embedding similarity S, if S >= threshold then cache lookup for Q2 after caching Q1 SHALL return Q1's results; if S < threshold then cache lookup SHALL return None.

**Validates: Requirements 2.2, 2.3**

### Property 4: Cache LRU Eviction

*For any* cache at max capacity, adding a new entry SHALL evict the least recently accessed entry, and the new entry SHALL be retrievable.

**Validates: Requirements 2.6**

### Property 5: Cache Round-Trip

*For any* query and results, storing in cache then retrieving with the same query embedding SHALL return equivalent results.

**Validates: Requirements 2.1**

### Property 6: Language Detection Accuracy

*For any* document in a supported language with at least 50 characters, the Language_Detector SHALL correctly identify the language with confidence >= 0.95 for at least 95% of test cases.

**Validates: Requirements 3.1**

### Property 7: Cross-Lingual Search

*For any* semantically equivalent document pair (one in language A, one in language B), searching with a query in language A SHALL return the document in language B with a relevance score within 20% of the same-language result.

**Validates: Requirements 3.3**

### Property 8: Language Filter Correctness

*For any* search with a language filter, all returned documents SHALL have the specified language in their metadata.

**Validates: Requirements 3.5**

### Property 9: Duplicate Detection Consistency

*For any* two documents D1 and D2 with Jaccard similarity S computed via MinHash, if S >= threshold then D2 SHALL be flagged as duplicate of D1 (or vice versa); if S < threshold then neither SHALL be flagged as duplicate of the other.

**Validates: Requirements 4.2**

### Property 10: Duplicate Action Correctness

*For any* detected duplicate with action=SKIP, the document SHALL NOT appear in the index; with action=FLAG, the document SHALL appear with is_duplicate=True; with action=MERGE, the canonical document SHALL have combined metadata.

**Validates: Requirements 4.3**

### Property 11: Metadata Filter Boolean Logic

*For any* filter expression with AND, OR, NOT operators and any document set, the filtered results SHALL exactly match the set of documents satisfying the boolean expression evaluated against their metadata.

**Validates: Requirements 5.2, 5.3, 5.4, 5.5**

### Property 12: Metadata Round-Trip

*For any* document with metadata, indexing then retrieving SHALL return equivalent metadata.

**Validates: Requirements 5.1**

### Property 13: Async Job Lifecycle

*For any* batch indexing request > 100 documents, the API SHALL return within 1 second with a valid job_id, and subsequent status queries SHALL return valid JobStatus with progress information.

**Validates: Requirements 6.1, 6.2, 6.6**

### Property 14: Job Retry Behavior

*For any* failing indexing job, the system SHALL retry up to max_retries times, with each retry delay >= previous_delay * 2 (exponential backoff).

**Validates: Requirements 6.4**

### Property 15: Job Cancellation

*For any* job in PENDING state, calling cancel SHALL transition it to CANCELLED state, and the job's documents SHALL NOT be indexed.

**Validates: Requirements 6.5**

### Property 16: Concurrent Job Limit

*For any* number of concurrent job submissions exceeding max_concurrent, excess jobs SHALL be queued (PENDING) rather than immediately processing.

**Validates: Requirements 6.7**

## Error Handling

| Component | Error Type | Handling |
|-----------|------------|----------|
| Autocomplete | Empty input | Return empty list |
| Autocomplete | Timeout | Return partial results |
| SemanticCache | Redis connection failure | Fall back to in-memory or bypass cache |
| SemanticCache | Corrupted entry | Delete entry, log warning |
| LanguageDetector | Undetectable language | Default to "unknown", log warning |
| LanguageDetector | Model load failure | Raise startup error |
| Deduplication | LSH index corruption | Rebuild index from fingerprints |
| MetadataFilter | Invalid syntax | Return 400 with parse error details |
| MetadataFilter | Unknown field | Return 400 with available fields |
| JobQueue | Backend unavailable | Return 503, retry connection |
| JobQueue | Job timeout | Mark failed, trigger retry |
| JobQueue | Webhook failure | Log error, don't block job completion |

## Testing Strategy

### Unit Tests
- Trie insert/search operations
- Filter expression parsing
- MinHash fingerprint computation
- LSH index add/query
- Job state transitions

### Property-Based Tests (using Hypothesis)
- Autocomplete response constraints (Property 1)
- Cache similarity threshold behavior (Property 3)
- Cache LRU eviction (Property 4)
- Duplicate detection consistency (Property 9)
- Metadata filter boolean logic (Property 11)
- Job lifecycle (Property 13)

### Integration Tests
- End-to-end autocomplete with real embeddings
- Cache with Redis backend
- Multi-language indexing and search
- Async job completion with webhook
- Metadata filtering with DuckDB

### Performance Tests
- Autocomplete < 50ms for 10k query history
- Cache lookup O(1) with 100k entries
- Deduplication O(1) with 1M documents
- Filter pre-application speedup measurement

### Test Configuration
- Property tests: minimum 100 iterations per property
- Use `hypothesis` library for Python property-based testing
- Tag format: **Feature: high-impact-features, Property {number}: {property_text}**
