"""
Semantic caching for search results.

Caches query embeddings and results based on semantic similarity.
"""
import hashlib
import json
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from loguru import logger

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


@dataclass
class SearchResult:
    """Cached search result."""
    score: float
    content: str
    doc_id: int


@dataclass
class CachedResult:
    """Cache entry with query and results."""
    query: str
    results: List[SearchResult] = field(default_factory=list)
    similarity: float = 0.0
    cached_at: datetime = field(default_factory=datetime.now)
    embedding: Optional[np.ndarray] = None


class LSHIndex:
    """
    Locality-Sensitive Hashing for fast similarity lookup.
    
    Uses random hyperplane LSH for cosine similarity.
    """
    
    def __init__(self, dimension: int, num_tables: int = 10, hash_size: int = 8):
        """
        Initialize LSH index.
        
        Args:
            dimension: Embedding dimension
            num_tables: Number of hash tables (more = better recall)
            hash_size: Bits per hash (more = better precision)
        """
        self.dimension = dimension
        self.num_tables = num_tables
        self.hash_size = hash_size
        
        # Random hyperplanes for hashing
        self.hyperplanes = [
            np.random.randn(hash_size, dimension).astype(np.float32)
            for _ in range(num_tables)
        ]
        
        # Hash tables: table_idx -> hash_value -> list of (key, embedding)
        self.tables: List[Dict[int, List[Tuple[str, np.ndarray]]]] = [
            {} for _ in range(num_tables)
        ]
        
        self._keys: Dict[str, np.ndarray] = {}
    
    def _hash(self, embedding: np.ndarray, table_idx: int) -> int:
        """Compute hash value for embedding."""
        projections = self.hyperplanes[table_idx] @ embedding
        bits = (projections > 0).astype(int)
        return int(''.join(map(str, bits)), 2)
    
    def add(self, key: str, embedding: np.ndarray) -> None:
        """Add embedding to index."""
        embedding = embedding.astype(np.float32)
        
        # Normalize for cosine similarity
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        self._keys[key] = embedding
        
        for i, table in enumerate(self.tables):
            h = self._hash(embedding, i)
            if h not in table:
                table[h] = []
            table[h].append((key, embedding))
    
    def remove(self, key: str) -> bool:
        """Remove key from index."""
        if key not in self._keys:
            return False
        
        embedding = self._keys[key]
        del self._keys[key]
        
        for i, table in enumerate(self.tables):
            h = self._hash(embedding, i)
            if h in table:
                table[h] = [(k, e) for k, e in table[h] if k != key]
                if not table[h]:
                    del table[h]
        
        return True
    
    def query(self, embedding: np.ndarray, threshold: float = 0.95) -> List[Tuple[str, float]]:
        """
        Find similar embeddings above threshold.
        
        Args:
            embedding: Query embedding
            threshold: Minimum cosine similarity
            
        Returns:
            List of (key, similarity) tuples
        """
        embedding = embedding.astype(np.float32)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        # Collect candidates from all tables
        candidates = set()
        for i, table in enumerate(self.tables):
            h = self._hash(embedding, i)
            if h in table:
                for key, _ in table[h]:
                    candidates.add(key)
        
        # Compute exact similarity for candidates
        results = []
        for key in candidates:
            stored = self._keys[key]
            similarity = float(np.dot(embedding, stored))
            if similarity >= threshold:
                results.append((key, similarity))
        
        # Sort by similarity descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    @property
    def size(self) -> int:
        """Number of items in index."""
        return len(self._keys)


class CacheBackend(ABC):
    """Abstract cache backend."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        pass
    
    @abstractmethod
    def set(self, key: str, value: Dict[str, Any], ttl: Optional[int] = None) -> None:
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        pass
    
    @abstractmethod
    def clear(self) -> int:
        pass
    
    @abstractmethod
    def size(self) -> int:
        pass


class InMemoryBackend(CacheBackend):
    """In-memory cache with LRU eviction."""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self._cache: OrderedDict[str, Tuple[Dict[str, Any], float]] = OrderedDict()
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        if key not in self._cache:
            return None
        
        value, expiry = self._cache[key]
        
        # Check TTL
        if expiry > 0 and time.time() > expiry:
            del self._cache[key]
            return None
        
        # Move to end (most recently used)
        self._cache.move_to_end(key)
        return value
    
    def set(self, key: str, value: Dict[str, Any], ttl: Optional[int] = None) -> None:
        expiry = time.time() + ttl if ttl else 0
        
        # Evict if at capacity
        while len(self._cache) >= self.max_size:
            self._cache.popitem(last=False)  # Remove oldest
        
        self._cache[key] = (value, expiry)
        self._cache.move_to_end(key)
    
    def delete(self, key: str) -> bool:
        if key in self._cache:
            del self._cache[key]
            return True
        return False
    
    def clear(self) -> int:
        count = len(self._cache)
        self._cache.clear()
        return count
    
    def size(self) -> int:
        return len(self._cache)


class RedisBackend(CacheBackend):
    """Redis cache backend."""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        prefix: str = "search_cache:"
    ):
        if not REDIS_AVAILABLE:
            raise ImportError("redis required. Install with: pip install redis")
        
        self.client = redis.Redis(host=host, port=port, db=db)
        self.prefix = prefix
    
    def _key(self, key: str) -> str:
        return f"{self.prefix}{key}"
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        data = self.client.get(self._key(key))
        if data:
            return json.loads(data)
        return None
    
    def set(self, key: str, value: Dict[str, Any], ttl: Optional[int] = None) -> None:
        data = json.dumps(value)
        if ttl:
            self.client.setex(self._key(key), ttl, data)
        else:
            self.client.set(self._key(key), data)
    
    def delete(self, key: str) -> bool:
        return self.client.delete(self._key(key)) > 0
    
    def clear(self) -> int:
        keys = self.client.keys(f"{self.prefix}*")
        if keys:
            return self.client.delete(*keys)
        return 0
    
    def size(self) -> int:
        return len(self.client.keys(f"{self.prefix}*"))


class SemanticCache:
    """
    Cache search results based on semantic similarity.
    
    Uses LSH for O(1) approximate similarity lookup.
    """
    
    def __init__(
        self,
        backend: Optional[CacheBackend] = None,
        similarity_threshold: float = 0.95,
        ttl_seconds: int = 3600,
        max_size: int = 10000,
        embedding_dim: int = 384
    ):
        """
        Initialize semantic cache.
        
        Args:
            backend: Cache backend (InMemory or Redis)
            similarity_threshold: Minimum similarity for cache hit
            ttl_seconds: Time-to-live for cache entries
            max_size: Maximum cache size
            embedding_dim: Embedding dimension for LSH
        """
        self.backend = backend or InMemoryBackend(max_size=max_size)
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        
        # LSH index for similarity lookup
        self.lsh = LSHIndex(
            dimension=embedding_dim,
            num_tables=10,
            hash_size=8
        )
        
        # Stats
        self._hits = 0
        self._misses = 0
        
        logger.info(f"Initialized SemanticCache (threshold={similarity_threshold}, ttl={ttl_seconds}s)")
    
    def _query_key(self, query: str) -> str:
        """Generate cache key from query."""
        return hashlib.md5(query.lower().strip().encode()).hexdigest()
    
    def get(
        self,
        query: str,
        query_embedding: np.ndarray
    ) -> Optional[CachedResult]:
        """
        Check cache for similar query.
        
        Args:
            query: Search query
            query_embedding: Query embedding vector
            
        Returns:
            CachedResult if cache hit, None otherwise
        """
        # Check LSH for similar queries
        similar = self.lsh.query(query_embedding, threshold=self.similarity_threshold)
        
        if not similar:
            self._misses += 1
            return None
        
        # Get best match
        best_key, similarity = similar[0]
        cached = self.backend.get(best_key)
        
        if not cached:
            self._misses += 1
            return None
        
        self._hits += 1
        logger.debug(f"Cache hit for '{query}' (similarity={similarity:.3f})")
        
        return CachedResult(
            query=cached['query'],
            results=[SearchResult(**r) for r in cached['results']],
            similarity=similarity,
            cached_at=datetime.fromisoformat(cached['cached_at'])
        )
    
    def set(
        self,
        query: str,
        query_embedding: np.ndarray,
        results: List[SearchResult]
    ) -> None:
        """
        Store query and results in cache.
        
        Args:
            query: Search query
            query_embedding: Query embedding vector
            results: Search results to cache
        """
        key = self._query_key(query)
        
        # Store in backend
        cache_data = {
            'query': query,
            'results': [{'score': r.score, 'content': r.content, 'doc_id': r.doc_id} for r in results],
            'cached_at': datetime.now().isoformat()
        }
        self.backend.set(key, cache_data, ttl=self.ttl_seconds)
        
        # Add to LSH index
        self.lsh.add(key, query_embedding)
        
        logger.debug(f"Cached results for '{query}'")
    
    def invalidate(self, pattern: Optional[str] = None) -> int:
        """
        Invalidate cache entries.
        
        Args:
            pattern: Optional pattern to match (not implemented for LSH)
            
        Returns:
            Number of entries invalidated
        """
        if pattern:
            logger.warning("Pattern-based invalidation not supported, clearing all")
        
        count = self.backend.clear()
        self.lsh = LSHIndex(
            dimension=self.lsh.dimension,
            num_tables=self.lsh.num_tables,
            hash_size=self.lsh.hash_size
        )
        
        logger.info(f"Invalidated {count} cache entries")
        return count
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Cache statistics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0
        
        return {
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': hit_rate,
            'size': self.backend.size(),
            'lsh_size': self.lsh.size
        }
