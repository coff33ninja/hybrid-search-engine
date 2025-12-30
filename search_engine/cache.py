"""
Caching layer for search results.

Supports in-memory and Redis backends.
"""
import hashlib
import json
import time
from typing import List, Tuple, Optional, Any, Dict
from dataclasses import dataclass
from functools import wraps
from loguru import logger


@dataclass
class CacheEntry:
    """Cached search result."""
    results: List[Tuple[float, str, int]]
    timestamp: float
    query: str
    params_hash: str


class InMemoryCache:
    """Simple in-memory LRU cache."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        """
        Args:
            max_size: Maximum number of cached queries.
            ttl_seconds: Time-to-live for cache entries.
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order: List[str] = []
    
    def _make_key(self, query: str, **params) -> str:
        """Create cache key from query and parameters."""
        params_str = json.dumps(params, sort_keys=True)
        combined = f"{query}:{params_str}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _evict_if_needed(self):
        """Evict oldest entries if cache is full."""
        while len(self._cache) >= self.max_size:
            if self._access_order:
                oldest_key = self._access_order.pop(0)
                self._cache.pop(oldest_key, None)
    
    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if entry has expired."""
        return time.time() - entry.timestamp > self.ttl_seconds
    
    def get(self, query: str, **params) -> Optional[List[Tuple[float, str, int]]]:
        """Get cached results."""
        key = self._make_key(query, **params)
        entry = self._cache.get(key)
        
        if entry is None:
            return None
        
        if self._is_expired(entry):
            self._cache.pop(key, None)
            return None
        
        # Update access order
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
        
        logger.debug(f"Cache hit for query: {query[:50]}...")
        return entry.results
    
    def set(self, query: str, results: List[Tuple[float, str, int]], **params):
        """Cache results."""
        self._evict_if_needed()
        
        key = self._make_key(query, **params)
        self._cache[key] = CacheEntry(
            results=results,
            timestamp=time.time(),
            query=query,
            params_hash=key
        )
        self._access_order.append(key)
        logger.debug(f"Cached results for query: {query[:50]}...")
    
    def invalidate(self, query: str = None, **params):
        """Invalidate cache entries."""
        if query:
            key = self._make_key(query, **params)
            self._cache.pop(key, None)
            if key in self._access_order:
                self._access_order.remove(key)
        else:
            # Clear all
            self._cache.clear()
            self._access_order.clear()
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "ttl_seconds": self.ttl_seconds
        }


class RedisCache:
    """Redis-backed cache for distributed deployments."""
    
    def __init__(
        self, 
        host: str = "localhost", 
        port: int = 6379,
        db: int = 0,
        prefix: str = "search:",
        ttl_seconds: int = 3600
    ):
        """
        Args:
            host: Redis host.
            port: Redis port.
            db: Redis database number.
            prefix: Key prefix for namespacing.
            ttl_seconds: TTL for cache entries.
        """
        self.prefix = prefix
        self.ttl_seconds = ttl_seconds
        self._client = None
        self._config = {"host": host, "port": port, "db": db}
    
    @property
    def client(self):
        """Lazy Redis connection."""
        if self._client is None:
            try:
                import redis
                self._client = redis.Redis(**self._config)
                self._client.ping()
                logger.info("Connected to Redis")
            except ImportError:
                raise ImportError("Install redis: pip install redis")
            except Exception as e:
                logger.error(f"Redis connection failed: {e}")
                raise
        return self._client
    
    def _make_key(self, query: str, **params) -> str:
        params_str = json.dumps(params, sort_keys=True)
        combined = f"{query}:{params_str}"
        hash_key = hashlib.md5(combined.encode()).hexdigest()
        return f"{self.prefix}{hash_key}"
    
    def get(self, query: str, **params) -> Optional[List[Tuple[float, str, int]]]:
        key = self._make_key(query, **params)
        data = self.client.get(key)
        
        if data is None:
            return None
        
        logger.debug(f"Redis cache hit for query: {query[:50]}...")
        return json.loads(data)
    
    def set(self, query: str, results: List[Tuple[float, str, int]], **params):
        key = self._make_key(query, **params)
        data = json.dumps(results)
        self.client.setex(key, self.ttl_seconds, data)
        logger.debug(f"Redis cached results for query: {query[:50]}...")
    
    def invalidate(self, query: str = None, **params):
        if query:
            key = self._make_key(query, **params)
            self.client.delete(key)
        else:
            # Clear all with prefix
            keys = self.client.keys(f"{self.prefix}*")
            if keys:
                self.client.delete(*keys)
    
    def stats(self) -> Dict[str, Any]:
        info = self.client.info("memory")
        keys = self.client.keys(f"{self.prefix}*")
        return {
            "size": len(keys),
            "memory_used": info.get("used_memory_human", "unknown"),
            "ttl_seconds": self.ttl_seconds
        }


class SemanticCache:
    """
    Semantic caching - returns cached results for similar queries.
    
    Uses embedding similarity to find cached queries that are
    semantically similar to the current query.
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.95,
        max_size: int = 1000,
        ttl_seconds: int = 3600
    ):
        """
        Args:
            model_name: Embedding model for query similarity.
            similarity_threshold: Minimum similarity to use cached result.
            max_size: Maximum cached queries.
            ttl_seconds: TTL for entries.
        """
        self.similarity_threshold = similarity_threshold
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._model = None
        self._model_name = model_name
        
        # Store: query_embedding -> (query, results, timestamp)
        self._cache: List[Tuple[Any, str, List, float]] = []
    
    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
        return self._model
    
    def _cosine_sim(self, a, b) -> float:
        import numpy as np
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    
    def get(self, query: str, **params) -> Optional[List[Tuple[float, str, int]]]:
        import numpy as np
        
        query_emb = self.model.encode([query], convert_to_numpy=True)[0]
        now = time.time()
        
        best_match = None
        best_sim = 0.0
        
        # Find most similar cached query
        valid_entries = []
        for emb, cached_query, results, timestamp in self._cache:
            if now - timestamp > self.ttl_seconds:
                continue  # Expired
            
            valid_entries.append((emb, cached_query, results, timestamp))
            sim = self._cosine_sim(query_emb, emb)
            
            if sim > best_sim and sim >= self.similarity_threshold:
                best_sim = sim
                best_match = results
        
        # Update cache with only valid entries
        self._cache = valid_entries
        
        if best_match:
            logger.debug(f"Semantic cache hit (sim={best_sim:.3f}) for: {query[:50]}...")
        
        return best_match
    
    def set(self, query: str, results: List[Tuple[float, str, int]], **params):
        # Evict if full
        while len(self._cache) >= self.max_size:
            self._cache.pop(0)
        
        query_emb = self.model.encode([query], convert_to_numpy=True)[0]
        self._cache.append((query_emb, query, results, time.time()))
    
    def invalidate(self, query: str = None, **params):
        if query is None:
            self._cache.clear()
    
    def stats(self) -> Dict[str, Any]:
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "similarity_threshold": self.similarity_threshold
        }


def cached_search(cache):
    """Decorator to add caching to search method."""
    def decorator(search_func):
        @wraps(search_func)
        def wrapper(self, query: str, *args, **kwargs):
            # Try cache first
            cache_params = {
                "top_k": kwargs.get("top_k", 5),
                "semantic_weight": kwargs.get("semantic_weight"),
                "lexical_weight": kwargs.get("lexical_weight")
            }
            
            cached = cache.get(query, **cache_params)
            if cached is not None:
                return cached
            
            # Execute search
            results = search_func(self, query, *args, **kwargs)
            
            # Cache results
            cache.set(query, results, **cache_params)
            
            return results
        return wrapper
    return decorator
