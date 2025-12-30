"""
Query autocomplete with prefix matching, fuzzy search, and semantic suggestions.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import time
from loguru import logger

try:
    from rapidfuzz import fuzz, process
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False


@dataclass
class Suggestion:
    """Autocomplete suggestion."""
    text: str
    score: float
    source: str = field(default="history")  # "history", "semantic", "fuzzy"
    frequency: int = field(default=0)


class TrieNode:
    """Node in prefix trie."""
    def __init__(self):
        self.children: Dict[str, TrieNode] = defaultdict(TrieNode)
        self.is_end: bool = False
        self.frequency: int = 0
        self.query: Optional[str] = None


class PrefixTrie:
    """
    Prefix tree for efficient query completion.
    
    Supports prefix matching and fuzzy search.
    """
    
    def __init__(self):
        self.root = TrieNode()
        self._all_queries: Dict[str, int] = {}  # query -> frequency
    
    def insert(self, query: str, frequency: int = 1) -> None:
        """
        Insert query into trie.
        
        Args:
            query: Query string to insert
            frequency: Query frequency/popularity
        """
        query = query.lower().strip()
        if not query:
            return
        
        node = self.root
        for char in query:
            node = node.children[char]  # defaultdict auto-creates nodes
        
        node.is_end = True
        node.frequency += frequency
        node.query = query
        
        self._all_queries[query] = self._all_queries.get(query, 0) + frequency
    
    def search_prefix(self, prefix: str, limit: int = 10) -> List[Tuple[str, int]]:
        """
        Find queries matching prefix.
        
        Args:
            prefix: Prefix to search for
            limit: Maximum results to return
            
        Returns:
            List of (query, frequency) tuples sorted by frequency
        """
        prefix = prefix.lower().strip()
        if not prefix:
            return []
        
        # Navigate to prefix node
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
        
        # Collect all queries under this prefix
        results = []
        self._collect_queries(node, results)
        
        # Sort by frequency and limit
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]
    
    def _collect_queries(self, node: TrieNode, results: List[Tuple[str, int]]) -> None:
        """Recursively collect queries from node."""
        if node.is_end and node.query:
            results.append((node.query, node.frequency))
        
        for child in node.children.values():
            self._collect_queries(child, results)
    
    def fuzzy_search(self, query: str, max_distance: int = 2, limit: int = 10) -> List[str]:
        """
        Find queries within edit distance of query.
        
        Args:
            query: Query to match
            max_distance: Maximum edit distance
            limit: Maximum results
            
        Returns:
            List of matching queries
        """
        if not RAPIDFUZZ_AVAILABLE:
            return []
        
        query = query.lower().strip()
        if not query or not self._all_queries:
            return []
        
        # Use rapidfuzz for fuzzy matching
        matches = process.extract(
            query,
            list(self._all_queries.keys()),
            scorer=fuzz.ratio,
            limit=limit * 2  # Get more candidates for filtering
        )
        
        # Filter by minimum similarity (roughly corresponds to edit distance)
        min_similarity = max(0, 100 - (max_distance * 20))
        results = [m[0] for m in matches if m[1] >= min_similarity]
        
        return results[:limit]
    
    def update_frequency(self, query: str, increment: int = 1) -> None:
        """Update frequency of existing query."""
        query = query.lower().strip()
        if query in self._all_queries:
            self._all_queries[query] += increment
            # Update trie node
            node = self.root
            for char in query:
                if char not in node.children:
                    return
                node = node.children[char]
            if node.is_end:
                node.frequency += increment
    
    @property
    def size(self) -> int:
        """Number of queries in trie."""
        return len(self._all_queries)
    
    def get_top_queries(self, limit: int = 100) -> List[Tuple[str, int]]:
        """Get most frequent queries."""
        sorted_queries = sorted(
            self._all_queries.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_queries[:limit]


class AutocompleteEngine:
    """
    Query autocomplete combining prefix matching, fuzzy search, and semantic similarity.
    """
    
    def __init__(
        self,
        max_suggestions: int = 10,
        min_prefix_length: int = 2,
        fuzzy_threshold: int = 2
    ):
        """
        Initialize autocomplete engine.
        
        Args:
            max_suggestions: Maximum suggestions to return
            min_prefix_length: Minimum characters before suggesting
            fuzzy_threshold: Maximum edit distance for fuzzy matching
        """
        self.trie = PrefixTrie()
        self.max_suggestions = max_suggestions
        self.min_prefix_length = min_prefix_length
        self.fuzzy_threshold = fuzzy_threshold
        
        # Optional: semantic embedder for content-based suggestions
        self._embedder = None
        self._content_terms: List[str] = []
        
        logger.info(f"Initialized AutocompleteEngine (max_suggestions={max_suggestions})")
    
    def add_query(self, query: str, frequency: int = 1) -> None:
        """Add query to history."""
        self.trie.insert(query, frequency)
    
    def add_queries_batch(self, queries: List[Tuple[str, int]]) -> None:
        """Add multiple queries with frequencies."""
        for query, freq in queries:
            self.trie.insert(query, freq)
    
    def set_content_terms(self, terms: List[str]) -> None:
        """Set content terms for semantic suggestions."""
        self._content_terms = list(set(terms))
    
    def suggest(
        self,
        partial_query: str,
        limit: Optional[int] = None
    ) -> List[Suggestion]:
        """
        Get suggestions for partial query.
        
        Args:
            partial_query: Partial query string
            limit: Override max_suggestions
            
        Returns:
            List of suggestions sorted by score
        """
        start_time = time.time()
        limit = limit or self.max_suggestions
        
        partial = partial_query.lower().strip()
        
        if len(partial) < self.min_prefix_length:
            return []
        
        suggestions: Dict[str, Suggestion] = {}
        
        # 1. Prefix matches from history (highest priority)
        prefix_matches = self.trie.search_prefix(partial, limit=limit)
        for query, freq in prefix_matches:
            score = 1.0 + (freq / 100)  # Base score + frequency bonus
            suggestions[query] = Suggestion(
                text=query,
                score=score,
                source="history",
                frequency=freq
            )
        
        # 2. Fuzzy matches for typo tolerance
        if len(suggestions) < limit:
            fuzzy_matches = self.trie.fuzzy_search(
                partial,
                max_distance=self.fuzzy_threshold,
                limit=limit
            )
            for query in fuzzy_matches:
                if query not in suggestions:
                    freq = self.trie._all_queries.get(query, 0)
                    suggestions[query] = Suggestion(
                        text=query,
                        score=0.7 + (freq / 200),  # Lower base score
                        source="fuzzy",
                        frequency=freq
                    )
        
        # 3. Content-based suggestions (if available)
        if len(suggestions) < limit and self._content_terms:
            content_matches = self._get_content_suggestions(partial, limit)
            for term in content_matches:
                if term not in suggestions:
                    suggestions[term] = Suggestion(
                        text=term,
                        score=0.5,
                        source="semantic",
                        frequency=0
                    )
        
        # Sort by score and limit
        result = sorted(suggestions.values(), key=lambda s: s.score, reverse=True)
        result = result[:limit]
        
        elapsed = (time.time() - start_time) * 1000
        logger.debug(f"Autocomplete for '{partial}': {len(result)} suggestions in {elapsed:.1f}ms")
        
        return result
    
    def _get_content_suggestions(self, partial: str, limit: int) -> List[str]:
        """Get suggestions from indexed content terms."""
        if not RAPIDFUZZ_AVAILABLE:
            # Fallback to simple prefix matching
            return [t for t in self._content_terms if t.lower().startswith(partial)][:limit]
        
        matches = process.extract(
            partial,
            self._content_terms,
            scorer=fuzz.partial_ratio,
            limit=limit
        )
        return [m[0] for m in matches if m[1] >= 60]
    
    def record_selection(self, partial: str, selected: str) -> None:
        """
        Record that user selected a suggestion.
        
        Args:
            partial: The partial query that was typed
            selected: The suggestion that was selected
        """
        # Boost frequency of selected query
        self.trie.update_frequency(selected, increment=1)
        
        # Also add the selected query if not present
        if selected.lower() not in self.trie._all_queries:
            self.trie.insert(selected, frequency=1)
        
        logger.debug(f"Recorded selection: '{partial}' -> '{selected}'")
    
    @property
    def query_count(self) -> int:
        """Number of queries in history."""
        return self.trie.size
