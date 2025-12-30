import numpy as np
import polars as pl
import duckdb
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Optional, Dict, Any
from loguru import logger

from .utils import batch_cosine_sim, normalize_scores, cosine_sim
from .extractor import extract_tokens

# Optional FAISS import
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class QueryMemory:
    """Tracks query history and learns optimal weights."""
    
    def __init__(self, db_path: str = "index.duckdb"):
        self.db_path = db_path
    
    def log_query(
        self, 
        query: str, 
        semantic_weight: float, 
        lexical_weight: float
    ) -> int:
        """Log a query and return its ID."""
        with duckdb.connect(self.db_path) as con:
            result = con.execute("""
                INSERT INTO query_history (query_id, query_text, semantic_weight, lexical_weight)
                VALUES (nextval('query_id_seq'), ?, ?, ?)
                RETURNING query_id
            """, [query, semantic_weight, lexical_weight]).fetchone()
            return result[0]
    
    def log_feedback(
        self, 
        query_id: int, 
        doc_id: int, 
        relevance_score: int = 0,
        clicked: bool = False
    ):
        """Log user feedback for a search result."""
        with duckdb.connect(self.db_path) as con:
            con.execute("""
                INSERT INTO feedback (feedback_id, query_id, doc_id, relevance_score, clicked)
                VALUES (nextval('feedback_id_seq'), ?, ?, ?, ?)
            """, [query_id, doc_id, relevance_score, clicked])
    
    def get_optimal_weights(self) -> Tuple[float, float]:
        """
        Calculate optimal weights based on feedback history.
        Returns default weights if insufficient data.
        """
        try:
            with duckdb.connect(self.db_path) as con:
                # Get average weights from queries with positive feedback
                result = con.execute("""
                    SELECT 
                        AVG(qh.semantic_weight) as avg_semantic,
                        AVG(qh.lexical_weight) as avg_lexical,
                        COUNT(*) as count
                    FROM query_history qh
                    JOIN feedback f ON qh.query_id = f.query_id
                    WHERE f.relevance_score > 0 OR f.clicked = TRUE
                """).fetchone()
                
                if result and result[2] >= 10:  # Need at least 10 feedback entries
                    return result[0], result[1]
        except Exception as e:
            logger.debug(f"Could not get optimal weights: {e}")
        
        return 0.7, 0.3  # Default weights
    
    def get_query_stats(self) -> Dict[str, Any]:
        """Get statistics about query history."""
        try:
            with duckdb.connect(self.db_path) as con:
                result = con.execute("""
                    SELECT 
                        COUNT(*) as total_queries,
                        COUNT(DISTINCT query_text) as unique_queries,
                        AVG(semantic_weight) as avg_semantic_weight
                    FROM query_history
                """).fetchone()
                
                feedback_result = con.execute("""
                    SELECT 
                        COUNT(*) as total_feedback,
                        SUM(CASE WHEN clicked THEN 1 ELSE 0 END) as total_clicks,
                        AVG(relevance_score) as avg_relevance
                    FROM feedback
                """).fetchone()
                
                return {
                    'total_queries': result[0],
                    'unique_queries': result[1],
                    'avg_semantic_weight': result[2],
                    'total_feedback': feedback_result[0],
                    'total_clicks': feedback_result[1],
                    'avg_relevance': feedback_result[2]
                }
        except Exception:
            return {}


class Searcher:
    """
    Handles the search logic, combining semantic and lexical search.
    """
    def __init__(
        self, 
        model_name: str = "all-MiniLM-L6-v2",
        db_path: str = "index.duckdb",
        use_faiss: bool = False,
        faiss_index_path: str = "index.faiss",
        enable_query_memory: bool = True
    ):
        """
        Initializes the Searcher.

        Args:
            model_name: The sentence-transformer model for query embedding.
            db_path: Path to DuckDB database.
            use_faiss: Whether to use FAISS for ANN search.
            faiss_index_path: Path to FAISS index file.
            enable_query_memory: Whether to track queries and learn weights.
        """
        self.model = SentenceTransformer(model_name)
        self.db_path = db_path
        self.use_faiss = use_faiss and FAISS_AVAILABLE
        self.faiss_index = None
        self.faiss_index_path = faiss_index_path
        self.enable_query_memory = enable_query_memory
        self.query_memory = QueryMemory(db_path) if enable_query_memory else None
        
        # Load FAISS index if available
        if self.use_faiss:
            self._load_faiss_index()
        
        logger.info(f"Initialized Searcher with model '{model_name}'.")

    def _load_faiss_index(self):
        """Load FAISS index from disk if it exists."""
        try:
            from pathlib import Path
            if Path(self.faiss_index_path).exists():
                self.faiss_index = faiss.read_index(self.faiss_index_path)
                logger.info(f"Loaded FAISS index from {self.faiss_index_path}")
        except Exception as e:
            logger.warning(f"Could not load FAISS index: {e}")
            self.faiss_index = None

    def _semantic_search_faiss(
        self, 
        q_vec: np.ndarray, 
        top_k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Perform semantic search using FAISS."""
        q_vec = q_vec.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(q_vec)
        distances, indices = self.faiss_index.search(q_vec, top_k)
        return distances[0], indices[0]

    def _semantic_search_brute(
        self, 
        q_vec: np.ndarray, 
        vectors: np.ndarray
    ) -> np.ndarray:
        """Perform brute-force semantic search."""
        return batch_cosine_sim(q_vec, vectors)

    def _lexical_scores(self, query: str, docs: List[str]) -> np.ndarray:
        """Calculate lexical similarity scores using fuzzy matching and token overlap."""
        query_tokens = set(extract_tokens(query.lower()))
        scores = []
        
        for doc in docs:
            # Fuzzy match score
            fuzzy_score = fuzz.partial_ratio(query.lower(), doc.lower()) / 100.0
            
            # Token overlap bonus
            doc_tokens = set(extract_tokens(doc.lower()))
            if query_tokens and doc_tokens:
                overlap = len(query_tokens & doc_tokens) / len(query_tokens)
                combined = (fuzzy_score * 0.7) + (overlap * 0.3)
            else:
                combined = fuzzy_score
            
            scores.append(combined)
        
        return np.array(scores, dtype=np.float32)

    def search(
        self,
        query: str,
        docs_df: pl.DataFrame,
        vectors: np.ndarray,
        top_k: int = 5,
        semantic_weight: Optional[float] = None,
        lexical_weight: Optional[float] = None,
        use_learned_weights: bool = False
    ) -> List[Tuple[float, str, int]]:
        """
        Performs a hybrid search combining semantic and lexical scores.

        Args:
            query: The search query string.
            docs_df: Polars DataFrame with 'doc_id' and 'content' columns.
            vectors: NumPy array of document embeddings.
            top_k: Number of top results to return.
            semantic_weight: Weight for semantic score (0-1).
            lexical_weight: Weight for lexical score (0-1).
            use_learned_weights: Use weights learned from feedback.

        Returns:
            List of tuples: (score, content, doc_id).
        """
        # Determine weights
        if use_learned_weights and self.query_memory:
            semantic_weight, lexical_weight = self.query_memory.get_optimal_weights()
            logger.info(f"Using learned weights: semantic={semantic_weight:.2f}, lexical={lexical_weight:.2f}")
        else:
            semantic_weight = semantic_weight if semantic_weight is not None else 0.7
            lexical_weight = lexical_weight if lexical_weight is not None else 0.3

        if not np.isclose(semantic_weight + lexical_weight, 1.0):
            raise ValueError("semantic_weight and lexical_weight must sum to 1.0")

        logger.info(f"Searching for: '{query}' (top_k={top_k})")

        # Embed query
        q_vec = self.model.encode([query], convert_to_numpy=True)[0].astype(np.float32)

        docs = docs_df['content'].to_list()
        doc_ids = docs_df['doc_id'].to_list()
        
        # Semantic scores
        if self.use_faiss and self.faiss_index is not None:
            # FAISS returns top-k directly
            faiss_scores, faiss_indices = self._semantic_search_faiss(q_vec, min(top_k * 2, len(docs)))
            semantic_scores = np.zeros(len(docs), dtype=np.float32)
            for score, idx in zip(faiss_scores, faiss_indices):
                if idx >= 0:
                    semantic_scores[idx] = score
        else:
            semantic_scores = self._semantic_search_brute(q_vec, vectors)
        
        # Also compute individual cosine_sim for detailed logging if needed
        if logger._core.min_level <= 10:  # DEBUG level
            for i in range(min(3, len(docs))):
                individual_score = cosine_sim(q_vec, vectors[i])
                logger.debug(f"Doc {i} cosine_sim: {individual_score:.4f}")

        # Lexical scores
        lexical_scores = self._lexical_scores(query, docs)

        # Normalize scores
        semantic_scores = normalize_scores(semantic_scores)
        lexical_scores = normalize_scores(lexical_scores)

        # Hybrid scores
        hybrid_scores = (semantic_scores * semantic_weight) + (lexical_scores * lexical_weight)

        # Sort and get top-k
        sorted_indices = np.argsort(hybrid_scores)[::-1][:top_k]
        
        results = [
            (float(hybrid_scores[i]), docs[i], doc_ids[i])
            for i in sorted_indices
        ]

        # Log query if memory enabled
        query_id = None
        if self.query_memory:
            query_id = self.query_memory.log_query(query, semantic_weight, lexical_weight)
            logger.debug(f"Logged query with ID {query_id}")

        logger.info(f"Returning {len(results)} results.")
        return results

    def record_feedback(
        self, 
        query_id: int, 
        doc_id: int, 
        relevance_score: int = 0,
        clicked: bool = False
    ):
        """
        Record user feedback for a search result.

        Args:
            query_id: The query ID from search results.
            doc_id: The document ID that received feedback.
            relevance_score: Relevance rating (e.g., 1-5).
            clicked: Whether the result was clicked.
        """
        if self.query_memory:
            self.query_memory.log_feedback(query_id, doc_id, relevance_score, clicked)
            logger.info(f"Recorded feedback for query {query_id}, doc {doc_id}")

    def get_stats(self) -> Dict[str, Any]:
        """Get search statistics."""
        if self.query_memory:
            return self.query_memory.get_query_stats()
        return {}
