"""
BM25 scoring for proper lexical ranking.
"""
import math
import numpy as np
from typing import List, Dict
from collections import Counter

from .extractor import extract_tokens


class BM25:
    """
    BM25 (Best Matching 25) ranking function for lexical search.
    
    Better than fuzzy matching for document retrieval.
    """
    
    def __init__(
        self, 
        k1: float = 1.5, 
        b: float = 0.75,
        remove_stopwords: bool = True
    ):
        """
        Initialize BM25.
        
        Args:
            k1: Term frequency saturation parameter (1.2-2.0 typical).
            b: Length normalization parameter (0.75 typical).
            remove_stopwords: Whether to remove stopwords.
        """
        self.k1 = k1
        self.b = b
        self.remove_stopwords = remove_stopwords
        
        # Corpus statistics (set during fit)
        self.doc_count = 0
        self.avg_doc_len = 0.0
        self.doc_lengths: List[int] = []
        self.doc_freqs: Dict[str, int] = {}  # term -> num docs containing term
        self.term_freqs: List[Dict[str, int]] = []  # per-doc term frequencies
        self.idf: Dict[str, float] = {}
    
    def fit(self, documents: List[str]):
        """
        Fit BM25 on a corpus of documents.
        
        Args:
            documents: List of document strings.
        """
        self.doc_count = len(documents)
        self.doc_lengths = []
        self.doc_freqs = {}
        self.term_freqs = []
        
        # Tokenize and compute statistics
        for doc in documents:
            tokens = extract_tokens(doc, remove_stopwords=self.remove_stopwords)
            self.doc_lengths.append(len(tokens))
            
            # Term frequencies for this doc
            tf = Counter(tokens)
            self.term_freqs.append(dict(tf))
            
            # Document frequencies (count each term once per doc)
            for term in set(tokens):
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1
        
        # Average document length
        self.avg_doc_len = sum(self.doc_lengths) / self.doc_count if self.doc_count > 0 else 0
        
        # Precompute IDF for all terms
        self._compute_idf()
    
    def _compute_idf(self):
        """Compute IDF scores for all terms."""
        self.idf = {}
        for term, df in self.doc_freqs.items():
            # IDF with smoothing to avoid division by zero
            self.idf[term] = math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1)
    
    def score(self, query: str, doc_idx: int) -> float:
        """
        Compute BM25 score for a query against a specific document.
        
        Args:
            query: Query string.
            doc_idx: Index of the document.
        
        Returns:
            BM25 score.
        """
        query_tokens = extract_tokens(query, remove_stopwords=self.remove_stopwords)
        doc_len = self.doc_lengths[doc_idx]
        tf_dict = self.term_freqs[doc_idx]
        
        score = 0.0
        for term in query_tokens:
            if term not in self.idf:
                continue
            
            tf = tf_dict.get(term, 0)
            idf = self.idf[term]
            
            # BM25 formula
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_len))
            
            score += idf * (numerator / denominator) if denominator > 0 else 0
        
        return score
    
    def score_batch(self, query: str) -> np.ndarray:
        """
        Compute BM25 scores for a query against all documents.
        
        Args:
            query: Query string.
        
        Returns:
            Array of BM25 scores.
        """
        scores = np.array([
            self.score(query, i) for i in range(self.doc_count)
        ], dtype=np.float32)
        return scores
    
    def search(self, query: str, top_k: int = 10) -> List[tuple]:
        """
        Search and return top-k results.
        
        Args:
            query: Query string.
            top_k: Number of results.
        
        Returns:
            List of (doc_idx, score) tuples.
        """
        scores = self.score_batch(query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices]


class BM25Okapi(BM25):
    """BM25 Okapi variant (same as base BM25)."""
    pass


class BM25Plus(BM25):
    """
    BM25+ variant that adds a small constant delta to prevent
    zero scores for very long documents.
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75, delta: float = 1.0, **kwargs):
        super().__init__(k1=k1, b=b, **kwargs)
        self.delta = delta
    
    def score(self, query: str, doc_idx: int) -> float:
        query_tokens = extract_tokens(query, remove_stopwords=self.remove_stopwords)
        doc_len = self.doc_lengths[doc_idx]
        tf_dict = self.term_freqs[doc_idx]
        
        score = 0.0
        for term in query_tokens:
            if term not in self.idf:
                continue
            
            tf = tf_dict.get(term, 0)
            idf = self.idf[term]
            
            # BM25+ formula with delta
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_len))
            
            score += idf * ((numerator / denominator) + self.delta) if denominator > 0 else 0
        
        return score
