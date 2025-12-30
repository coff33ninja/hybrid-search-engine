"""
Cross-encoder reranking for improved result quality.
"""
import numpy as np
from typing import List, Tuple, Optional
from loguru import logger


class CrossEncoderReranker:
    """
    Reranks search results using a cross-encoder model.
    
    Cross-encoders are more accurate than bi-encoders but slower,
    so we use them to rerank top-k results from initial retrieval.
    """
    
    def __init__(
        self, 
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None
    ):
        """
        Initialize the reranker.
        
        Args:
            model_name: Cross-encoder model from HuggingFace.
            device: Device to run on ('cpu', 'cuda', or None for auto).
        
        Recommended models:
            - cross-encoder/ms-marco-MiniLM-L-6-v2 (fast, good quality)
            - cross-encoder/ms-marco-MiniLM-L-12-v2 (better quality)
            - cross-encoder/ms-marco-TinyBERT-L-2-v2 (fastest)
            - BAAI/bge-reranker-base (state-of-art)
            - BAAI/bge-reranker-large (best quality)
        """
        self.model_name = model_name
        self.device = device
        self._model = None
        logger.info(f"CrossEncoderReranker initialized with model: {model_name}")
    
    @property
    def model(self):
        """Lazy load the model."""
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name, device=self.device)
            logger.info(f"Loaded cross-encoder model: {self.model_name}")
        return self._model
    
    def rerank(
        self, 
        query: str, 
        results: List[Tuple[float, str, int]],
        top_k: Optional[int] = None
    ) -> List[Tuple[float, str, int]]:
        """
        Rerank search results using cross-encoder.
        
        Args:
            query: Search query.
            results: List of (score, content, doc_id) from initial retrieval.
            top_k: Number of results to return (None = all).
        
        Returns:
            Reranked list of (score, content, doc_id).
        """
        if not results:
            return results
        
        # Prepare query-document pairs
        pairs = [(query, content) for _, content, _ in results]
        
        # Get cross-encoder scores
        scores = self.model.predict(pairs, show_progress_bar=False)
        
        # Combine with original results
        reranked = [
            (float(score), content, doc_id)
            for score, (_, content, doc_id) in zip(scores, results)
        ]
        
        # Sort by new scores
        reranked.sort(key=lambda x: x[0], reverse=True)
        
        if top_k:
            reranked = reranked[:top_k]
        
        logger.debug(f"Reranked {len(results)} results")
        return reranked
    
    def rerank_with_fusion(
        self,
        query: str,
        results: List[Tuple[float, str, int]],
        original_weight: float = 0.3,
        rerank_weight: float = 0.7,
        top_k: Optional[int] = None
    ) -> List[Tuple[float, str, int]]:
        """
        Rerank with score fusion between original and cross-encoder scores.
        
        Args:
            query: Search query.
            results: List of (score, content, doc_id).
            original_weight: Weight for original retrieval score.
            rerank_weight: Weight for cross-encoder score.
            top_k: Number of results to return.
        
        Returns:
            Reranked results with fused scores.
        """
        if not results:
            return results
        
        pairs = [(query, content) for _, content, _ in results]
        ce_scores = self.model.predict(pairs, show_progress_bar=False)
        
        # Normalize scores to [0, 1]
        original_scores = np.array([s for s, _, _ in results])
        if original_scores.max() > original_scores.min():
            original_scores = (original_scores - original_scores.min()) / (original_scores.max() - original_scores.min())
        
        ce_scores = np.array(ce_scores)
        if ce_scores.max() > ce_scores.min():
            ce_scores = (ce_scores - ce_scores.min()) / (ce_scores.max() - ce_scores.min())
        
        # Fuse scores
        fused_scores = original_weight * original_scores + rerank_weight * ce_scores
        
        reranked = [
            (float(score), content, doc_id)
            for score, (_, content, doc_id) in zip(fused_scores, results)
        ]
        
        reranked.sort(key=lambda x: x[0], reverse=True)
        
        if top_k:
            reranked = reranked[:top_k]
        
        return reranked


class ColBERTReranker:
    """
    ColBERT-style late interaction reranking.
    More efficient than full cross-encoder for longer documents.
    """
    
    def __init__(self, model_name: str = "colbert-ir/colbertv2.0"):
        """
        Note: Requires the 'colbert' package.
        pip install colbert-ai
        """
        self.model_name = model_name
        self._model = None
        logger.warning("ColBERT reranker requires 'colbert-ai' package")
    
    def rerank(
        self, 
        query: str, 
        results: List[Tuple[float, str, int]],
        top_k: Optional[int] = None
    ) -> List[Tuple[float, str, int]]:
        """Placeholder for ColBERT reranking."""
        # ColBERT implementation would go here
        # For now, return original results
        logger.warning("ColBERT reranking not implemented, returning original results")
        return results[:top_k] if top_k else results


def create_reranker(
    reranker_type: str = "cross-encoder",
    model_name: Optional[str] = None,
    **kwargs
):
    """
    Factory function to create a reranker.
    
    Args:
        reranker_type: 'cross-encoder' or 'colbert'.
        model_name: Model name override.
        **kwargs: Additional arguments.
    
    Returns:
        Reranker instance.
    """
    if reranker_type == "cross-encoder":
        model = model_name or "cross-encoder/ms-marco-MiniLM-L-6-v2"
        return CrossEncoderReranker(model_name=model, **kwargs)
    elif reranker_type == "colbert":
        model = model_name or "colbert-ir/colbertv2.0"
        return ColBERTReranker(model_name=model)
    else:
        raise ValueError(f"Unknown reranker type: {reranker_type}")
