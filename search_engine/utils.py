import numpy as np
from numba import njit
from typing import List, Tuple

@njit(fastmath=True, cache=True)
def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calculates the cosine similarity between two vectors.

    Args:
        a: The first vector.
        b: The second vector.

    Returns:
        The cosine similarity score.
    """
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    # Prevent division by zero
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot_product / (norm_a * norm_b)


@njit(fastmath=True, cache=True)
def batch_cosine_sim(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """
    Calculates cosine similarity between a query and multiple vectors.

    Args:
        query: The query vector (1D).
        vectors: Matrix of document vectors (2D).

    Returns:
        Array of similarity scores.
    """
    n = vectors.shape[0]
    scores = np.empty(n, dtype=np.float32)
    query_norm = np.linalg.norm(query)
    
    if query_norm == 0.0:
        return np.zeros(n, dtype=np.float32)
    
    for i in range(n):
        vec_norm = np.linalg.norm(vectors[i])
        if vec_norm == 0.0:
            scores[i] = 0.0
        else:
            scores[i] = np.dot(query, vectors[i]) / (query_norm * vec_norm)
    
    return scores


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """
    Normalizes scores to [0, 1] range using min-max normalization.

    Args:
        scores: Array of scores.

    Returns:
        Normalized scores.
    """
    min_score = scores.min()
    max_score = scores.max()
    if max_score - min_score == 0:
        return np.ones_like(scores)
    return (scores - min_score) / (max_score - min_score)


def top_k_indices(scores: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get top-k indices and their scores.

    Args:
        scores: Array of scores.
        k: Number of top results.

    Returns:
        Tuple of (top_k_scores, top_k_indices).
    """
    k = min(k, len(scores))
    indices = np.argsort(scores)[::-1][:k]
    return scores[indices], indices


def pairwise_cosine_sim(vectors: np.ndarray) -> List[Tuple[int, int, float]]:
    """
    Compute pairwise cosine similarity for all vectors.

    Args:
        vectors: Matrix of vectors (N x D).

    Returns:
        List of (i, j, similarity) tuples for all pairs.
    """
    n = vectors.shape[0]
    results = []
    for i in range(n):
        for j in range(i + 1, n):
            sim = cosine_sim(vectors[i], vectors[j])
            results.append((i, j, float(sim)))
    return results
