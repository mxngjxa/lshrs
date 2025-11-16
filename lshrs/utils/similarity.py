"""
Vector Similarity Utilities for Nearest Neighbor Search

This module provides efficient implementations of cosine similarity computation,
commonly used in information retrieval, recommendation systems, and semantic search.

Cosine similarity measures the angle between vectors, ranging from -1 (opposite)
to +1 (identical direction), with 0 indicating orthogonality. It's particularly
useful because it's independent of vector magnitude, focusing purely on direction.

Key functions:
    - l2_norm: Normalize vectors to unit length
    - cosine_similarity: Compute similarity between query and multiple candidates
    - top_k_cosine: Find k most similar vectors efficiently
"""

from __future__ import annotations

from typing import Iterable, Sequence
import numpy as np

from lshrs.utils.norm import l2_norm


def cosine_similarity(
    query: np.ndarray, candidates: Sequence[np.ndarray]
) -> np.ndarray:
    """
    Compute cosine similarity between a query vector and multiple candidate vectors.

    Cosine similarity measures the cosine of the angle between two vectors:
        cos(θ) = (a · b) / (||a|| × ||b||)

    By normalizing all vectors to unit length first, this simplifies to just
    the dot product. The result ranges from:
        +1.0 = identical direction (most similar)
         0.0 = orthogonal (no similarity)
        -1.0 = opposite direction (most dissimilar)

    This implementation is optimized for computing similarity between ONE query
    and MANY candidates simultaneously using vectorized operations.

    Args:
        query: Query vector to compare against candidates.
              Can be any shape (will be normalized and flattened).
        candidates: Sequence of candidate vectors (list, tuple, etc.).
                   All candidates should have the same dimension as query.

    Returns:
        1D numpy array of cosine similarities, one per candidate.
        Array length equals len(candidates).
        Values are in range [-1.0, 1.0].

    Raises:
        ValueError: If query or any candidate is a zero vector (via l2_norm).

    Example:
        >>> query = np.array([1.0, 0.0, 0.0])
        >>> candidates = [
        ...     [1.0, 0.0, 0.0],  # Same direction: similarity = 1.0
        ...     [0.0, 1.0, 0.0],  # Perpendicular: similarity = 0.0
        ...     [-1.0, 0.0, 0.0], # Opposite: similarity = -1.0
        ...     [1.0, 1.0, 0.0],  # 45 degrees: similarity ≈ 0.707
        ... ]
        >>> similarities = cosine_similarity(query, candidates)
        >>> similarities
        array([ 1.0,  0.0, -1.0,  0.707], dtype=float32)

    Performance notes:
        - O(n × d) where n = number of candidates, d = vector dimension
        - Uses matrix multiplication for efficient batch processing
        - More efficient than computing similarities in a loop

    Implementation details:
        Uses normalized_candidates @ normalized_query which is equivalent to:
        [dot(candidate, query) for candidate in candidates]
        But vectorized for speed.
    """
    # Normalize query to unit length
    # This makes cosine similarity = dot product
    normalized_query = l2_norm(query)

    # Normalize all candidates and stack into matrix
    # Shape: (num_candidates, dimension)
    # Each row is a normalized candidate vector
    normalized_candidates = np.stack([l2_norm(vec) for vec in candidates])

    # Compute all dot products at once using matrix-vector multiplication
    # Result[i] = dot(normalized_candidates[i], normalized_query)
    # This is the cosine similarity since all vectors are unit length
    return normalized_candidates @ normalized_query


def top_k_cosine(
    query: np.ndarray,
    candidates: Sequence[np.ndarray],
    *,
    k: int,
) -> Iterable[tuple[int, float]]:
    """
    Find the top-k most similar vectors to query using cosine similarity.

    This function efficiently identifies the k candidates with highest cosine
    similarity to the query vector. It uses partial sorting (argpartition) for
    better performance when k << len(candidates).

    Results are returned in descending order of similarity (most similar first).

    Algorithm:
        1. Compute cosine similarity for all candidates
        2. Use argpartition to find top-k indices (O(n) average case)
        3. Sort only the top-k results (O(k log k))
        4. Return (index, similarity) tuples

    Args:
        query: Query vector to compare against candidates.
        candidates: Sequence of candidate vectors to rank.
        k: Number of top results to return (keyword-only).
           Must be positive. If k > len(candidates), returns all candidates.

    Returns:
        List of (index, similarity) tuples, sorted by similarity descending.
        - index: Position of candidate in original candidates sequence (int)
        - similarity: Cosine similarity score as float in [-1.0, 1.0]

        Returns empty list if candidates is empty.

    Raises:
        ValueError: If k <= 0.

    Example:
        >>> query = np.array([1.0, 0.0])
        >>> candidates = [
        ...     [1.0, 0.1],   # index 0: high similarity
        ...     [0.0, 1.0],   # index 1: low similarity
        ...     [1.0, 0.0],   # index 2: perfect match
        ...     [-1.0, 0.0],  # index 3: opposite
        ...     [0.9, 0.2],   # index 4: high similarity
        ... ]
        >>> results = top_k_cosine(query, candidates, k=3)
        >>> results
        [(2, 1.0), (0, 0.995), (4, 0.976)]
        # Returns indices and scores of 3 most similar vectors

    Performance notes:
        - Time complexity: O(n + k log k) where n = len(candidates)
        - Space complexity: O(n) for similarity array
        - Much faster than full sort when k << n
        - For k close to n, consider np.argsort instead

    Use cases:
        - Semantic search: find most relevant documents
        - Recommendation systems: find similar items
        - Nearest neighbor search in embedding spaces
        - LSH candidate reranking (after hash-based filtering)
    """
    # Validate k parameter
    if k <= 0:
        raise ValueError("k must be > 0")

    # Compute similarity scores for all candidates
    # Shape: (num_candidates,)
    similarities = cosine_similarity(query, candidates)

    # Handle empty candidate set
    if len(similarities) == 0:
        return []

    # Find indices of top-k elements using partial sort
    # argpartition(-similarities, kth=k) partitions array so that:
    #   - Elements [:k] are the k largest (but not necessarily sorted)
    #   - Elements [k:] are all smaller than elements [:k]
    # We use -similarities to get largest values (partition finds smallest by default)
    # min(k, len(similarities) - 1) handles case where k >= len(candidates)
    top_indices = np.argpartition(-similarities, kth=min(k, len(similarities) - 1))[:k]

    # Sort only the top-k indices by their similarity scores (descending)
    # argsort returns indices that would sort the array
    # We use -similarities[top_indices] to sort in descending order
    sorted_indices = top_indices[np.argsort(-similarities[top_indices])]

    # Build result list of (index, score) tuples
    # Convert numpy types to native Python types for cleaner output
    return [(int(idx), float(similarities[idx])) for idx in sorted_indices]
