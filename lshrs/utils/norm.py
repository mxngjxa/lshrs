import numpy as np


def l2_norm(vector: np.ndarray) -> np.ndarray:
    """
    Return the L2-normalized (unit length) version of a vector.

    L2 normalization divides each element by the vector's Euclidean norm (length),
    producing a unit vector that points in the same direction. This is essential
    for cosine similarity because cos(θ) = (a·b) / (||a|| ||b||), and with unit
    vectors this simplifies to just the dot product a·b.

    Mathematical formula:
        normalized_vector = vector / ||vector||₂
        where ||vector||₂ = sqrt(sum(vector²))

    Args:
        vector: Input vector of any shape (will be flattened to 1D).
               Can be list, tuple, or numpy array.

    Returns:
        L2-normalized vector as 1D float32 numpy array.
        The output will have unit length: ||output||₂ = 1.0

    Raises:
        ValueError: If the input is a zero vector (all elements are 0).
                   Zero vectors have no direction and cannot be normalized.

    Example:
        >>> vec = np.array([3.0, 4.0])
        >>> normalized = l2_normalize(vec)
        >>> normalized
        array([0.6, 0.8], dtype=float32)
        >>> np.linalg.norm(normalized)  # Verify unit length
        1.0

        >>> # Zero vector raises error
        >>> l2_normalize([0, 0, 0])
        ValueError: Cannot normalize zero vector

    Performance notes:
        - Uses float32 for memory efficiency (sufficient for most ML applications)
        - O(n) time complexity where n is vector length
        - Allocates new array (does not modify input)
    """
    # Convert to float32 and flatten to 1D
    # reshape(-1) handles any input shape: scalars, lists, multi-dim arrays
    vec = np.asarray(vector, dtype=np.float32).reshape(-1)

    # Compute L2 norm (Euclidean length of vector)
    # This is sqrt(x₁² + x₂² + ... + xₙ²)
    norm = np.linalg.norm(vec)

    # Check for zero vector (undefined normalization)
    # Catching this explicitly prevents NaN/Inf in results
    if norm == 0:
        raise ValueError("Cannot normalize zero vector")

    # Divide by norm to get unit vector
    # Result has same direction but length = 1
    return vec / norm
