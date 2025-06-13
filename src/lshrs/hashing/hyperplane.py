# Converts vectorized embeddings and TF-IDF to a binary vector using
# hyperplane cosine similarity.

import numpy as np
import scipy.sparse


def generate_hyperplane(
    plane_count: int, plane_dimensions: int
) -> scipy.sparse.csr_matrix:
    """
    Generates a random hyperplane with the specified number of planes.

    Input:
        plane_count (int): The number of hyperplanes to generate.
        plane_dimensions (int): The number of dimensions for each hyperplane.

    Output:
        scipy.sparse.csr_matrix: A sparse matrix representing the hyperplanes.
    """
    # Generate random vectors for each plane
    planes = np.random.randn(plane_count, plane_dimensions)
    return scipy.sparse.csr_matrix(planes)

"""
Input: encoding (A matrix come from embedding or tfidf),
       n_planes (The number of planes)
Output: result (A matrix with binary elements)


Function Hyperplane (encoding, n_planes):
    Initialize planes = empty list

    For i in range(n_planes):
        Append a random vector to planes

    Initialize result = empty list

    For vector in encoding:
        Initialize signature = empty list

        For plane in planes:
            dot_product = vector · plane
            If dot_product >= 0:
                binary = 1
            Else:
                binary = 0
            Append binary to signature

        Append signature to result

    Return result
"""
