# function to implement minhashing as the next step in LSH after onehot
# encoding and before locality sensitive hashing

import numpy as np


def generate_permutations(num_features: int, num_hashes: int) -> np.array:
    """
    Generates a set of random permutations for MinHashing.

    :param num_features: The number of features (columns) in the binary matrix.
    :param num_hashes: The number of hash functions to generate.

    :return: A 2D numpy array where each row is a permutation of feature
      indices.
    """
    return np.array(
        [np.random.permutation(num_features) for _ in range(num_hashes)]
    )


def minhash(
    vectorized_text: np.array,
    permutation_matrix: np.array,
    hash_size: int = 1024,
) -> np.array:
    """
    Function that takes in a binary matrix and returns a matrix of documents'
    signatures using MinHashing.

    :param vectorized_text: A binary matrix (numpy array) representing the
      one-hot encoded documents.
    :param permutation_matrix: A 2D numpy array where each row is a
      permutation of feature indices.
    :param hash_size: The number of hash functions to use for MinHashing.

    :return: A matrix of shape (hash_size, number of documents) containing
      the MinHash signatures.
    """
    pass
