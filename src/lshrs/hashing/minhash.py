# function to implement minhashing as the next step in LSH after onehot encoding and before locality sensitive hashing

import numpy as np
import pandas as pd


def generate_permutations(num_features: int, num_hashes: int) -> np.array:
    """
    Generates a set of random permutations for MinHashing.

    :param num_features: The number of features (columns) in the binary matrix.
    :param num_hashes: The number of hash functions to generate.

    :return: A 2D numpy array where each row is a permutation of feature indices.
    """
    return np.array([np.random.permutation(num_features) for _ in range(num_hashes)])

def minhash(vectorized_text: np.array, permutation_matrix: np.array, hash_size: int = 1024) -> np.array:
    """
    Function that takes in a binary matrix and returns a matrix of documents' signatures using MinHashing.

    :param vectorized_text: A binary matrix (numpy array) representing the one-hot encoded documents.
    :param permutation_matrix: A 2D numpy array where each row is a permutation of feature indices.
    :param hash_size: The number of hash functions to use for MinHashing.

    :return: A matrix of shape (hash_size, number of documents) containing the MinHash signatures.
    """

    pass


"""
Input: one_hot (A binary matrix coming from OneHotEncoding function)
       hash_size (The number of hash functions)
Output: result (A matrix of documents' signitures)

Function MinHash (one_hot, hash_size):
    Initialize hash_functions = A list of permutations (from 0 to one_hot.shape[1] - 1) with number of hash_size
    Initialize result = Two dimensions list of shape (number of documents * hash_size)

    For row in 0 to hash_size - 1:
        For col in 0 to number of documents - 1:
            For index in hash_functions[row]:
                If one_hot[col][index] == 1:
                    Set result[row][col] = index
                    Break

    Return result
"""

# Old Code:


def generate_permutation_matrix():
    """
    Creates a permutation matrix for MinHashing.

    The matrix has `permutations` rows, and each row is a random permutation
    of shingle indices (from 0 to `shingle_count` - 1).

    Returns:
        pandas.DataFrame: The permutation matrix, where each row is a permutation.
    """
    pm = list()
    # For each desired permutation, generate a shuffled array of shingle indices
    for i in range(permutations):
        pm.append(perm_array(shingle_count))

    # Convert the list of permutations into a Pandas DataFrame
    pm = pd.DataFrame(pm)

    #print(pm)
    print("Permutation Matrix Generated")
    return pm

#use minhashing to permute data into a signature matrix
def index(permutations: int):
    """
    Creates a Signature Matrix using MinHashing.

    The Signature Matrix approximates the Jaccard similarity between documents.
    Each column represents a document, and each row corresponds to a hash function
    (permutation). The value at `(i, j)` is the minimum hash value of shingle indices
    present in document `j` according to permutation `i`.

    Args:
        permutations (int): The number of permutations (hash functions) to use for MinHashing.
                            This determines the number of rows in the signature matrix.

    Sets:
        permutations (int): The number of permutations used.
        doc_count (int): The number of documents.
        shingle_count (int): The number of unique shingles.
        signature_matrix (pandas.DataFrame): The resulting signature matrix.
        one_hot (scipy.sparse.csr_matrix): The one-hot encoded representation of shingles.
        perm_matrix (pandas.DataFrame): The permutation matrix used for MinHashing.
    """
    print("MinHashing initiated.")
    permutations = permutations

    #set some variables for easy iteration
    doc_count = len(post_shingle)
    len(shingle_array)

    # Initialize an empty signature matrix with permutations as rows and documents as columns
    signature_matrix = pd.DataFrame(index=range(permutations), columns=range(doc_count))

    # Perform one-hot encoding of shingles for all documents
    one_hot = one_hot_encode()
    # Generate the matrix of random permutations of shingle indices
    perm_matrix = generate_permutation_matrix()

    # Convert the sparse one-hot matrix to a dense NumPy array for efficient row-wise operations
    # This can be memory-intensive for very large datasets.
    one_hot_np = one_hot.toarray()

    # Iterate over each document to compute its MinHash signature
    for doc_idx in range(doc_count):
        # Find the indices of shingles present in the current document (where one_hot_np is 1)
        shingle_indices_in_doc = np.where(one_hot_np[doc_idx] == 1)[0]

        # Iterate over each permutation (hash function)
        for perm_idx, perm_shingle_order in perm_matrix.iterrows():
            # For the current permutation, find the minimum permuted index of a shingle present in the document
            # This is the core MinHash computation for one (permutation, document) pair.
            min_hash_val = perm_shingle_order[shingle_indices_in_doc].min()
            signature_matrix.at[perm_idx, doc_idx] = min_hash_val

    # Ensure the signature matrix contains integer values
    signature_matrix = signature_matrix.astype(int)
    # Print the signature matrix
    # print(signature_matrix)
    gc.collect()
    print("Minhashing processing complete, proceed to LSH.")
