# Convertes the vectorized output of the embeddings and TF-IDF to a binary vector using hyperplane cosine similarity

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