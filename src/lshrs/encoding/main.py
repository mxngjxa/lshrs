# convertes the vectorized output of the embeddings to a binary vector using hyperplane cosine similarity
# uses the encoding methods from other files in this directory to pull in functions to compute embeddings

"""
Input:
    - document in one of the following format
        - json
        - list
        - single string of documents
Output:
    - returns the embecdded version of their doucment encoded to a vector format

Used in: lshrs/main
"""
