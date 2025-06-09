# File for using embeddings to encode text data into binary vectors
# embedding models to consider on huggingface include 
# google-bert/bert-base-uncased
# jinaai/jina-embeddings-v2-base-en
# jinaai/jina-embeddings-v2-small-en
# ...etc, can also include their own models

"""
Input: documents (The format depends on models)
Output: result (A matrix with encoding numbers. Different rows are different documents)

Function Embedding (documents):
    result = Use models to transform documents to vectors
    
    Return result
"""