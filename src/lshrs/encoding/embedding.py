# File for using embeddings to encode text data into binary vectors
# embedding models to consider on huggingface include
# google-bert/bert-base-uncased
# jinaai/jina-embeddings-v2-base-en
# jinaai/jina-embeddings-v2-small-en
# ...etc, can also include their own models

"""
Input:
    - one document
    - document dataframe with index
Output: returns
    - tuple with document and the embedding representation
    - document dataframe with index but additional column for embedding representation
Used in: lshrs.encoding.main

Logic: using a pretrianed model and intergrating with the HuggingFace API
could potentially use the user's own embedding model
"""
