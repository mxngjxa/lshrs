import numpy as np


class lshrs:
    """
    lsh class that has a few attirbutes, including
        redis connector,
        embedding dimension,
        number of bands and buckets,
    """

    def __init__(self):
        """
        need to define logic to 
            initialize the connection to the redis database
            establish connection to the database that stores the text data and vectors(postgres for now)

        """

    def __getstate__(self):
        pass

    def create_signatures(self, format: str = "postgres"):
        """
        load in the vectorized text and indices and hash them to the redis database
            alternatively, load it in from parquet
            call the respective functions from lshrs/io
        """
        pass

    def get_top_k(self, vector: np.ndarray, topk: int = 10):
        """
        Get the top k signatures based on the vector input
        """
        pass

    def get_above_p(self, vector: np.array, p: float = float(0.95)):
        """
        get the top p signatures based on the vector input
        """
        pass

    def save_to_disk(self):
        """
            add another option for saving an object for lshrs from storage as a persisted object. 
        """
        pass

    @property
    def load_from_disk(self):
        """
            add another option for saving an object for lshrs from storage as a persisted object. 
            technically should just load in the random states associated with regenerating the object, so we can recreate this from scratch. 
        """
        pass