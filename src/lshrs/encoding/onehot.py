# implemente shingling and one-hot encoding.

from sklearn.preprocessing import OneHotEncoder as ohe

from lshrs.utils.helpers import shingling


class OneHotEncoder(ohe):
    """
    OneHotEncoder class that extends sklearn's OneHotEncoder
    to handle shingling and one-hot encoding of documents.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, documents):
        # Shingle the documents first
        shingles = shingling(documents, self.n_features_in_)
        # Fit the OneHotEncoder on the shingles
        return super().fit(shingles)

    def transform(self, documents):
        # Shingle the documents first
        shingles = shingling(documents, self.n_features_in_)
        # Transform the shingles using OneHotEncoder
        return super().transform(shingles)

    def fit_transform(self, documents):
        # Shingle the documents first
        shingles = shingling(documents, self.n_features_in_)
        # Fit and transform the shingles using OneHotEncoder
        return super().fit_transform(shingles)

"""
Input: documents (2D dataframe: documents x words), size (shingling size)
Output: result (2D dataframe: documents x shingles)

Function Shingling (documents, size):
    Initialize result = empty list

    For each doc in documents:
        Initialize shingle_set as an empty set

        For i from 0 to len(doc) - size:
            Add substring from i to i + size into shingle_set

        Append shingle_set to result

    return result


Input: shingles (A list comming from Shingling function)
Output: result (A matrix containing binary numbers)

Function OneHotEncoding (shingles):
    Initialize entire_set = empty set
    For each set in shingles:
        Add set's elements into entire_set

    Initialize result = empty list
    For each set in shingles:
        Initialize vector = empty list
        For each i in entire_set:
            If i in set:
                Append 1 to vector
            Else:
                Append 0 to vector
        Append vector to result

    Return result
"""
