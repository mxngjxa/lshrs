# file/function to implement TF-IDF encoding

"""
Use Scikit-learn's library for Tf-Idf vectorizer
"""

from sklearn.feature_extraction.text import TfidfVectorizer as tiv


class TfidfVectorizer(tiv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Add custom initialization if needed

    def fit(self, documents):

        result = super().fit(documents)

        return result

    def fit_transform(self, documents):
        # Call parent method
        result = super().fit_transform(documents)
        # Add custom processing here
        return result

    def transform(self, documents):
        # Custom transform logic
        result = super().transform(documents)
        # Apply additional transformations
        return result


r"""
Input: doc (A list of words of a document),
Output: result (A dictionary. The TF of each word in a document):

Function TF (doc):
    result = empty dictionary

    For word in doc:
        result[word] += 1

    For word in result:
        result[word] \= length of document

    Return result


Input: documents (2D list: documents x words)
Output: result (A dictionary. The IDF of each word)

Function IDF (documents):
    result = empty dictionary

    For doc in documents:
        word_set = convert doc into a set (remove duplicates)
        For word in word_set:
            result[word] += 1

    For word in result:
        result[word] = log(N / result[word])

    Return result


Input: None
Output: result (TF-IDF of each word)

Function TF-IDF ():
    idf = IDF(documents)
    result = empty list

    For doc in documents:
        tf = TF(doc)
        tfidf = empty dictionary

        For word in tf:
            tfidf[word] = tf[word] * idf[word]
        Append tfidf to result

    Return result
"""
