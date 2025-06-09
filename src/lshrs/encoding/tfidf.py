# file/function to implement TF-IDF encoding

"""
Input: doc (A list of words of a document),
Output: result (A dictionary. The TF of each word in a document):

Function TF (doc):
    result = empty dictionary

    For word in doc:
        result[word] += 1

    For word in result:
        result[word] \= length of document

    Return result


Input: documents (Two dimensions list. Dimension 1 is documents. Dimension 2 is words of a documents)
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