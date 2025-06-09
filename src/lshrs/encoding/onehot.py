# implemente shingling and one-hot encoding.

"""
Input: documents (Two dimensions list. Dimension 1 is documents. Dimension 2 is words of a documents),
       size (shingling size)
Output: result (Two dimensions list. Dimension 1 is documents. Dimension 2 is shingles of a documents)

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