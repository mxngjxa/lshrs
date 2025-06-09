# function to implement minhashing as the next step in LSH after onehot encoding and before locality sensitive hashing

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