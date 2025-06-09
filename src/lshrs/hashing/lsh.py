# main function to implement LSH

"""
Input: signature
       r (How many rows in a band)
       signature_type (OneHotEncoding or HyperplaneEncoding)
Output: buckets

Function LSH (signature, r, signature_type):
    Initialize buckets = empty dictionary

    For id, signature in enumerate(signatures):
        For index from 0 to len(signature) - 1:
            if signature_type is binary:
                band = int(signature[index * r : (index + 1) * r], 2) # Convert to decimal
            else:
                band = signature[index * r : (index + 1) * r]

            If band not in buckets:
                buckets[band] = empty list
            Append id to buckets[band]

    Return buckets
"""