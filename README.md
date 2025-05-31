# Recommendation System

This recommendation system package is designed to preprocess input text, create indices, compute minHash values, and provide similarity-based recommendations using the 20 Newsgroups dataset.

## Functionality

1. **Preprocessing**: Input text undergoes preprocessing techniques including tokenization, stopword removal, and stemming/lemmatization.

2. **Parameter Options**:
   - Accept 'b' and 'r' as parameters, representing the number of bands and the number of rows in each band respectively.
   - Alternatively, accept 'n' as a parameter, which is the total number of permutations. If 'n' is supplied, the system computes 'b' and 'r' automatically using the minimization of False Positive Rate (FPR) + False Negative Rate (FNR).
   
3. **Index Creation**: The system creates indices for each of the 'n' permutations.

4. **Article Processing**: Input articles are processed one at a time after preprocessing.

5. **Index Method**: The 'index' method computes the minHash value for each article and for each permutation. It uses 'hashlib.sha256' to hash 'r' integers in each band for each column into a dictionary key, with a set as the value, and the article ID as one of the set members.

6. **Query Method**: The 'query' method takes input text and a parameter 'topK' (e.g., topK=5) to return the 'topK' most similar articles.

## Demonstration with 20 Newsgroups Dataset

The 20 Newsgroups dataset is utilized for demonstrating the functionality of this package. The dataset contains training and testing sets. However, only the training set should be used for this demonstration.

### About 20 Newsgroups Dataset

A newsgroup is akin to an email discussion group focused on a specific topic. This dataset, reminiscent of the early social media platforms in the 1990s, provides a collection of documents categorized into different newsgroups.

### Usage

To demonstrate the package/class using the 20 Newsgroups dataset:

1. Obtain the dataset from [here](https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html).
2. Use the training set for training and testing purposes.
3. Integrate the recommendation system package/class with the dataset to showcase its functionality.

## Acknowledgments

The functionality of this recommendation system is inspired by previous assignments and projects. Special thanks to the developers and contributors of the 20 Newsgroups dataset and scikit-learn library for making this demonstration possible.
