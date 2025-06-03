import re
import os
import gc
import nltk
import hashlib
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.sparse import csr_matrix
try:
    from .optimal_br import OptimalBR
except:
    from optimal_br import OptimalBR


lem = WordNetLemmatizer()

# Set the NLTK_DATA environment variable to the desired directory
current_directory = os.getcwd()
desired_directory = f'{current_directory}/.venv/nltk_data'

nltk.download('stopwords', download_dir=desired_directory)
nltk.download('wordnet', download_dir=desired_directory)
stop_words = set(stopwords.words('english'))

class recommendation_system:
    """
    A system for recommending similar documents based on their text content
    using Locality Sensitive Hashing (LSH).

    The process involves:
    1. Preprocessing: Cleaning text, removing stopwords, and lemmatization.
    2. Shingling: Converting documents into sets of k-shingles (contiguous sequences of k words).
    3. Indexing (MinHashing): Creating a signature matrix representing documents with MinHash signatures.
    4. LSH: Hashing signatures into buckets to find candidate similar pairs.
    5. Querying: Finding documents similar to a given query text.
    """

    def __init__(self, data, target):
        """
        Initializes the recommendation system.

        Args:
            data (list): A list of strings, where each string is a document.
            target (list): A list of identifiers or labels corresponding to each document in `data`.
        """
        self.raw_data = data
        self.target = target

    def __repr__(self):
        """
        Provides a string representation of the recommendation_system object's current state.
        """
        if not hasattr(self, 'preprocessed') or self.preprocessed is None:
            return f"Raw text of length {len(self.raw_data)}. Awaiting preprocessing."
        elif not hasattr(self, 'shingled') or self.shingled is None:
            return f"Preprocessed text of length {len(self.preprocessed)}. Awaiting shingling."
        elif not hasattr(self, 'signature_matrix') or self.signature_matrix is None:
            return f"Shingled into {self.k}-token shingles. Awaiting MinHash indexing."
        elif not hasattr(self, 'lsh_buckets') or self.lsh_buckets is None:
            return f"MinHash indexed with {self.permutations} permutations. Awaiting Locality Sensitive Hashing (LSH)."
        else:
            return f"Text preprocessed, shingled into {self.k}-token shingles, indexed using MinHash with {self.permutations} permutations, and ready for querying with LSH using {self.b} bands and {self.r} rows per band."


    def preprocess(self):
        """
        Cleans, removes stopwords, and lemmatizes the raw input data.

        The raw data (list of document strings) is processed by:
        1. Splitting into words.
        2. Removing non-alphanumeric characters from each word.
        3. Converting words to lowercase.
        4. Removing common English stopwords.
        5. Lemmatizing words to their base form.

        Sets `self.preprocessed` to a list of lists, where each inner list
        contains the processed tokens for a document.
        """
        print("Preprocessing.")
        data = list()
        for i in range(len(self.raw_data)):
            # Split document into words
            current_doc_words = self.raw_data[i].split()
            processed_words = []
            for word in current_doc_words:
                # Remove non-alphanumeric characters
                cleaned_word = re.sub(r'\W+', '', word)
                if cleaned_word: # Ensure word is not empty after cleaning
                    processed_words.append(cleaned_word)
            
            # Convert to lowercase and remove stopwords
            processed_words = [w.lower() for w in processed_words if w.lower() not in stop_words]
            # Lemmatize words
            processed_words = [lem.lemmatize(w) for w in processed_words]
            data.append(processed_words)

        self.preprocessed = data
        print("Preprocessing Complete, please apply shingling function.")


    def shingle(self, k: int):
        """
        Transforms preprocessed documents into k-shingles.

        A k-shingle is a contiguous sequence of k tokens. This method generates
        shingles for each document based on the preprocessed tokens.

        Args:
            k (int): The number of tokens in each shingle.

        Sets:
            self.k (int): The shingle size used.
            self.post_shingle (list): A list of lists, where each inner list
                                      contains the shingles for a document.
            self.shingle_set (set): A set of all unique shingles found across all documents.
            self.shingle_array (list): A list of all unique shingles (derived from `shingle_set`).
        """
        self.k = k
        print(f"Applying shingling function.")
        self.post_shingle = list()
        self.shingle_set = set()

        for i in range(len(self.preprocessed)):
            doc_shingles = []
            # Iterate through the tokens of the document to create k-shingles
            for j in range(len(self.preprocessed[i]) - self.k + 1): # Adjusted loop to include last possible shingle
                # Extract k tokens to form a shingle
                shingle_list = self.preprocessed[i][j:j + self.k]
                # Join tokens to form the shingle string
                shingle = " ".join([s for s in shingle_list])
                # Add to document's shingle list
                doc_shingles.append(shingle)
                # Add to the global set of unique shingles
                if shingle not in self.shingle_set:
                    self.shingle_set.add(shingle)
            self.post_shingle.append(doc_shingles)

        # Convert the set of unique shingles to a list for consistent ordering (though order isn't strictly necessary for set operations)
        self.shingle_array = list(self.shingle_set)
        #print("shingled_data", self.post_shingle)
        #[[0, ['This first document', 'first document sure']], [1, ['This document second', 'document second document', 'second document whatever']]]
        print(f"Shingling complete.")


    def perm_array(self, array_size):
        """
        Generates a permutation array of a given size.

        This is a helper function for `generate_permutation_matrix`.

        Args:
            array_size (int): The size of the array to permute (typically the total number of unique shingles).

        Returns:
            numpy.ndarray: A shuffled array of integers from 0 to `array_size` - 1.
        """
        a = np.arange(array_size)
        np.random.shuffle(a)
        return a


    def generate_permutation_matrix(self):
        """
        Creates a permutation matrix for MinHashing.

        The matrix has `self.permutations` rows, and each row is a random permutation
        of shingle indices (from 0 to `self.shingle_count` - 1).

        Returns:
            pandas.DataFrame: The permutation matrix, where each row is a permutation.
        """
        pm = list()
        # For each desired permutation, generate a shuffled array of shingle indices
        for i in range(self.permutations):
            pm.append(self.perm_array(self.shingle_count))

        # Convert the list of permutations into a Pandas DataFrame
        pm = pd.DataFrame(pm)

        #print(pm)
        print("Permutation Matrix Generated")
        return pm


    def one_hot_encode(self):
        """
        One-hot encodes the shingled data.

        Transforms the list of shingled documents (`self.post_shingle`) into a
        sparse matrix where rows represent documents and columns represent unique
        shingles. A '1' indicates the presence of a shingle in a document.

        Returns:
            scipy.sparse.csr_matrix: A sparse matrix representing the one-hot encoded shingled documents.
        """
        # Convert the list of shingled documents to a Pandas Series for MultiLabelBinarizer
        pd_data = pd.Series(self.post_shingle)

        # Initialize MultiLabelBinarizer to convert lists of shingles into a binary matrix
        mlb = MultiLabelBinarizer()

        # Fit and transform the data to get the one-hot encoded matrix
        # `mlb.classes_` will be the unique shingles, serving as column names
        res = pd.DataFrame(mlb.fit_transform(pd_data),
                        columns=mlb.classes_,
                        index=pd_data.index) # Documents are rows
        # Convert the dense DataFrame to a sparse CSR matrix for memory efficiency
        sparse = csr_matrix(res)
        del res # Free up memory from the dense DataFrame

        #print(sparse)
        print("One-Hot Encoding Complete")
        gc.collect()
        return sparse

    #use minhashing to permute data into a signature matrix
    def index(self, permutations: int):
        """
        Creates a Signature Matrix using MinHashing.

        The Signature Matrix approximates the Jaccard similarity between documents.
        Each column represents a document, and each row corresponds to a hash function
        (permutation). The value at `(i, j)` is the minimum hash value of shingle indices
        present in document `j` according to permutation `i`.

        Args:
            permutations (int): The number of permutations (hash functions) to use for MinHashing.
                                This determines the number of rows in the signature matrix.

        Sets:
            self.permutations (int): The number of permutations used.
            self.doc_count (int): The number of documents.
            self.shingle_count (int): The number of unique shingles.
            self.signature_matrix (pandas.DataFrame): The resulting signature matrix.
            self.one_hot (scipy.sparse.csr_matrix): The one-hot encoded representation of shingles.
            self.perm_matrix (pandas.DataFrame): The permutation matrix used for MinHashing.
        """
        print("MinHashing initiated.")
        self.permutations = permutations

        #set some variables for easy iteration
        self.doc_count = len(self.post_shingle)
        self.shingle_count = len(self.shingle_array)

        # Initialize an empty signature matrix with permutations as rows and documents as columns
        self.signature_matrix = pd.DataFrame(index=range(self.permutations), columns=range(self.doc_count))

        # Perform one-hot encoding of shingles for all documents
        self.one_hot = self.one_hot_encode()
        # Generate the matrix of random permutations of shingle indices
        self.perm_matrix = self.generate_permutation_matrix()
        
        # Convert the sparse one-hot matrix to a dense NumPy array for efficient row-wise operations
        # This can be memory-intensive for very large datasets.
        one_hot_np = self.one_hot.toarray()

        # Iterate over each document to compute its MinHash signature
        for doc_idx in range(self.doc_count):
            # Find the indices of shingles present in the current document (where one_hot_np is 1)
            shingle_indices_in_doc = np.where(one_hot_np[doc_idx] == 1)[0]
            
            # Iterate over each permutation (hash function)
            for perm_idx, perm_shingle_order in self.perm_matrix.iterrows():
                # For the current permutation, find the minimum permuted index of a shingle present in the document
                # This is the core MinHash computation for one (permutation, document) pair.
                min_hash_val = perm_shingle_order[shingle_indices_in_doc].min()
                self.signature_matrix.at[perm_idx, doc_idx] = min_hash_val

        # Ensure the signature matrix contains integer values
        self.signature_matrix = self.signature_matrix.astype(int)
        # Print the signature matrix
        # print(self.signature_matrix)
        gc.collect()
        print("Minhashing processing complete, proceed to LSH.")


    def pre_lsh(self, n_permutations: int):
        """
        Computes the optimal number of bands (b) and rows per band (r) for LSH.

        This method uses the `OptimalBR` class to find `b` and `r` such that
        `b * r` is close to `n_permutations`.

        Args:
            n_permutations (int): The total number of permutations (rows in the signature matrix).

        Sets:
            self.b (int): The optimal number of bands.
            self.r (int): The optimal number of rows per band.

        Returns:
            str: A message indicating the computed b and r values.
        """
        # Compute optimal b and r based on n
        best_br = OptimalBR()
        self.b, self.r = best_br.br(n_permutations)
        return f"{self.b} bands and {self.r} rows per band computed."


    def lsh_256(self, b: int = None, r: int = None):
        """
        Applies Locality Sensitive Hashing (LSH) to the signature matrix.

        Divides the signature matrix into `b` bands, each with `r` rows.
        For each band, documents are hashed into buckets. Documents that hash
        to the same bucket in at least one band are considered candidate similar pairs.

        Args:
            b (int, optional): The number of bands. If None, it's computed automatically.
            r (int, optional): The number of rows per band. If None, it's computed automatically.

        Sets:
            self.b (int): The number of bands used.
            self.r (int): The number of rows per band used.
            self.lsh_buckets (dict): A dictionary where keys are hash values (band-specific)
                                     and values are sets of document targets (identifiers)
                                     that hashed to that key.

        Raises:
            ValueError: If the signature matrix is not initialized or if `b * r`
                        does not equal `self.permutations` when `b` and `r` are provided.
        """
        if self.signature_matrix is None:
            raise ValueError("Signature matrix is not initialized.")
        print("LSH initiated.")

        # Check if b (bands) and r (rows per band) are provided
        if b and r:
            # Use provided b and r
            self.b, self.r = b, r
            # Validate that b * r equals the total number of permutations
            if self.b * self.r != self.permutations:
                raise ValueError(f"Number of Bands and Rows invalid, product must be equal to {self.permutations}.")
        else:
            # Automatically calculate optimal b and r if not provided
            self.pre_lsh(self.permutations)
        
        # Initialize a dictionary to store LSH buckets
        # Keys will be hash values of bands, values will be sets of document targets
        self.lsh_buckets = dict()

        # Iterate over each document's signature
        for doc_id in range(self.doc_count):
            # Extract the signature (column) for the current document
            signature_column = self.signature_matrix[doc_id].tolist()
            # Get the target/identifier for the current document
            doc_target_id = self.target[doc_id]

            # Divide the signature into bands
            for band_idx in range(self.b):
                # Determine the start and end row for the current band
                start_row = band_idx * self.r
                end_row = start_row + self.r
                # Extract the portion of the signature corresponding to the current band
                band_values = signature_column[start_row:end_row]
                
                # Hash the band values to a single key (e.g., using SHA256)
                # Concatenate string representations of integers in the band
                band_string_representation = "".join([str(val) for val in band_values])
                # Encode to bytes and then hash
                band_hash_key = hashlib.sha256(band_string_representation.encode('utf-8')).hexdigest()
                
                # Add the document's target ID to the bucket corresponding to this band's hash key
                if band_hash_key in self.lsh_buckets:
                    self.lsh_buckets[band_hash_key].add(doc_target_id)
                else:
                    self.lsh_buckets[band_hash_key] = {doc_target_id}
        print(f"LSH complete with {self.b} bands and {self.r} rows.")


    def query(self, data_test: str, topk: int):
        """
        Finds the top-k most similar documents to a given query string.

        The query string undergoes the same preprocessing, shingling, and MinHashing
        steps as the original dataset. Then, LSH is used to identify candidate
        similar documents from the `self.lsh_buckets`.

        Args:
            data_test (str): The query string for which to find similar documents.
            topk (int): The number of most similar documents to return.

        Returns:
            dict: A dictionary where keys are document targets (identifiers) and
                  values are their similarity scores (counts of shared LSH buckets
                  with the query). The dictionary contains the top-k candidates.

        Raises:
            ValueError: If shingling and LSH parameters (k, permutations, b, r)
                        are not initialized.
        """
        if not all(hasattr(self, attr) for attr in ['k', 'permutations', 'b', 'r']):
            raise ValueError("Shingling and LSH parameters are not initialized.")

        # --- Preprocess the query text ---
        print(f"Querying with text: '{data_test[:50]}...'")
        query_words = data_test.split()
        processed_query_words = []
        for word in query_words:
            cleaned_word = re.sub(r'\W+', '', word)
            if cleaned_word:
                processed_query_words.append(cleaned_word)
        processed_query_words = [w.lower() for w in processed_query_words if w.lower() not in stop_words]
        lemmatized_query_words = [lem.lemmatize(w) for w in processed_query_words]

        # --- Shingle the preprocessed query text ---
        query_shingles_list = []
        if len(lemmatized_query_words) >= self.k: # Ensure there are enough words to form at least one shingle
            for i in range(len(lemmatized_query_words) - self.k + 1):
                shingle_tokens = lemmatized_query_words[i:i + self.k]
                query_shingles_list.append(" ".join(shingle_tokens))
        
        # --- One-hot encode the query shingles against the known shingle universe ---
        # `self.shingle_array` contains all unique shingles from the training data
        query_one_hot_vector = np.full((self.shingle_count), 0, dtype=int)
        # Create a mapping from shingle string to its index in self.shingle_array for quick lookup
        shingle_to_idx_map = {shingle: i for i, shingle in enumerate(self.shingle_array)}
        for shingle in query_shingles_list:
            if shingle in shingle_to_idx_map:
                query_one_hot_vector[shingle_to_idx_map[shingle]] = 1

        # --- Compute MinHash signature for the query text ---
        query_signature = np.full((self.permutations), np.inf) # Initialize with infinity

        # Find indices of shingles present in the query
        shingle_indices_in_query = np.where(query_one_hot_vector == 1)[0]

        if shingle_indices_in_query.size > 0: # Proceed only if the query has known shingles
            for perm_idx, perm_shingle_order in self.perm_matrix.iterrows():
                # Get the permuted order for shingles present in the query
                permuted_indices_for_query_shingles = perm_shingle_order[shingle_indices_in_query]
                # The MinHash value for this permutation is the minimum of these permuted indices
                query_signature[perm_idx] = permuted_indices_for_query_shingles.min()
        
        # Replace np.inf with a value indicating no match if any permutation didn't find a shingle
        # (e.g., if a query has no common shingles, its signature might remain all np.inf)
        # For LSH, these high values will likely not match anything.
        # A more robust approach might be to handle cases where shingle_indices_in_query is empty earlier.
        query_signature[query_signature == np.inf] = self.shingle_count # Or some other large, out-of-bounds value

        # --- Apply LSH to the query's signature ---
        query_lsh_keys = set()
        for band_idx in range(self.b):
            start_row = band_idx * self.r
            end_row = start_row + self.r
            band_values = query_signature[start_row:end_row]
            
            band_string_representation = "".join([str(int(val)) for val in band_values]) # Ensure integer strings
            band_hash_key = hashlib.sha256(band_string_representation.encode('utf-8')).hexdigest()
            query_lsh_keys.add(band_hash_key)
        
        print(f"Query LSH keys: {query_lsh_keys}")

        # --- Find candidate documents based on matching LSH keys ---
        candidate_docs = self.find_candidates(query_lsh_keys)
        
        # --- Select top-k candidates ---
        # `find_candidates` already sorts, so we can take the top items.
        # Note: `dict.items()` in Python 3.x returns a view, convert to list for slicing if needed,
        # or iterate and build the topk dict.
        # `sorted_candidates` in `find_candidates` is already a dict.
        
        # Convert to list of items to sort and then take topk
        # find_candidates returns a dict sorted by value (similarity score)
        # To get topk, we need to sort by score in descending order.
        # The current find_candidates sorts in ascending order. Let's fix that or re-sort here.
        # Assuming find_candidates returns dict sorted ascending by count.
        
        # Re-sorting here for clarity, ensuring descending order of similarity
        sorted_candidate_list = sorted(candidate_docs.items(), key=lambda item: item[1], reverse=True)
        
        topk_results = dict(sorted_candidate_list[:topk])
        
        return topk_results


    def find_candidates(self, query_keys: set):
        """
        Finds candidate documents from LSH buckets that match the query's LSH keys.

        Compares the LSH keys generated for a query document with the LSH buckets
        of the main dataset. Documents sharing one or more LSH keys with the query
        are considered candidates. Their scores are incremented based on the number
        of shared keys.

        Args:
            query_keys (set): A set of LSH hash keys generated for the query document.

        Returns:
            dict: A dictionary of candidate document targets (identifiers) and their
                  similarity scores (number of matching LSH keys), sorted by score
                  in descending order.

        Raises:
            ValueError: If LSH buckets are not initialized.
        """
        if self.lsh_buckets is None:
            raise ValueError("LSH buckets are not initialized.")

        candidates = {}

        # Iterate over the LSH buckets of the main dataset
        for lsh_key, doc_targets_in_bucket in self.lsh_buckets.items():
            # If this LSH key from the dataset matches one of the query's LSH keys
            if lsh_key in query_keys:
                # Increment the score for each document target found in this matching bucket
                for doc_target in doc_targets_in_bucket:
                    if doc_target not in candidates:
                        candidates[doc_target] = 0
                    candidates[doc_target] += 1

        # Sort the candidate documents by their similarity score (number of matching LSH keys)
        # Sorting in descending order to get the most similar items first.
        sorted_candidates = dict(sorted(candidates.items(), key=lambda item: item[1], reverse=True))

        return sorted_candidates


def main():
    """
    Main function to demonstrate the usage of the recommendation_system.

    It initializes the system with sample data, performs preprocessing,
    shingling, MinHashing, LSH, and then queries for similar articles.
    """
    # Sample raw data
    raw_data = ['From: v064mb9k@ubvmsd.cc.buffalo.edu (NEIL B. GANDLER)\nSubject: Need info on 88-89 Bonneville\nOrganization: University at Buffalo\nLines: 10\nNews-Software: VAX/VMS VNEWS 1.41\nNntp-Posting-Host: ubvmsd.cc.buffalo.edu\n\n\n I am a little confused on all of the models of the 88-89 bonnevilles.\nI have heard of the LE SE LSE SSE SSEI. Could someone tell me the\ndifferences are far as features or performance. I am also curious to\nknow what the book value is for prefereably the 89 model. And how much\nless than book value can you usually get them for. In other words how\nmuch are they in demand this time of year. I have heard that the mid-spring\nearly summer is the best time to buy.\n\n\t\t\tNeil Gandler\n',

                'From: Rick Miller <rick@ee.uwm.edu>\nSubject: X-Face?\nOrganization: Just me.\nLines: 17\nDistribution: world\nNNTP-Posting-Host: 129.89.2.33\nSummary: Go ahead... swamp me.  <EEP!>\n\nI\'m not familiar at all with the format of these "X-Face:" thingies, but\nafter seeing them in some folks\' headers, I\'ve *got* to *see* them (and\nmaybe make one of my own)!\n\nI\'ve got "dpg-view" on my Linux box (which displays "uncompressed X-Faces")\nand I\'ve managed to compile [un]compface too... but now that I\'m *looking*\nfor them, I can\'t seem to find any X-Face:\'s in anyones news headers!  :-(\n\nCould you, would you, please send me your "X-Face:" header?\n\nI *know* I\'ll probably get a little swamped, but I can handle it.\n\n\t...I hope.\n\nRick Miller  <rick@ee.uwm.edu> | <ricxjo@discus.mil.wi.us>   Ricxjo Muelisto\nSend a postcard, get one back! | Enposxtigu bildkarton kaj vi ricevos alion!\n          RICK MILLER // 16203 WOODS // MUSKEGO, WIS. 53150 // USA\n',

                'From: mathew <mathew@mantis.co.uk>\nSubject: Re: STRONG & weak Atheism\nOrganization: Mantis Consultants, Cambridge. UK.\nX-Newsreader: rusnews v1.02\nLines: 9\n\nacooper@mac.cc.macalstr.edu (Turin Turambar, ME Department of Utter Misery) writes:\n> Did that FAQ ever got modified to re-define strong atheists as not those who\n> assert the nonexistence of God, but as those who assert that they BELIEVE in \n> the nonexistence of God?\n\nIn a word, yes.\n\n\nmathew\n',

                'From: bakken@cs.arizona.edu (Dave Bakken)\nSubject: Re: Saudi clergy condemns debut of human rights group!\nKeywords: international, non-usa government, government, civil rights, \tsocial issues, politics\nOrganization: U of Arizona CS Dept, Tucson\nLines: 101\n\nIn article <benali.737307554@alcor> benali@alcor.concordia.ca ( ILYESS B. BDIRA ) writes:\n>It looks like Ben Baz\'s mind and heart are also blind, not only his eyes.\n>I used to respect him, today I lost the minimal amount of respect that\n>I struggled to keep for him.\n>To All Muslim netters: This is the same guy who gave a "Fatwah" that\n>Saudi Arabia can be used by the United Ststes to attack Iraq . \n\nThey were attacking the Iraqis to drive them out of Kuwait,\na country whose citizens have close blood and business ties\nto Saudi citizens.  And me thinks if the US had not helped out\nthe Iraqis would have swallowed Saudi Arabia, too (or at \nleast the eastern oilfields).  And no Muslim country was doing\nmuch of anything to help liberate Kuwait and protect Saudi\nArabia; indeed, in some masses of citizens were demonstrating\nin favor of that butcher Saddam (who killed lotsa Muslims),\njust because he was killing, raping, and looting relatively\nrich Muslims and also thumbing his nose at the West.\n\nSo how would have *you* defended Saudi Arabia and rolled\nback the Iraqi invasion, were you in charge of Saudi Arabia???\n\n>Fatwah is as legitimate as this one. With that kind of "Clergy", it might\n>be an Islamic duty to separate religion and politics, if religion\n>means "official Clergy".\n\nI think that it is a very good idea to not have governments have an\nofficial religion (de facto or de jure), because with human nature\nlike it is, the ambitious and not the pious will always be the\nones who rise to power.  There are just too many people in this\nworld (or any country) for the citizens to really know if a \nleader is really devout or if he is just a slick operator.\n\n>\n>  \tCAIRO, Egypt (UPI) -- The Cairo-based Arab Organization for Human\n>  Rights (AOHR) Thursday welcomed the establishement last week of the\n>  Committee for Defense of Legal Rights in Saudi Arabia and said it was\n>  necessary to have such groups operating in all Arab countries.\n\nYou make it sound like these guys are angels, Ilyess.  (In your\nclarinet posting you edited out some stuff; was it the following???)\nFriday\'s New York Times reported that this group definitely is\nmore conservative than even Sheikh Baz and his followers (who\nthink that the House of Saud does not rule the country conservatively\nenough).  The NYT reported that, besides complaining that the\ngovernment was not conservative enough, they have:\n\n\t- asserted that the (approx. 500,000) Shiites in the Kingdom\n\t  are apostates, a charge that under Saudi (and Islamic) law\n\t  brings the death penalty.  \n\n\t  Diplomatic guy (Sheikh bin Jibrin), isn\'t he Ilyess?\n\n\t- called for severe punishment of the 40 or so women who\n\t  drove in public a while back to protest the ban on\n\t  women driving.  The guy from the group who said this,\n\t  Abdelhamoud al-Toweijri, said that these women should\n\t  be fired from their jobs, jailed, and branded as\n\t  prostitutes.\n\n\t  Is this what you want to see happen, Ilyess?  I\'ve\n\t  heard many Muslims say that the ban on women driving\n\t  has no basis in the Qur\'an, the ahadith, etc.\n\t  Yet these folks not only like the ban, they want\n\t  these women falsely called prostitutes?  \n\n\t  If I were you, I\'d choose my heroes wisely,\n\t  Ilyess, not just reflexively rally behind\n\t  anyone who hates anyone you hate.\n\n\t- say that women should not be allowed to work.\n\n\t- say that TV and radio are too immoral in the Kingdom.\n\nNow, the House of Saud is neither my least nor my most favorite government\non earth; I think they restrict religious and political reedom a lot, among\nother things.  I just think that the most likely replacements\nfor them are going to be a lot worse for the citizens of the country.\nBut I think the House of Saud is feeling the heat lately.  In the\nlast six months or so I\'ve read there have been stepped up harassing\nby the muttawain (religious police---*not* government) of Western women\nnot fully veiled (something stupid for women to do, IMO, because it\nsends the wrong signals about your morality).  And I\'ve read that\nthey\'ve cracked down on the few, home-based expartiate religious\ngatherings, and even posted rewards in (government-owned) newspapers\noffering money for anyone who turns in a group of expartiates who\ndare worship in their homes or any other secret place. So the\ngovernment has grown even more intolerant to try to take some of\nthe wind out of the sails of the more-conservative opposition.\nAs unislamic as some of these things are, they\'re just a small\ntaste of what would happen if these guys overthrow the House of\nSaud, like they\'re trying to in the long run.\n\nIs this really what you (and Rached and others in the general\nwest-is-evil-zionists-rule-hate-west-or-you-are-a-puppet crowd)\nwant, Ilyess?\n\n--\nDave Bakken\n==>"the President is doing a fine job, but the problem is we don\'t know what\n    to do with her husband." James Carville (Clinton campaign strategist),2/93\n==>"Oh, please call Daddy. Mom\'s far too busy."  Chelsea to nurse, CSPAN, 2/93\n']
    index = [7, 5, 0, 17]

    # Instantiate recommendation system with sample data
    rec_sys = recommendation_system(raw_data, index)

    # Perform preprocessing
    rec_sys.preprocess()

    # Set shingle size
    k = 2
    rec_sys.shingle(k)

    # Set number of permutations for MinHash
    permutations = 256
    rec_sys.index(permutations)

    # Set parameters for LSH
    #n = 32  # Total number of permutations
    rec_sys.lsh_256()

    # Query text
    query_text = ">Saudi Arabia can be used by the United Ststes to attack Iraq ."

    # Define the value of topK
    topK = 5

    # Query the recommendation system
    top_similar_articles = rec_sys.query(query_text, topK)

    print("Top similar articles:", top_similar_articles)

    print(rec_sys.lsh_buckets)

if __name__ == "__main__":
    main()
    #pass
