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

    def __init__(self, data, target):
        self.raw_data = data
        self.target = target

    def __repr__(self):
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


    #clean, remove stopwords, and lemmatize data
    def preprocess(self):
        """
        Takes in raw data and converts it into a list of lemmatized shingles.
        """
        print("Preprocessing.")
        data = list()
        for i in range(len(self.raw_data)):
            #remove non-alphanumeric characters
            data.append(self.raw_data[i].split())
            for j in range(len(data[i])):
                data[i][j] = re.sub(r'\W+', '', data[i][j])
            #stopword removed
            data[i] = [w.lower() for w in data[i] if w not in stop_words]
            #lemmatized
            data[i] = [lem.lemmatize(w) for w in data[i]]

        self.preprocessed = data
        print("Preprocessing Complete, please apply shingling function.")


    #transform document into shingles
    def shingle(self, k):
        """
        Returns self.post_shingle, list of data [index:int, [list_of_shingles]]
        creates self.shingle_array: list of all shingles in the document
        """
        self.k = k
        print(f"Applying shingling function.")
        self.post_shingle = list()
        self.shingle_set = set()

        for i in range(len(self.preprocessed)):
            self.post_shingle.append(list())
            for j in range(len(self.preprocessed[i]) - self.k):
                #append new shingle as list
                shingle_list = self.preprocessed[i][j:j + self.k]
                shingle = " ".join([s for s in shingle_list])
                if shingle not in self.shingle_set:
                    self.shingle_set.add(shingle)
                self.post_shingle[i].append(shingle)

        self.shingle_array = list(self.shingle_set)
        #print("shingled_data", self.post_shingle)
        #[[0, ['This first document', 'first document sure']], [1, ['This document second', 'document second document', 'second document whatever']]]
        print(f"Shingling complete.")


    def perm_array(self, array_size):
        """
        Generates a permuation array with length of the total number of shingles.
        Helper function for generate_permutation_matrix
        """
        a = np.arange(array_size)
        np.random.shuffle(a)
        return a


    def generate_permutation_matrix(self):
        """
        Creates permutation matrix, with rows as permutations of length equal to number of total shingles.
        """
        pm = list()
        for i in range(self.permutations):
            pm.append(self.perm_array(self.shingle_count))

        pm = pd.DataFrame(pm)

        #print(pm)
        print("Permutation Matrix Generated")
        return pm


    def one_hot_encode(self):
        """
        One-Hot encode the list of shingled data. Returns self.one_hot_matrix with
        rows as documents and colums as one-hot indexes.
        """
        pd_data = pd.Series(self.post_shingle)

        mlb = MultiLabelBinarizer()

        res = pd.DataFrame(mlb.fit_transform(pd_data),
                        columns=mlb.classes_,
                        index=pd_data.index)
        sparse = csr_matrix(res)
        del res

        #print(sparse)
        print("One-Hot Encoding Complete")
        gc.collect()
        return sparse

    #use minhashing to permute data into a signature matrix
    def index(self, permutations: int):
        """
        Creates a Signature Matrix with columns as documents and rows as number of permutations
        after applying it to the
        """
        print("MinHashing initiated.")
        self.permutations = permutations

        #set some variables for easy iteration
        self.doc_count = len(self.post_shingle)
        self.shingle_count = len(self.shingle_array)

        #generate signature matrix and correct type
        self.signature_matrix = pd.DataFrame(index=range(self.permutations), columns=range(self.doc_count))


        self.one_hot = self.one_hot_encode()
        self.perm_matrix = self.generate_permutation_matrix()
        
        # Convert one_hot to a numpy array for faster processing
        one_hot_np = self.one_hot.toarray()

        # Iterate over permutations
        for xdoc_id in range(self.doc_count):
            # Get the positions of ones in the one-hot encoded matrix for the current document
            ones_positions = np.where(one_hot_np[xdoc_id] == 1)[0]
            for perm_index, perm_row in self.perm_matrix.iterrows():
                # Get the shingle locations in order
                perm_row_filtered = perm_row[ones_positions]
                min_perm = perm_row_filtered.min()
                self.signature_matrix.at[perm_index, xdoc_id] = min_perm

        self.signature_matrix = self.signature_matrix.astype(int)
        # Print the signature matrix
        # print(self.signature_matrix)
        gc.collect()
        print("Minhashing processing complete, proceed to LSH.")


    #compute the optimal number of bands and rows per band using a seperate function
    def pre_lsh(self, x: int):
        # Compute optimal b and r based on n
        best_br = OptimalBR()
        self.b, self.r = best_br.br(x)
        return f"{self.b} bands and {self.r} rows per band computed."


    #use lsh_256 to hash items into buckets. LSH processing is complete after this.
    def lsh_256(self, b = None, r = None):
       #complete lsh and returns a dictionary with lsh values as keys and set of documents sorted in as values
        if self.signature_matrix is None:
            raise ValueError("Signature matrix is not initialized.")
        print("LSH initiated.")

        if b and r:
        # If two parameters are passed, assume they are 'b' and 'r'
            self.b, self.r = b, r
            if self.b * self.r != self.permutations:
                raise ValueError(f"Number of Bands and Rows invalid, product must be equal to {self.permutations}.")
        else:
        #simply automatically calculate the numebr of b and r using the function
            self.pre_lsh(self.permutations)
        self.lsh_buckets = dict()

        for doc_id in range(self.doc_count):
            signature_array = [x for x in self.signature_matrix[doc_id]]
            doc_group = self.target[doc_id]

            for band_index in range(self.b):
                start = band_index * self.r
                band_key = hashlib.sha256("".join([str(line) for line in signature_array[start:start + self.r]])\
                    .encode('utf-8')).hexdigest()
                if band_key in self.lsh_buckets.keys():
                    self.lsh_buckets[band_key].add(doc_group)
                else:
                    self.lsh_buckets[band_key] = {doc_group}
        print(f"LSH complete with {self.b} bands and {self.r} rows.")


    #completes all the previous steps for a unique string and sees which data bucket it would likely fit into.
    def query(self, data_test: str, topk: int):

        if not all(hasattr(self, attr) for attr in ['k', 'permutations', 'b', 'r']):
            raise ValueError("Shingling and LSH parameters are not initialized.")

        #initiated document querying
        query_data = data_test.split()
        for i in range(len(query_data)):
            query_data[i] = re.sub(r'\W+', '', query_data[i])
        #stopword removed
        query_data = [w.lower() for w in query_data if w not in stop_words]
        #lemmatized
        query_data = [lem.lemmatize(w) for w in query_data]


        #shingling data
        query_shingles = list()

        for i in range(len(query_data)):
            shingle = query_data[i]
            query_shingles.append(shingle)

        #one hot encoding
        one_hot_encoded_list = np.full((self.shingle_count), 0)

        for i in range(len(self.shingle_array)):
            if self.shingle_array[i] in query_shingles:
                one_hot_encoded_list[i] = 1

        #create a mini permutation matrix
        signature_list = np.full((self.permutations), 0)

        for perm_index, perm_row in self.perm_matrix.iterrows():
            for c in range(self.shingle_count):
                p = perm_row[c]
                if one_hot_encoded_list[p] == 1:
                    signature_list[perm_index] = p
                    break

        #apply lsh and hash into lsh_buckets dictionary
        lsh_keys = set()

        for band_index in range(self.b):
            start = band_index * self.r
            band_key = hashlib.sha256("".join([str(line) for line in signature_list[start:start + self.r]])\
                .encode('utf-8')).hexdigest()
            lsh_keys.add(band_key)
        print(lsh_keys)

        # Find candidates using LSH
        candidates = self.find_candidates(lsh_keys)
        topk_candidates = dict(candidates.items()[:topk])
        # Return topK most similar articles
        return topk_candidates


    #after querying, find the most likely candidtates the queried text would fit into.
    def find_candidates(self, query_keys):
        if self.lsh_buckets is None:
            raise ValueError("LSH buckets are not initialized.")

        candidates = {}

        # Iterate over each item in the large dataset LSH buckets
        for key, bucket in self.lsh_buckets.items():
            if key in query_keys:
                for item in bucket:
                    if item not in candidates.keys():
                        candidates[item] = 1
                    else:
                        candidates[item] += 1

        # Sort the candidates in descending order
        sorted_candidates = dict(sorted(candidates.items(), key=lambda item: item[1]))

        return sorted_candidates


def main():
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
