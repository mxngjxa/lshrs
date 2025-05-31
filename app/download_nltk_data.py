import nltk

def download_nltk_data():
    # Download NLTK data
    nltk.download('stopwords')
    nltk.download('wordnet')

if __name__ == "__main__":
    print("Downloading NLTK data...")
    download_nltk_data()
    print("NLTK data downloaded successfully.")
