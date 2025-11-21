import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data (run this once)
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except nltk.downloader.DownloadError:
    nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    """
    Applies a series of cleaning and normalization steps to the input text.
    """
    text = str(text).lower()
    
    # 1. Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # 2. Remove user mentions and hash symbols
    text = re.sub(r'@\w+|#', '', text)
    
    # 3. Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # 4. Remove Emojis and other non-standard characters (optional but useful)
    # The following regex targets basic emojis/symbols; more robust methods exist
    text = re.sub(r'[^\w\s\d]', '', text) 
    
    # 5. Tokenization and Stopword Removal
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words and word.isalpha()]
    
    # 6. Lemmatization
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return " ".join(tokens)

if __name__ == '__main__':
    # Example usage:
    sample_text = "@user The new product is amazing! 💯 Check it out: http://example.com #bestproduct"
    cleaned = clean_text(sample_text)
    print(f"Original: {sample_text}")
    print(f"Cleaned: {cleaned}") 
    # Output: 'new product amazing check examplecom bestproduct' (or similar, depending on stopword list)
