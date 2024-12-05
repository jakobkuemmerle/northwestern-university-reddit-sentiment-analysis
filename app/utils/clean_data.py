import logging
import re
import os
import nltk

# Clear the NLTK cache to ensure it's using the fresh downloads
# nltk.data.clear_cache()

# Check if resources are available
def check_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt')
        print("Punkt tokenizer is available!")
    except LookupError:
        print("Punkt tokenizer not found, attempting to download...")
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('punkt_tab')

# Run the check function
check_nltk_resources()

# Set up logging
log = logging.getLogger("reddit_analysis")
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler())

# Function to filter and process the DataFrame
def filter_data(df, min_chars, keywords=None, start_year=None, end_year=None):
    log.info("Filtering data...")
    
    # Filter columns
    columns_to_keep = [
        "title", "selftext", "score", "archived", "author",
        "created_utc", "id", "media", "num_comments",
        "subreddit", "created_datetime"
    ]
    df = df[columns_to_keep]
    
    # Filter by text length
    df = df[df['selftext'].str.len() > min_chars]
    
    # Filter by keywords
    if keywords:
        keyword_regex = '|'.join([rf'\b{k}\b' for k in keywords])
        df = df[df['selftext'].str.contains(keyword_regex, flags=re.IGNORECASE, na=False)]
    
    # Filter by year range
    df['year'] = df['created_datetime'].dt.year
    if start_year:
        df = df[df['year'] >= start_year]
    if end_year:
        df = df[df['year'] <= end_year]
    
    log.info(f"Filtered data to {len(df):,} rows.")
    return df

def preprocess_text(text: str) -> str:
    """
    Preprocesses the given text by lowercasing, removing special characters, 
    tokenizing, removing stopwords, and lemmatizing.

    Args:
        text (str): The input text to preprocess.

    Returns:
        str: Preprocessed text.
    """
    # Lowercase the text
    text = text.lower()
    # Remove special characters, numbers, and punctuation
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    # Tokenize
    tokens = nltk.word_tokenize(text)

    # Remove stopwords
    stop_words = set(nltk.corpus.stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatize
    lemmatizer = nltk.WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)




# Linda added

from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

# Download WordNet data if not already downloaded
nltk.download('wordnet')
nltk.download('omw-1.4')

def generate_similar_words(keyword):
    
    # Initialize WordNet Lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # Find synonyms using WordNet
    synonyms = set()
    for syn in wordnet.synsets(keyword):
        for lemma in syn.lemmas():
            # Exclude compound words or phrases
            if "_" not in lemma.name() and len(lemma.name().split()) == 1:
                synonyms.add(lemma.name())
    
    # Generate morphological variations
    variations = set([
        keyword,  # Original word
        lemmatizer.lemmatize(keyword, pos='n'),  # Lemma (noun)
        lemmatizer.lemmatize(keyword, pos='v'),  # Lemma (verb)
        lemmatizer.lemmatize(keyword, pos='a'),  # Lemma (adjective)
        f"{keyword}s",  # Plural
        f"{keyword}ed",  # Past tense
        f"{keyword}ing"  # Present participle
    ])
    
    # Combine and deduplicate
    similar_words = sorted(synonyms.union(variations))
    return similar_words
