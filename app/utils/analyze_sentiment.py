import pandas as pd
from textblob import TextBlob

def analyze_sentiment(text):
    """
    Analyzes the sentiment of a given text using TextBlob.
    Returns Positive, Negative, or Neutral based on polarity.
    """
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.1:
        return 'Positive'
    elif polarity < -0.1:
        return 'Negative'
    else:
        return 'Neutral'

def assign_sentiments(df):
    """
    Assign sentiment labels to the posts in the DataFrame.
    """
    df['sentiment'] = df['selftext'].apply(analyze_sentiment)
    return df

def calculate_sentiment_distribution(df):
    """
    Groups posts by year and sentiment and calculates sentiment percentages per year.
    """
    # Extract year
    df['year'] = df['created_datetime'].dt.year
    
    # Count posts per sentiment per year
    sentiment_counts = df.groupby(['year', 'sentiment']).size().reset_index(name='count')
    
    # Total posts per year
    total_per_year = df.groupby('year').size().reset_index(name='total')
    
    # Merge counts with totals
    sentiment_distribution = pd.merge(sentiment_counts, total_per_year, on='year')
    
    # Calculate percentage
    sentiment_distribution['percentage'] = (
        sentiment_distribution['count'] / sentiment_distribution['total'] * 100
    )
    
    return sentiment_distribution

