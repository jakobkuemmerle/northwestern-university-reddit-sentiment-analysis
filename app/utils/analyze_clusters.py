import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from typing import  List, Dict, Tuple

def perform_lda(
    text_data: pd.Series, 
    n_topics: int = 5, 
    max_features: int = 5000
) -> Tuple[LatentDirichletAllocation, CountVectorizer, np.ndarray]:
    """
    Performs LDA topic modeling on the provided text data.

    Args:
        text_data (pd.Series): A pandas Series of text data to analyze.
        n_topics (int): Number of topics to generate.
        max_features (int): Maximum number of features for vectorization.

    Returns:
        Tuple: LDA model, CountVectorizer, and topic assignments.
    """
    vectorizer = CountVectorizer(max_features=max_features, stop_words='english')
    text_matrix = vectorizer.fit_transform(text_data)

    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(text_matrix)

    return lda, vectorizer, lda.transform(text_matrix)

def display_topics(lda: LatentDirichletAllocation, feature_names: List[str], n_top_words: int = 10) -> Dict[str, List[str]]:
    """
    Extracts and displays the top words for each topic from the LDA model.

    Args:
        lda (LatentDirichletAllocation): Fitted LDA model.
        feature_names (List[str]): List of feature names from vectorization.
        n_top_words (int): Number of words to display for each topic.

    Returns:
        Dict[str, List[str]]: A dictionary of topics and their top words.
    """
    topics = {}
    for topic_idx, topic in enumerate(lda.components_):
        topics[f"Topic {topic_idx + 1}"] = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
    return topics

def analyze_topics_over_time(
    df: pd.DataFrame, 
    topic_assignments: np.ndarray
) -> pd.DataFrame:
    """
    Analyzes topic trends over time by calculating the number of posts per topic per year.

    Args:
        df (pd.DataFrame): DataFrame containing the original data with a 'created_datetime' column.
        topic_assignments (np.ndarray): Topic assignments for each post.

    Returns:
        pd.DataFrame: Topic trends over time.
    """
    df['dominant_topic'] = topic_assignments.argmax(axis=1) + 1  # Add 1 for 1-based indexing
    df['year'] = df['created_datetime'].dt.year

    # Count posts per topic per year
    topic_trends = df.groupby(['year', 'dominant_topic']).size().unstack(fill_value=0)

    return topic_trends

def detect_spikes(
    topic_trends: pd.DataFrame, 
    overall_trends: pd.Series
) -> pd.DataFrame:
    """
    Detects spikes in topic popularity relative to the overall number of posts.

    Args:
        topic_trends (pd.DataFrame): Trends of topics over time.
        overall_trends (pd.Series): Overall number of posts per year.

    Returns:
        pd.DataFrame: DataFrame of percentage changes (spikes).
    """
    # Normalize by overall number of posts
    normalized_trends = topic_trends.div(overall_trends, axis=0).fillna(0)
    percentage_change = normalized_trends.pct_change().fillna(0)

    # Highlight spikes (e.g., >50% growth)
    return percentage_change[percentage_change > 0.5]

def get_cluster_descriptions(
    df: pd.DataFrame, 
    lda: LatentDirichletAllocation, 
    vectorizer: CountVectorizer, 
    n_top_words: int = 10, 
    n_examples: int = 3
) -> Dict[str, Dict[str, List[str]]]:
    """
    Creates meaningful descriptions for each cluster based on top words and example posts.

    Args:
        df (pd.DataFrame): The input DataFrame with posts.
        lda (LatentDirichletAllocation): Fitted LDA model.
        vectorizer (CountVectorizer): Fitted CountVectorizer.
        n_top_words (int): Number of top words to include in the description.
        n_examples (int): Number of example posts to include for each topic.

    Returns:
        Dict[str, Dict[str, List[str]]]: Dictionary containing top words and examples for each topic.
    """
    feature_names = vectorizer.get_feature_names_out()
    cluster_descriptions = {}

    for topic_idx, topic in enumerate(lda.components_):
        # Get top words
        top_words = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        
        # Get example posts
        topic_posts = df[df['dominant_topic'] == topic_idx + 1]  # 1-based indexing for topics
        example_posts = topic_posts['selftext'].head(n_examples).tolist()
        
        cluster_descriptions[f"Topic {topic_idx + 1}"] = {
            "Top Words": top_words,
            "Example Posts": example_posts
        }
    
    return cluster_descriptions

def get_trending_topic(
    df: pd.DataFrame, 
    lda: LatentDirichletAllocation, 
    vectorizer: CountVectorizer, 
    year: int, 
    month: int
) -> Tuple[int, List[str]]:
    """
    Identifies the trending topic for a specific month and year.

    Args:
        df (pd.DataFrame): The input DataFrame with posts.
        lda (LatentDirichletAllocation): Fitted LDA model.
        vectorizer (CountVectorizer): Fitted CountVectorizer.
        year (int): The year to filter by.
        month (int): The month to filter by.

    Returns:
        Tuple[int, List[str]]: Trending topic index and its top words.
    """
    feature_names = vectorizer.get_feature_names_out()

    # Filter posts by the specified month and year
    monthly_posts = df[(df['created_datetime'].dt.year == year) & 
                       (df['created_datetime'].dt.month == month)]
    
    if monthly_posts.empty:
        return None, ["No data available for this period."]
    
    # Count the number of posts for each topic
    topic_counts = monthly_posts['dominant_topic'].value_counts()

    # Get the most popular topic
    trending_topic = topic_counts.idxmax()

    # Get the top words for the trending topic
    top_words = [
        feature_names[i] 
        for i in lda.components_[trending_topic - 1].argsort()[:-11:-1]  # Adjust for 1-based indexing
    ]
    
    return trending_topic, top_words