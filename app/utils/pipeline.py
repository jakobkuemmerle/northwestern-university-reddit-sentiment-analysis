import logging
import pandas as pd
from utils.read_data import load_reddit_data
from utils.clean_data import filter_data, preprocess_text
from utils.plots import plot_posts_per_year, plot_sentiment_distribution, plot_trends, plot_spikes
from utils.analyze_clusters import perform_lda, display_topics, analyze_topics_over_time, detect_spikes, get_cluster_descriptions, get_trending_topic
from utils.analyze_sentiment import assign_sentiments, calculate_sentiment_distribution
from utils.api import generate_summary_for_topics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Set up logging
log = logging.getLogger("reddit_analysis")
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler())

# Main pipeline function
def prepare_data_pipeline(file_path, min_chars, keywords=None, start_year=None, end_year=None):
    log.info("Starting Reddit Data Analysis...")
    
    # Step 1: Load data
    df = load_reddit_data(file_path)
    
    # Step 2: Filter data
    filtered_df = filter_data(df, min_chars, keywords, start_year, end_year)
    
    # Step 3: Generate plot
    fig = plot_posts_per_year(filtered_df)
    
    return filtered_df, fig

def sentiment_analysis_pipeline(submissions_df):
    """
    Full pipeline to analyze sentiment and plot distribution over time.
    """
    # Step 1: Assign sentiment to posts
    submissions_with_sentiment = assign_sentiments(submissions_df)
    
    # Step 2: Calculate sentiment distribution
    sentiment_distribution = calculate_sentiment_distribution(submissions_with_sentiment)
    
    # Step 3: Plot the sentiment distribution
    fig = plot_sentiment_distribution(sentiment_distribution)

    return submissions_with_sentiment, sentiment_distribution, fig

def topic_modeling_pipeline(
    df: pd.DataFrame, 
    n_topics: int = 5, 
    max_features: int = 5000, 
    n_top_words: int = 10, 
    n_examples: int = 3
) -> None:
    """
    Executes the entire pipeline: preprocessing, topic modeling, trend analysis, and visualization.

    Args:
        df (pd.DataFrame): The input DataFrame with posts.
        n_topics (int): Number of topics for LDA.
        max_features (int): Maximum features for vectorization.
        n_top_words (int): Number of top words to display per topic.
        n_examples (int): Number of example posts to display for each topic.
    """
    # Preprocess text
    df['cleaned_text'] = df['selftext'].apply(preprocess_text)

    # Perform LDA
    lda, vectorizer, topic_assignments = perform_lda(df['cleaned_text'], n_topics, max_features)
    
    # Add topic assignments to DataFrame
    df['dominant_topic'] = topic_assignments.argmax(axis=1) + 1  # 1-based indexing
    
    # Display topics
    topics = display_topics(lda, vectorizer.get_feature_names_out(), n_top_words)
    summarized_topics = generate_summary_for_topics(topics)
    print("Identified Topics:")
    for topic, words in summarized_topics.items():
        print(f"{topic}: {words}")
    
    # Get cluster descriptions
    cluster_descriptions = get_cluster_descriptions(df, lda, vectorizer, n_top_words, n_examples)
    print("\nCluster Descriptions:")
    for topic, details in cluster_descriptions.items():
        print(f"{topic} - Top Words: {', '.join(details['Top Words'])}")
        for idx, example in enumerate(details['Example Posts']):
            print(f"Example {idx + 1}: {example[:200]}...")  # Print first 200 chars for readability

    # Analyze trends over time
    topic_trends = analyze_topics_over_time(df, topic_assignments)

    # Plot topic trends
    fig1 = plot_trends(topic_trends)

    # Detect and plot spikes
    overall_trends = df.groupby('year').size()
    spikes = detect_spikes(topic_trends, overall_trends)
    print("Detected Spikes in Topics:")
    print(spikes)
    fig2 = plot_spikes(spikes)

    return fig1, fig2

def trending_topic_pipeline(
    df: pd.DataFrame, 
    year: int, 
    month: int, 
    n_topics: int = 5, 
    max_features: int = 5000
) -> None:
    """
    Standalone pipeline to get the trending topic for a specific month and year.

    Args:
        df (pd.DataFrame): The input DataFrame with posts.
        year (int): The year to filter by.
        month (int): The month to filter by.
        n_topics (int): Number of topics for LDA model.
        max_features (int): Maximum number of features for the vectorizer.
    """
    # Vectorize the cleaned text data
    vectorizer = CountVectorizer(max_features=max_features, stop_words='english')
    text_data = df['cleaned_text'].values
    data_vectorized = vectorizer.fit_transform(text_data)

    # Train LDA model on the entire data
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(data_vectorized)

    # Get topic distributions for the posts
    topic_distribution = lda.transform(data_vectorized)

    # Assign the most likely topic to each post
    df['dominant_topic'] = topic_distribution.argmax(axis=1)

    # Call `get_trending_topic` for the specified month and year
    trending_topic, top_words = get_trending_topic(df, lda, vectorizer, year, month)

    if trending_topic is not None:
        print(f"Trending Topic for {year}-{month:02d}: Topic {trending_topic}")
        print(f"Top words for Topic {trending_topic}: {', '.join(top_words)}")
    else:
        print(f"No data available for {year}-{month:02d}.")

    return trending_topic