import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Function to plot the number of posts per year
def plot_posts_per_year(df: pd.DataFrame):
    posts_per_year = df['year'].value_counts().sort_index()

    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=posts_per_year.index, y=posts_per_year.values, color='skyblue', ax=ax)
    ax.set_title('Number of Posts per Year', fontsize=16)
    ax.set_xlabel('Year', fontsize=14)
    ax.set_ylabel('Number of Posts', fontsize=14)
    ax.set_xticklabels(ax.get_xticks(), rotation=45)
    ax.grid(True)

    # Tight layout ensures everything fits in the plot
    fig.tight_layout()
    return fig

def plot_sentiment_distribution(sentiment_distribution: pd.DataFrame):
    """
    Plots the sentiment distribution over time.

    Args:
        sentiment_distribution (pd.DataFrame): DataFrame with sentiment, year, and percentage columns.

    Returns:
        matplotlib.figure.Figure: The generated plot figure.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(
        data=sentiment_distribution,
        x='year', y='percentage', hue='sentiment', marker='o', ax=ax
    )
    ax.set_title('Sentiment Distribution Over Time', fontsize=16)
    ax.set_xlabel('Year', fontsize=14)
    ax.set_ylabel('Percentage of Posts', fontsize=14)
    ax.legend(title='Sentiment', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    fig.tight_layout()
    return fig

def plot_trends(topic_trends: pd.DataFrame) -> plt.Figure:
    """
    Plots topic trends over time.

    Args:
        topic_trends (pd.DataFrame): Trends of topics over time.

    Returns:
        matplotlib.figure.Figure: The generated plot figure.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    topic_trends.plot(kind='line', marker='o', colormap='tab10', ax=ax)
    ax.set_title('Topic Trends Over Time', fontsize=16)
    ax.set_xlabel('Year', fontsize=14)
    ax.set_ylabel('Number of Posts', fontsize=14)
    ax.legend(title='Topics', fontsize=12)
    ax.grid(True)
    fig.tight_layout()
    return fig

def plot_spikes(spikes: pd.DataFrame) -> plt.Figure:
    """
    Plots a heatmap of spikes in topic popularity.

    Args:
        spikes (pd.DataFrame): DataFrame of percentage changes (spikes).

    Returns:
        matplotlib.figure.Figure: The generated plot figure.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(spikes, cmap="coolwarm", annot=True, fmt=".2f", linewidths=0.5, ax=ax)
    ax.set_title("Spikes in Topic Popularity (Percentage Change)", fontsize=16)
    ax.set_xlabel("Topic", fontsize=14)
    ax.set_ylabel("Year", fontsize=14)
    fig.tight_layout()
    return fig
