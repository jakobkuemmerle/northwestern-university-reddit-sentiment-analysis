import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set seaborn style for the new background color
sns.set_theme(style="whitegrid", context="talk")

# Function to plot the number of posts per year
def plot_posts_per_year(df: pd.DataFrame):
    posts_per_year = df['year'].value_counts().sort_index()

    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        x=posts_per_year.index, 
        y=posts_per_year.values, 
        color='#4CAF50',  # Greenish color
        ax=ax
    )
    ax.set_title('Number of Posts per Year', fontsize=18, color='white')
    ax.set_xlabel('Year', fontsize=14, color='white')
    ax.set_ylabel('Number of Posts', fontsize=14, color='white')
    ax.set_xticklabels(posts_per_year.index, rotation=45, color='white')
    ax.set_yticks(range(0, int(posts_per_year.max() + 1), max(1, int(posts_per_year.max() // 10))))
    ax.set_yticklabels(map(str, range(0, int(posts_per_year.max() + 1), max(1, int(posts_per_year.max() // 10)))), color='white')

    # Add labels to the bars
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=12, color='white', 
                    xytext=(0, 10), textcoords='offset points')

    ax.grid(color='gray', linestyle='--', linewidth=0.5)

    # Update background color
    fig.patch.set_facecolor('#0F2C4C')  
    ax.set_facecolor('#0F2C4C')  
    fig.tight_layout()
    return fig

def plot_sentiment_distribution(sentiment_distribution: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(
        data=sentiment_distribution,
        x='year', y='percentage', hue='sentiment', marker='o', palette='Set2', ax=ax
    )
    ax.set_title('Sentiment Distribution Over Time', fontsize=18, color='white')
    ax.set_xlabel('Year', fontsize=14, color='white')
    ax.set_ylabel('Percentage of Posts', fontsize=14, color='white')
    ax.legend(
        title='Sentiment', fontsize=12, title_fontsize=14, frameon=False, labelcolor='white'
    )
    ax.get_legend().get_title().set_color('white')  # Ensure legend title is white
    ax.tick_params(axis='x', rotation=45, colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.grid(color='gray', linestyle='--', linewidth=0.5)

    fig.patch.set_facecolor('#0F2C4C')  
    ax.set_facecolor('#0F2C4C')  
    fig.tight_layout()
    return fig

def plot_trends(topic_trends: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 6))
    topic_trends.plot(kind='line', marker='o', colormap='tab10', ax=ax)
    ax.set_title('Topic Trends Over Time', fontsize=18, color='white')
    ax.set_xlabel('Year', fontsize=14, color='white')
    ax.set_ylabel('Number of Posts', fontsize=14, color='white')
    ax.legend(
        title='Topics', fontsize=12, title_fontsize=14, frameon=False, labelcolor='white'
    )
    ax.get_legend().get_title().set_color('white')  # Ensure legend title is white
    ax.tick_params(axis='x', rotation=45, colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.grid(color='gray', linestyle='--', linewidth=0.5)

    fig.patch.set_facecolor('#0F2C4C')  
    ax.set_facecolor('#0F2C4C')  
    fig.tight_layout()
    return fig

def plot_spikes(spikes: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 6))
    heatmap = sns.heatmap(
        spikes, 
        cmap="coolwarm", 
        annot=True, 
        fmt=".2f", 
        linewidths=0.5, 
        ax=ax,
        annot_kws={"fontsize": 10, "color": "black"}
    )
    ax.set_title("Spikes in Topic Popularity (Percentage Change)", fontsize=18, color='white')
    ax.set_xlabel("Topic", fontsize=14, color='white')
    ax.set_ylabel("Year", fontsize=14, color='white')
    colorbar = heatmap.collections[0].colorbar
    colorbar.ax.yaxis.set_tick_params(color='white')  # Color for colorbar ticks
    plt.setp(colorbar.ax.yaxis.get_majorticklabels(), color='white')  # Color for tick labels

    fig.patch.set_facecolor('#0F2C4C')  
    ax.set_facecolor('#0F2C4C')  
    for text in ax.get_xticklabels() + ax.get_yticklabels():
        text.set_color('white')
    fig.tight_layout()
    return fig
