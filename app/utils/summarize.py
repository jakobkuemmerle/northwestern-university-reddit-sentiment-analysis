from transformers import pipeline
import nltk
import matplotlib.pyplot as plt

def summarize_first_row(df):
    """
    Summarizes the most controversial (highest score) post from the DataFrame.
    """
    # Initialize the summarizer
    summarizer = pipeline("summarization", model="t5-large", tokenizer="t5-large")

    # Filter and sort DataFrame by score
    filtered_df = df[df['Text'].str.len() > 1000]
    sorted_df = filtered_df.sort_values(by='Score', ascending=False)

    # Check if DataFrame has valid data
    if sorted_df.empty:
        return "No suitable posts found for summarization."

    # Summarize the highest score post
    text_to_summarize = sorted_df.iloc[0]['Text']
    summary = summarizer(text_to_summarize, max_length=150, min_length=50, do_sample=False)
    return summary[0]['summary_text']

def sentiment_analysis_by_paragraph(df):
    """
    Performs sentiment analysis by paragraph on the most controversial post.
    """
    # Filter and sort DataFrame
    filtered_df = df[df['Text'].str.len() > 1000]
    sorted_df = filtered_df.sort_values(by='Score', ascending=False)

    # Check if DataFrame has valid data
    if sorted_df.empty:
        return None

    # Get text for analysis
    text = sorted_df.iloc[0]['Text']

    # Initialize emotion classification pipeline
    emotion_model = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

    # Chunk text into paragraphs
    nltk.download('punkt')
    paragraphs = nltk.sent_tokenize(text)

    # Perform emotion detection
    emotion_counts = {}
    for para in paragraphs:
        emotion = emotion_model(para)[0]['label']
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(emotion_counts.keys(), emotion_counts.values(), color='skyblue')
    ax.set_title("Sentiment Analysis by Paragraph")
    ax.set_xlabel("Emotion")
    ax.set_ylabel("Frequency")
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig