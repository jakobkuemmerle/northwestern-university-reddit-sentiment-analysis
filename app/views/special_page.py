import streamlit as st
import os
from transformers import pipeline
import nltk

from utils.pipeline import prepare_data_pipeline, sentiment_analysis_pipeline, topic_modeling_pipeline, trending_topic_pipeline, summarize_pipeline
from utils.clean_data import generate_similar_words
from utils.summarize import summarize_first_row
from utils.read_data import get_api_data

def sentiment_analysis_by_paragraph_streamlit(df):
    """
    Perform sentiment analysis on the highest-scored post and display it color-coded in Streamlit.
    """
    # Filter rows where length of Text > 1000
    filtered_df = df[df['Text'].str.len() > 1000]

    # Sort the DataFrame by Score in descending order
    sorted_df = filtered_df.sort_values(by='Score', ascending=False)

    # Ensure there's at least one valid row
    if sorted_df.empty:
        st.warning("No posts with sufficient length for sentiment analysis.")
        return

    # Get the Text from the highest-scored post
    text = sorted_df.iloc[0]['Text']
    
    # Load the pre-trained emotion classification model
    emotion_model = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

    # Define color mapping for emotions
    emotion_colors = {
        'anger': 'red',
        'disgust': 'green',
        'fear': 'orange',
        'joy': 'yellow',
        'love': 'pink',
        'sadness': 'blue',
        'surprise': 'purple',
        'neutral': 'gray'
    }

    # Chunk the text into paragraphs
    nltk.download('punkt')
    paragraphs = text.split('\n')

    # Remove empty paragraphs (if any)
    paragraphs = [p for p in paragraphs if p.strip()]

    # Perform emotion detection on each paragraph
    chunk_emotions = []
    for chunk in paragraphs:
        emotions = emotion_model(chunk)
        chunk_emotions.append((chunk, emotions[0]['label']))  # Save the paragraph and its emotion label

    # Generate HTML with highlighted paragraphs
    html_output = "<div style='font-family: Arial, sans-serif;'>"
    for idx, (para, emotion) in enumerate(chunk_emotions):
        color = emotion_colors.get(emotion.lower(), 'gray')  # Default to gray if emotion not found
        html_output += f'<p style="color:{color};"><b>Paragraph {idx+1} ({emotion}):</b><br>{para}</p>'
    html_output += "</div>"

    # Display the HTML content in Streamlit
    st.markdown(html_output, unsafe_allow_html=True)


# Function to list subreddit files
def list_subreddit_files(folder: str) -> list:
    """Lists subreddit files in the specified folder."""
    return [file for file in os.listdir(folder) if file.endswith('.zst')]

# Define the Streamlit app

# Title and subtitle
st.title("Special Analysis Dashboard for Staff")
st.subheader("Summarize the most controversial posts of a given Category")

# Get the list of subreddit files
folder_path = "downloads/reddit-downloads"
subreddits = list_subreddit_files(folder_path)

# Extract the part before the first underscore for each file
subreddits = [file.split('_')[0] for file in subreddits]


# Dropdown to select subreddit
selected_subreddit = st.selectbox(
    "Select a subreddit to analyze:",
    options=subreddits,
    help="Choose the subreddit data file you want to analyze."
)

# Text input for keyword
keyword = st.text_input(
    "Enter a keyword to analyze:",
    placeholder="Enter a keyword (e.g., 'admission')"
)

if st.button("Find and Summarize"):
    if selected_subreddit and keyword:
        st.write(f"Analyzing subreddit '{selected_subreddit}' for keyword '{keyword}'...")

        # Define folder path
        folder_path = "downloads/reddit-downloads"

        # Generate file path
        subreddit_path = os.path.join(folder_path, f"{selected_subreddit}_submissions.zst")

        try:
            # Call the pipeline function
            df = get_api_data(selected_subreddit, keyword, limit=1000)

            # Summarize and analyze
            summary = summarize_first_row(df)
            st.subheader("Summarization of the Most Controversial Post:")
            st.write(summary)

            st.subheader("Sentiment Analysis by Paragraph (Color-Coded):")
            sentiment_analysis_by_paragraph_streamlit(df)

        except Exception as e:
            st.error(f"An error occurred during analysis: {e}")
    else:
        st.warning("Please enter both a subreddit and a keyword to proceed.")
