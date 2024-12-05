import streamlit as st
import os
import matplotlib.pyplot as plt

from utils.pipeline import prepare_data_pipeline, sentiment_analysis_pipeline, topic_modeling_pipeline, trending_topic_pipeline

# Linda
from utils.clean_data import generate_similar_words

# Import your pipeline function
# from your_pipeline_module import pipeline  # Uncomment and replace with your actual import

# Function to list subreddit files
def list_subreddit_files(folder: str) -> list:
    """Lists subreddit files in the specified folder."""
    return [file for file in os.listdir(folder) if file.endswith('.zst')]

# Define the Streamlit app

# Title and subtitle
st.title("Reddit Analysis Dashboard for Northwestern")
st.subheader("Explore subreddit trends and insights related to NU")

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

# Add a slider for selecting the start and end year between 2000 and 2024
start_year, end_year = st.slider(
    "Select the year range for analysis",
    min_value=2000,
    max_value=2024,
    value=(2000, 2024),
    step=1,
)

# Analyze button
if st.button("Analyze"):
    if selected_subreddit and keyword:
        st.write(f"Analyzing {selected_subreddit} for keyword '{keyword}'...")
        
        # Call the pipeline function
        subreddit_path = os.path.join(folder_path, f"{selected_subreddit}_submissions.zst")

        try:
            # Assuming pipeline returns a list of plots
            file_path = 'downloads/reddit-downloads/Northwestern_submissions.zst'
            min_chars = 100
            
            # Linda
            # keywords = ["northwestern", "NU"]
            keywords = generate_similar_words(keyword)
            
            result_df, plot_fig = prepare_data_pipeline(subreddit_path, min_chars, keywords, start_year, end_year)
            st.pyplot(plot_fig)

            sentiment_results, sentiment_distribution, plot_fig2 = sentiment_analysis_pipeline(result_df)
            st.pyplot(plot_fig2)

            plot_fig3, plot_fig4 = topic_modeling_pipeline(result_df, n_topics=5, max_features=5000, n_top_words=10)
            st.pyplot(plot_fig3)
            st.pyplot(plot_fig4)

            trending_topic = trending_topic_pipeline(result_df, year=2020, month=9)
            st.write(trending_topic)

            # result_plots = pipeline(subreddit_path, keyword)
            
            # Display the plots
            #for plot in result_plots:
            #    st.pyplot(plot)

            st.write("Analysis completed. Try again with a new keyboard or check out the other features")

        except Exception as e:
            st.error(f"An error occurred during analysis: {e}")
    else:
        st.warning("Please select a subreddit and enter a keyword to proceed.")