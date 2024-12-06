import streamlit as st
import os

from utils.pipeline import prepare_data_pipeline, sentiment_analysis_pipeline, topic_modeling_pipeline, trending_topic_pipeline
from utils.clean_data import generate_similar_words

# Function to list subreddit files
def list_subreddit_files(folder: str) -> list:
    """Lists subreddit files in the specified folder."""
    return [file for file in os.listdir(folder) if file.endswith('.zst')]

# Initialize session state for role selection
if "role" not in st.session_state:
    st.session_state["role"] = "user"  # Default role is 'user'

# Sidebar for role selection
st.sidebar.title("Role Selection")
previous_role = st.session_state["role"]  # Store the previous role
selected_role = st.sidebar.selectbox(
    "Choose your role:",
    options=["user", "special"],
    index=["user", "special"].index(st.session_state["role"]),  # Use the current role as default
)

st.sidebar.image("assets/images/nu.jpeg", caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto", use_container_width=False)

# Update the role and trigger rerun if it changes
if selected_role != previous_role:
    st.session_state["role"] = selected_role
    st.rerun()

# Title and subtitle
st.title("Trend Analysis Dashboard for Northwestern")
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
    if selected_subreddit:
        st.write(f"Analyzing {selected_subreddit} for keyword '{keyword if keyword else 'all keywords'}'...")
        
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

            st.write("Analysis completed. Try again with a new keyword or subreddit.")

        except Exception as e:
            st.error(f"An error occurred during analysis: {e}")
    else:
        st.warning("Please select a subreddit to proceed.")