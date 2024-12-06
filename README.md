# northwestern-university-reddit-sentiment-analysis
- A project by Jakob Kuemmerle, Linda Liang, Cailey Farrell
## Use Cases
### 1. Analyze Northwestern-related Reddit Posts and Trends
- Includes analysis of historic/deleted posts
- displays number of posts over time for given keyword
- classfies sentiment of posts over time (positive/negative/neutral)
- finds meaningful topic cluster and analyzes them over time
- displays spikes of topics

### 2. Find the most controversial Reddit posts of a given keyword and analyze in detail
- on live data of the reddit API
- lets the user access posts of a given subreddit and keyword
- finds the post controversial post
- summarized the post
- analyzes the sentiment of each chunk of the text

## How to Use:
### Get Started
1. clone repo, cd into repo & pull latest changes
2. run "make" in terminal
3. run "poetry install"
4. poetry run streamlit run app/dashboard.py
5. Try it out!

### historical Reddit Data
- you need to add the data files from the website:
- https://the-eye.eu/redarcs/
- save them in the folder "downloads/reddit-downloads"
- Download https://github.com/chapmanjacobd/reddit_mining/blob/main/top_text_subreddits.csv
- save under downloads/subreddit-list

### ENVIRONMENT VAR
- add the following to your .eve file:

OPENAI_API_KEY= ""

REDDIT_CLIENT_ID = ""

REDDIT_CLIENT_SECRET = ""

### Troubleshoot nltk-data
- if you run into any issue realted to nltk_data you should delete the nlt_data folder under your user and run streamlit again
- issue are due to a new version of the tokenizers