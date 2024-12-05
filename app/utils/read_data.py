import zstandard
import os
import json
import pandas as pd
import logging
from datetime import datetime
import re
import praw
from dotenv import load_dotenv


# Set up logging
log = logging.getLogger("reddit_analysis")
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler())


# Function to decode zstandard files
def read_and_decode(reader, chunk_size, max_window_size, previous_chunk=None, bytes_read=0):
    chunk = reader.read(chunk_size)
    bytes_read += chunk_size
    if previous_chunk is not None:
        chunk = previous_chunk + chunk
    try:
        return chunk.decode()
    except UnicodeDecodeError:
        if bytes_read > max_window_size:
            raise UnicodeError(f"Unable to decode frame after reading {bytes_read:,} bytes")
        log.info(f"Decoding error with {bytes_read:,} bytes, reading another chunk")
        return read_and_decode(reader, chunk_size, max_window_size, chunk, bytes_read)
    
# Function to read lines from a zst file
def read_lines_zst(file_name):
    with open(file_name, 'rb') as file_handle:
        buffer = ''
        reader = zstandard.ZstdDecompressor(max_window_size=2**31).stream_reader(file_handle)
        while True:
            chunk = read_and_decode(reader, 2**27, (2**29) * 2)
            if not chunk:
                break
            lines = (buffer + chunk).split("\n")
            for line in lines[:-1]:
                yield line, file_handle.tell()
            buffer = lines[-1]
        reader.close()


# Function to process the raw file into a DataFrame
def load_reddit_data(file_path):
    log.info(f"Loading data from: {file_path}")
    file_size = os.stat(file_path).st_size
    data = []
    bad_lines = 0
    for line, _ in read_lines_zst(file_path):
        try:
            obj = json.loads(line)
            obj['created_datetime'] = datetime.utcfromtimestamp(int(obj['created_utc']))
            data.append(obj)
        except (KeyError, json.JSONDecodeError) as err:
            bad_lines += 1
    log.info(f"Data loading complete with {len(data):,} rows and {bad_lines:,} bad lines.")
    
    df = pd.DataFrame(data)
    if 'northwestern' not in file_path.lower():
        NU_keywords = r'\b(Northwestern|NU|wildcat|wildcats)\b'  # Regular expression for the keywords
        df = df[df['selftext'].str.contains(NU_keywords, flags=re.IGNORECASE, na=False)]

    return df

def get_api_data(subreddit_name, search_keyword, limit=1000):
    """
    Function to fetch Reddit posts from a given subreddit based on a search keyword.

    Parameters:
    - subreddit_name (str): The name of the subreddit to search in.
    - search_keyword (str): The keyword to search for in the subreddit.
    - limit (int, optional): The maximum number of posts to fetch. Default is 1000.

    Returns:
    - pd.DataFrame: A DataFrame containing post data (Title, Score, URL, Created, Subreddit, Text).
    """
    # Load environment variables
    load_dotenv()

    # Set up Reddit API client
    reddit = praw.Reddit(
        client_id=os.environ.get("REDDIT_CLIENT_ID"),
        client_secret=os.environ.get("REDDIT_CLIENT_SECRET"),
        user_agent='your_user_agent'
    )

    # Search for posts in the specified subreddit with the given keyword
    posts = reddit.subreddit(subreddit_name).search(search_keyword, sort='relevance', limit=limit)

    # Create an empty list to store post data
    post_data = []

    # Extract relevant data from each post
    for post in posts:
        post_data.append({
            'Title': post.title,
            'Score': post.score,
            'URL': post.url,
            'Created': post.created_utc,
            'Subreddit': post.subreddit.display_name,
            'Text': post.selftext
        })

    # Convert the list of post data to a DataFrame
    return pd.DataFrame(post_data)