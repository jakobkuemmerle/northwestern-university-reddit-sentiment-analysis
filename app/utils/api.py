import openai
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key = openai.api_key)

def generate_summary_for_topics(topics: dict) -> dict:
    """
    Generates one-word summaries for the given topics using OpenAI API (GPT-4).
    
    Args:
        topics (dict): Dictionary with topic names and their top words.
    
    Returns:
        dict: Dictionary with topic names and their one-word summaries.
    """
    # Create a prompt with all the topics and their words
    prompt = "For each of the following topics, summarize the key idea in one word:\n\n"
    for topic, words in topics.items():
        prompt += f"{topic}: {', '.join(words)}\n"
    
    try:
        # Make the API call to OpenAI (using GPT-4)
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=50,  # Limit response length for better formatting
            temperature=0.5  # Adjust creativity
        )
        
        # Extract the response
        summary = response.choices[0].message.content
        
        # Parse the summary into a dictionary
        summarized_topics = {}
        topic_summaries = summary.split("\n")
        for line in topic_summaries:
            if ':' in line:
                topic_name, topic_summary = line.split(":", 1)
                summarized_topics[topic_name.strip()] = topic_summary.strip()
        
        return summarized_topics

    except Exception as e:
        print(f"Error generating summary: {e}")
        return {}

