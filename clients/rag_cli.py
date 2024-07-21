import os
import requests
from loguru import logger

def query_knowledge_base(query, kb_name):
    """
    Call the FastAPI endpoint to query the knowledge base.

    Args:
        query (str): The query to send to the knowledge base.
        kb_name (str): The name of the knowledge base.
    
    Returns:
        dict: The response from the API.
    """
    url = "http://127.0.0.1:8000/v1/kb/chat_kb"  # Adjust the URL to your FastAPI server

    data = {
        "query": query,
        "kb_name": kb_name
    }
    logger.info(f"Sending request to {url} with data: {data}")
    response = requests.post(url, json=data)

    if response.status_code == 200:
        logger.info("Request successful.")
        return response.json()
    else:
        logger.error(f"Request failed with status code {response.status_code}: {response.text}")
        raise Exception(f"Failed to query knowledge base: {response.text}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Client script to call FastAPI chat_kb endpoint.")
    parser.add_argument("-q", "--query", type=str, required=True, help="The query to send to the knowledge base.")
    parser.add_argument("-k", "--kb_name", type=str, required=True, help="The name of the knowledge base.")

    args = parser.parse_args()

    try:
        result = query_knowledge_base(args.query, args.kb_name)
        print("Response from API:")
        print(result)
    except Exception as e:
        print(f"Error: {e}")
