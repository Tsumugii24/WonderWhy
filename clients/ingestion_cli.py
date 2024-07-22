import os
import requests
from loguru import logger

def vectorize_document(document_dir, kb_name, save_imgs):
    """
    Call the FastAPI endpoint to vectorize and store documents.

    Args:
        document_dir (str): The path to the directory containing the documents.
        kb_name (str): The name of the knowledge base.
        save_imgs(bool): Flag to indicate whether to save images.
    
    Returns:
        dict: The response from the API.
    """
    url = "http://127.0.0.1:8000/v1/kb/ingestion"  # Adjust the URL to your FastAPI server

    document_dir = os.path.abspath(document_dir)

    data = {
        "document_dir": document_dir,
        "kb_name": kb_name,
        "save_images": save_imgs,

    }
    logger.info(f"Sending request to {url} with data: {data}")
    response = requests.post(url, json=data)

    if response.status_code == 200:
        logger.info("Request successful.")
        return response.json()
    else:
        logger.error(f"Request failed with status code {response.status_code}: {response.text}")
        raise Exception(f"Failed to vectorize document: {response.text}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Client script to call FastAPI ingestion endpoint.")
    parser.add_argument("-d","--document_dir", type=str, help="The path to the directory containing the documents.")
    parser.add_argument("-k","--kb_name", type=str, help="The name of the knowledge base.")
    parser.add_argument("--save_imgs", action="store_true", help="Flag to indicate whether to save images.")

    args = parser.parse_args()
    
    # Validate the document path
    if not os.path.isdir(args.document_dir):
        raise ValueError(f"The provided document path '{args.document_dir}' is not a valid directory.")
    
    try:
        result = vectorize_document(args.document_dir, args.kb_name, args.save_imgs)
        print("Response from API:")
        print(result)
    except Exception as e:
        print(f"Error: {e}")
