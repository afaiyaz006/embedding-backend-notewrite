import requests
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load HuggingFace API key from .env
load_dotenv()
HF_API_KEY = os.getenv('HF_API_KEY')

HF_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,       # Max tokens/chars per chunk
    chunk_overlap=50,     # Overlap between chunks
)

def get_embedding(text: str):
    url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{HF_MODEL_NAME}"
    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {"inputs": text}

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code != 200:
        raise Exception(f"HuggingFace API Error: {response.text}")

    embedding = response.json()
    return embedding

def split_and_embed_text(input_texts: list[str]):
    """
    Splits input text into chunks and generates embeddings for each chunk.
    
    Returns:
        chunks (List[str]): List of text chunks.
        embeddings (List[List[float]]): Corresponding embeddings.
    """
   
    chunks=[]
    for text in input_texts:
        chunks.extend(text_splitter.split_text(text))

    embeddings = []
    for chunk in chunks:
        emb = get_embedding(chunk)
        embeddings.append(emb)
    return chunks, embeddings

# Example usage
if __name__ == "__main__":
    """
    Testing code
    """
    input_texts = ["""A quick brown fox jumps over the lazy dog. 
    Then it runs into the forest where it meets other animals. 
    They play together all afternoon before returning home at sunset."""]

    chunks, embeddings = split_and_embed_text(input_texts)

    print(f"Split into {len(chunks)} chunks.")
    for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        print(f"\nChunk {idx+1}: {chunk}")
        print(f"Embedding size: {len(embedding[0]) if isinstance(embedding[0], list) else len(embedding)}")
