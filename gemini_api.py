import requests
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from google import genai
from google.genai import types

# Load HuggingFace API key from .env
load_dotenv()
HF_API_KEY = os.getenv('HF_API_KEY')
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HF_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,       # Max tokens/chars per chunk
    chunk_overlap=100,     # Overlap between chunks
)
gemini_client = genai.Client(api_key=GEMINI_API_KEY)
# def get_embedding(text: str):
#     url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{HF_MODEL_NAME}"
#     headers = {
#         "Authorization": f"Bearer {HF_API_KEY}",
#         "Content-Type": "application/json"
#     }
#     payload = {"inputs": text}

#     response = requests.post(url, headers=headers, json=payload)

#     if response.status_code != 200:
#         raise Exception(f"HuggingFace API Error: {response.text}")

#     embedding = response.json()
#     return embedding
def get_embedding(text:str):
    result = gemini_client.models.embed_content(
        model = "text-embedding-004",
        contents=text,
        config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
    )
    
    return result.embeddings[0].values
def split_and_embed_text(input_texts: list[str]):
    """
    Splits input text into chunks and generates embeddings for each chunk.
    
    Returns:
        chunks (List[str]): List of text chunks.
        embeddings (List[List[float]]): Corresponding embeddings.
    """
   
    chunks=[]
    total_texts=""
    for text in input_texts:
        total_texts+=text
    chunks.extend(text_splitter.split_text(total_texts))

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
    input_texts = ["""A quick brown fox darted through the tall grass, its sleek body cutting through the early morning mist like an arrow. It leapt gracefully over a sleeping dog, careful not to disturb its peaceful slumber, and continued on its way toward the edge of the dense forest. There, the trees stood like ancient sentinels, their branches whispering secrets to the wind. As the fox ventured deeper, sunlight filtered through the canopy in golden rays, illuminating patches of wildflowers and mossy stones. Soon, the fox encountered a gathering of animals—rabbits, deer, squirrels, and birds—all drawn to a small clearing where laughter and joy echoed beneath the boughs. They played games of chase and hide-and-seek, their movements a blur of fur and feathers. Hours passed in a blur of happiness until the sky blushed with the hues of dusk. One by one, the animals said their goodbyes, returning to their burrows and nests, while the fox, content and tired, padded quietly home under the soft light of the setting sun."""]

    chunks, embeddings = split_and_embed_text(input_texts)

    print(f"Split into {len(chunks)} chunks.")
    for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        print(f"\nChunk {idx+1}: {chunk}")
        print(f"Embedding size: {len(embedding[0]) if isinstance(embedding[0], list) else len(embedding)}")
