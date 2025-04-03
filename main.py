from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from qdrant_client import QdrantClient
from typing import List
import os
from dotenv import load_dotenv
load_dotenv()
# Initialize FastAPI app with Swagger and Redoc documentation
app = FastAPI(
    title="Text Embedding API",
    description="An API to process batch text input->generate embeddings -> store embeddings.",
    version="1.0.0",
    
)

# Initialize remote Qdrant client
client = QdrantClient(
    url=os.getenv('Q_URL'),
    api_key=os.getenv('Q_API_KEY'),
    port=os.getenv('Q_PORT')
)

# Define request model for batch input
class EmbeddingRequest(BaseModel):
    user: str
    texts: List[str]

class QueryEmbedding(BaseModel):
    user:str
    query_texts:list[str]

# API endpoint to process batch text input
@app.post("/embed", summary="Generate embeddings from text batch for each user", tags=["Embeddings"])
def embed_texts(request: EmbeddingRequest):
    """
    Process a batch of texts generate and store their embeddings and return their ids.
    
    - **texts**: A list of input strings.
    - **Returns**: A list of ids
    """
    try:
        user_id = request.user
        texts = request.texts
        ids = client.add(collection_name=user_id,documents=texts)
        return ids
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/query",summary ="Query text",tags=["query"])
def query_embeds(request:QueryEmbedding):
    """
    Query a text similarity based on vector embeddings
    """
    try:
        user = request.user
        texts=request.query_texts
        embeddings = client.query_batch(collection_name=user,query_texts=texts)
        return embeddings
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))