from uuid import uuid4
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from qdrant_client import QdrantClient
from typing import List
import os
from hf_api import split_and_embed_text
from dotenv import load_dotenv
from pathlib import Path
from qdrant_client.models import PointStruct
from qdrant_client.models import VectorParams, Distance
print(os.getcwd())
load_dotenv(dotenv_path=".env", override=True)
# Initialize FastAPI app with Swagger and Redoc documentation
app = FastAPI(
    title="Text Embedding API",
    description="An API to process batch text input->generate embeddings -> store embeddings.",
    version="1.0.0",
    
)
vector_size=384
# Initialize remote Qdrant client
client = QdrantClient(
    url=os.getenv('Q_URL'),
    api_key=os.getenv('Q_API_KEY'),
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


@app.post('/hf-embed', summary="Generate embeddings from text and store vectors", tags=["Embeddings"])
def process_and_embed_text(request: EmbeddingRequest):
    """
    Generate embeddings using Hugging Face API, then store vectors in Qdrant.

    - **texts**: List of input texts
    - **Returns**: List of vector IDs
    """

    try:
        user_id = request.user
        texts = request.texts

        chunks,embeddings = split_and_embed_text(texts)  # Should return a list of vectors

        vectors_to_store = []
        ids = []
        if  not client.collection_exists(user_id):
            client.create_collection(user_id,vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE))
        for chunk,emb in zip(chunks,embeddings):
            vector_id = str(uuid4())
            
            point = PointStruct(
                id=vector_id,
                vector=emb,
                payload={"user_id": user_id,"content":chunk}
            )
            vectors_to_store.append(point)
            ids.append(vector_id)

        print("Inserting embeddings into vector store...")

        client.upsert(
            collection_name=user_id,
            points=vectors_to_store
        )

        return {"ids": ids}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/hf-query", summary="Query text", tags=["Query"])
def query_embeds(request: QueryEmbedding):
    """
    Query similar vectors from Qdrant based on input texts.
    Returns the most relevant payloads.
    """
    try:
        user = request.user
        query_texts = request.query_texts  # List of query texts
        # print(query_texts)
        chunks,query_embeddings = split_and_embed_text(query_texts)

        all_results = []

        for query_vector in query_embeddings:
            search_result = client.search(
                collection_name=user,
                query_vector=query_vector,
                limit=10,  # Number of similar vectors to fetch
                with_payload=True,  # VERY IMPORTANT: ask Qdrant to return payloads
            )

            # Extract relevant payloads
            hits = []
            for point in search_result:
                payload = point.payload  # This contains your stored metadata
                score = point.score      # Similarity score
                hits.append({
                    "payload": payload,
                    "score": score
                })

            all_results.append(hits)
        # print(all_results)
        relevant_chunks=[]
        for result in all_results[0]:
            relevant_chunks.append(result['payload']['content'])

        return relevant_chunks

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
