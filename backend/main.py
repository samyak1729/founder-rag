
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import cohere
import qdrant_client
from qdrant_client.http import models
from rank_bm25 import BM25Okapi
import pandas as pd
import os
from dotenv import load_dotenv
from pathlib import Path

# Configuration
load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
COLLECTION_NAME = "founders"

# --- Data and Model Loading --- 

# Initialize clients
co = cohere.Client(COHERE_API_KEY)
qdrant_client = qdrant_client.QdrantClient(
    url=QDRANT_URL, 
    api_key=QDRANT_API_KEY,
    timeout=30.0
)

# Load data for BM25
SCRIPT_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = SCRIPT_DIR / "data" / "founders.csv"
df = pd.read_csv(CSV_PATH)
df = df.fillna('')
df['chunk'] = df.apply(
    lambda row: f"{row.get('founder_name', '')}, {row.get('role', '')} at {row.get('company', '')}, {row.get('location', '')}. Idea: {row.get('idea', '')}. Bio: {row.get('about', '')}. Keywords: {row.get('keywords', '')}.",
    axis=1
)

# Create BM25 index from the documents
tokenized_corpus = [doc.split(" ") for doc in df['chunk']]
bm25 = BM25Okapi(tokenized_corpus)

# --- FastAPI Application ---

app = FastAPI(
    title="Founder RAG Chatbot API",
    description="API for searching startup founders using hybrid search."
)

class SearchQuery(BaseModel):
    query: str
    top_k: int = 5

@app.post("/search")
def search(search_query: SearchQuery):
    """Performs hybrid search (vector + keyword) over the dataset."""
    query = search_query.query
    top_k = search_query.top_k

    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    # 1. Vector Search (Semantic)
    try:
        query_embedding = co.embed(
            texts=[query],
            model="embed-english-v3.0",
            input_type="search_query"
        ).embeddings[0]

        vector_search_results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            limit=top_k,
            with_payload=True
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vector search failed: {e}")

    # 2. Keyword Search (BM25)
    tokenized_query = query.split(" ")
    bm25_scores = bm25.get_scores(tokenized_query)
    
    # Get top k results for BM25
    top_n_indices = bm25_scores.argsort()[-top_k:][::-1]
    bm25_results = []
    for idx in top_n_indices:
        if bm25_scores[idx] > 0:
            bm25_results.append({
                "score": bm25_scores[idx],
                "payload": df.iloc[idx].to_dict()
            })

    # 3. Hybrid Ranking (Reciprocal Rank Fusion)
    # A simple RRF implementation
    ranked_results = {}
    
    # Process vector search results
    for i, result in enumerate(vector_search_results):
        doc_id = result.payload['id']
        if doc_id not in ranked_results:
            ranked_results[doc_id] = {"score": 0, "payload": result.payload}
        ranked_results[doc_id]["score"] += 1 / (i + 1) # Rank-based score

    # Process BM25 results
    for i, result in enumerate(bm25_results):
        doc_id = result["payload"]['id']
        if doc_id not in ranked_results:
            ranked_results[doc_id] = {"score": 0, "payload": result["payload"]}
        ranked_results[doc_id]["score"] += 0.5 / (i + 1) # Lower weight for BM25

    # Sort by the combined score
    sorted_results = sorted(ranked_results.values(), key=lambda x: x["score"], reverse=True)

    # Limit to top_k and format the output
    final_results = [item["payload"] for item in sorted_results[:top_k]]

    if not final_results:
        return {"message": "No relevant results found."}

    return {"results": final_results}

@app.get("/")
def read_root():
    return {"message": "Welcome to the Founder RAG Chatbot API. Use the /docs endpoint to see the API documentation."}

# To run this app: uvicorn backend.main:app --reload
