
import pandas as pd
import qdrant_client
from qdrant_client.http import models
import cohere
from rank_bm25 import BM25Okapi
import numpy as np
import json
import os
import uuid
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

# 1. Configuration
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

# Robust path handling
SCRIPT_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = SCRIPT_DIR / "data" / "founders.csv"
COLLECTION_NAME = "founders"
BM25_INDEX_PATH = SCRIPT_DIR / "data" / "bm25_index.json"


# Initialize clients
co = cohere.Client(COHERE_API_KEY)
qdrant_client = qdrant_client.QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=60.0  # Set a longer timeout to 60 seconds
)

def create_qdrant_collection():
    """Create the Qdrant collection if it doesn't exist."""
    try:
        qdrant_client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=1024,  # Cohere's default embedding size for embed-english-v3.0
                distance=models.Distance.COSINE
            )
        )
        print(f"Collection '{COLLECTION_NAME}' created.")
    except Exception as e:
        # A more specific exception might be better, but this is fine for a demo
        print(f"Collection already exists or an error occurred: {e}")

def load_and_prepare_data():
    """Load data from CSV and create the text for embedding."""
    if not CSV_PATH.exists():
        print(f"Error: CSV file not found at {CSV_PATH}")
        return None
        
    df = pd.read_csv(CSV_PATH)
    # Use a consistent ID based on row index if 'id' is not present
    if 'id' not in df.columns:
        df['id'] = [str(uuid.uuid4()) for _ in range(len(df))]
    else:
        df['id'] = df['id'].apply(lambda x: str(x) if pd.notna(x) else str(uuid.uuid4()))
        
    df = df.fillna('')

    # Concatenate fields for a rich text chunk
    df['chunk'] = df.apply(
        lambda row: f"{row.get('founder_name', '')}, {row.get('role', '')} at {row.get('company', '')}, {row.get('location', '')}. Idea: {row.get('idea', '')}. Bio: {row.get('about', '')}. Keywords: {row.get('keywords', '')}.",
        axis=1
    )
    return df

def create_and_save_bm25_index(df):
    """Create and save the BM25 index."""
    if df is None:
        return
    tokenized_corpus = [doc.split(" ") for doc in df['chunk']]
    bm25 = BM25Okapi(tokenized_corpus)
    
    print("BM25 index created. It will be recreated on app startup.")
    # In a production scenario, you would pickle and save the bm25 object
    # with open(BM25_INDEX_PATH, 'wb') as f:
    #     pickle.dump(bm25, f)

def embed_and_upsert(df):
    """Embed data using Cohere and upsert to Qdrant."""
    if df is None:
        return
        
    batch_size = 32  # Reduced batch size for stability
    total_rows = len(df)

    for i in range(0, total_rows, batch_size):
        batch_df = df.iloc[i:i + batch_size]
        texts = batch_df['chunk'].tolist()
        
        print(f"Embedding batch {i//batch_size + 1}/{(total_rows + batch_size - 1)//batch_size}...")
        response = co.embed(
            texts=texts,
            model="embed-english-v3.0",
            input_type="search_document"
        )
        embeddings = response.embeddings

        # Prepare points for Qdrant
        points = []
        for idx, row in batch_df.iterrows():
            # Get the corresponding embedding
            embedding_index = idx - i
            embedding = embeddings[embedding_index]
            
            payload = row.to_dict()
            # Ensure payload values are JSON serializable
            for key, value in payload.items():
                if isinstance(value, (np.int64, np.float64)):
                    payload[key] = str(value) # or int(value), float(value)
                elif pd.isna(value):
                    payload[key] = None

            points.append(
                models.PointStruct(
                    id=row['id'],
                    vector=embedding,
                    payload=payload
                )
            )
        
        # Upsert points to Qdrant
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
            wait=True
        )
        print(f"Upserted {len(points)} points to Qdrant.")

def main():
    """Main function to run the ingestion process."""
    print("Starting data ingestion process...")
    create_qdrant_collection()
    df = load_and_prepare_data()
    create_and_save_bm25_index(df)
    embed_and_upsert(df)
    print("Data ingestion complete.")

if __name__ == "__main__":
    main()
