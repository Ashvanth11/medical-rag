from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


# -------------------------
# Config
# -------------------------

CHUNKS_PATH = Path("data/docs/chunks/medical_chunks.jsonl")

COLLECTION_NAME = "medical_guidelines"
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
BATCH_SIZE = 32


# -------------------------
# Helpers
# -------------------------

def load_chunks() -> List[Dict[str, Any]]:
    chunks = []
    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


# -------------------------
# Main
# -------------------------

def main() -> None:
    print("Loading chunks...")
    chunks = load_chunks()
    texts = [c["text"] for c in chunks]

    print(f"Loaded {len(texts)} chunks")

    print("Loading embedding model...")
    model = SentenceTransformer(EMBED_MODEL_NAME)

    print("Embedding chunks...")
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    dim = embeddings.shape[1]
    print(f"Embedding dimension: {dim}")

    print("Connecting to Qdrant...")
    client = QdrantClient(host="localhost", port=6333)

    print("Creating collection (if not exists)...")
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=dim,
            distance=Distance.COSINE,
        ),
    )

    print("Upserting vectors...")
    points: List[PointStruct] = []

    for i, (vector, chunk) in enumerate(zip(embeddings, chunks)):
        payload = {
            "chunk_id": chunk["chunk_id"],
            "doc_id": chunk["doc_id"],
            "title": chunk["title"],
            "url": chunk["url"],
            "source": chunk["source"],
            "text": chunk["text"],
        }

        points.append(
            PointStruct(
                id=i,
                vector=vector.tolist(),
                payload=payload,
            )
        )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
    )

    print(f"Indexed {len(points)} chunks into Qdrant collection '{COLLECTION_NAME}'")
    print("Step 2 indexing complete.")


if __name__ == "__main__":
    main()
