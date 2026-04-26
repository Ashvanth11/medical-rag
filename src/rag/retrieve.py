import os

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

COLLECTION_NAME = "medical_guidelines"
MODEL_NAME = "BAAI/bge-small-en-v1.5"


client = QdrantClient(host=os.environ.get("QDRANT_HOST", "localhost"), port=6333)
model = SentenceTransformer(MODEL_NAME)


def retrieve_top_k(query: str, top_k: int = 5):
    qvec = model.encode(query, normalize_embeddings=True)

    res = client.query_points(
        collection_name=COLLECTION_NAME,
        query=qvec.tolist(),
        limit=top_k,
    )

    results = res.points

    chunks = []
    scores = []

    for r in results:
        payload = r.payload or {}
        chunks.append({
            "title": payload.get("title"),
            "url": payload.get("url"),
            "text": payload.get("text", "")
        })
        scores.append(r.score)

    return chunks, scores
