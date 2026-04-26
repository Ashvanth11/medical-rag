import sys
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

COLLECTION_NAME = "medical_guidelines"
MODEL_NAME = "BAAI/bge-small-en-v1.5"

def main():
    client = QdrantClient(host="localhost", port=6333)
    model = SentenceTransformer(MODEL_NAME)

    query = sys.argv[1] if len(sys.argv) > 1 else "What causes low blood sugar?"
    qvec = model.encode(query, normalize_embeddings=True)

    res = client.query_points(
        collection_name=COLLECTION_NAME,
        query=qvec.tolist(),
        limit=5,
    )
    results = res.points  # list of ScoredPoint

    print("QUESTION:", query)
    print()

    for i, r in enumerate(results, 1):
        payload = r.payload or {}
        print(f"{i}. score={r.score:.4f}")
        print(f"   title: {payload.get('title')}")
        print(f"   url: {payload.get('url')}")
        text = payload.get("text", "")
        print(f"   text: {text[:200]}...")
        print()

if __name__ == "__main__":
    main()
