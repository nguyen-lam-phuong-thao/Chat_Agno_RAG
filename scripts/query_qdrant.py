# scripts/query_qdrant.py
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# === CONFIG ===
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "medical_diseases"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def main():
    # Step 1: Connect
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    print("Connected to Qdrant")

    # Step 2: Load model
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Step 3: Encode query
    query_text = "What are the symptoms of Fungal infection?"
    query_vector = model.encode(query_text).tolist()

    # Step 4: Search
    hits = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=3  # top 3 results
    )

    # Step 5: Print results
    print("\n--- QUERY ---")
    print(query_text)
    print("\n--- RESULTS ---")
    for i, hit in enumerate(hits.points, start=1):
        payload = hit.payload
        print(f"\nResult {i} (score={hit.score:.4f}):")
        print(payload.get("page_content", "No content"))

if __name__ == "__main__":
    main()
