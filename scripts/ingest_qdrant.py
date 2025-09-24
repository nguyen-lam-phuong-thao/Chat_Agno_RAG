# scripts/ingest_qdrant.py
import json
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

# === CONFIG ===
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "medical_diseases"
DOCUMENTS_FILE = Path("data/documents.jsonl")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def connect_qdrant():
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

def create_collection_if_not_exists(client: QdrantClient, size: int):
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=size, distance=Distance.COSINE)
        )
        print(f"Created collection: {COLLECTION_NAME} with dim={size}")
    else:
        print(f"Collection {COLLECTION_NAME} already exists")

def load_documents(file_path: Path):
    documents = []
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            documents.append(json.loads(line))
    print(f"Loaded {len(documents)} documents from {file_path}")
    return documents

def embed_and_upsert(docs, client, model, batch_size=64):
    vectors = model.encode([doc["page_content"] for doc in docs], show_progress_bar=True)
    total = len(docs)
    for start in range(0, total, batch_size):
        end = start + batch_size
        batch_docs = docs[start:end]
        batch_vectors = vectors[start:end]

        points = []
        for idx, (doc, vector) in enumerate(zip(batch_docs, batch_vectors), start=start):
            points.append(
                PointStruct(
                    id=idx,
                    vector=vector.tolist(),
                    payload=doc["metadata"] | {"page_content": doc["page_content"]}
                )
            )

        client.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"Upserted {len(points)} documents (batch {start}–{end})")

if __name__ == "__main__":
    # Step 1: Connect
    client = connect_qdrant()
    print("Connected to Qdrant")

    # Step 2: Load embedding model
    model = SentenceTransformer(EMBEDDING_MODEL)
    vector_size = model.get_sentence_embedding_dimension()

    # Step 3: Ensure collection
    create_collection_if_not_exists(client, vector_size)

    # Step 4: Load documents
    docs = load_documents(DOCUMENTS_FILE)

    # Step 5: Embed + upsert
    embed_and_upsert(docs, client, model)
