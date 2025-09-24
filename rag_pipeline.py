import os
from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.google.gemini import Gemini
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http.exceptions import UnexpectedResponse
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

load_dotenv()

COLLECTION_NAME = "medical_diseases"
client = QdrantClient(path="qdrant_data")
embeddings = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ensure collection tồn tại
try:
    client.get_collection(collection_name=COLLECTION_NAME)
except (UnexpectedResponse, ValueError):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    )

# ---------------------------
# Adapter Qdrant -> Agent
# ---------------------------
class QdrantAdapter:
    def __init__(self, client: QdrantClient, collection_name: str, embedding, max_results: int = 5):
        self.client = client
        self.collection_name = collection_name
        self.embedding = embedding
        self.max_results = max_results

    def validate_filters(self, filters):
        return None, []

    def search(self, query: str, max_results: int = None, **kwargs):
        if max_results is None:
            max_results = self.max_results

        vector = self.embedding.embed_query(query)
        hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=vector,
            limit=max_results
        )
        return [hit.payload or {} for hit in hits]

knowledge_base = QdrantAdapter(client, COLLECTION_NAME, embeddings, max_results=5)

# ---------------------------
# Init Agent
# ---------------------------
agent = Agent(
    model=Gemini(id="gemini-2.0-flash-001", api_key=os.getenv("GOOGLE_API_KEY")),
    knowledge=knowledge_base,
    description="Answer user questions based on medical knowledge base",
    markdown=True,
    search_knowledge=True,
)
