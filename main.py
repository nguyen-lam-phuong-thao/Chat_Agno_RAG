import os
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse

from agno.agent import Agent
from agno.models.google.gemini import Gemini
from qdrant_client import QdrantClient
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

BASE = Path(__file__).parent
load_dotenv()

app = FastAPI(title="Agno Medical RAG API")
google_api_key = os.getenv("GOOGLE_API_KEY")

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
        """
        Agno agent gọi method này, nhưng Qdrant không dùng filters.
        Trả về (None, []) để báo hợp lệ.
        """
        return None, []

    def search(self, query: str, max_results: int = None, **kwargs):
        if max_results is None:
            max_results = self.max_results

        # chuyển query sang vector
        vector = self.embedding.embed_query(query)

        # tìm kiếm trong Qdrant
        hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=vector,
            limit=max_results
        )

        # trả về payload như list document
        results = []
        for hit in hits:
            payload = hit.payload or {}
            results.append(payload)
        return results

# ---------------------------
# Kết nối Qdrant
# ---------------------------
client = QdrantClient(path="qdrant_data")
collection_name = "medical_diseases"
embeddings = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

knowledge_base = QdrantAdapter(client, collection_name, embeddings, max_results=5)

# ---------------------------
# Init Agno Agent
# ---------------------------
agent = Agent(
    model=Gemini(id="gemini-2.0-flash-001", api_key=google_api_key),
    knowledge=knowledge_base,
    description="Answer user questions from medical dataset",
    markdown=True,
    search_knowledge=True,
)

# --- Endpoints ---
@app.get("/ask")
def ask(question: str):
    result = agent.run(question)
    return {"question": question, "answer": str(result)}

@app.post("/chat_api")
async def chat_api(request: Request):
    data = await request.json()
    question = data.get("message", "")
    result = agent.run(question)
    return {"answer": str(result)}

@app.get("/", response_class=HTMLResponse)
def index():
    return BASE.joinpath("static/index.html").read_text(encoding="utf-8")

