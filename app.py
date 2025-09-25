import os
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from agno.agent import Agent
from agno.models.google import Gemini
from dotenv import load_dotenv
load_dotenv()

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "medical_diseases")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")  

qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
embedder = SentenceTransformer(EMBEDDING_MODEL)

def qdrant_retriever(
    query: str,
    agent: Optional[Agent] = None,
    num_documents: int = 5,
    **kwargs
) -> List[Dict[str, Any]]:
    vec = embedder.encode(query).tolist()
    hits = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=vec,
        limit=num_documents
    )
    refs: List[Dict[str, Any]] = []
    for h in getattr(hits, "points", []) or []:
        payload = h.payload or {}
        text = payload.get("page_content") or payload.get("text") or ""
        refs.append({
            "text": text,
            "score": float(getattr(h, "score", 0.0)),
            "metadata": {k: v for k, v in payload.items() if k != "page_content"}
        })
    return refs

SYSTEM_INSTRUCTIONS = (
    "You are a helpful medical info assistant. "
    "Only answer from provided context; if not enough info, say you don't know. "
    "Answer in the same language as the user."
)

agent = Agent(
    name="Medical RAG Agent",
    model=Gemini(id=GEMINI_MODEL),      
    instructions=SYSTEM_INSTRUCTIONS,
    add_knowledge_to_context=True,
    knowledge_retriever=qdrant_retriever,  
    markdown=False
)

if __name__ == "__main__":
    print(agent.run("What are the symptoms of Fungal infection?").content)
