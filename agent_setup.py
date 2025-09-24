from agno.agent import Agent
from agno.knowledge.knowledge import Knowledge
from agno.vectordb.pgvector import PgVector, SearchType
from agno.models.google.gemini import Gemini
from gemini_embedder import GeminiEmbedder  # <-- use GeminiEmbedder
import os
from dotenv import load_dotenv

load_dotenv()

DB_URL = "postgresql+psycopg://ai:ai@localhost:5532/ai"

def build_agent():
    embedder = GeminiEmbedder(model="models/embedding-001")  # <-- match ingest.py

    knowledge = Knowledge(
        vector_db=PgVector(
            table_name="disease_docs",
            db_url=DB_URL,
            search_type=SearchType.hybrid,
            embedder=embedder,
        ),
    )

    agent = Agent(
        model=Gemini(id="gemini-1.5-flash"),
        knowledge=knowledge,
        search_knowledge=True,
        markdown=True,
    )
    return agent

if __name__ == "__main__":
    agent = build_agent()
    results = agent.knowledge.vector_db.search("Fungal infection symptoms")
    print("Retrieved docs:", [r.content for r in results])
    agent.print_response("what is the symptom of Fungal infection?", stream=True)