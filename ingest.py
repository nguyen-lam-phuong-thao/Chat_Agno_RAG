import pandas as pd
from agno.knowledge.knowledge import Knowledge
from agno.vectordb.pgvector import PgVector, SearchType
from gemini_embedder import GeminiEmbedder  # custom embedder cho Gemini
from dotenv import load_dotenv
import os
from sqlalchemy import create_engine, text

load_dotenv()

DB_URL = "postgresql+psycopg://ai:ai@localhost:5532/ai"


# -------------------------
# Build documents từ CSV
# -------------------------
def build_documents(disease_csv, precaution_csv):
    df = pd.read_csv(disease_csv)
    prec = pd.read_csv(precaution_csv)

    merged = df.merge(prec, on="Disease", how="left")
    docs = []

    for _, row in merged.iterrows():
        disease = row["Disease"]
        text_parts = [f"Disease: {disease}"]

        # Symptoms
        text_parts.append("Symptoms:")
        for i in range(1, 7):
            col = f"Symptom_{i}"
            if col in row and pd.notna(row[col]):
                text_parts.append(f"- {row[col]}")

        # Precautions
        text_parts.append("Precautions:")
        for i in range(1, 5):
            col = f"Precaution_{i}"
            if col in row and pd.notna(row[col]):
                text_parts.append(f"- {row[col]}")

        docs.append("\n".join(text_parts))
    return docs


# -------------------------
# Check DB row count
# -------------------------
def check_db():
    engine = create_engine(DB_URL)
    with engine.connect() as conn:
        cols = conn.execute(text("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'disease_docs'
        """)).fetchall()
        print("Columns in disease_docs:", [c[0] for c in cols])

        try:
            rows = conn.execute(text("SELECT content FROM disease_docs LIMIT 3")).fetchall()
            print("Sample content:", rows)
        except Exception:
            rows = conn.execute(text("SELECT document FROM disease_docs LIMIT 3")).fetchall()
            print("Sample document:", rows)

# -------------------------
# Main ingest function
# -------------------------
def main():
    docs = build_documents(
        "data/DiseaseAndSymptoms.csv",
        "data/Disease precaution.csv"
    )

    knowledge = Knowledge(
        vector_db=PgVector(
            table_name="disease_docs",
            db_url=DB_URL,
            search_type=SearchType.hybrid,
            embedder=GeminiEmbedder(model="models/embedding-001"),
        ),
    )

    for d in docs:
        knowledge.add_content(
            text_content=d,
            metadata={"source": "csv_ingest"}
        )

    print(f"Indexed {len(docs)} documents into PgVector.")

if __name__ == "__main__":
    main()
    check_db()
