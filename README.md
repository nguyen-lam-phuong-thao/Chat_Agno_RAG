# 🧠 RAG Medical (Qdrant + Agno Agent + Gemini)

A small **Retrieval-Augmented Generation (RAG)** app that answers medical questions from a **fixed database** stored in **Qdrant**. It uses **Sentence-Transformers** for embeddings and an **Agno Agent** to retrieve relevant documents and generate answers with **Gemini**. A simple web UI is provided at `static/index.html`.

📂 **Dataset:** https://www.kaggle.com/datasets/choongqianzheng/disease-and-symptoms-dataset  
⚠️ **Disclaimer:** For learning/demo only. **Not medical advice.**

---

## ⚙️ Architecture

```
User → Web UI
      │
      FastAPI (api_qa.py)
      │
      Agno Agent
      │
SentenceTransformer embeddings
      │
      Qdrant vector DB
      │
      Gemini LLM
```

Flow: **Question → retrieve documents → add context → Gemini generates answer**

---

## 📁 Project Structure

```
scripts/                # data processing scripts
static/index.html       # chat UI
data/                   # dataset CSV / JSONL
qdrant_data/            # Qdrant storage
rag_agent_qdrant.py     # agent + retriever
api_qa.py               # FastAPI endpoints
app.py                  # console test
```

---

## 📦 Requirements

- Python **3.10+**
- **Docker** (for Qdrant)

Install dependencies

```bash
pip install fastapi uvicorn[standard] qdrant-client sentence-transformers agno google-generativeai
```

---

## 🚀 Run

Start **Qdrant**

```bash
docker run -p 6333:6333 -p 6334:6334 \
-v "${PWD}/qdrant_data:/qdrant/storage" \
--name qdrant qdrant/qdrant
```

Prepare dataset and ingest vectors

```bash
python scripts/create_document.py
python scripts/ingest_qdrant.py
```

Run API

```bash
uvicorn api_qa:app --reload --port 8000
```

Open

```
http://127.0.0.1:8000/
```

Example query

```
What are the symptoms of Fungal infection?
```

---

## 🔗 API

**POST /qa**

Request

```json
{ "query": "string", "top_k": 5 }
```

Response

```json
{ "answer": "string" }
```

Optional retrieval endpoint

```
POST /search
```

---

## 🧰 Stack

- **Embeddings:** sentence-transformers/all-MiniLM-L6-v2  
- **Vector DB:** Qdrant  
- **Agent:** Agno  
- **LLM:** Gemini  

---

## ⚠️ Medical Disclaimer

This project is for **educational/demonstration purposes only** and **does not provide medical advice**.  
**Always consult a qualified healthcare professional.**
