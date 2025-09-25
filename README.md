# RAG Medical (Qdrant + Agno Agent + Gemini)

A small **Retrieval-Augmented Generation (RAG)** app that answers medical questions from a **fixed database** ingested into **Qdrant**. It uses **Sentence-Transformers** for embeddings and an **Agno Agent** to: accept a question → retrieve relevant documents from Qdrant → add them as context → ask an LLM (**Gemini**) for a concise answer. A clean single‑file web UI is provided at `static/index.html`.

> **Dataset source:** https://www.kaggle.com/datasets/choongqianzheng/disease-and-symptoms-dataset  
> **Disclaimer:** For learning/demo only. **Not medical advice.**

---

## Quick Architecture

```
User → Web UI (static/index.html)
          │  POST /qa            (final answer)
          └─ POST /search (opt.) (references/passages)
                │
          FastAPI (api_qa.py)
                │
     Agno Agent (rag_agent_qdrant.py, Qdrant retriever)
                │
 SentenceTransformer (all-MiniLM-L6-v2) → Qdrant (medical_diseases)
                │
                      Gemini LLM (GOOGLE_API_KEY)
```

---

## Folder Structure

```
Agno/
├─ scripts/
│  ├─ create_document.py   # CSV → JSONL documents
│  ├─ ingest_qdrant.py     # Embed & upsert → Qdrant
│  └─ query_qdrant.py      # Manual query example (for reference)
├─ static/
│  └─ index.html           # Sleek chat UI (single-file)
├─ qdrant_data/            # Qdrant storage (Docker volume)
│  ├─ collection/          # (created by Qdrant)
│  └─ meta.json
├─ data/                   # Place dataset CSV / generated JSONL here
├─ rag_agent_qdrant.py     # Agno Agent + Qdrant retriever (sync version)
├─ api_qa.py               # FastAPI: /, /qa, /search, /health
├─ app.py                  # (optional) quick console test
├─ .env                    # (optional) environment variables
└─ README.md
```

---

## Prerequisites

- Python **3.10+**
- **Docker** (to run Qdrant)
- GPU **not required** (embedding model is lightweight)

---

## Installation

### 1) Create a virtual environment & install deps

**Windows PowerShell**

```powershell
python -m venv .venv
.venv\Scripts\activate

pip install -U fastapi uvicorn[standard] qdrant-client sentence-transformers agno google-generativeai
```

(Optional) If you prefer a file:

```txt
# requirements.txt
fastapi
uvicorn[standard]
qdrant-client
sentence-transformers
agno
google-generativeai
```

### 2) Run Qdrant (Docker)

**Windows PowerShell**

```powershell
mkdir qdrant_data
docker run -p 6333:6333 -p 6334:6334 ^
  -v "${PWD}\qdrant_data:/qdrant/storage" ^
  --name qdrant qdrant/qdrant
```

> If it already exists: `docker start qdrant`

### 3) Prepare data & ingest

1. Download the Kaggle dataset and put the CSV under `data/`.
2. Convert CSV → JSONL documents:

```powershell
python scripts/create_document.py
```

3. Embed & upsert into Qdrant (default collection: `medical_diseases`):

```powershell
python scripts/ingest_qdrant.py
```

> The project uses the embedding model `sentence-transformers/all-MiniLM-L6-v2`. Keep the same model for queries to match the vector space.

### 4) Environment variables

Using **Gemini** (recommended):

```powershell
$env:GOOGLE_API_KEY="YOUR_KEY"
$env:GEMINI_MODEL="gemini-1.5-flash"       # or gemini-2.0-flash-001
$env:QDRANT_HOST="localhost"
$env:QDRANT_PORT="6333"
$env:QDRANT_COLLECTION="medical_diseases"
$env:EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
```

(You may also put these in a `.env` and load them yourself.)

### 5) Run API + UI

```powershell
uvicorn api_qa:app --reload --port 8000
```

Open: **http://127.0.0.1:8000/**

Try: *“What are the symptoms of Fungal infection?”*

---

## REST API

### `GET /`
Serves the UI (`static/index.html`).

### `GET /health`
Simple health/status info.

### `POST /qa`
**Body**
```json
{ "query": "string", "top_k": 5 }
```
**Response**
```json
{ "answer": "string" }
```

### `POST /search` *(optional — used by the UI to show sources)*
**Body**
```json
{ "query": "string", "top_k": 5 }
```
**Response**
```json
{ "results": [ { "text": "...", "score": 0.83, "metadata": { "disease": "..." } } ] }
```

**Example**
```bash
curl -X POST "http://127.0.0.1:8000/qa"   -H "Content-Type: application/json"   -d '{"query":"What are the symptoms of Fungal infection?", "top_k": 3}'
```

---

## Customization

- **Change collection / embedding model**: adjust env `QDRANT_COLLECTION`, `EMBEDDING_MODEL` (then **re‑ingest**).
- **Top‑K control**: the UI exposes a Top‑K select; backend passes `top_k` to the retriever.
- **Reranking**: add a cross‑encoder step to rerank top‑k before LLM.
- **CORS / Security**: narrow `allow_origins` in `api_qa.py` for real deployments.
- **No LLM mode**: you can fallback to “extractive” answers from top context if you wish (not included by default).

---

## Troubleshooting

- **“I don’t have enough information to answer…”**  
  Usually means the retriever did not return context (Qdrant empty/not running) or `GOOGLE_API_KEY` is missing.

- **Async retriever warning (`coroutine was never awaited`)**  
  Use the **sync** version of `qdrant_retriever` (as in `rag_agent_qdrant.py`) **or** call `agent.arun()` instead of `agent.run()`.

- **CORS errors** when serving UI from another origin/port  
  Keep the `CORSMiddleware` enabled and set `allow_origins` appropriately.

- **Qdrant has no data**  
  Re‑run `create_document.py` → `ingest_qdrant.py`. Ensure the collection is `medical_diseases` and the embedding model matches.

---

## Credits & Licenses

- **Dataset**: *Disease and Symptoms Dataset* by Choong Qian Zheng (Kaggle). Please follow the dataset’s license and terms.  
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`. Check the Sentence‑Transformers project license.  
- **Qdrant**: Open‑source vector database.  
- **Agno**: Agent framework used to orchestrate retrieval + LLM.  
- **Gemini**: Requires a valid `GOOGLE_API_KEY` (Google AI Studio terms apply).

---

## Medical Disclaimer

This project is for **educational/demonstration** purposes only and **does not provide medical advice**. Always consult a qualified healthcare professional for medical decisions.

---

## Roadmap Ideas

- Streaming answers (SSE) for token‑by‑token UI updates.  
- Offline evaluation set (hit@k, ROUGE, answerability).  
- Query filters (disease/category) + highlight matched spans in UI.  
- Docker Compose for Qdrant + API + static server.
