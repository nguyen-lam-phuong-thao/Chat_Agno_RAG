import os
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from app import agent
class QAReq(BaseModel):
    query: str
    top_k: int = 5

app = FastAPI(title="Agno Agent RAG")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    return FileResponse("static/index.html")

@app.get("/health")
def health():
    return {"status": "ok", "model": os.getenv("GEMINI_MODEL", "gemini-1.5-flash")}

@app.post("/qa")
def qa(req: QAReq):

    run = agent.run(req.query, num_documents=req.top_k)
    return {
        "answer": run.content,
    }
