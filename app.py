from fastapi import FastAPI
from pydantic import BaseModel
from agent_setup import build_agent

app = FastAPI()
agent = build_agent()

class Query(BaseModel):
    question: str

@app.post("/query")
async def query(q: Query):
    answer = agent.run(q.question)
    return {"question": q.question, "answer": str(answer)}

