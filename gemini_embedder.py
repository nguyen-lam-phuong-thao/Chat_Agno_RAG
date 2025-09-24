from agno.knowledge.embedder.base import Embedder
import google.generativeai as genai
import os

class GeminiEmbedder(Embedder):
    def __init__(self, model: str = "models/embedding-001"):
        self.model = model
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set in environment")
        genai.configure(api_key=api_key)

    def get_embedding(self, text: str):
        """Return embedding vector for a single text string."""
        resp = genai.embed_content(model=self.model, content=text)
        return resp["embedding"]

    def embed_documents(self, texts):
        """Batch embed documents (list of strings)."""
        return [self.get_embedding(t) for t in texts]

    def embed_query(self, text: str):
        """Embed query text (used in hybrid search)."""
        return self.get_embedding(text)
