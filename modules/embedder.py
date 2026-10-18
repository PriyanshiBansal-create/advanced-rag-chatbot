"""Shared local embedding model singleton."""
from langchain_huggingface import HuggingFaceEmbeddings

_embedding_model = None

def get_embeddings():
    """Single shared local embedding model (free, no API key)."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return _embedding_model
