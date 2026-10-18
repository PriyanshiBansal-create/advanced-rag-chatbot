"""ChromaDB creation and loading."""
from langchain_chroma import Chroma
from .embedder import get_embeddings
from .config import PERSIST_DIRECTORY

def create_vector_store(chunks, persist_directory=None):
    """Create and persist ChromaDB vector store."""
    path = str(persist_directory or PERSIST_DIRECTORY)
    embedding_model = get_embeddings()
    return Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=path,
        collection_metadata={"hnsw:space": "cosine"},
    )

def load_vector_store(persist_directory=None):
    """Load an existing ChromaDB collection."""
    return Chroma(
        persist_directory=str(persist_directory or PERSIST_DIRECTORY),
        embedding_function=get_embeddings(),
        collection_metadata={"hnsw:space": "cosine"},
    )
