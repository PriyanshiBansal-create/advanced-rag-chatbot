"""Ingestion orchestrator. Run with ``python ingestion_pipeline.py``."""
from modules.pdf_loader import load_documents
from modules.chunker import split_documents
from modules.vector_store import create_vector_store, load_vector_store
from modules.config import DOCS_PATH, PERSIST_DIRECTORY

def main():
    """Load documents, chunk them, and create or reuse the vector store."""
    if PERSIST_DIRECTORY.exists():
        return load_vector_store()
    documents = load_documents(str(DOCS_PATH))
    chunks = split_documents(documents)
    return create_vector_store(chunks, PERSIST_DIRECTORY)

if __name__ == "__main__":
    main()
