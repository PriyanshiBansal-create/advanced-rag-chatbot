"""Semantic and recursive document chunking."""
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from .embedder import get_embeddings

def split_documents(documents, max_chunk_size=1500, chunk_overlap=200):
    """
    Semantic chunking using free local embeddings.

    - Plain text docs -> SemanticChunker (splits where meaning shifts).
    - Docs containing [TABLE] / [IMAGE DESCRIPTION] -> recursive splitter, so
      HTML tables aren't cut by sentence-based splitting.
    - Any semantic chunk longer than max_chunk_size gets a recursive second pass
      (SemanticChunker has no size cap, and all-MiniLM truncates long inputs).
    """
    # print("Splitting documents into chunks (semantic)...")

    semantic_splitter = SemanticChunker(
        embeddings=get_embeddings(),
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=70,   # lowered from 90: keeps related sections together
        min_chunk_size=100,
    )

    recursive_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=max_chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = []
    for doc in documents:
        has_structured = (
            "[TABLE]" in doc.page_content or "[IMAGE DESCRIPTION]" in doc.page_content
        )

        if has_structured:
            chunks.extend(recursive_splitter.split_documents([doc]))
            continue

        try:
            semantic_chunks = semantic_splitter.split_documents([doc])
        except Exception as e:
            print(f"  ⚠️ Semantic split failed ({e}); using recursive splitter.")
            semantic_chunks = [doc]

        for sc in semantic_chunks:
            if len(sc.page_content) > max_chunk_size:
                chunks.extend(recursive_splitter.split_documents([sc]))
            else:
                chunks.append(sc)

    MIN_CHUNK_LENGTH = 50
    chunks = [c for c in chunks if len(c.page_content.strip()) >= MIN_CHUNK_LENGTH]

    if chunks:
        for i, chunk in enumerate(chunks[:5]):
            # print(f"\n--- Chunk {i+1} ---")
            # print(f"Source: {chunk.metadata['source']}")
            # print(f"Length: {len(chunk.page_content)} characters")
            # print("Content:")
            # print(chunk.page_content)
            # print("-" * 50)
            pass

        if len(chunks) > 5:
            # print(f"\n... and {len(chunks) - 5} more chunks")
            pass

    return chunks
