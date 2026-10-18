<<<<<<< HEAD
# Local RAG Assistant

A document question-answering project with local Hugging Face embeddings, ChromaDB persistence, BM25 plus MMR retrieval, query variation fusion, cross-encoder reranking, and Groq generation. PDF ingestion supports Unstructured-based tables and image captioning, with PyMuPDF text extraction as a fallback.

## Project layout

```text
ingestion_pipeline.py       # ingestion entry point
retrieval_pipeline.py       # retrieval, generation, and interactive chat
modules/
  pdf_loader.py             # text/PDF loading and image captioning
  chunker.py                # semantic and recursive chunking
  embedder.py               # shared embedding model
  vector_store.py           # ChromaDB create/load helpers
  hybrid_retriever.py       # BM25 + MMR ensemble
  query_expander.py         # structured query variations
  rrf.py                    # query-variation reciprocal rank fusion
  reranker.py               # cross-encoder reranking
  history.py                # question rewriting and chat history
  config.py                 # project-relative, overridable paths
```

## Setup

Use Python 3.10 or newer. Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set `GROQ_API_KEY`. The same key is used by Groq chat and image captioning. Do not commit `.env`.

For multimodal PDF extraction, install the system tools required by Unstructured and Tesseract. When Unstructured is unavailable, PDF loading falls back to text extraction via PyMuPDF. Image captioning failures are reported as warnings and the affected image is skipped. Tesseract is discovered through `PATH`, or can be configured with `TESSERACT_PATH`.

## Add documents and run

Place your own `.txt` and `.pdf` files under `docs/`. Document files, ChromaDB data, and local environments are ignored by Git by default.

Build or reuse the persisted vector store:

```bash
python ingestion_pipeline.py
```

Start interactive question answering:

```bash
python retrieval_pipeline.py
```

The defaults store the ChromaDB collection in `db/chroma_db`. Set `RAG_DOCS_PATH` or `CHROMA_DB_PATH` to use other directories. The first embedding and cross-encoder use may download their models.

## Behavior notes

The retrieval pipeline keeps the original model choices, prompts, chunk sizes, retrieval weights, candidate counts, and fusion settings. Conversation history exists in memory for one process and is not persisted.
=======
# advanced-rag-chatbot
An advanced RAG chatbot featuring multimodal document ingestion, semantic chunking, hybrid retrieval, Reciprocal Rank Fusion (RRF), reranking, and conversational history for context-aware question answering.
>>>>>>> ce020f9e79082cd3f4bcc971fd472aecaa3f3079
