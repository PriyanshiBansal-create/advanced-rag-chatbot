"""Shared project configuration."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_PATH = Path(os.getenv("RAG_DOCS_PATH", str(PROJECT_ROOT / "docs")))
PERSIST_DIRECTORY = Path(os.getenv("CHROMA_DB_PATH", str(PROJECT_ROOT / "db" / "chroma_db")))
TESSERACT_PATH = os.getenv("TESSERACT_PATH")
