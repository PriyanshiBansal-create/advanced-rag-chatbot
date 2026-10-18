"""Text and multimodal PDF loading with optional image captioning."""
import os
from functools import partial
import fitz
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from .config import DOCS_PATH, TESSERACT_PATH

try:
    from unstructured.partition.pdf import partition_pdf
    from unstructured.chunking.title import chunk_by_title
    UNSTRUCTURED_AVAILABLE = True
    # print("✅ Unstructured imported successfully")
except Exception:
    print("❌ Failed to import unstructured; text-only PDF fallback enabled.")
    import traceback
    traceback.print_exc()
    UNSTRUCTURED_AVAILABLE = False

load_dotenv()
GROQ_VISION_MODEL = os.getenv("GROQ_VISION_MODEL", "qwen/qwen3.8-27b")
if TESSERACT_PATH:
    os.environ["TESSERACT_PATH"] = TESSERACT_PATH
    tesseract_dir = os.path.dirname(TESSERACT_PATH)
    if tesseract_dir and tesseract_dir not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = os.environ.get("PATH", "") + os.pathsep + tesseract_dir

def describe_image_with_groq(image_base64: str) -> str:
    """
    Use Groq's free-tier vision model to turn an embedded image into
    searchable text, instead of paying for gpt-4o.
    """
    try:
        vision_llm = ChatGroq(model=GROQ_VISION_MODEL, temperature=0, max_tokens=800)

        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": (
                        "Describe this image in detail for a document search "
                        "system. If it contains a chart, diagram, or figure, "
                        "explain what it shows. If it contains text, transcribe "
                        "the key text as well."
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                },
            ]
        )

        result = vision_llm.invoke([message])
        return result.content.strip()

    except Exception as e:
        print(f"  ⚠️ Image captioning failed ({e}); skipping image content.")
        return ""

def load_pdf_documents_multimodal(pdf_path):
    """
    Reads a PDF using `unstructured`, extracting text, tables (as HTML),
    and images (captioned via free Groq vision model). Falls back to
    plain text extraction if unstructured/tesseract aren't installed.
    """
    if not UNSTRUCTURED_AVAILABLE:
        print("  ⚠️ `unstructured` not installed — falling back to text-only extraction.")
        return load_pdf_documents_textonly(pdf_path)

    # print(f"📄 Partitioning (multimodal): {pdf_path}")

    elements = partition_pdf(
        filename=pdf_path,
        strategy="hi_res",
        infer_table_structure=True,
        extract_image_block_types=["Image"],
        extract_image_block_to_payload=True,
    )

    chunks = chunk_by_title(
        elements,
        max_characters=3000,
        new_after_n_chars=2400,
        combine_text_under_n_chars=500,
    )

    documents = []

    for i, chunk in enumerate(chunks):
        text_parts = [chunk.text] if chunk.text else []
        content_types = ["text"]

        orig_elements = getattr(chunk.metadata, "orig_elements", []) or []

        for element in orig_elements:
            element_type = type(element).__name__

            if element_type == "Table":
                table_html = getattr(element.metadata, "text_as_html", element.text)
                text_parts.append(f"[TABLE]\n{table_html}")
                content_types.append("table")

            elif element_type == "Image":
                image_b64 = getattr(element.metadata, "image_base64", None)
                if image_b64:
                    # print(f"  🖼️  Captioning image in chunk {i+1}...")
                    caption = describe_image_with_groq(image_b64)
                    if caption:
                        text_parts.append(f"[IMAGE DESCRIPTION]\n{caption}")
                        content_types.append("image")

        combined_text = "\n\n".join(text_parts).strip()
        if not combined_text:
            continue

        documents.append(
            Document(
                page_content=combined_text,
                metadata={
                    "source": pdf_path,
                    "chunk": i + 1,
                    "type": "pdf",
                    "content_types": ",".join(sorted(set(content_types))),
                },
            )
        )

    # print(f"✅ Extracted {len(documents)} multimodal chunks")
    return documents

def load_pdf_documents_textonly(pdf_path):
    """
    Original PyMuPDF-based extraction: one Document per page, text only.
    Used as a fallback when `unstructured` isn't available.
    """
    pdf = fitz.open(pdf_path)
    documents = []

    for page_num, page in enumerate(pdf):
        text = page.get_text("text")
        if text.strip():
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": pdf_path,
                        "page": page_num + 1,
                        "type": "pdf",
                    },
                )
            )

    pdf.close()
    return documents

USE_MULTIMODAL_PDF = True

def load_documents(docs_path="docs"):

    # print(f"Loading documents from {docs_path}...")

    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"{docs_path} does not exist.")

    documents = []

    # -------- TXT FILES --------
    txt_loader = DirectoryLoader(
        path=docs_path,
        glob="**/*.txt",
        loader_cls=partial(TextLoader, encoding="utf-8"),
    )
    documents.extend(txt_loader.load())

    # -------- PDF FILES --------
    for root, _, files in os.walk(docs_path):
        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_path = os.path.join(root, file)
                # print(f"Loading PDF: {pdf_path}")

                if USE_MULTIMODAL_PDF:
                    documents.extend(load_pdf_documents_multimodal(pdf_path))
                else:
                    documents.extend(load_pdf_documents_textonly(pdf_path))

    if not documents:
        raise FileNotFoundError("No TXT or PDF files found.")

    # print(f"\nLoaded {len(documents)} documents.\n")
    return documents
