"""BM25 + MMR hybrid retrieval using LangChain's EnsembleRetriever."""
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

def build_hybrid_retriever(vectorstore, k: int = 10):
    all_docs = vectorstore.get()
    corpus = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(all_docs["documents"], all_docs["metadatas"])
    ]
    # print(f"BM25 index built over {len(corpus)} stored chunks.")
 
    bm25_retriever = BM25Retriever.from_documents(corpus)
    bm25_retriever.k = k  # BM25 returns its own keyword-ranked list (no RRF yet)
 
    vector_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": k * 3, "lambda_mult": 0.7},
    )
 
    # EnsembleRetriever internally applies RRF to merge [vector_ranked, bm25_ranked]
    # Weight 0.7 / 0.3 biases toward semantic search while keeping keyword boost.
    return EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.6, 0.4],
    )
