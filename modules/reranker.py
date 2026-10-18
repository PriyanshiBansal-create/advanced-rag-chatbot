"""Cross-encoder reranking."""
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank_docs(query: str, docs, top_n: int = 5):
    if not docs:
        return docs
 
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    scored = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
 
    # print(f"\nCross-encoder reranking {len(docs)} candidates → keeping top {top_n}:")
    for rank, (score, doc) in enumerate(scored, 1):
        # print(f"  #{rank} score={score:.4f}: {doc.page_content[:70]}...")
        pass
 
    return [doc for _, doc in scored[:top_n]]
