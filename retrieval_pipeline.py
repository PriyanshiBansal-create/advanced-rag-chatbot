"""Retrieval and generation orchestrator. Run with ``python retrieval_pipeline.py``."""
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_groq import ChatGroq
from modules.config import PERSIST_DIRECTORY
from modules.embedder import get_embeddings
from modules.vector_store import load_vector_store
from modules.history import chat_history, rewrite_question
from modules.query_expander import generate_query_variations
from modules.hybrid_retriever import build_hybrid_retriever
from modules.rrf import reciprocal_rank_fusion
from modules.reranker import rerank_docs

load_dotenv()
model = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
db = load_vector_store()
hybrid_retriever = build_hybrid_retriever(db, k=20)

def ask_question(user_question):
    print(f"You asked: {user_question}")
    search_question = rewrite_question(model, user_question)
    query_variations = generate_query_variations(model, search_question)
    all_retrieval_results = [hybrid_retriever.invoke(query) for query in query_variations]
    fused = reciprocal_rank_fusion(all_retrieval_results, k=60)
    rrf_candidates = [doc for doc, score in fused[:30]]
    docs = rerank_docs(search_question, rrf_candidates, top_n=8)

    combined_input = f"""Answer this question using the documents below: {user_question}
 
Documents:
{chr(10).join([f"- {doc.page_content}" for doc in docs])}
 
Instructions:
- Read ALL documents carefully before deciding.
- Answer using only information present in the documents.
- Even if the answer is spread across multiple documents, piece it together.
- Only say "I don't have enough information" if the answer is genuinely absent from every document.
- Do not refuse if the answer is partially present — use what is there.
"""
    messages = [
        SystemMessage(content="You are a helpful assistant that answers questions based on provided documents and conversation history."),
    ] + chat_history + [HumanMessage(content=combined_input)]
    result = model.invoke(messages)
    answer = result.content
    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))
    print(f"Answer: {answer}")
    return answer

def start_chat():
    # print("Ask me questions! Type 'quit' to exit.")
    while True:
        question = input("Your question: ")
        if question.lower() == "quit":
            # print("Goodbye!")
            break
        ask_question(question)

if __name__ == "__main__":
    start_chat()
