"""Standalone-question rewriting and in-memory chat history."""
from langchain_core.messages import HumanMessage, SystemMessage

chat_history = []

def rewrite_question(model, user_question: str, history=None) -> str:
    """Rewrite a follow-up as a standalone searchable question."""
    history = chat_history if history is None else history
    if not history:
        return user_question
    messages = [
        SystemMessage(content=(
            "Given the chat history, rewrite the new question to be standalone "
            "and searchable. Just return the rewritten question. "
            "DO NOT answer it. DO NOT add any facts. DO NOT infer information. "
            "Return ONLY the rewritten question."
        )),
    ] + history + [HumanMessage(content=f"New question: {user_question}")]
    result = model.invoke(messages)
    return result.content.strip()
