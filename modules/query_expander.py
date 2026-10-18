"""Structured multi-query generation."""
from typing import List
from pydantic import BaseModel

class QueryVariations(BaseModel):
    queries: List[str]

def generate_query_variations(model, search_question: str):
    """Generate three alternatives and return them with the original query."""
    llm_structured = model.with_structured_output(QueryVariations, method="json_mode")
    variation_prompt = f"""
Generate exactly 3 different variations of this query that would help retrieve relevant documents.
 
Original query: {search_question}
 
Return ONLY valid JSON in exactly this format:
{{
    "queries": [
        "first alternative query",
        "second alternative query",
        "third alternative query"
    ]
}}
 
Do not include markdown, explanations, or any other text.
"""
    response = llm_structured.invoke(variation_prompt)
    return [search_question] + response.queries
