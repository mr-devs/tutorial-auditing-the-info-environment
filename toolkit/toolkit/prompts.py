"""
Store LLM prompts for data analysis tasks.
"""

SIMPLE_PROMPT = (
    "Fact check this claim. You must use your web search tool to look up the "
    "latest information from the web before answering; do not rely on prior "
    "knowledge alone."
)

# Tutorial addition: closed-book counterpart of SIMPLE_PROMPT for the
# parametric-memory-only condition (no tools attached to the request).
CLOSED_BOOK_PROMPT = (
    "Fact check this claim using only your own knowledge. You have no web "
    "search tool; do not pretend to search. If you lack the information "
    "needed to decide, answer 'Not enough information'."
)
