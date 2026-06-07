"""app/chatbot/chain.py

Builds a RAG chain using Ollama (local LLM) + FAISS (local vector store).
No API key required — everything runs on the local machine.

Requirements:
    1. Ollama installed and running: https://ollama.com
    2. Model pulled: ollama pull llama3.2
    3. pip install langchain-ollama langchain-core
"""

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from app.retriever import build_retriever

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Prompt ─────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an assistant specialized in churn prediction for \
a B2C telecom company.

Answer questions using only the context provided below.
Be concise and direct. If the answer is not in the context, say you don't know.

Context:
{context}"""

_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}"),
])


def _format_docs(docs: list) -> str:
    """Concatenates retrieved document chunks into a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)


def build_chain(model: str = "llama3.2") -> tuple:
    """Builds a RAG chain using a local Ollama model and a FAISS retriever.

    Args:
        model: Ollama model name. Must be pulled locally before use.
            Defaults to "llama3.2". Other options: "mistral", "phi3", "gemma2".

    Returns:
        A tuple of (chain, retriever):
            - chain: LCEL runnable that accepts a question string and returns
              an answer string.
            - retriever: The underlying FAISS retriever, exposed separately
              so the caller can fetch source documents for attribution.

    Example:
        >>> chain, retriever = build_chain()
        >>> answer = chain.invoke("What is the churn rate?")
    """
    llm       = ChatOllama(model=model, temperature=0)
    retriever = build_retriever()

    chain = (
        {
            "context":  retriever | _format_docs,
            "question": RunnablePassthrough(),
        }
        | _PROMPT
        | llm
        | StrOutputParser()
    )

    return chain, retriever