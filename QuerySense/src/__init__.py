"""QuerySense: intelligent semantic search and FAQ retrieval.

The package is split so that data access, retrieval, and evaluation logic stay
independent of the Streamlit presentation layer.
"""

__version__ = "1.0.0"

__all__ = [
    "data_loader",
    "preprocessing",
    "tfidf_retriever",
    "semantic_retriever",
    "metrics",
    "evaluation",
]
