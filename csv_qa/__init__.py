"""
CSV Question Answering Bot package
"""

import logging

from csv_qa.exceptions import (
    CSVQAError,
    DataLoadError,
    OllamaError,
    OllamaConnectionError,
    OllamaTimeoutError,
    OllamaResponseError,
    OllamaModelNotFoundError,
    QueryEngineError,
    StructuredQueryError,
    VectorSearchError,
    HybridEngineError,
    ContextMemoryError,
    QueryResult,
    QUERY_ERROR_MESSAGES,
)

# Set up package-level logger with NullHandler (let applications configure handlers)
logging.getLogger("csv_qa").addHandler(logging.NullHandler())

__all__ = [
    "CSVQAError",
    "DataLoadError",
    "OllamaError",
    "OllamaConnectionError",
    "OllamaTimeoutError",
    "OllamaResponseError",
    "OllamaModelNotFoundError",
    "QueryEngineError",
    "StructuredQueryError",
    "VectorSearchError",
    "HybridEngineError",
    "ContextMemoryError",
    "QueryResult",
    "QUERY_ERROR_MESSAGES",
]
