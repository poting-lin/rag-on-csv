from dataclasses import dataclass, field


class CSVQAError(Exception):
    """Base exception for all CSV QA system errors."""

    def __init__(
        self,
        error_code: str,
        user_message: str,
        detail: str = "",
    ) -> None:
        self.error_code = error_code
        self.user_message = user_message
        self.detail = detail
        super().__init__(user_message)


class DataLoadError(CSVQAError):
    """CSV file cannot be loaded (missing, encoding, corrupt format)."""

    def __init__(self, detail: str = "", path: str = "") -> None:
        super().__init__(
            error_code="DATA_LOAD_ERR",
            user_message="Failed to load the CSV file. Please check the file path and format.",
            detail=detail,
        )
        self.path = path


class OllamaError(CSVQAError):
    """Base exception for Ollama API errors."""


class OllamaConnectionError(OllamaError):
    """Ollama service is not running or unreachable."""

    def __init__(self, detail: str = "") -> None:
        super().__init__(
            error_code="OLLAMA_CONN_ERR",
            user_message="Cannot connect to Ollama. Make sure it is running with 'ollama serve'.",
            detail=detail,
        )


class OllamaTimeoutError(OllamaError):
    """Request to Ollama timed out."""

    def __init__(self, detail: str = "") -> None:
        super().__init__(
            error_code="OLLAMA_TIMEOUT",
            user_message="Request timed out. Try a simpler question or check if the model is loaded.",
            detail=detail,
        )


class OllamaResponseError(OllamaError):
    """Ollama returned a malformed or empty response."""

    def __init__(self, detail: str = "") -> None:
        super().__init__(
            error_code="OLLAMA_RESP_ERR",
            user_message="Received an invalid response from Ollama. Try again or switch models.",
            detail=detail,
        )


class OllamaModelNotFoundError(OllamaError):
    """Specified model does not exist in Ollama."""

    def __init__(self, model_name: str = "", detail: str = "") -> None:
        super().__init__(
            error_code="OLLAMA_MODEL_404",
            user_message=f"Model '{model_name}' not found. Run 'ollama pull {model_name}' to download it.",
            detail=detail,
        )
        self.model_name = model_name


class QueryEngineError(CSVQAError):
    """Base exception for query engine failures."""


class StructuredQueryError(QueryEngineError):
    """Pandas operation failed during structured query execution."""

    def __init__(self, detail: str = "") -> None:
        super().__init__(
            error_code="STRUCT_QUERY_ERR",
            user_message="Failed to execute the structured query. Try rephrasing your question.",
            detail=detail,
        )


class VectorSearchError(QueryEngineError):
    """Vector index build or retrieval failed."""

    def __init__(self, detail: str = "") -> None:
        super().__init__(
            error_code="VECTOR_SEARCH_ERR",
            user_message="Vector search failed. Try rephrasing your question.",
            detail=detail,
        )


class HybridEngineError(QueryEngineError):
    """All engines failed to produce a result."""

    def __init__(self, detail: str = "") -> None:
        super().__init__(
            error_code="HYBRID_ENGINE_ERR",
            user_message="Unable to process this question with any engine. Try rephrasing or simplifying.",
            detail=detail,
        )


class ContextMemoryError(CSVQAError):
    """Conversation memory operation failed."""

    def __init__(self, detail: str = "") -> None:
        super().__init__(
            error_code="CTX_MEMORY_ERR",
            user_message="Conversation memory encountered an error. Your question will still be processed.",
            detail=detail,
        )


@dataclass
class QueryResult:
    """Unified result object for all query engine responses.

    Used for expected query-level outcomes, not system errors.
    System errors should raise exceptions instead.
    """

    success: bool
    data: str = ""
    engine: str = ""
    error_code: str | None = None
    error_message: str | None = None
    confidence: float = 0.0
    metadata: dict = field(default_factory=dict)

    @staticmethod
    def ok(data: str, engine: str = "", confidence: float = 1.0, metadata: dict | None = None) -> "QueryResult":
        """Create a successful result."""
        return QueryResult(
            success=True,
            data=data,
            engine=engine,
            confidence=confidence,
            metadata=metadata or {},
        )

    @staticmethod
    def fail(error_code: str, error_message: str, engine: str = "") -> "QueryResult":
        """Create a failed result for expected query-level issues."""
        return QueryResult(
            success=False,
            data="",
            engine=engine,
            error_code=error_code,
            error_message=error_message,
        )


QUERY_ERROR_MESSAGES: dict[str, str] = {
    "NO_MATCH": "No matching data found. Try rephrasing or check if the value exists.",
    "NO_CONTEXT": "Could not find relevant context for your question. Try being more specific.",
    "AMBIGUOUS_QUERY": "Your question is too broad. Try asking about a specific column or value.",
    "EMPTY_RESULT": "Query executed successfully but returned no results.",
    "ALL_ENGINES_FAILED": "Unable to process this question. Try rephrasing or simplifying it.",
}
