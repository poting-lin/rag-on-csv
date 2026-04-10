"""Unit tests for the custom exception hierarchy and QueryResult dataclass."""

import pytest

from csv_qa.exceptions import (
    CSVQAError,
    ContextMemoryError,
    DataLoadError,
    HybridEngineError,
    OllamaConnectionError,
    OllamaError,
    OllamaModelNotFoundError,
    OllamaResponseError,
    OllamaTimeoutError,
    QueryEngineError,
    QueryResult,
    QUERY_ERROR_MESSAGES,
    StructuredQueryError,
    VectorSearchError,
)


# --- Exception hierarchy tests ---


class TestCSVQAError:
    """Tests for the base exception class."""

    def test_attributes(self):
        """Verify error_code, user_message, and detail are stored correctly."""
        err = CSVQAError(error_code="TEST_ERR", user_message="Something broke", detail="extra info")
        assert err.error_code == "TEST_ERR"
        assert err.user_message == "Something broke"
        assert err.detail == "extra info"

    def test_str_returns_user_message(self):
        """str() should return the user_message."""
        err = CSVQAError(error_code="X", user_message="visible message")
        assert str(err) == "visible message"

    def test_is_exception(self):
        """CSVQAError should be catchable as a standard Exception."""
        with pytest.raises(Exception):
            raise CSVQAError(error_code="X", user_message="boom")


class TestDataLoadError:
    """Tests for DataLoadError."""

    def test_defaults(self):
        """Default error_code and user_message should be set."""
        err = DataLoadError()
        assert err.error_code == "DATA_LOAD_ERR"
        assert "CSV file" in err.user_message
        assert err.path == ""

    def test_custom_detail_and_path(self):
        """Detail and path should be stored."""
        err = DataLoadError(detail="encoding issue", path="/tmp/bad.csv")
        assert err.detail == "encoding issue"
        assert err.path == "/tmp/bad.csv"

    def test_inheritance(self):
        """Should be catchable as CSVQAError."""
        with pytest.raises(CSVQAError):
            raise DataLoadError()


class TestOllamaErrors:
    """Tests for Ollama exception subclasses."""

    def test_connection_error(self):
        """OllamaConnectionError should have correct error_code."""
        err = OllamaConnectionError(detail="refused")
        assert err.error_code == "OLLAMA_CONN_ERR"
        assert isinstance(err, OllamaError)
        assert isinstance(err, CSVQAError)

    def test_timeout_error(self):
        """OllamaTimeoutError should have correct error_code."""
        err = OllamaTimeoutError()
        assert err.error_code == "OLLAMA_TIMEOUT"
        assert "timed out" in err.user_message.lower()

    def test_response_error(self):
        """OllamaResponseError should have correct error_code."""
        err = OllamaResponseError(detail="empty body")
        assert err.error_code == "OLLAMA_RESP_ERR"
        assert err.detail == "empty body"

    def test_model_not_found(self):
        """OllamaModelNotFoundError should embed model name in user_message."""
        err = OllamaModelNotFoundError(model_name="llama99")
        assert err.error_code == "OLLAMA_MODEL_404"
        assert "llama99" in err.user_message
        assert err.model_name == "llama99"

    def test_catch_by_base_ollama(self):
        """All Ollama errors should be catchable as OllamaError."""
        for exc_cls in (OllamaConnectionError, OllamaTimeoutError, OllamaResponseError, OllamaModelNotFoundError):
            with pytest.raises(OllamaError):
                if exc_cls is OllamaModelNotFoundError:
                    raise exc_cls(model_name="x")
                else:
                    raise exc_cls()


class TestQueryEngineErrors:
    """Tests for query engine exception subclasses."""

    def test_structured_query_error(self):
        """StructuredQueryError should have correct error_code."""
        err = StructuredQueryError(detail="division by zero")
        assert err.error_code == "STRUCT_QUERY_ERR"
        assert isinstance(err, QueryEngineError)

    def test_vector_search_error(self):
        """VectorSearchError should have correct error_code."""
        err = VectorSearchError()
        assert err.error_code == "VECTOR_SEARCH_ERR"

    def test_hybrid_engine_error(self):
        """HybridEngineError should have correct error_code."""
        err = HybridEngineError()
        assert err.error_code == "HYBRID_ENGINE_ERR"

    def test_catch_by_base_query_engine(self):
        """All query engine errors should be catchable as QueryEngineError."""
        for exc_cls in (StructuredQueryError, VectorSearchError, HybridEngineError):
            with pytest.raises(QueryEngineError):
                raise exc_cls()


class TestContextMemoryError:
    """Tests for ContextMemoryError."""

    def test_defaults(self):
        """Default error_code should be CTX_MEMORY_ERR."""
        err = ContextMemoryError()
        assert err.error_code == "CTX_MEMORY_ERR"
        assert isinstance(err, CSVQAError)


# --- QueryResult tests ---


class TestQueryResultOk:
    """Tests for QueryResult.ok() factory."""

    def test_success_fields(self):
        """ok() should set success=True and populate data."""
        result = QueryResult.ok(data="42 records", engine="structured", confidence=0.9)
        assert result.success is True
        assert result.data == "42 records"
        assert result.engine == "structured"
        assert result.confidence == 0.9
        assert result.error_code is None
        assert result.error_message is None

    def test_defaults(self):
        """ok() with only data should have sensible defaults."""
        result = QueryResult.ok(data="answer")
        assert result.engine == ""
        assert result.confidence == 1.0
        assert result.metadata == {}

    def test_metadata(self):
        """ok() should store metadata dict."""
        meta = {"row_count": 5}
        result = QueryResult.ok(data="x", metadata=meta)
        assert result.metadata == {"row_count": 5}


class TestQueryResultFail:
    """Tests for QueryResult.fail() factory."""

    def test_failure_fields(self):
        """fail() should set success=False and populate error fields."""
        result = QueryResult.fail(error_code="NO_MATCH", error_message="nothing found")
        assert result.success is False
        assert result.data == ""
        assert result.error_code == "NO_MATCH"
        assert result.error_message == "nothing found"

    def test_engine_on_failure(self):
        """fail() should allow specifying the engine that failed."""
        result = QueryResult.fail(error_code="X", error_message="oops", engine="semantic")
        assert result.engine == "semantic"


class TestQueryErrorMessages:
    """Tests for the QUERY_ERROR_MESSAGES mapping."""

    def test_known_codes_present(self):
        """All expected error codes should be in the mapping."""
        expected_codes = ["NO_MATCH", "NO_CONTEXT", "AMBIGUOUS_QUERY", "EMPTY_RESULT", "ALL_ENGINES_FAILED"]
        for code in expected_codes:
            assert code in QUERY_ERROR_MESSAGES, f"Missing error code: {code}"

    def test_messages_are_nonempty_strings(self):
        """Every message should be a non-empty string."""
        for code, msg in QUERY_ERROR_MESSAGES.items():
            assert isinstance(msg, str) and len(msg) > 0, f"Invalid message for {code}"
