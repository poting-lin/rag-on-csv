"""Tests for the semantic search module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import requests as _requests

from csv_qa.exceptions import OllamaConnectionError
from csv_qa.semantic_search import (
    Chunk,
    ChunkBuilder,
    OllamaEmbedder,
    SemanticSearch,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_embedding(dim: int = 4) -> list[float]:
    """Return a deterministic fake embedding vector."""
    return [0.1, 0.2, 0.3, 0.4][:dim]


def _mock_post_response(embeddings: list[list[float]], status_code: int = 200):
    """Build a mock requests.post return value."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = {"embeddings": embeddings}
    resp.text = "mock response body"
    return resp


def _sample_df() -> pd.DataFrame:
    """Return a small mixed-type DataFrame for testing."""
    return pd.DataFrame(
        {
            "name": ["Alice", "Bob", "Carol"],
            "age": [30, 25, 35],
            "city": ["NYC", "LA", "NYC"],
        }
    )


# ---------------------------------------------------------------------------
# TestOllamaEmbedder
# ---------------------------------------------------------------------------


class TestOllamaEmbedder:
    """Tests for OllamaEmbedder."""

    def test_default_config(self) -> None:
        """Model name and URL should use defaults when env is not set."""
        embedder = OllamaEmbedder()
        assert embedder.model_name == "nomic-embed-text"
        assert "localhost:11434" in embedder.embed_url

    @patch.dict("os.environ", {"OLLAMA_EMBED_MODEL": "custom-model", "OLLAMA_BASE_URL": "http://myhost:9999"})
    def test_custom_env_config(self) -> None:
        """Model name and URL should come from env vars."""
        embedder = OllamaEmbedder()
        assert embedder.model_name == "custom-model"
        assert "myhost:9999" in embedder.embed_url

    @patch("csv_qa.semantic_search.requests.post")
    def test_embed_single(self, mock_post: MagicMock) -> None:
        """embed_single should return a single embedding vector."""
        mock_post.return_value = _mock_post_response([[0.1, 0.2, 0.3]])
        embedder = OllamaEmbedder()
        result = embedder.embed_single("hello")
        assert result == [0.1, 0.2, 0.3]

    @patch("csv_qa.semantic_search.requests.post")
    def test_embed_batch(self, mock_post: MagicMock) -> None:
        """embed should return one vector per input text."""
        vecs = [[0.1, 0.2], [0.3, 0.4]]
        mock_post.return_value = _mock_post_response(vecs)
        embedder = OllamaEmbedder()
        result = embedder.embed(["a", "b"])
        assert result == vecs

    @patch("csv_qa.semantic_search.requests.post")
    def test_caching(self, mock_post: MagicMock) -> None:
        """Duplicate text should hit cache; requests.post called only once."""
        mock_post.return_value = _mock_post_response([[0.5, 0.6]])
        embedder = OllamaEmbedder()
        embedder.embed_single("same text")
        embedder.embed_single("same text")
        assert mock_post.call_count == 1

    @patch("csv_qa.semantic_search.requests.post", side_effect=_requests.exceptions.ConnectionError("refused"))
    def test_connection_error(self, mock_post: MagicMock) -> None:
        """ConnectionError should raise OllamaConnectionError."""
        embedder = OllamaEmbedder()
        with pytest.raises(OllamaConnectionError):
            embedder.embed(["test"])

    @patch("csv_qa.semantic_search.requests.post")
    def test_404_model_not_found(self, mock_post: MagicMock) -> None:
        """404 response should raise OllamaConnectionError mentioning 'sidebar'."""
        resp = MagicMock()
        resp.status_code = 404
        resp.text = "model not found"
        mock_post.return_value = resp
        embedder = OllamaEmbedder()
        with pytest.raises(OllamaConnectionError) as exc_info:
            embedder.embed(["test"])
        assert "sidebar" in exc_info.value.detail


# ---------------------------------------------------------------------------
# TestChunk
# ---------------------------------------------------------------------------


class TestChunk:
    """Tests for the Chunk dataclass."""

    def test_basic_creation(self) -> None:
        """Chunk should store text, type, and metadata."""
        chunk = Chunk(text="hello", chunk_type="row_group", metadata={"key": "val"})
        assert chunk.text == "hello"
        assert chunk.chunk_type == "row_group"
        assert chunk.metadata == {"key": "val"}

    def test_default_metadata(self) -> None:
        """Metadata should default to empty dict."""
        chunk = Chunk(text="x", chunk_type="test")
        assert chunk.metadata == {}


# ---------------------------------------------------------------------------
# TestChunkBuilder
# ---------------------------------------------------------------------------


class TestChunkBuilder:
    """Tests for ChunkBuilder."""

    def test_returns_list_of_chunks(self) -> None:
        """build_chunks should return a list of Chunk objects."""
        builder = ChunkBuilder()
        chunks = builder.build_chunks(_sample_df())
        assert isinstance(chunks, list)
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_all_four_chunk_types_present(self) -> None:
        """All four chunk types should be present in the result."""
        builder = ChunkBuilder()
        chunks = builder.build_chunks(_sample_df())
        types = {c.chunk_type for c in chunks}
        assert types == {"row_group", "column_description", "statistical_summary", "categorical_distribution"}

    def test_no_row_cap(self) -> None:
        """All 200 rows should be covered by row_group chunks."""
        df = pd.DataFrame({"a": range(200), "b": range(200)})
        builder = ChunkBuilder()
        chunks = builder.build_chunks(df)
        row_chunks = [c for c in chunks if c.chunk_type == "row_group"]
        # every row index should appear in at least one chunk
        covered = set()
        for c in row_chunks:
            for i in range(c.metadata["start_row"], c.metadata["end_row"] + 1):
                covered.add(i)
        assert covered == set(range(200))

    def test_adaptive_window_narrow(self) -> None:
        """2 columns should give a large window size."""
        builder = ChunkBuilder()
        assert builder._calculate_window_size(2) == 15  # 60//2=30, clamped to 15

    def test_adaptive_window_wide(self) -> None:
        """20 columns should give a small window size."""
        builder = ChunkBuilder()
        assert builder._calculate_window_size(20) == 3  # 60//20=3

    def test_empty_df(self) -> None:
        """Empty DataFrame should produce no chunks."""
        builder = ChunkBuilder()
        assert builder.build_chunks(pd.DataFrame()) == []

    def test_one_column_description_per_column(self) -> None:
        """There should be exactly one column_description per column."""
        df = _sample_df()
        builder = ChunkBuilder()
        chunks = builder.build_chunks(df)
        desc_chunks = [c for c in chunks if c.chunk_type == "column_description"]
        assert len(desc_chunks) == len(df.columns)

    def test_statistical_summary_only_for_numeric(self) -> None:
        """statistical_summary chunks should only exist for numeric columns."""
        df = _sample_df()
        builder = ChunkBuilder()
        chunks = builder.build_chunks(df)
        stat_chunks = [c for c in chunks if c.chunk_type == "statistical_summary"]
        numeric_cols = set(df.select_dtypes(include="number").columns)
        stat_cols = {c.metadata["column"] for c in stat_chunks}
        assert stat_cols == numeric_cols


# ---------------------------------------------------------------------------
# TestSemanticSearch
# ---------------------------------------------------------------------------


class TestSemanticSearch:
    """Tests for SemanticSearch."""

    @patch("csv_qa.semantic_search.requests.post")
    def test_index_and_search_returns_chunks(self, mock_post: MagicMock) -> None:
        """After indexing, search should return Chunk objects."""
        dim = 4
        df = _sample_df()

        # Build chunks to verify the DataFrame is chunkable
        builder = ChunkBuilder()
        builder.build_chunks(df)

        # Each call to requests.post returns embeddings for all texts sent
        def side_effect(*args, **kwargs):
            texts = kwargs.get("json", args[1] if len(args) > 1 else {}).get("input", [])
            if isinstance(texts, list):
                vecs = [np.random.default_rng(i).random(dim).tolist() for i in range(len(texts))]
            else:
                vecs = [np.random.default_rng(0).random(dim).tolist()]
            return _mock_post_response(vecs)

        mock_post.side_effect = side_effect

        ss = SemanticSearch()
        ss.index(df)
        results = ss.search("what is the average age?", top_k=3)

        assert len(results) == 3
        assert all(isinstance(r, Chunk) for r in results)

    def test_search_before_index_returns_empty(self) -> None:
        """Searching before indexing should return an empty list."""
        ss = SemanticSearch()
        assert ss.search("anything") == []

    @patch("csv_qa.semantic_search.requests.post")
    def test_clear_resets_state(self, mock_post: MagicMock) -> None:
        """clear() should reset chunks, vectors, and cache."""

        def side_effect(*args, **kwargs):
            texts = kwargs.get("json", {}).get("input", [])
            if isinstance(texts, list):
                vecs = [[0.1] * 4 for _ in range(len(texts))]
            else:
                vecs = [[0.1] * 4]
            return _mock_post_response(vecs)

        mock_post.side_effect = side_effect

        ss = SemanticSearch()
        ss.index(_sample_df())
        assert ss._chunks is not None

        ss.clear()
        assert ss._chunks is None
        assert ss._vectors is None
        assert ss._embedder._cache == {}

    def test_custom_embed_model(self) -> None:
        """SemanticSearch should pass custom model to embedder."""
        ss = SemanticSearch(embed_model="my-model")
        assert ss._embedder.model_name == "my-model"
