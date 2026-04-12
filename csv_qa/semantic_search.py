"""Semantic search module for RAG-on-CSV.

Provides embedding generation, chunk building, and vector similarity search
over CSV data for retrieval-augmented generation.
"""

import hashlib
import logging
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import requests

from .config import DEFAULT_EMBED_MODEL
from .exceptions import OllamaConnectionError

logger = logging.getLogger(__name__)


class OllamaEmbedder:
    """Wraps Ollama's /api/embed endpoint for generating dense vector embeddings."""

    def __init__(self, model_name: str | None = None) -> None:
        """Initialize the embedder with model name and base URL from env or defaults."""
        self.model_name = model_name or os.environ.get("OLLAMA_EMBED_MODEL", DEFAULT_EMBED_MODEL)
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
        self.embed_url = f"{base_url}/api/embed"
        self._cache: dict[str, list[float]] = {}

    def embed_single(self, text: str) -> list[float]:
        """Embed a single text string, using cache when available."""
        cache_key = hashlib.md5(text.encode()).hexdigest()
        if cache_key in self._cache:
            return self._cache[cache_key]
        result = self.embed([text])[0]
        self._cache[cache_key] = result
        return result

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts via the Ollama API.

        Raises:
            OllamaConnectionError: If Ollama is unreachable or returns an error.
        """
        try:
            response = requests.post(
                self.embed_url,
                json={"model": self.model_name, "input": texts},
                timeout=120,
            )
        except requests.exceptions.ConnectionError as e:
            raise OllamaConnectionError(detail=str(e)) from e
        except requests.exceptions.RequestException as e:
            raise OllamaConnectionError(detail=str(e)) from e

        if response.status_code == 404:
            raise OllamaConnectionError(detail=f"Model '{self.model_name}' not available. Pull it via the sidebar.")
        if response.status_code != 200:
            raise OllamaConnectionError(detail=f"HTTP {response.status_code}: {response.text}")

        return response.json()["embeddings"]


@dataclass
class Chunk:
    """A text chunk representing a piece of CSV data for semantic search."""

    text: str
    chunk_type: str
    metadata: dict = field(default_factory=dict)


class ChunkBuilder:
    """Builds text chunks from a DataFrame for embedding and retrieval."""

    def build_chunks(self, df: pd.DataFrame) -> list[Chunk]:
        """Build all chunk types from a DataFrame."""
        if df.empty:
            return []

        chunks: list[Chunk] = []
        chunks.extend(self._build_row_group_chunks(df))
        chunks.extend(self._build_column_description_chunks(df))
        chunks.extend(self._build_statistical_summary_chunks(df))
        chunks.extend(self._build_categorical_distribution_chunks(df))
        return chunks

    def _calculate_window_size(self, num_columns: int) -> int:
        """Calculate adaptive sliding window size based on column count."""
        return max(3, min(15, 60 // max(num_columns, 1)))

    def _build_row_group_chunks(self, df: pd.DataFrame) -> list[Chunk]:
        """Build chunks from sliding windows over all rows."""
        chunks: list[Chunk] = []
        window = self._calculate_window_size(len(df.columns))
        columns = list(df.columns)

        for start in range(0, len(df), window):
            end = min(start + window, len(df))
            rows_text = []
            for idx in range(start, end):
                row = df.iloc[idx]
                pairs = ", ".join(f"{col}: {row[col]}" for col in columns)
                rows_text.append(f"Row {idx}: {pairs}")

            chunks.append(
                Chunk(
                    text="\n".join(rows_text),
                    chunk_type="row_group",
                    metadata={
                        "start_row": start,
                        "end_row": end - 1,
                        "columns": columns,
                    },
                )
            )
        return chunks

    def _build_column_description_chunks(self, df: pd.DataFrame) -> list[Chunk]:
        """Build one description chunk per column."""
        chunks: list[Chunk] = []
        for col in df.columns:
            series = df[col]
            missing = series.isna().sum()

            if pd.api.types.is_numeric_dtype(series):
                text = (
                    f"Column '{col}' is numeric. "
                    f"Range: {series.min()} to {series.max()}. "
                    f"Mean: {series.mean():.2f}. "
                    f"Missing: {missing}."
                )
            else:
                unique_count = series.nunique()
                top_values = series.value_counts().head(5).index.tolist()
                text = (
                    f"Column '{col}' is categorical. "
                    f"Unique values: {unique_count}. "
                    f"Top 5: {top_values}. "
                    f"Missing: {missing}."
                )

            chunks.append(
                Chunk(
                    text=text,
                    chunk_type="column_description",
                    metadata={"column": col},
                )
            )
        return chunks

    def _build_statistical_summary_chunks(self, df: pd.DataFrame) -> list[Chunk]:
        """Build one statistical summary chunk per numeric column."""
        chunks: list[Chunk] = []
        numeric_cols = df.select_dtypes(include="number").columns

        for col in numeric_cols:
            series = df[col]
            text = (
                f"Statistics for '{col}': "
                f"min={series.min()}, max={series.max()}, "
                f"mean={series.mean():.2f}, std={series.std():.2f}, "
                f"median={series.median()}."
            )
            chunks.append(
                Chunk(
                    text=text,
                    chunk_type="statistical_summary",
                    metadata={"column": col},
                )
            )
        return chunks

    def _build_categorical_distribution_chunks(self, df: pd.DataFrame) -> list[Chunk]:
        """Build distribution chunks for categorical columns with <= 50 unique values."""
        chunks: list[Chunk] = []
        cat_cols = df.select_dtypes(exclude="number").columns

        for col in cat_cols:
            if df[col].nunique() > 50:
                continue

            counts = df[col].value_counts().head(10)
            lines = [f"  {val}: {cnt}" for val, cnt in counts.items()]
            text = f"Distribution of '{col}' (top 10):\n" + "\n".join(lines)

            chunks.append(
                Chunk(
                    text=text,
                    chunk_type="categorical_distribution",
                    metadata={"column": col},
                )
            )
        return chunks


class SemanticSearch:
    """Indexes DataFrame chunks and performs cosine similarity search."""

    def __init__(self, embed_model: str | None = None) -> None:
        """Initialize with an embedder and chunk builder."""
        self._embedder = OllamaEmbedder(model_name=embed_model)
        self._chunk_builder = ChunkBuilder()
        self._chunks: list[Chunk] | None = None
        self._vectors: np.ndarray | None = None

    def index(self, df: pd.DataFrame) -> None:
        """Build chunks from the DataFrame and embed them."""
        self._chunks = self._chunk_builder.build_chunks(df)
        if not self._chunks:
            self._vectors = None
            return

        texts = [c.text for c in self._chunks]
        embeddings = self._embedder.embed(texts)
        self._vectors = np.array(embeddings)

    def search(self, query: str, top_k: int = 5, min_score: float = 0.3) -> list[Chunk]:
        """Search for chunks most similar to the query.

        Drops chunks below min_score to avoid polluting the LLM context with
        irrelevant rows, and dedupes by chunk text hash. Returns chunks
        sorted by similarity (highest first), or an empty list if no index
        has been built or nothing clears the threshold.
        """
        if self._chunks is None or self._vectors is None:
            return []

        query_vec = np.array(self._embedder.embed_single(query))

        norms = np.linalg.norm(self._vectors, axis=1) * np.linalg.norm(query_vec)
        norms = np.where(norms == 0, 1e-10, norms)
        similarities = self._vectors @ query_vec / norms

        ranked = np.argsort(similarities)[::-1]

        seen: set[str] = set()
        results: list[Chunk] = []
        for idx in ranked:
            if len(results) >= top_k:
                break
            score = float(similarities[idx])
            if score < min_score:
                break
            chunk = self._chunks[idx]
            key = hashlib.md5(chunk.text.encode()).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            results.append(chunk)

        if not results:
            logger.debug(
                "No chunks passed min_score=%.2f (best=%.3f)",
                min_score,
                float(similarities[ranked[0]]) if len(ranked) else 0.0,
            )
        return results

    def clear(self) -> None:
        """Reset the index and clear the embedding cache."""
        self._chunks = None
        self._vectors = None
        self._embedder._cache.clear()
