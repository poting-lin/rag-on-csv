"""
Vector Search Engine module for semantic search in CSV data.
"""
import hashlib
import logging

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .exceptions import VectorSearchError

logger = logging.getLogger(__name__)


class VectorSearchEngine:
    """Provides vector-based semantic search capabilities for CSV data with caching."""

    def __init__(self) -> None:
        """Initialize the vector search engine."""
        self.vectorizer: TfidfVectorizer | None = None
        self.vectors = None
        self.chunk_list: list[str] | None = None

        self._cache: dict = {}
        self._current_cache_key: str | None = None

    def _generate_cache_key(self, chunks: list[str]) -> str:
        """Generate a unique cache key for the given chunks."""
        chunks_str = "".join(chunks)
        return hashlib.md5(chunks_str.encode()).hexdigest()

    def _is_cache_valid(self, chunks: list[str]) -> bool:
        """Check if we have valid cached data for the given chunks."""
        cache_key = self._generate_cache_key(chunks)
        return cache_key in self._cache

    def build_vector_index(
        self, chunks: list[str], force_rebuild: bool = False
    ) -> tuple[TfidfVectorizer, object, list[str]]:
        """Build a vector index from text chunks with caching support.

        Raises:
            VectorSearchError: If vector index cannot be built.
        """
        try:
            cache_key = self._generate_cache_key(chunks)

            if not force_rebuild and cache_key in self._cache:
                cached_data = self._cache[cache_key]
                self.vectorizer = cached_data["vectorizer"]
                self.vectors = cached_data["vectors"]
                self.chunk_list = cached_data["chunk_list"]
                self._current_cache_key = cache_key

                logger.debug("Using cached vector index with %d chunks", len(chunks))

                return self.vectorizer, self.vectors, self.chunk_list

            self.vectorizer = TfidfVectorizer()
            self.vectors = self.vectorizer.fit_transform(chunks)
            self.chunk_list = chunks
            self._current_cache_key = cache_key

            self._cache[cache_key] = {
                "vectorizer": self.vectorizer,
                "vectors": self.vectors,
                "chunk_list": self.chunk_list.copy(),
            }

            logger.debug("Built and cached vector index with %d chunks", len(chunks))
            logger.debug("Cache now contains %d entries", len(self._cache))

            return self.vectorizer, self.vectors, self.chunk_list
        except Exception as e:
            logger.error("Error in build_vector_index: %s", e, exc_info=True)
            raise VectorSearchError(detail=str(e)) from e

    def retrieve_context(self, question: str, top_k: int = 3) -> str:
        """Retrieve the most relevant context for a question.

        Raises:
            VectorSearchError: If retrieval fails unexpectedly.
        """
        if self.vectorizer is None or self.vectors is None or self.chunk_list is None:
            raise ValueError("Vector index not built. Call build_vector_index first.")

        try:
            question_vector = self.vectorizer.transform([question])
            similarities = cosine_similarity(question_vector, self.vectors)[0]
            top_indices = np.argsort(similarities)[-top_k:][::-1]

            context = "\n\n".join([self.chunk_list[i] for i in top_indices])

            logger.debug("Retrieved %d chunks as context", top_k)
            logger.debug(
                "Top similarity scores: %s",
                [similarities[i] for i in top_indices],
            )

            return context
        except Exception as e:
            logger.error("Error in retrieve_context: %s", e, exc_info=True)
            raise VectorSearchError(detail=str(e)) from e

    def clear_cache(self) -> None:
        """Clear all cached vector data."""
        self._cache.clear()
        self._current_cache_key = None
        self.vectorizer = None
        self.vectors = None
        self.chunk_list = None

        logger.debug("Cleared vector search cache")

    def get_cache_info(self) -> dict:
        """Get information about the current cache state."""
        return {
            "cache_entries": len(self._cache),
            "current_cache_key": self._current_cache_key,
            "has_active_index": self.vectorizer is not None,
        }
