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

    def retrieve_context(
        self,
        question: str,
        top_k: int = 3,
        min_score: float = 0.15,
        char_budget: int = 1500,
    ) -> str:
        """Retrieve relevant context for a question.

        Filters below min_score to drop noise, dedupes identical chunks,
        and caps total size at char_budget. Most relevant chunk is placed
        LAST so it sits closest to the question in the final prompt
        (mitigates lost-in-the-middle).

        Raises:
            VectorSearchError: If retrieval fails unexpectedly.
        """
        if self.vectorizer is None or self.vectors is None or self.chunk_list is None:
            raise ValueError("Vector index not built. Call build_vector_index first.")

        try:
            question_vector = self.vectorizer.transform([question])
            similarities = cosine_similarity(question_vector, self.vectors)[0]
            ranked_indices = np.argsort(similarities)[::-1]

            seen: set[str] = set()
            selected: list[tuple[int, float]] = []
            total_chars = 0

            for idx in ranked_indices:
                if len(selected) >= top_k:
                    break
                score = float(similarities[idx])
                if score < min_score:
                    break
                chunk = self.chunk_list[idx]
                key = hashlib.md5(chunk.encode()).hexdigest()
                if key in seen:
                    continue
                added = len(chunk) + (2 if selected else 0)
                if total_chars + added > char_budget:
                    break
                seen.add(key)
                selected.append((idx, score))
                total_chars += added

            if not selected:
                logger.debug(
                    "No chunks passed min_score=%.2f (best=%.3f)",
                    min_score,
                    float(similarities[ranked_indices[0]]) if len(ranked_indices) else 0.0,
                )
                return ""

            ordered = list(reversed(selected))
            context = "\n\n".join(self.chunk_list[i] for i, _ in ordered)

            logger.debug(
                "Retrieved %d chunks (%d chars) scores=%s",
                len(ordered),
                total_chars,
                [round(s, 3) for _, s in ordered],
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
