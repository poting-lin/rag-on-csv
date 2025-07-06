"""
Vector Search Engine module for semantic search in CSV data
"""
import numpy as np
import hashlib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import traceback


class VectorSearchEngine:
    """
    Provides vector-based semantic search capabilities for CSV data with caching
    """

    def __init__(self, debug_mode=False):
        """Initialize the vector search engine"""
        self.debug_mode = debug_mode
        self.vectorizer = None
        self.vectors = None
        self.chunk_list = None

        # Caching attributes
        self._cache = {}
        self._current_cache_key = None

    def _generate_cache_key(self, chunks):
        """
        Generate a unique cache key for the given chunks

        Args:
            chunks: List of text chunks to generate key for

        Returns:
            str: Unique cache key
        """
        # Create a hash of the chunks content to use as cache key
        chunks_str = ''.join(chunks)
        return hashlib.md5(chunks_str.encode()).hexdigest()

    def _is_cache_valid(self, chunks):
        """
        Check if we have valid cached data for the given chunks

        Args:
            chunks: List of text chunks to check cache for

        Returns:
            bool: True if cache is valid, False otherwise
        """
        cache_key = self._generate_cache_key(chunks)
        return cache_key in self._cache

    def build_vector_index(self, chunks, force_rebuild=False):
        """
        Build a vector index from text chunks with caching support

        Args:
            chunks: List of text chunks to index
            force_rebuild: If True, rebuild even if cache exists

        Returns:
            tuple: (vectorizer, vectors, chunks)
        """
        try:
            cache_key = self._generate_cache_key(chunks)

            # Check if we can use cached data
            if not force_rebuild and cache_key in self._cache:
                cached_data = self._cache[cache_key]
                self.vectorizer = cached_data['vectorizer']
                self.vectors = cached_data['vectors']
                self.chunk_list = cached_data['chunk_list']
                self._current_cache_key = cache_key

                if self.debug_mode:
                    print(
                        f"Using cached vector index with {len(chunks)} chunks")

                return self.vectorizer, self.vectors, self.chunk_list

            # Build new index
            self.vectorizer = TfidfVectorizer()
            self.vectors = self.vectorizer.fit_transform(chunks)
            self.chunk_list = chunks
            self._current_cache_key = cache_key

            # Cache the results
            self._cache[cache_key] = {
                'vectorizer': self.vectorizer,
                'vectors': self.vectors,
                'chunk_list': self.chunk_list.copy()
            }

            if self.debug_mode:
                print(
                    f"Built and cached vector index with {len(chunks)} chunks")
                print(f"Cache now contains {len(self._cache)} entries")

            return self.vectorizer, self.vectors, self.chunk_list
        except Exception as e:
            print(f"Error in build_vector_index: {e}")
            traceback.print_exc()
            raise

    def retrieve_context(self, question, top_k=3):
        """
        Retrieve the most relevant context for a question

        Args:
            question: The question to find context for
            top_k: Number of top results to return

        Returns:
            str: Combined context from top matches
        """
        if self.vectorizer is None or self.vectors is None or self.chunk_list is None:
            raise ValueError(
                "Vector index not built. Call build_vector_index first.")

        try:
            # Transform the question into the same vector space
            question_vector = self.vectorizer.transform([question])

            # Calculate similarity between the question and all chunks
            similarities = cosine_similarity(question_vector, self.vectors)[0]

            # Get the indices of the top-k most similar chunks
            top_indices = np.argsort(similarities)[-top_k:][::-1]

            # Combine the top chunks into a single context
            context = "\n\n".join([self.chunk_list[i] for i in top_indices])

            if self.debug_mode:
                print(f"Retrieved {top_k} chunks as context")
                print(
                    f"Top similarity scores: {[similarities[i] for i in top_indices]}")

            return context
        except Exception as e:
            print(f"Error in retrieve_context: {e}")
            traceback.print_exc()
            return ""

    def clear_cache(self):
        """
        Clear all cached vector data
        """
        self._cache.clear()
        self._current_cache_key = None
        self.vectorizer = None
        self.vectors = None
        self.chunk_list = None

        if self.debug_mode:
            print("Cleared vector search cache")

    def get_cache_info(self):
        """
        Get information about the current cache state

        Returns:
            dict: Cache information including size and current key
        """
        return {
            'cache_entries': len(self._cache),
            'current_cache_key': self._current_cache_key,
            'has_active_index': self.vectorizer is not None
        }
