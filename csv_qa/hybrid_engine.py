"""
Hybrid Engine - combines structured queries, semantic search, and LLM analysis.
"""

import logging
from typing import Any

import pandas as pd

from .data_handler import build_schema_card
from .exceptions import QueryResult
from .structured_query_engine import StructuredQueryEngine
from .semantic_search import SemanticSearch
from .ollama_client import OllamaAPIClient

logger = logging.getLogger(__name__)


class HybridCSVEngine:
    """Combines multiple approaches for comprehensive CSV analysis."""

    def __init__(self, model_name: str = "llama3.2:1b") -> None:
        """Initialize the hybrid engine."""
        self.structured_engine = StructuredQueryEngine()
        self.semantic_search = SemanticSearch()
        self.llm_client = OllamaAPIClient(model_name=model_name)

    def answer_question(self, question: str, df: pd.DataFrame) -> QueryResult:
        """Use multiple approaches to answer the question.

        Args:
            question: The user's question.
            df: The pandas DataFrame to analyze.

        Returns:
            QueryResult with the answer and metadata.
        """
        logger.debug("Hybrid engine processing: %s", question)

        # 1. Try structured approach first (fastest)
        structured_result = self._try_structured_query(question, df)
        if structured_result.success and structured_result.confidence > 0.8:
            logger.debug("Structured query succeeded with high confidence")
            return structured_result

        # 2. Try semantic analysis
        semantic_result = self._try_semantic_analysis(question, df)
        if semantic_result.success:
            logger.debug("Semantic analysis succeeded")

            # Combine with partial structured result if available
            if structured_result.success:
                combined = self._combine_results(structured_result, semantic_result)
                return QueryResult.ok(
                    data=combined,
                    engine="hybrid",
                    confidence=0.85,
                    metadata=structured_result.metadata,
                )

            return semantic_result

        # 3. Fall back to LLM with basic context
        llm_result = self._try_llm_analysis(question, df)
        return llm_result

    def _try_structured_query(self, question: str, df: pd.DataFrame) -> QueryResult:
        """Attempt direct DataFrame operations."""
        try:
            result = self.structured_engine.execute_query(question, df)

            if result.success:
                confidence = self._calculate_structured_confidence(question, result)
                return QueryResult.ok(
                    data=result.data,
                    engine="structured",
                    confidence=confidence,
                    metadata=result.metadata,
                )

            return result

        except Exception as e:
            logger.debug("Structured query failed: %s", e)
            return QueryResult.fail(
                error_code="NO_MATCH",
                error_message=str(e),
                engine="structured",
            )

    def _try_semantic_analysis(self, question: str, df: pd.DataFrame) -> QueryResult:
        """Attempt semantic analysis using embedding-based search."""
        try:
            self.semantic_search.index(df)
            chunks = self.semantic_search.search(question, top_k=5)

            if not chunks:
                return QueryResult.fail(
                    error_code="NO_CONTEXT",
                    error_message="No relevant context found.",
                    engine="semantic",
                )

            retrieved = "\n\n".join(c.text for c in reversed(chunks))
            context = f"## Schema\n{build_schema_card(df)}\n\n## Retrieved rows\n{retrieved}"
            llm_response = self._analyze_with_llm(question, context)

            if llm_response:
                confidence = self._calculate_semantic_confidence(question, context, llm_response)
                return QueryResult.ok(
                    data=llm_response,
                    engine="semantic",
                    confidence=confidence,
                    metadata={"context": context},
                )

            return QueryResult.fail(
                error_code="ALL_ENGINES_FAILED",
                error_message="LLM analysis of semantic context failed.",
                engine="semantic",
            )

        except Exception as e:
            logger.warning("Semantic analysis failed: %s", e)
            return QueryResult.fail(
                error_code="NO_CONTEXT",
                error_message=str(e),
                engine="semantic",
            )

    def _try_llm_analysis(self, question: str, df: pd.DataFrame) -> QueryResult:
        """Fall back to LLM with basic DataFrame information."""
        try:
            basic_context = self._create_basic_context(df)
            llm_response = self._analyze_with_llm(question, basic_context)

            if llm_response:
                return QueryResult.ok(
                    data=llm_response,
                    engine="llm",
                    confidence=0.6,
                    metadata={"context": basic_context},
                )

            return QueryResult.fail(
                error_code="ALL_ENGINES_FAILED",
                error_message="LLM analysis with basic context failed.",
                engine="llm",
            )

        except Exception as e:
            logger.warning("LLM analysis failed: %s", e)
            return QueryResult.fail(
                error_code="ALL_ENGINES_FAILED",
                error_message=str(e),
                engine="llm",
            )

    def _analyze_with_llm(self, question: str, context: str) -> str | None:
        """Use LLM to analyze context and answer question."""
        try:
            response = self.llm_client.ask(context, question)
            logger.debug("LLM response: %s...", response[:200])
            return response
        except Exception as e:
            logger.warning("LLM analysis error: %s", e)
            return None

    def _create_basic_context(self, df: pd.DataFrame) -> str:
        """Create basic context about the DataFrame for LLM."""
        return f"## Schema\n{build_schema_card(df)}"

    def _calculate_structured_confidence(self, question: str, result: QueryResult) -> float:
        """Calculate confidence score for structured query results."""
        base_confidence = 0.9

        data = result.metadata.get("data")
        if data is not None and hasattr(data, "__len__") and len(data) == 0:
            base_confidence *= 0.7

        if len(result.data) < 20:
            base_confidence *= 0.8

        question_lower = question.lower()
        if any(word in question_lower for word in ["max", "min", "count", "show"]):
            base_confidence *= 1.1

        return min(base_confidence, 1.0)

    def _calculate_semantic_confidence(self, question: str, context: str, response: str) -> float:
        """Calculate confidence score for semantic analysis."""
        base_confidence = 0.7

        if any(char.isdigit() for char in response):
            base_confidence += 0.1

        for line in context.split("\n"):
            if "Column" in line and ":" in line:
                col_name = line.split(":")[0].replace("Column ", "").strip()
                if col_name.lower() in response.lower():
                    base_confidence += 0.05

        generic_phrases = ["i don't have", "cannot determine", "not enough information"]
        if any(phrase in response.lower() for phrase in generic_phrases):
            base_confidence *= 0.5

        analysis_keywords = ["statistics", "average", "maximum", "minimum", "records"]
        if any(keyword in response.lower() for keyword in analysis_keywords):
            base_confidence += 0.1

        return min(base_confidence, 1.0)

    def _combine_results(self, structured_result: QueryResult, semantic_result: QueryResult) -> str:
        """Combine insights from structured and semantic analysis."""
        structured_answer = structured_result.data
        semantic_answer = semantic_result.data

        if len(structured_answer) < 50 and len(semantic_answer) > 50:
            return f"{structured_answer}\n\nAdditional context: {semantic_answer}"

        if len(structured_answer) > 50 and len(semantic_answer) > 50:
            return f"Direct analysis: {structured_answer}\n\nDetailed analysis: {semantic_answer}"

        return structured_answer if len(structured_answer) > len(semantic_answer) else semantic_answer

    def get_engine_status(self) -> dict[str, Any]:
        """Get status information about the engines."""
        status: dict[str, Any] = {
            "structured_engine": "ready",
            "semantic_search": "ready" if self.semantic_search._chunks is not None else "not_indexed",
            "llm_client": "ready",
        }
        return status

    def clear_cache(self) -> None:
        """Clear all caches."""
        self.semantic_search.clear()
        logger.debug("Hybrid engine cache cleared")
