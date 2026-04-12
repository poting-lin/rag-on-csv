"""
CSV Question Answerer module - main class that orchestrates the question answering process.
"""

import logging
import os
import re
import json
from collections import OrderedDict
import pandas as pd
from csv_qa.data_handler import CSVDataHandler
from csv_qa.fuzzy_matcher import FuzzyMatcher
from csv_qa.question_parser import QuestionParser
from csv_qa.semantic_search import SemanticSearch
from csv_qa.ollama_client import OllamaAPIClient
from csv_qa.context_memory import ConversationContext
from csv_qa.question_router import QuestionRouter
from csv_qa.structured_query_engine import StructuredQueryEngine
from csv_qa.hybrid_engine import HybridCSVEngine
from csv_qa.exceptions import (
    OllamaConnectionError,
    OllamaTimeoutError,
    OllamaResponseError,
)

logger = logging.getLogger(__name__)


RESPONSE_CACHE_MAX_SIZE = 100


class CSVQuestionAnswerer:
    """Main class that orchestrates the CSV question answering process."""

    def __init__(
        self,
        model_name: str = "llama3.2:1b",
        embed_model: str | None = None,
        enable_context_memory: bool = True,
        use_enhanced_engines: bool = True,
    ) -> None:
        """Initialize the CSV Question Answerer.

        Args:
            model_name: Name of the Ollama model to use.
            embed_model: Name of the embedding model for semantic search.
            enable_context_memory: Whether to enable conversation context memory.
            use_enhanced_engines: Whether to use enhanced multi-engine approach.
        """
        self.model_name = model_name
        self.enable_context_memory = enable_context_memory
        self.use_enhanced_engines = use_enhanced_engines

        # Flag to prevent duplicate conversation storage
        self._conversation_stored = False

        # Initialize components
        self.data_handler = CSVDataHandler()
        self.fuzzy_matcher = FuzzyMatcher()

        # Initialize enhanced engines if enabled
        if self.use_enhanced_engines:
            self.question_router = QuestionRouter()
            self.structured_engine = StructuredQueryEngine()
            self.hybrid_engine = HybridCSVEngine(model_name=model_name)

        # Semantic search replaces both vector search and enhanced vector search
        self.semantic_search = SemanticSearch(embed_model=embed_model)
        self.ollama_client = OllamaAPIClient(model_name=model_name)
        self.question_parser = None  # Will be initialized after loading CSV

        # Response cache: (question_lower, csv_path, mtime, model_name) -> answer
        self._response_cache: OrderedDict = OrderedDict()

        # Register cache clearing callback with data handler
        self.data_handler.add_cache_clear_callback(self.semantic_search.clear)
        self.data_handler.add_cache_clear_callback(self._response_cache.clear)

        if self.use_enhanced_engines:
            self.data_handler.add_cache_clear_callback(self.hybrid_engine.clear_cache)

        # Initialize context memory
        if self.enable_context_memory:
            self.context_memory = ConversationContext(
                max_turns=10,
                max_age_minutes=30,
            )
        else:
            self.context_memory = None

    def load_csv(self, csv_path: str) -> list[str]:
        """Load the CSV file and initialize components that depend on CSV data.

        Args:
            csv_path: Path to the CSV file.

        Returns:
            List of column names in the CSV.
        """
        self.data_handler.load_csv(csv_path)
        self.question_parser = QuestionParser(
            csv_columns=self.data_handler.get_columns(),
        )

        logger.debug("CSV loaded, semantic search index will be built on first query")

        return self.data_handler.get_columns()

    def check_for_quick_suggestions(
        self, question: str, csv_path: str | None = None
    ) -> tuple[str, list[str]] | None:
        """Fast local typo check before streaming.

        Mirrors the fuzzy-match branch in _get_answer_original so the
        streaming UI can surface suggestions without running the full
        LLM pipeline. Returns (message, suggestions) on hit, or None if
        the question should fall through to normal answering.
        """
        if csv_path and not self.data_handler.is_loaded():
            self.load_csv(csv_path)
        df = self.data_handler.get_dataframe()
        if df is None:
            return None

        search_term = question.lower().strip()
        if len(search_term) >= 10:
            return None

        if any(search_term in str(row).lower() for _, row in df.iterrows()):
            return None

        similar_values = self.fuzzy_matcher.find_similar_values(search_term, df)
        if not similar_values:
            return None

        if len(similar_values) == 1:
            message = (
                f"I couldn't find '{question.strip()}' in the CSV data. "
                f"Did you mean '{similar_values[0]}'? If yes, please ask about that instead."
            )
            return message, [similar_values[0]]

        suggestions_str = ", ".join(f"'{v}'" for v in similar_values[:3])
        message = (
            f"I couldn't find '{question.strip()}' in the CSV data. "
            f"Did you mean one of these: {suggestions_str}? "
            "If yes, please ask about that instead."
        )
        return message, similar_values[:3]

    def answer_question_stream(self, question: str, csv_path: str | None = None):
        """Yield answer tokens for a question, streaming from the LLM.

        Uses the semantic search path end-to-end so the LLM's generation
        can be streamed. Structured or direct-lookup answers should be
        obtained via the non-streaming answer_question().

        Yields:
            Strings (token chunks). Network/LLM errors propagate as the
            usual OllamaConnectionError / OllamaTimeoutError / OllamaResponseError
            so the caller can distinguish them.
        """
        if csv_path and not self.data_handler.is_loaded():
            self.load_csv(csv_path)
        if not self.data_handler.is_loaded():
            yield "CSV not loaded."
            return

        df = self.data_handler.get_dataframe()
        self.semantic_search.index(df)
        chunks = self.semantic_search.search(question, top_k=5)
        if not chunks:
            yield "I couldn't find any matching information in the CSV data."
            return

        retrieved = "\n\n".join(c.text for c in reversed(chunks))
        context = self._build_llm_context(retrieved)
        yield from self.ollama_client.ask_stream(context, question)

    def _build_llm_context(self, retrieved: str) -> str:
        """Prepend the cached schema card to retrieved context.

        Gives the LLM a stable data anchor (column types, ranges, example
        row) in addition to whatever rows were retrieved for this question.
        """
        if not self.data_handler.is_loaded():
            return retrieved
        try:
            card = self.data_handler.get_schema_card()
        except Exception as e:
            logger.debug("Could not build schema card: %s", e)
            return retrieved
        if not retrieved:
            return f"## Schema\n{card}"
        return f"## Schema\n{card}\n\n## Retrieved rows\n{retrieved}"

    def format_matches(self, matches, target_column=None, id_column=None, id_value=None):
        """Format matching rows into a readable answer.

        Args:
            matches: DataFrame of matching rows.
            target_column: The column the user is asking about.
            id_column: The column used to identify the row.
            id_value: The value used to identify the row.

        Returns:
            Formatted answer with all matches.
        """
        if matches.empty:
            if id_column and id_value:
                return f"I don't have information about {id_column} {id_value}."
            return "No matches found."

        # Format the results
        if len(matches) == 1:
            # Single match
            if target_column:
                # User asked about a specific column
                return f"{target_column} for {id_column} {id_value} is: {matches.iloc[0][target_column]}"
            else:
                # User didn't specify a target column, show all columns
                result = []
                for col in self.data_handler.get_columns():
                    result.append(f"- {col}: {matches.iloc[0][col]}")
                return "\n".join(result)
        else:
            # Multiple matches
            if id_column and id_value:
                header = f"Found {len(matches)} matches for {id_column} {id_value}:"
            elif target_column:
                header = f"Found {len(matches)} matches for {target_column}:"
            else:
                header = f"Found {len(matches)} matches:"

            result = [header, ""]

            # Add each match
            for i, (_, row) in enumerate(matches.iterrows(), 1):
                result.append(f"Match {i}:")
                for col in self.data_handler.get_columns():
                    result.append(f"- {col}: {row[col]}")
                if i < len(matches):
                    result.append("")  # Add blank line between matches

            return "\n".join(result)

    def answer_question(self, question: str, csv_path: str | None = None) -> str | tuple:
        """Answer a question based on CSV data with context memory support.

        Args:
            question: Question to answer.
            csv_path: Path to CSV file (optional, if already loaded).

        Returns:
            Either a string answer or a tuple (answer, suggestions, is_suggestion)
            for spell corrections and other suggestions.
        """
        if not question:
            return "Please ask a question."

        logger.info("Question received: %s", question)

        cache_key = self._cache_key(question, csv_path)
        if cache_key is not None and cache_key in self._response_cache:
            self._response_cache.move_to_end(cache_key)
            cached = self._response_cache[cache_key]
            logger.info("Response cache hit")
            return cached

        # Reset conversation storage flag
        self._conversation_stored = False

        # Check for context memory and handle follow-up questions
        context_enhanced_question = question
        context_info = None

        if self.enable_context_memory and self.context_memory:
            # Detect if this is a follow-up question
            context_info = self.context_memory.detect_follow_up_intent(question)

            if context_info["is_follow_up"]:
                logger.debug(
                    "Detected follow-up question: %s",
                    context_info["reference_type"],
                )

                # Enhance the question with context
                if context_info["reference_type"] == "pronoun":
                    # For pronoun references, add context to help resolve them
                    context_enhanced_question = self._resolve_pronouns_with_context(
                        question, context_info["referenced_turns"]
                    )
                elif context_info["reference_type"] == "continuation":
                    # For continuation, the original question is fine but we'll use context in LLM
                    pass

        # Get the answer using the potentially enhanced question
        answer = self._get_answer_internal(context_enhanced_question, csv_path, context_info)

        # Store this interaction in context memory only if not already stored
        should_store = (
            self.enable_context_memory
            and self.context_memory
            and not isinstance(answer, tuple)
            and not self._conversation_stored
        )
        if should_store:
            self._store_conversation_turn(question, answer, context_info)

        # Log response status and content preview
        if isinstance(answer, tuple):
            answer_text, _, is_suggestion = answer
            logger.info("Response status: suggestion, preview: %s", str(answer_text)[:200])
        else:
            logger.info("Response status: ok, preview: %s", str(answer)[:200])

        if cache_key is not None and not isinstance(answer, tuple):
            self._cache_response(cache_key, answer)

        return answer

    def _cache_key(self, question: str, csv_path: str | None) -> tuple | None:
        """Build a cache key, or return None if caching should be skipped.

        Caching is skipped when context memory is active (stateful answers)
        or the CSV path is unknown (cannot detect file changes).
        """
        if self.enable_context_memory and self.context_memory:
            return None
        path = csv_path or self.data_handler.current_csv_path
        if not path:
            return None
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            return None
        return (question.strip().lower(), path, mtime, self.model_name)

    def _cache_response(self, key: tuple, answer) -> None:
        """Insert an answer into the LRU cache, evicting oldest if full."""
        self._response_cache[key] = answer
        self._response_cache.move_to_end(key)
        while len(self._response_cache) > RESPONSE_CACHE_MAX_SIZE:
            self._response_cache.popitem(last=False)

    def _get_answer_internal(self, question: str, csv_path: str | None = None, context_info=None) -> str | tuple:
        """Internal method to get answer with enhanced multi-engine approach."""

        # Use enhanced engines if available
        if self.use_enhanced_engines and self.data_handler.get_dataframe() is not None:
            logger.info("Trying enhanced engines")
            enhanced_result = self._get_answer_with_enhanced_engines(question, context_info)
            if enhanced_result:
                return enhanced_result

        # Fall back to original implementation
        logger.info("Enhanced engines returned no result, falling back to original engine")
        return self._get_answer_original(question, csv_path, context_info)

    def _answer_via_semantic_search(self, question: str, df: pd.DataFrame) -> str | None:
        """Answer a question using semantic search over the DataFrame."""
        try:
            self.semantic_search.index(df)
            chunks = self.semantic_search.search(question, top_k=5)
            if not chunks:
                logger.debug("No relevant chunks found for question")
                return None
            retrieved = "\n\n".join(c.text for c in reversed(chunks))
            return self.ollama_client.ask(self._build_llm_context(retrieved), question)
        except OllamaConnectionError as e:
            logger.error("Embedding model error: %s", e)
            return None
        except (OllamaTimeoutError, OllamaResponseError) as e:
            logger.error("Ollama error during semantic search: %s", e, exc_info=True)
            return None

    def _get_answer_with_enhanced_engines(self, question: str, context_info=None):
        """Get answer using the enhanced multi-engine approach."""
        try:
            df = self.data_handler.get_dataframe()
            if df is None:
                return None

            # Apply context filter if needed
            filtered_df = df
            if self.enable_context_memory and self.context_memory and context_info:
                context_filter = self.context_memory.get_context_data_filter(question)
                if context_filter and "filter_column" in context_filter:
                    filtered_result = self._apply_context_filter(context_filter)
                    if filtered_result is not None:
                        filtered_df = filtered_result
                        logger.debug("Applied context filter: %d records", len(filtered_df))

            # Route the question to the appropriate engine
            columns = self.data_handler.get_columns()
            engine_type = self.question_router.route_question(question, columns)

            logger.info("Question routed to engine: %s", engine_type)

            # Handle different engine types
            if engine_type == "structured":
                result = self.structured_engine.execute_query(question, filtered_df)
                if result.success:
                    # Store result data for context memory
                    if self.enable_context_memory and self.context_memory:
                        # Extract filter info from question for context memory
                        result_count = self._count_results_in_answer(result.data)
                        extracted_result_data = None
                        if result_count > 0:
                            extracted_result_data = self._extract_filter_info_from_question(question, result_count)

                        self._store_conversation_turn(
                            question, result.data, context_info, result_data=extracted_result_data
                        )
                        self._conversation_stored = True
                    return result.data

            elif engine_type == "semantic":
                response = self._answer_via_semantic_search(question, filtered_df)

                if response:
                    if self.enable_context_memory and self.context_memory:
                        # Extract filter info from question for context memory
                        result_count = self._count_results_in_answer(response)
                        extracted_result_data = None
                        if result_count > 0:
                            extracted_result_data = self._extract_filter_info_from_question(question, result_count)

                        self._store_conversation_turn(
                            question, response, context_info, result_data=extracted_result_data
                        )
                        self._conversation_stored = True
                    return response

            elif engine_type == "hybrid":
                result = self.hybrid_engine.answer_question(question, filtered_df)
                if result.success:
                    if self.enable_context_memory and self.context_memory:
                        # Extract filter info from question for context memory
                        result_count = self._count_results_in_answer(result.data)
                        extracted_result_data = None
                        if result_count > 0:
                            extracted_result_data = self._extract_filter_info_from_question(question, result_count)

                        self._store_conversation_turn(
                            question, result.data, context_info, result_data=extracted_result_data
                        )
                        self._conversation_stored = True
                    return result.data

            # If enhanced engines fail, return None to fall back to original
            return None

        except Exception as e:
            logger.error("Enhanced engine error: %s", e, exc_info=True)
            return None

    def _get_answer_original(self, question: str, csv_path: str | None = None, context_info=None) -> str | tuple:
        """Original implementation as fallback."""
        # Check if this is a complex query that needs to be broken down into steps
        if self._is_complex_query(question):
            logger.debug("Detected complex query: %s", question)
            return self._execute_query_steps(question, csv_path)

        try:
            # Make sure CSV is loaded
            if self.data_handler.get_dataframe() is None:
                self.load_csv(csv_path)

            # Check if this is a follow-up question that should use filtered context
            context_filter = None
            filtered_dataframe = None
            if self.enable_context_memory and self.context_memory:
                context_filter = self.context_memory.get_context_data_filter(question)
                logger.debug("Context filter result: %s", context_filter)

                # Apply the context filter to get the subset of data
                if context_filter and "filter_column" in context_filter:
                    filtered_dataframe = self._apply_context_filter(context_filter)
                    logger.debug(
                        "Filtered dataframe result: %s",
                        filtered_dataframe is not None,
                    )
                    if filtered_dataframe is not None:
                        logger.debug(
                            "Context filter applied: %d records from filter",
                            len(filtered_dataframe),
                        )

            # Create chunks for vector search (use filtered data if available)
            if filtered_dataframe is not None:
                # Temporarily replace the dataframe for this query
                original_df = self.data_handler.get_dataframe()
                self.data_handler._dataframe = filtered_dataframe
                chunks = self.data_handler.create_chunks()
                # We'll restore the original dataframe at the end
            else:
                chunks = self.data_handler.create_chunks()

            # Extract information from the question
            question_info = self.question_parser.extract_question_info(question)

            # Check if this is an aggregation question on filtered data
            logger.debug(
                "Checking aggregation: filtered_dataframe is None: %s",
                filtered_dataframe is None,
            )
            logger.debug(
                "Is aggregation question: %s",
                self._is_aggregation_question(question),
            )

            if filtered_dataframe is not None and self._is_aggregation_question(question):
                logger.debug(
                    "Processing aggregation question on filtered data: %d records",
                    len(filtered_dataframe),
                )
                result = self._handle_aggregation_on_filtered_data(question, filtered_dataframe)
                if result:
                    return result

            # Check if this is a help query
            if question_info.get("is_help_query", False):
                # Return suggested questions
                return self.generate_suggested_questions(csv_path)

            # Check if this is an analysis request
            if self._is_analysis_request(question):
                return self._perform_analysis(csv_path, question)

            # Check if this is a command with auto-correction
            if question_info.get("command_correction"):
                corrected_command = question_info["command_correction"]
                return f"Did you mean '{corrected_command}'?", [corrected_command], True

            # Extract detailed information for regular questions
            target_column = question_info.get("target_column")
            id_column = question_info.get("id_column")
            id_value = question_info.get("id_value")
            comparison_info = question_info.get("comparison_info")
            command_info = question_info

            # Check if we have a special command query
            if command_info and "command" in command_info:
                # Check if this is a spell correction that should be auto-corrected
                if command_info.get("auto_correct"):
                    # Remove the auto_correct flag to prevent infinite recursion
                    command_info.pop("auto_correct", None)
                    command_info.pop("original", None)

                command = command_info["command"]
                operation = command_info.get("operation")
                columns = command_info.get("columns", [])

                logger.debug(
                    "Processing command: %s, operation: %s, columns: %s",
                    command,
                    operation,
                    columns,
                )

                # Handle help queries
                if command == "help" and operation == "suggest_questions":
                    return self.generate_suggested_questions(csv_path)

                # Create a query plan based on the command
                query_plan = {"operation": operation}

                # Check for 'where' clause in the question
                where_match = re.search(r"where\s+([\w\s]+)\s+(?:is|=)\s+([\w\s]+)", question.lower())
                if where_match:
                    filter_column = where_match.group(1).strip()
                    filter_value = where_match.group(2).strip()

                    # Find the actual column name with correct case
                    actual_column = None
                    for col in self.data_handler.get_columns():
                        if col.lower() == filter_column:
                            actual_column = col
                            break

                    if actual_column:
                        # Check if the column has the exact value (case-insensitive)
                        # First, get all unique values in the column
                        unique_values = self.data_handler.get_dataframe()[actual_column].astype(str).unique()

                        # Find the closest match (case-insensitive)
                        actual_value = filter_value
                        for val in unique_values:
                            if val.lower() == filter_value.lower():
                                actual_value = val
                                break

                        query_plan["filters"] = [{"column": actual_column, "operator": "=", "value": actual_value}]

                        # Make sure we're including ALL columns in the result
                        if operation == "list":
                            # Empty list means all columns in our implementation
                            # This is critical for 'where' queries to return all columns
                            query_plan["columns"] = []

                        logger.debug("Added filter: %s = %s", actual_column, filter_value)

                # Add columns if specified
                if columns:
                    query_plan["columns"] = columns
                else:
                    # If no specific columns, use all columns
                    query_plan["columns"] = self.data_handler.get_columns()

                # Execute the query plan
                result_df, description = self.data_handler.execute_query_plan(query_plan)

                if not result_df.empty:
                    if query_plan["operation"] == "summarize":
                        # Summarize by column if specified
                        if query_plan.get("columns"):
                            column = query_plan["columns"][0]
                            # Get unique values and their counts
                            value_counts = self.data_handler.get_value_counts(column)
                            if value_counts is not None:
                                return self.format_value_counts(column, value_counts)
                            else:
                                return f"Could not summarize by {column}. Column not found."
                        else:
                            # Summarize all data
                            return self.format_dataframe(self.data_handler.get_dataframe())

                    elif query_plan["operation"] == "list":
                        # List records, filtered if specified
                        if "filters" in query_plan:
                            logger.debug("Filtered result_df type: %s", type(result_df))
                            if hasattr(result_df, "shape"):
                                logger.debug("Filtered result_df shape: %s", result_df.shape)
                            if hasattr(result_df, "columns"):
                                logger.debug(
                                    "Filtered result_df columns: %s",
                                    result_df.columns.tolist(),
                                )

                            # Always ensure we have all columns for filtered results
                            if hasattr(result_df, "ndim"):
                                if result_df.ndim == 1:  # Series case
                                    # Get the filtered indices
                                    filtered_indices = result_df.index
                                    # Get the full dataframe with all columns for those indices
                                    result_df = self.data_handler.get_dataframe().loc[filtered_indices]
                                # DataFrame with single column
                                elif len(result_df.columns) == 1:
                                    # Get the filtered indices
                                    filtered_indices = result_df.index
                                    # Get the full dataframe with all columns for those indices
                                    result_df = self.data_handler.get_dataframe().loc[filtered_indices]

                                logger.debug(
                                    "Reconstructed full dataframe with shape: %s",
                                    result_df.shape,
                                )
                                logger.debug("Columns: %s", result_df.columns.tolist())

                            return self.format_dataframe(result_df)
                        else:
                            return self.format_dataframe(self.data_handler.get_dataframe())

                    elif query_plan["operation"] == "count":
                        # Count records
                        count = len(self.data_handler.get_dataframe())
                        return f"Found {count} records in the dataset."
                else:
                    return "No data found for your query."
            else:
                # If we get here, it's not a special command query.
                pass

            # Check if we have a numerical comparison query
            if comparison_info:
                column = comparison_info["column"]
                operator = comparison_info["operator"]
                value = comparison_info["value"]

                # Use the data handler to filter by comparison
                matches = self.data_handler.filter_by_comparison(column, operator, value)

                if not matches.empty:
                    return self.format_matches(matches, target_column, column, f"{operator} {value}")

            # Check for keyword search patterns like "all records are festival"
            keyword_match = re.search(r"all\s+records\s+(?:are|with|containing|having|about)\s+(\w+)", question.lower())
            if keyword_match:
                keyword = keyword_match.group(1).strip()
                logger.debug("Detected keyword search for: %s", keyword)
                matches = self.data_handler.search_value_in_all_columns(keyword)
                if not matches.empty:
                    return self.format_dataframe(matches)
                else:
                    return f"No records found containing '{keyword}'."

            # Check if the query is very simple (just 1-2 words)
            search_term = question.strip().lower()
            simple_query = len(search_term.split()) <= 2

            # For simple queries, try direct search across all columns
            if simple_query:
                # Try to find matches across all columns
                matches = self.data_handler.search_value_in_all_columns(search_term)
                if not matches.empty:
                    return self.format_dataframe(matches)

            # For more complex queries, use LLM to generate a query plan
            if len(search_term.split()) > 2:
                # Get sample data for context
                sample_data = self.data_handler.get_sample_data(5)
                columns = self.data_handler.get_columns()

                # Use LLM to analyze the question and generate a query plan
                query_plan = self.ollama_client.analyze_question(question, columns, sample_data)

                if query_plan:
                    # Execute the query plan
                    result_df, description = self.data_handler.execute_query_plan(query_plan)

                    if not result_df.empty:
                        # Format the results
                        if len(result_df) > 10:
                            # For large result sets, summarize
                            summary = (
                                f"{description}\n\nFound {len(result_df)} matching records. Here are the first 10:\n\n"
                            )
                            return summary + self.format_dataframe(result_df.head(10))
                        else:
                            # For smaller result sets, show all
                            return f"{description}\n\n" + self.format_dataframe(result_df)
                    else:
                        return f"No matching records found for your query about {question}"

            # If we have enough information for a direct lookup
            if id_column and id_value:
                # Try direct lookup first
                matches = self.data_handler.find_rows_by_value(id_column, id_value)

                if not matches.empty:
                    return self.format_matches(matches, target_column, id_column, id_value)

            # Check if the query is just a simple ID or value that doesn't exist in our data
            csv_dataframe = self.data_handler.get_dataframe()

            if len(search_term) < 10 and not any(
                search_term in str(row).lower() for _, row in csv_dataframe.iterrows()
            ):
                # Try to find similar values using fuzzy matching
                similar_values = self.fuzzy_matcher.find_similar_values(search_term, csv_dataframe)
                if similar_values:
                    # Format the suggestion message
                    if len(similar_values) == 1:
                        suggestion = similar_values[0]
                        return (
                            f"I couldn't find '{question.strip()}' in the CSV data. "
                            f"Did you mean '{suggestion}'? If yes, please ask about that instead."
                        )
                    else:
                        suggestions = ", ".join([f"'{v}'" for v in similar_values[:3]])
                        return (
                            f"I couldn't find '{question.strip()}' in the CSV data. "
                            f"Did you mean one of these: {suggestions}? "
                            f"If yes, please ask about that instead."
                        )
                return (
                    f"I couldn't find '{question.strip()}' in the CSV data. "
                    "Please check if the value exists or try a different query."
                )

            # Use semantic search for context retrieval
            response = self._answer_via_semantic_search(question, self.data_handler.get_dataframe())
            if response:
                return response
            context = ""

            # Try to extract answer from context if we have target and ID columns
            if target_column and id_column and id_value:
                # We already tried direct lookup and failed, so use the context
                if context:
                    try:
                        return self.ollama_client.ask(
                            self._build_llm_context(context), question
                        )
                    except (OllamaConnectionError, OllamaTimeoutError, OllamaResponseError) as e:
                        logger.error("Ollama error during context lookup: %s", e, exc_info=True)
                        return (
                            f"I'm not able to find a proper answer. Error: {e.user_message}. "
                            "Would you like to ask another way?"
                        )

            # For general questions about the data, only use the LLM if we have relevant context
            if not context.strip():
                return (
                    "I couldn't find any matching information in the CSV data. "
                    "Please try rephrasing your question or check if the value "
                    "you're looking for exists in the data."
                )

            # Ask the LLM with the context we found
            try:
                result = self.ollama_client.ask(
                    self._build_llm_context(context), question
                )
            except (OllamaConnectionError, OllamaTimeoutError, OllamaResponseError) as e:
                logger.error("Ollama error during question answering: %s", e, exc_info=True)
                result = (
                    f"I'm not able to find a proper answer. Error: {e.user_message}. Would you like to ask another way?"
                )

        except Exception as e:
            logger.error("Error processing question: %s", e, exc_info=True)
            result = f"Error processing your question: {str(e)}. Please try again."

        finally:
            # Restore original dataframe if we temporarily replaced it
            if filtered_dataframe is not None and "original_df" in locals():
                self.data_handler._dataframe = original_df
                logger.debug("Restored original dataframe")

        return result

    def format_dataframe(self, df):
        """Format a dataframe for display.

        Args:
            df: DataFrame to format.

        Returns:
            Formatted dataframe as string.
        """
        if df.empty:
            return "No data available"

        logger.debug("DataFrame shape: %s", df.shape)
        logger.debug("DataFrame columns: %s", df.columns.tolist())
        logger.debug("DataFrame head:\n%s", df.head())

        # Make sure we have a DataFrame with multiple columns, not a Series
        if hasattr(df, "ndim") and df.ndim == 1:
            # Convert Series to DataFrame
            df = df.to_frame()

        # Ensure we have a proper DataFrame object
        if not hasattr(df, "columns") or len(df.columns) == 0:
            return "Error: Invalid data format"

        # For small dataframes, use a more readable format
        if len(df) <= 20:
            # Create a nicely formatted table
            col_widths = [max(len(str(x)) for x in df[col].astype(str).tolist() + [col]) for col in df.columns]
            header = "  ".join(col.ljust(width) for col, width in zip(df.columns, col_widths))
            separator = "-" * len(header)

            rows = []
            for _, row in df.iterrows():
                formatted_row = "  ".join(str(val).ljust(width) for val, width in zip(row.values, col_widths))
                rows.append(formatted_row)

            return "\n".join([header, separator] + rows)

        # For larger dataframes, format as markdown table for better readability
        header = "| " + " | ".join(df.columns) + " |"
        separator = "| " + " | ".join(["---" for _ in df.columns]) + " |"

        rows = []
        for _, row in df.head(20).iterrows():  # Limit to 20 rows max
            formatted_row = "| " + " | ".join([str(val) for val in row.values]) + " |"
            rows.append(formatted_row)

        if len(df) > 20:
            rows.append(f"| ... and {len(df) - 20} more rows ... |")

        return "\n".join([header, separator] + rows)

    def generate_suggested_questions(self, csv_path=None):
        """Generate a list of suggested questions based on the CSV content.

        Args:
            csv_path: Path to CSV file (optional, if already loaded).

        Returns:
            Formatted list of suggested questions.
        """
        if csv_path:
            self.data_handler.load_csv(csv_path)

        if self.data_handler.csv_dataframe is None:
            return "Please load a CSV file first."

        # Get column information
        columns = self.data_handler.get_columns()

        # Extract unique values for categorical columns
        unique_values = {}
        categorical_columns = []
        numerical_columns = []
        date_columns = []

        # Identify column types and gather unique values for categorical columns
        for col in columns:
            # Check if column is numeric
            if self.data_handler.csv_dataframe[col].dtype.kind in "ifc":
                numerical_columns.append(col)
                continue

            # Check if column might be a date (simple heuristic)
            date_pattern = r"\d{4}[-/]\d{1,2}[-/]\d{1,2}"
            sample_val = str(self.data_handler.csv_dataframe[col].iloc[0])
            if re.search(date_pattern, sample_val):
                date_columns.append(col)
                continue

            # For categorical columns, get unique values (limit to 10)
            unique_vals = self.data_handler.csv_dataframe[col].unique()
            if len(unique_vals) <= 15:  # Only for columns with reasonable number of unique values
                categorical_columns.append(col)
                unique_values[col] = unique_vals[:5]  # Limit to 5 examples

        # Generate questions
        questions = ["Here are some questions you can ask about this data:"]

        # Basic questions
        questions.append("\n🔍 Basic Questions:")
        questions.append("- summarize")
        questions.append("- list all records")
        questions.append("- count records")

        # Questions about categorical columns
        if categorical_columns:
            questions.append("\n📊 Filtering Questions:")
            # Limit to 3 columns to avoid too many examples
            for col in categorical_columns[:3]:
                # Limit to 2 values per column
                for val in unique_values.get(col, [])[:2]:
                    questions.append(f"- show me {col} with value {val}")
                    questions.append(f"- list records where {col} is {val}")

        # Questions about numerical columns
        if numerical_columns:
            questions.append("\n📈 Numerical Analysis:")
            for col in numerical_columns[:3]:  # Limit to 3 columns
                questions.append(f"- what is the average {col}?")
                questions.append(f"- show records with {col} greater than [value]")
                questions.append(f"- what is the highest {col}?")

        # Questions about date columns
        if date_columns:
            questions.append("\n📅 Time-Based Questions:")
            for col in date_columns[:2]:  # Limit to 2 date columns
                questions.append(f"- show records from [specific date] in {col}")
                questions.append(f"- list records between [date1] and [date2] in {col}")

        # Advanced questions
        questions.append("\n🔬 Advanced Questions:")
        if len(categorical_columns) >= 2 and len(categorical_columns) > 0:
            col1 = categorical_columns[0]
            col2 = categorical_columns[min(1, len(categorical_columns) - 1)]
            questions.append(f"- count records grouped by {col1}")
            questions.append(f"- summarize data by {col2}")

        if len(numerical_columns) >= 1 and len(categorical_columns) >= 1:
            num_col = numerical_columns[0]
            cat_col = categorical_columns[0]
            questions.append(f"- what is the average {num_col} for each {cat_col}?")
            questions.append(f"- which {cat_col} has the highest {num_col}?")

        return "\n".join(questions)

    def _is_analysis_request(self, question: str) -> bool:
        """Check if the question is requesting data analysis.

        Args:
            question: The question to check.

        Returns:
            True if this is an analysis request.
        """
        analysis_keywords = [
            "analysis",
            "analyze",
            "statistics",
            "statistical",
            "stats",
            "stat",
            "summary",
            "summarize",
            "overview",
            "insights",
            "patterns",
            "outliers",
            "anomalies",
            "distribution",
            "trend",
            "average",
            "mean",
            "median",
            "mode",
            "min",
            "max",
            "standard deviation",
            "std",
            "variance",
        ]

        question_lower = question.lower()

        # Check if any analysis keyword is in the question
        for keyword in analysis_keywords:
            if keyword in question_lower:
                # Don't trigger on "summarize by" which is handled by the query plan system
                if keyword == "summarize" and "summarize by" in question_lower:
                    continue
                return True

        return False

    def _is_complex_query(self, question: str) -> bool:
        """Check if the question is a complex query that needs to be broken down.

        Args:
            question: The question to check.

        Returns:
            True if this is a complex query.
        """
        question_lower = question.lower()

        # Check for analysis combined with filtering
        analysis_keywords = ["analysis", "analyze", "statistics", "stats"]
        filter_patterns = ["where", "with", "containing", "all records are", "records are", "for records"]

        # Check if question contains both analysis and filtering
        has_analysis = any(keyword in question_lower for keyword in analysis_keywords)
        has_filter = any(pattern in question_lower for pattern in filter_patterns)

        if has_analysis and has_filter:
            return True

        # Check for other complex patterns that might need LLM parsing
        complex_patterns = [
            r"records\s+(?:with|containing|having)\s+\w+",
            r"all\s+records\s+(?:are|with|containing|having|about)\s+\w+",
            r"analysis\s+.*\s+where\s+.*\s+is\s+\w+",
        ]

        for pattern in complex_patterns:
            if re.search(pattern, question_lower):
                return True

        return False

    def _parse_query_steps(self, question: str, csv_path: str | None = None) -> dict:
        """Use LLM to break down a complex query into steps.

        Args:
            question: The question to parse.
            csv_path: Path to CSV file (optional, if already loaded).

        Returns:
            Query steps plan with operations to perform.
        """
        if csv_path and not self.data_handler.is_loaded():
            self.load_csv(csv_path)

        if not self.data_handler.is_loaded():
            return {"error": "CSV data not loaded"}

        df = self.data_handler.get_dataframe()
        sample_data = df.head(3).to_string(index=False)
        columns = self.data_handler.get_columns()

        question_lower = question.lower().strip()

        if question_lower == "analysis all records are festival":
            return {
                "steps": [
                    {
                        "operation": "filter",
                        "type": "keyword_search",
                        "keyword": "festival",
                        "description": "Find records containing festival",
                    },
                    {
                        "operation": "analyze",
                        "type": "statistical",
                        "target": "filtered_results",
                        "description": "Perform statistical analysis on filtered records",
                    },
                ],
                "description": "Filter records containing 'festival' and then analyze them",
            }
        elif question_lower == "analysis all records are not festival":
            return {
                "steps": [
                    {
                        "operation": "filter",
                        "type": "keyword_search_exclude",
                        "keyword": "festival",
                        "description": "Find records NOT containing festival",
                    },
                    {
                        "operation": "analyze",
                        "type": "statistical",
                        "target": "filtered_results",
                        "description": "Perform statistical analysis on filtered records",
                    },
                ],
                "description": "Filter records NOT containing 'festival' and then analyze them",
            }

        logger.debug("Breaking down complex query with LLM...")

        try:
            query_plan = self.ollama_client.parse_query_steps(
                question=question,
                columns=columns,
                sample_data=sample_data,
            )
        except (OllamaConnectionError, OllamaTimeoutError, OllamaResponseError) as e:
            logger.error("Ollama error while parsing query steps: %s", e, exc_info=True)
            return self._create_default_query_plan(question)

        if query_plan is None:
            logger.warning("LLM did not return a valid query plan; using default")
            return self._create_default_query_plan(question)

        logger.debug("Query plan: %s", json.dumps(query_plan, indent=2))
        return query_plan

    def _create_default_query_plan(self, question: str) -> dict:
        """Create a default query plan when LLM parsing fails.

        Args:
            question: The original question.

        Returns:
            A default query plan based on the question.
        """
        question_lower = question.lower()

        # Check for negation in the query
        has_negation = any(neg in question_lower for neg in ["not", "aren't", "aren't", "excluding", "except"])

        # Check for OR conditions
        has_or_condition = " or " in question_lower

        # Extract potential keywords for filtering
        words = question_lower.split()
        keywords = []

        # Skip common words and find potential keywords
        skip_words = [
            "analysis",
            "analyze",
            "all",
            "records",
            "are",
            "where",
            "with",
            "containing",
            "is",
            "the",
            "a",
            "an",
            "not",
            "aren't",
            "aren't",
            "excluding",
            "except",
            "or",
        ]
        for word in words:
            if word not in skip_words and len(word) > 3:
                keywords.append(word)

        # Check if this is an analysis request
        is_analysis = self._is_analysis_request(question)

        steps = []

        # Handle OR conditions
        if has_or_condition:
            # Split by "or" and extract keywords on each side
            or_parts = question_lower.split(" or ")
            or_keywords = []

            for part in or_parts:
                part_words = part.split()
                for word in part_words:
                    if word not in skip_words and len(word) > 3:
                        or_keywords.append(word)

            if or_keywords:
                steps.append(
                    {
                        "operation": "filter",
                        "type": "compound_search",
                        "keywords": or_keywords,
                        "logical_operator": "or",
                        "description": f"Find records containing any of: {', '.join(or_keywords)}",
                    }
                )
        # Handle regular keyword search
        elif keywords:
            keyword = keywords[-1]  # Use the last keyword if available
            if has_negation:
                steps.append(
                    {
                        "operation": "filter",
                        "type": "keyword_search_exclude",
                        "keyword": keyword,
                        "description": f"Find records NOT containing {keyword}",
                    }
                )
            else:
                steps.append(
                    {
                        "operation": "filter",
                        "type": "keyword_search",
                        "keyword": keyword,
                        "description": f"Find records containing {keyword}",
                    }
                )

        # Add analysis step if this is an analysis request
        if is_analysis:
            steps.append(
                {
                    "operation": "analyze",
                    "type": "statistical",
                    "target": "filtered_results" if keywords or has_or_condition else "all_records",
                    "description": "Perform statistical analysis on records",
                }
            )
        else:
            # If not an analysis request, add a list step
            steps.append({"operation": "list", "type": "display", "description": "Display the filtered records"})

        return {"steps": steps, "description": f"Process query: {question}"}

    def _execute_query_steps(self, question: str, csv_path: str | None = None) -> str:
        """Execute a complex query by breaking it down into steps.

        Args:
            question: The complex query to execute.
            csv_path: Path to CSV file (optional, if already loaded).

        Returns:
            The result of executing the query steps.
        """
        # Load CSV if needed
        if csv_path and not self.data_handler.is_loaded():
            self.load_csv(csv_path)

        if not self.data_handler.is_loaded():
            return "CSV data not loaded. Please provide a CSV file."

        # Parse the query into steps
        query_plan = self._parse_query_steps(question, csv_path)

        if "error" in query_plan:
            return f"Error parsing query: {query_plan['error']}"

        if "steps" not in query_plan or not query_plan["steps"]:
            return "Could not break down the query into steps."

        # Execute each step in sequence
        result_df = self.data_handler.get_dataframe()
        step_results = []

        logger.debug("Executing %d steps", len(query_plan["steps"]))

        for i, step in enumerate(query_plan["steps"]):
            operation = step.get("operation")
            step_type = step.get("type")

            logger.debug("Step %d: %s - %s", i + 1, operation, step_type)

            # Handle different operations
            if operation == "filter":
                if step_type == "keyword_search":
                    keyword = step.get("keyword")
                    if keyword:
                        result_df = self.data_handler.search_value_in_all_columns(keyword)
                        step_results.append(f"Found {len(result_df)} records containing '{keyword}'")
                elif step_type == "compound_search":
                    keywords = step.get("keywords")
                    logical_operator = step.get("logical_operator", "or")
                    if keywords:
                        result_df = self.data_handler.search_multiple_values(keywords, logical_operator)
                        step_results.append(
                            f"Found {len(result_df)} records matching compound search with {logical_operator} condition"
                        )
                elif step_type == "keyword_search_exclude":
                    keyword = step.get("keyword")
                    if keyword:
                        # Get all records first
                        all_records = self.data_handler.get_dataframe()
                        # Get records containing the keyword
                        matching_records = self.data_handler.search_value_in_all_columns(keyword)
                        # Get records NOT containing the keyword by filtering out matching records
                        if not matching_records.empty:
                            result_df = all_records[~all_records.index.isin(matching_records.index)]
                        else:
                            result_df = all_records
                        step_results.append(f"Found {len(result_df)} records NOT containing '{keyword}'")
                elif step_type == "column_filter":
                    column = step.get("column")
                    value = step.get("value")
                    if column and value:
                        # Find the actual column name with correct case
                        actual_column = None
                        for col in self.data_handler.get_columns():
                            if col.lower() == column.lower():
                                actual_column = col
                                break

                        if actual_column:
                            # Create a query plan for the data handler
                            filter_query = {
                                "operation": "list",
                                "columns": [],  # Empty list means all columns
                                "filters": [{"column": actual_column, "operator": "=", "value": value}],
                            }
                            result_df, _ = self.data_handler.execute_query_plan(filter_query)
                            step_results.append(
                                f"Filtered to {len(result_df)} records where {actual_column} is {value}"
                            )
                        else:
                            step_results.append(f"Column '{column}' not found")
                    else:
                        step_results.append("Missing column or value for filter")

            elif operation == "analyze":
                # If we have filtered results, analyze those
                if result_df is not None and not result_df.empty:
                    # Temporarily replace the dataframe in the data handler with our filtered results
                    original_df = self.data_handler.get_dataframe()
                    self.data_handler.csv_dataframe = result_df

                    # Perform analysis on the filtered data
                    analysis_results = self.data_handler.analyze_data()
                    formatted_results = self._format_analysis_results(analysis_results, question)

                    # Restore the original dataframe
                    self.data_handler.csv_dataframe = original_df

                    # Return the analysis results
                    return formatted_results
                else:
                    return "No data to analyze after filtering."

            elif operation == "list":
                # Just return the current filtered results
                return self.format_dataframe(result_df)

        # If we get here, return the final step results or the filtered data
        if result_df is not None and not result_df.empty:
            return self.format_dataframe(result_df)
        else:
            return "\n".join(step_results) if step_results else "No results found."

    def _perform_analysis(self, csv_path: str | None, question: str) -> str:
        """Perform statistical analysis on the data.

        Args:
            csv_path: Path to the CSV file.
            question: The analysis question.

        Returns:
            Formatted analysis results.
        """
        # Load CSV if needed
        if csv_path:
            self.load_csv(csv_path)

        # Extract potential filter criteria from the question
        filter_column = None
        filter_value = None
        target_columns = None

        # Check if the question mentions specific columns or filters
        question_lower = question.lower()

        # Try to extract column names from the question
        columns = self.data_handler.get_columns()
        mentioned_columns = [col for col in columns if col.lower() in question_lower]

        # Get all possible values for categorical columns to match in the question
        df = self.data_handler.get_dataframe()
        categorical_values = {}
        for col in mentioned_columns:
            if col in df.columns:
                # Get unique values for this column
                unique_vals = df[col].astype(str).unique()
                categorical_values[col] = unique_vals

        # Check for specific column mentions
        for col in mentioned_columns:
            # Look for patterns like "analysis of DecibelsA" or "analyze EventType"
            if any(
                f"{keyword} {col.lower()}" in question_lower
                or f"{keyword} of {col.lower()}" in question_lower
                or f"{keyword} for {col.lower()}" in question_lower
                for keyword in ["analysis", "analyze", "statistics", "stats"]
            ):
                target_columns = [col]
            # Look for patterns like "analysis where Location is City Center"
            elif f"where {col.lower()}" in question_lower or f"for {col.lower()}" in question_lower:
                filter_column = col

                # Try to find values from this column in the question
                if col in categorical_values:
                    for val in categorical_values[col]:
                        if val.lower() in question_lower:
                            filter_value = val
                            break

                # If we couldn't find a value, try to extract it from the question structure
                if not filter_value:
                    try:
                        words = question_lower.split()
                        col_index = words.index(col.lower())
                        if col_index < len(words) - 2 and words[col_index + 1] in ["is", "equals", "="]:
                            filter_value = words[col_index + 2]
                    except (ValueError, IndexError):
                        # If we can't extract a value, just continue
                        pass

        # Perform the analysis
        analysis_results = self.data_handler.analyze_data(
            filter_column=filter_column, filter_value=filter_value, target_columns=target_columns
        )

        # Format the results
        return self._format_analysis_results(analysis_results, question)

    def _format_analysis_results(self, results: dict, question: str) -> str:
        """Format analysis results into a readable string.

        Args:
            results: Analysis results dictionary.
            question: The original question.

        Returns:
            Formatted analysis results.
        """
        if "error" in results:
            return f"Analysis Error: {results['error']}"

        output = ["Statistical Analysis Results"]
        output.append(f"Total Records: {results['record_count']}")

        # Add numeric column statistics
        if results["columns"]:
            output.append("\nNumeric Column Statistics:")
            for col, stats in results["columns"].items():
                output.append(f"\n{col}:")
                output.append(f"  - Min: {stats['min']:.2f}")
                output.append(f"  - Max: {stats['max']:.2f}")
                output.append(f"  - Mean: {stats['mean']:.2f}")
                output.append(f"  - Median: {stats['median']:.2f}")
                output.append(f"  - Standard Deviation: {stats['std']:.2f}")

        # Add outlier information
        if results["outliers"]:
            output.append("\nOutliers Detected:")
            for col, outlier_info in results["outliers"].items():
                output.append(f"\n{col}: {outlier_info['count']} outliers detected")
                if outlier_info["count"] > 0:
                    output.append(
                        f"  Outlier values: {', '.join([str(round(v, 2)) for v in outlier_info['values'][:5]])}"
                    )
                    if len(outlier_info["values"]) > 5:
                        output.append(f"  ... and {len(outlier_info['values']) - 5} more")

                    # Add sample outlier records
                    if outlier_info["records"]:
                        output.append("  Sample outlier records:")
                        for i, record in enumerate(outlier_info["records"], 1):
                            record_str = ", ".join([f"{k}: {v}" for k, v in record.items()][:3])
                            output.append(f"    Record {i}: {record_str}...")

        # Add categorical distributions
        if results["categorical_counts"]:
            output.append("\nCategorical Distributions:")
            for col, counts in results["categorical_counts"].items():
                output.append(f"\n{col}:")
                for val, count in counts.items():
                    percentage = (count / results["record_count"]) * 100
                    output.append(f"  - {val}: {count} ({percentage:.1f}%)")

        return "\n".join(output)

    def process_question_with_suggestions(self, csv_file: str, question: str):
        """Process a question with interactive suggestions for similar values.

        Args:
            csv_file: Path to the CSV file.
            question: The question to answer.

        Returns:
            Tuple of (answer, suggested_values, is_suggestion).
        """
        result = self.answer_question(question, csv_file)

        # Check if the result is already a tuple (for command corrections)
        if isinstance(result, tuple) and len(result) == 3:
            # Already in the correct format, just return it
            return result

        # Otherwise, treat it as a regular answer string
        answer = result

        # Check if the answer contains a suggestion
        if "Did you mean" in answer:
            suggested_values = []

            if "Did you mean '" in answer:
                # Single suggestion
                suggested_value = re.search(r"Did you mean '([^']+)'\?", answer)
                if suggested_value:
                    suggested_values = [suggested_value.group(1)]
            else:
                # Multiple suggestions
                suggested_values = re.findall(r"'([^']+)'(?=[,\?])", answer)

            return answer, suggested_values, True

        # If no suggestions, just return the original answer
        return answer, [], False

    def _resolve_pronouns_with_context(self, question: str, referenced_turns) -> str:
        """Resolve pronouns in the question using context from previous turns."""
        if not referenced_turns:
            return question

        # Get the most recent relevant turn
        latest_turn = referenced_turns[-1]

        # Simple pronoun resolution - replace "it", "them", "that" with entities from context
        resolved_question = question

        # Try to extract meaningful context from the latest turn's question
        context_description = None

        # Look for filter patterns in the previous question
        prev_question = latest_turn.question.lower()

        # Pattern: "where EventType is Festival" -> "Festival records"
        where_match = re.search(r"where\s+(\w+)\s+is\s+(\w+)", prev_question)
        if where_match:
            _column = where_match.group(1)
            value = where_match.group(2)
            context_description = f"{value} records"

        # Pattern: "with Location Urban Park" -> "Urban Park records"
        with_match = re.search(r"with\s+(\w+)\s+([a-zA-Z\s]+)", prev_question)
        if with_match:
            _column = with_match.group(1)
            value = with_match.group(2).strip()
            context_description = f"{value} records"

        # If we found a good context description, use it
        if context_description:
            # Replace pronouns with the context description
            resolved_question = re.sub(r"\bthem\b", context_description, resolved_question, flags=re.IGNORECASE)
            resolved_question = re.sub(r"\bit\b", context_description, resolved_question, flags=re.IGNORECASE)
            resolved_question = re.sub(r"\bthat\b", context_description, resolved_question, flags=re.IGNORECASE)
            resolved_question = re.sub(r"\bthose\b", context_description, resolved_question, flags=re.IGNORECASE)
        elif latest_turn.entities_mentioned:
            # Fallback to the original logic
            main_entity = latest_turn.entities_mentioned[0]
            resolved_question = re.sub(r"\bit\b", main_entity, resolved_question, flags=re.IGNORECASE)
            resolved_question = re.sub(r"\bthat\b", main_entity, resolved_question, flags=re.IGNORECASE)
            resolved_question = re.sub(r"\bthose\b", main_entity, resolved_question, flags=re.IGNORECASE)

        logger.debug("Resolved '%s' to '%s'", question, resolved_question)

        return resolved_question

    def _store_conversation_turn(self, question: str, answer: str, context_info, result_data=None) -> None:
        """Store the conversation turn in context memory."""
        if not self.context_memory:
            return

        # Determine question type
        question_type = self._classify_question_type(question)

        # Extract entities mentioned in the question and answer
        entities_mentioned = self._extract_entities(question, answer)

        # Count results if possible
        result_count = self._count_results_in_answer(answer)

        # Calculate confidence score (simplified)
        confidence_score = self._calculate_confidence_score(answer)

        # Store metadata
        metadata = {
            "csv_loaded": self.data_handler.get_dataframe() is not None,
            "had_context": context_info is not None and context_info.get("is_follow_up", False),
        }

        # Create result_data if we have filtering information
        logger.debug(
            "Checking filter extraction: result_data=%s, question_type=%s, result_count=%d",
            result_data,
            question_type,
            result_count,
        )

        if result_data is None and question_type in ["filter", "list"] and result_count > 0:
            # Try to extract filter information from the question
            result_data = self._extract_filter_info_from_question(question, result_count)
            logger.debug(
                "Extracted result_data for question '%s': %s",
                question,
                result_data,
            )

        self.context_memory.add_turn(
            question=question,
            answer=answer,
            question_type=question_type,
            entities_mentioned=entities_mentioned,
            result_count=result_count,
            confidence_score=confidence_score,
            metadata=metadata,
            result_data=result_data,
        )

    def _classify_question_type(self, question: str) -> str:
        """Classify the type of question for context memory."""
        question_lower = question.lower()

        if any(word in question_lower for word in ["summarize", "summary"]):
            return "summary"
        elif any(word in question_lower for word in ["list", "show", "display"]):
            return "list"
        elif any(word in question_lower for word in ["count", "how many"]):
            return "count"
        elif any(word in question_lower for word in ["where", "filter"]):
            return "filter"
        elif any(word in question_lower for word in ["analyze", "analysis"]):
            return "analysis"
        elif "what" in question_lower:
            return "lookup"
        else:
            return "unknown"

    def _extract_entities(self, question: str, answer: str) -> list[str]:
        """Extract entities (columns, values) mentioned in question and answer."""
        entities = []

        # Add CSV columns mentioned in the question
        if self.data_handler.get_dataframe() is not None:
            columns = self.data_handler.get_columns()
            for col in columns:
                if col.lower() in question.lower() or col in answer:
                    entities.append(col)

        # Extract quoted values or capitalized words
        quoted_values = re.findall(r"'([^']+)'|\"([^\"]+)\"", question + " " + answer)
        for quote_tuple in quoted_values:
            for value in quote_tuple:
                if value:
                    entities.append(value)

        # Extract capitalized words (potential proper nouns)
        cap_words = re.findall(r"\b[A-Z][a-zA-Z]+\b", question)
        entities.extend(cap_words)

        return list(set(entities))  # Remove duplicates

    def _count_results_in_answer(self, answer: str) -> int:
        """Extract the number of results from the answer text."""
        # Look for patterns like "Found 5 matches", "3 records", etc.
        count_patterns = [
            r"found (\d+) (?:matches|records|results)",
            r"(\d+) (?:matches|records|results)",
            r"returned (\d+)",
            r"showing (\d+)",
        ]

        for pattern in count_patterns:
            match = re.search(pattern, answer.lower())
            if match:
                return int(match.group(1))

        # Count lines that look like data records
        lines = answer.split("\n")
        record_lines = [line for line in lines if line.strip() and ("-" in line or ":" in line)]
        if len(record_lines) > 2:  # More than just headers
            return len(record_lines) - 2  # Subtract headers

        return 0

    def _calculate_confidence_score(self, answer: str) -> float:
        """Calculate confidence score for the answer."""
        # Simple heuristic based on answer characteristics
        if any(phrase in answer.lower() for phrase in ["error", "could not", "not found", "no matches"]):
            return 0.3
        elif any(phrase in answer.lower() for phrase in ["found", "matches", "records"]):
            return 0.9
        elif len(answer) > 100:  # Substantial answer
            return 0.8
        else:
            return 0.6

    def _extract_filter_info_from_question(self, question: str, result_count: int) -> dict:
        """Extract filter information from the question to store for context."""
        filter_info = {"filter_applied": True, "original_question": question, "result_count": result_count}

        # Try to extract filter conditions from common patterns
        question_lower = question.lower()

        # Pattern: "where Column is Value"
        where_match = re.search(r"where\s+(\w+)\s+is\s+(\w+)", question_lower)
        if where_match:
            filter_info["filter_column"] = where_match.group(1)
            filter_info["filter_value"] = where_match.group(2)
            filter_info["filter_operator"] = "="

        # Pattern: "records where EventType is Traffic"
        event_match = re.search(r"where\s+(\w+)\s+is\s+([a-zA-Z]+)", question_lower)
        if event_match:
            filter_info["filter_column"] = event_match.group(1)
            filter_info["filter_value"] = event_match.group(2)
            filter_info["filter_operator"] = "="

        # Pattern: "with Column Value" (e.g., "show records with Location Urban Park")
        with_match = re.search(r"with\s+(\w+)\s+([a-zA-Z\s]+)", question_lower)
        if with_match:
            filter_info["filter_column"] = with_match.group(1)
            filter_info["filter_value"] = with_match.group(2).strip()
            filter_info["filter_operator"] = "="

        logger.debug("Extracted filter info: %s", filter_info)

        return filter_info

    def _apply_context_filter(self, context_filter: dict):
        """Apply context filter to get filtered dataframe."""
        try:
            if "filter_column" in context_filter and "filter_value" in context_filter:
                filter_column = context_filter["filter_column"]
                filter_value = context_filter["filter_value"]

                # Get the original dataframe
                df = self.data_handler.get_dataframe()

                # Find the actual column name (case insensitive)
                actual_column = None
                for col in df.columns:
                    if col.lower() == filter_column.lower():
                        actual_column = col
                        break

                if actual_column is None:
                    logger.debug("Column %s not found in dataframe", filter_column)
                    return None

                # Apply the filter
                filtered_df = df[df[actual_column].astype(str).str.lower() == filter_value.lower()]

                logger.debug(
                    "Filtered %d records to %d records using %s = %s",
                    len(df),
                    len(filtered_df),
                    actual_column,
                    filter_value,
                )

                return filtered_df

        except Exception as e:
            logger.error("Error applying context filter: %s", e, exc_info=True)
            return None

        return None

    def _is_aggregation_question(self, question: str) -> bool:
        """Check if the question is asking for aggregation (max, min, avg, etc.)."""
        question_lower = question.lower()
        aggregation_keywords = [
            "max",
            "maximum",
            "min",
            "minimum",
            "avg",
            "average",
            "mean",
            "sum",
            "total",
            "count",
            "highest",
            "lowest",
            "largest",
            "smallest",
        ]
        return any(keyword in question_lower for keyword in aggregation_keywords)

    def _handle_aggregation_on_filtered_data(self, question: str, filtered_df):
        """Handle aggregation questions on filtered data."""
        try:
            question_lower = question.lower()

            # Extract the column name from the question
            columns = self.data_handler.get_columns()
            target_column = None

            # More robust column detection
            for col in columns:
                if col.lower() in question_lower:
                    target_column = col
                    break

            if not target_column:
                logger.debug(
                    "Could not identify target column in question: %s",
                    question,
                )
                logger.debug("Available columns: %s", columns)
                return None

            logger.debug("Identified target column: %s", target_column)

            # Check if the column exists in filtered data
            if target_column not in filtered_df.columns:
                return f"Column '{target_column}' not found in the filtered data."

            # Try to convert to numeric if possible
            try:
                numeric_col = pd.to_numeric(filtered_df[target_column], errors="coerce")

                logger.debug(
                    "Column %s numeric conversion - NaN count: %d/%d",
                    target_column,
                    numeric_col.isna().sum(),
                    len(numeric_col),
                )

                if numeric_col.isna().all():
                    # Not a numeric column, handle as text
                    if any(word in question_lower for word in ["max", "maximum", "highest", "largest"]):
                        result = filtered_df[target_column].max()
                        return f"The maximum {target_column} in these {len(filtered_df)} records is: {result}"
                    elif any(word in question_lower for word in ["min", "minimum", "lowest", "smallest"]):
                        result = filtered_df[target_column].min()
                        return f"The minimum {target_column} in these {len(filtered_df)} records is: {result}"
                    else:
                        return f"Cannot perform numerical aggregation on non-numeric column '{target_column}'"
                else:
                    # Numeric column
                    if any(word in question_lower for word in ["max", "maximum", "highest", "largest"]):
                        result = numeric_col.max()
                        return f"The maximum {target_column} in these {len(filtered_df)} records is: {result}"
                    elif any(word in question_lower for word in ["min", "minimum", "lowest", "smallest"]):
                        result = numeric_col.min()
                        return f"The minimum {target_column} in these {len(filtered_df)} records is: {result}"
                    elif any(word in question_lower for word in ["avg", "average", "mean"]):
                        result = numeric_col.mean()
                        return f"The average {target_column} in these {len(filtered_df)} records is: {result:.2f}"
                    elif any(word in question_lower for word in ["sum", "total"]):
                        result = numeric_col.sum()
                        return f"The total {target_column} in these {len(filtered_df)} records is: {result}"
                    elif any(word in question_lower for word in ["count"]):
                        result = len(filtered_df)
                        return f"There are {result} records in the filtered data."

            except Exception as e:
                logger.error("Error in numeric conversion: %s", e, exc_info=True)
                return None

        except Exception as e:
            logger.error("Error in aggregation handling: %s", e, exc_info=True)
            return None

        return None

    def get_conversation_summary(self) -> str:
        """Get a summary of the current conversation."""
        if self.enable_context_memory and self.context_memory:
            return self.context_memory.get_conversation_summary()
        else:
            return "Context memory is not enabled."

    def clear_conversation_history(self) -> None:
        """Clear the conversation history."""
        if self.enable_context_memory and self.context_memory:
            self.context_memory.clear_context()
            logger.info("Conversation history cleared.")

    def save_conversation_history(self, filepath: str) -> None:
        """Save conversation history to a file."""
        if self.enable_context_memory and self.context_memory:
            self.context_memory.save_to_file(filepath)
            logger.info("Conversation history saved to %s", filepath)

    def load_conversation_history(self, filepath: str) -> None:
        """Load conversation history from a file."""
        if self.enable_context_memory and self.context_memory:
            try:
                self.context_memory.load_from_file(filepath)
                logger.info("Conversation history loaded from %s", filepath)
            except FileNotFoundError:
                logger.warning("Conversation history file %s not found", filepath)
            except Exception as e:
                logger.error("Error loading conversation history: %s", e, exc_info=True)
