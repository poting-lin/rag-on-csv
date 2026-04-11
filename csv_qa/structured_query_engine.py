"""
Structured Query Engine - handles direct DataFrame operations for precise queries.
"""

import logging
import re
from typing import Optional

import pandas as pd

from .exceptions import QueryResult

logger = logging.getLogger(__name__)


class StructuredQueryEngine:
    """Handles structured queries that can be executed directly on DataFrames."""

    def execute_query(self, question: str, df: pd.DataFrame) -> QueryResult:
        """Execute a structured query on the DataFrame.

        Args:
            question: The user's question.
            df: The pandas DataFrame to query.

        Returns:
            QueryResult with the query outcome.
        """
        try:
            question_lower = question.lower()
            columns = df.columns.tolist()

            logger.debug("Executing structured query: %s", question)

            # Try different query patterns in order of specificity

            # 1. Aggregation queries
            agg_result = self._handle_aggregation_query(question_lower, df, columns)
            if agg_result.success:
                return agg_result

            # 2. Filtering queries
            filter_result = self._handle_filter_query(question_lower, df, columns)
            if filter_result.success:
                return filter_result

            # 3. Column display queries
            column_result = self._handle_column_query(question_lower, df, columns)
            if column_result.success:
                return column_result

            # 4. Comparison queries
            comparison_result = self._handle_comparison_query(question_lower, df, columns)
            if comparison_result.success:
                return comparison_result

            # 5. Statistics queries
            stats_result = self._handle_statistics_query(question_lower, df, columns)
            if stats_result.success:
                return stats_result

            # 6. Count queries
            count_result = self._handle_count_query(question_lower, df, columns)
            if count_result.success:
                return count_result

            return QueryResult.fail(
                error_code="NO_MATCH",
                error_message="No matching structured query pattern found.",
                engine="structured",
            )

        except Exception as e:
            logger.error("Error in structured query execution: %s", e, exc_info=True)
            return QueryResult.fail(
                error_code="NO_MATCH",
                error_message=str(e),
                engine="structured",
            )

    # Maps natural language keywords to pandas aggregation functions.
    AGG_KEYWORDS: dict[str, list[str]] = {
        "sum": ["sum", "total", "how much", "add up", "combined", "altogether", "grand total"],
        "mean": ["average", "avg", "mean"],
        "median": ["median", "middle value"],
        "max": ["max", "maximum", "highest", "largest", "biggest", "top", "peak", "greatest"],
        "min": ["min", "minimum", "lowest", "smallest", "least", "bottom"],
        "std": ["std", "standard deviation", "stddev", "volatility", "deviation"],
        "count": ["count", "how many", "number of"],
    }

    def _handle_aggregation_query(self, question: str, df: pd.DataFrame, columns: list[str]) -> QueryResult:
        """Handle aggregation queries using regex patterns with a column-aware fallback.

        Supports natural phrasings like:
        - "what is the sum of amount_eur"
        - "how much in total in amount_eur"
        - "calculate the average of amount_eur"
        - "amount_eur total"
        - "highest amount_eur"
        - "give me the median amount_eur"
        """
        # Step 1: Try regex-based pattern matching (handles explicit syntax)
        # Build patterns programmatically to avoid repetition.
        _verb = r"(?:what is|what's|show|get|give me|calculate)"
        _prep = r"(?:of\s+|for\s+|in\s+)?(?:the\s+)?"
        _agg_terms = {
            "max": "max|maximum",
            "min": "min|minimum",
            "mean": "avg|average|mean",
            "sum": "sum|total",
            "median": "median",
            "std": "standard deviation|std|stddev",
        }
        _reverse_terms = {
            "sum": "sum|total",
            "mean": "average|avg|mean",
            "max": "max|maximum|highest",
            "min": "min|minimum|lowest",
            "median": "median",
        }

        agg_patterns: list[tuple[str, str]] = []
        for func, terms in _agg_terms.items():
            # "what is the sum of amount_eur"
            agg_patterns.append((
                rf"{_verb}\s+(?:the\s+)?({terms})\s+{_prep}(\w+)",
                func,
            ))
        for func, terms in _agg_terms.items():
            # "sum of amount_eur"
            agg_patterns.append((
                rf"({terms})\s+{_prep}(\w+)",
                func,
            ))
        for func, terms in _reverse_terms.items():
            # "amount_eur total" — column before keyword
            agg_patterns.append((rf"(\w+)\s+({terms})", func))

        for pattern, agg_func in agg_patterns:
            match = re.search(pattern, question)
            if not match:
                continue

            # For "column keyword" patterns, column is group 1; for others, it's the last group
            groups = match.groups()
            if agg_func in groups[0].lower() if groups[0] else False:
                column_name = groups[-1]
            else:
                column_name = groups[0]

            actual_column = self._find_column_name(column_name, columns)
            if not actual_column:
                # Try the other group in case of ambiguity
                other = groups[-1] if column_name == groups[0] else groups[0]
                actual_column = self._find_column_name(other, columns)
            if not actual_column:
                continue

            result = self._execute_aggregation(agg_func, actual_column, df)
            if result:
                return result

        # Step 2: Column-aware fallback — find mentioned columns + aggregation keywords
        detected_func = self._detect_aggregation_function(question)

        if detected_func:
            mentioned_columns = self._find_mentioned_columns(question, columns)
            for col in mentioned_columns:
                result = self._execute_aggregation(detected_func, col, df)
                if result:
                    logger.debug("Column-aware aggregation matched: %s(%s)", detected_func, col)
                    return result

        return QueryResult.fail("NO_MATCH", "No aggregation pattern matched.", engine="structured")

    def _detect_aggregation_function(self, question: str) -> str | None:
        """Detect the aggregation function from natural language in the question.

        Args:
            question: The lowercased question string.

        Returns:
            Aggregation function name or None.
        """
        for func, keywords in self.AGG_KEYWORDS.items():
            if any(kw in question for kw in keywords):
                return func
        return None

    def _find_mentioned_columns(self, question: str, columns: list[str]) -> list[str]:
        """Find DataFrame columns mentioned in the question.

        Matches columns case-insensitively and returns them sorted by name length
        (longest first) to prefer specific matches over accidental substrings.

        Args:
            question: The lowercased question string.
            columns: Available DataFrame column names.

        Returns:
            List of matched column names, longest first.
        """
        mentioned = [col for col in columns if col.lower() in question]
        mentioned.sort(key=len, reverse=True)
        return mentioned

    # Human-readable labels for aggregation functions.
    AGG_LABELS: dict[str, str] = {
        "sum": "total",
        "mean": "average",
        "median": "median",
        "max": "maximum",
        "min": "minimum",
        "std": "standard deviation",
        "count": "count",
    }

    def _execute_aggregation(self, agg_func: str, column: str, df: pd.DataFrame) -> QueryResult | None:
        """Execute an aggregation function on a column.

        Args:
            agg_func: The aggregation function name (sum, mean, median, max, min, std, count).
            column: The DataFrame column name.
            df: The DataFrame.

        Returns:
            QueryResult on success, None if the aggregation is not applicable.
        """
        try:
            label = self.AGG_LABELS.get(agg_func, agg_func)

            if agg_func == "count":
                non_null = df[column].count()
                total = len(df)
                data = f"The {label} of {column} is: {non_null} (out of {total} rows)"
                return QueryResult.ok(
                    data=data,
                    engine="structured",
                    confidence=0.9,
                    metadata={"data": non_null},
                )

            if not pd.api.types.is_numeric_dtype(df[column]):
                if agg_func in ["max", "min"]:
                    result = getattr(df[column], agg_func)()
                    return QueryResult.ok(
                        data=f"The {label} of {column} is: {result}",
                        engine="structured",
                        confidence=0.9,
                        metadata={"data": result},
                    )
                if agg_func == "count":
                    non_null = df[column].count()
                    return QueryResult.ok(
                        data=f"The {label} of {column} is: {non_null}",
                        engine="structured",
                        confidence=0.9,
                        metadata={"data": non_null},
                    )
                return None

            result = getattr(df[column], agg_func)()
            formatted = f"{result:.2f}" if isinstance(result, float) else str(result)

            return QueryResult.ok(
                data=f"The {label} of {column} is: {formatted}",
                engine="structured",
                confidence=0.9,
                metadata={"data": result},
            )

        except Exception as e:
            logger.debug("Error in aggregation %s for column %s: %s", agg_func, column, e)
            return None

    def _handle_filter_query(self, question: str, df: pd.DataFrame, columns: list[str]) -> QueryResult:
        """Handle filtering queries like 'show records where EventType is Festival'."""
        filter_patterns = [
            r"(?:show|list|find|get)\s+records\s+where\s+(\w+)\s+(?:is|equals?|=)\s+(\w+)",
            r"records\s+where\s+(\w+)\s+(?:is|equals?|=)\s+(\w+)",
            r"where\s+(\w+)\s+(?:is|equals?|=)\s+(\w+)",
        ]

        for pattern in filter_patterns:
            match = re.search(pattern, question)
            if not match:
                continue

            column_name, value = match.groups()
            actual_column = self._find_column_name(column_name, columns)
            if not actual_column:
                continue

            try:
                filtered_df = df[df[actual_column].astype(str).str.lower() == value.lower()]

                if filtered_df.empty:
                    return QueryResult.ok(
                        data=f"No records found where {actual_column} is {value}",
                        engine="structured",
                        confidence=0.9,
                        metadata={"data": filtered_df},
                    )

                result_text = f"Found {len(filtered_df)} records where {actual_column} is {value}:\n\n"
                result_text += self._format_dataframe(filtered_df)

                return QueryResult.ok(
                    data=result_text,
                    engine="structured",
                    confidence=0.9,
                    metadata={"data": filtered_df},
                )

            except Exception as e:
                logger.debug("Error filtering by %s = %s: %s", actual_column, value, e)
                continue

        return QueryResult.fail("NO_MATCH", "No filter pattern matched.", engine="structured")

    def _handle_column_query(self, question: str, df: pd.DataFrame, columns: list[str]) -> QueryResult:
        """Handle column display queries like 'show Location column'."""
        column_patterns = [
            r"(?:show|display|get)\s+(\w+)\s+(?:column|values?)",
            r"(?:unique|distinct)\s+(\w+)",
            r"(\w+)\s+(?:column|values?)",
        ]

        for pattern in column_patterns:
            match = re.search(pattern, question)
            if not match:
                continue

            column_name = match.group(1)
            actual_column = self._find_column_name(column_name, columns)
            if not actual_column:
                continue

            try:
                if "unique" in question or "distinct" in question:
                    unique_values = df[actual_column].unique()
                    result_text = f"Unique values in {actual_column}:\n"
                    result_text += "\n".join([f"- {val}" for val in unique_values[:20]])
                    if len(unique_values) > 20:
                        result_text += f"\n... and {len(unique_values) - 20} more values"

                    return QueryResult.ok(
                        data=result_text,
                        engine="structured",
                        confidence=0.9,
                        metadata={"data": unique_values},
                    )

                values = df[actual_column].tolist()
                result_text = f"Values in {actual_column} column:\n"
                result_text += "\n".join([f"- {val}" for val in values[:20]])
                if len(values) > 20:
                    result_text += f"\n... and {len(values) - 20} more values"

                return QueryResult.ok(
                    data=result_text,
                    engine="structured",
                    confidence=0.9,
                    metadata={"data": values},
                )

            except Exception as e:
                logger.debug("Error displaying column %s: %s", actual_column, e)
                continue

        return QueryResult.fail("NO_MATCH", "No column pattern matched.", engine="structured")

    def _handle_comparison_query(self, question: str, df: pd.DataFrame, columns: list[str]) -> QueryResult:
        """Handle comparison queries like 'DecibelsA above 80'."""
        comparison_patterns = [
            (r"(\w+)\s+(?:above|over|greater\s+than|>|>=)\s+(\d+(?:\.\d+)?)", ">"),
            (r"(\w+)\s+(?:below|under|less\s+than|<|<=)\s+(\d+(?:\.\d+)?)", "<"),
            (r"(\w+)\s+(?:equals?|=|==)\s+(\d+(?:\.\d+)?)", "="),
        ]

        for pattern, operator in comparison_patterns:
            match = re.search(pattern, question)
            if not match:
                continue

            column_name, value_str = match.groups()
            actual_column = self._find_column_name(column_name, columns)
            if not actual_column:
                continue

            try:
                value = float(value_str)

                if not pd.api.types.is_numeric_dtype(df[actual_column]):
                    continue

                if operator == ">":
                    filtered_df = df[df[actual_column] > value]
                    op_text = f"greater than {value}"
                elif operator == "<":
                    filtered_df = df[df[actual_column] < value]
                    op_text = f"less than {value}"
                else:
                    filtered_df = df[df[actual_column] == value]
                    op_text = f"equal to {value}"

                if filtered_df.empty:
                    return QueryResult.ok(
                        data=f"No records found where {actual_column} is {op_text}",
                        engine="structured",
                        confidence=0.9,
                        metadata={"data": filtered_df},
                    )

                result_text = f"Found {len(filtered_df)} records where {actual_column} is {op_text}:\n\n"
                result_text += self._format_dataframe(filtered_df)

                return QueryResult.ok(
                    data=result_text,
                    engine="structured",
                    confidence=0.9,
                    metadata={"data": filtered_df},
                )

            except Exception as e:
                logger.debug("Error in comparison query for %s: %s", actual_column, e)
                continue

        return QueryResult.fail("NO_MATCH", "No comparison pattern matched.", engine="structured")

    def _handle_statistics_query(self, question: str, df: pd.DataFrame, columns: list[str]) -> QueryResult:
        """Handle basic statistics queries."""
        if not any(word in question for word in ["statistics", "stats", "describe"]):
            return QueryResult.fail("NO_MATCH", "Not a statistics query.", engine="structured")

        numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()

        if not numeric_columns:
            return QueryResult.ok(
                data="No numeric columns found for statistical analysis",
                engine="structured",
                confidence=0.9,
            )

        stats_text = "Basic Statistics:\n\n"
        for col in numeric_columns[:5]:
            stats = df[col].describe()
            stats_text += f"{col}:\n"
            stats_text += f"  Count: {stats['count']:.0f}\n"
            stats_text += f"  Mean: {stats['mean']:.2f}\n"
            stats_text += f"  Std: {stats['std']:.2f}\n"
            stats_text += f"  Min: {stats['min']:.2f}\n"
            stats_text += f"  Max: {stats['max']:.2f}\n\n"

        return QueryResult.ok(
            data=stats_text,
            engine="structured",
            confidence=0.9,
            metadata={"data": df[numeric_columns].describe()},
        )

    def _handle_count_query(self, question: str, df: pd.DataFrame, columns: list[str]) -> QueryResult:
        """Handle count queries like 'how many records', 'count records'."""
        count_patterns = [
            r"(?:how many|count)\s+records",
            r"number of records",
            r"total records",
        ]

        for pattern in count_patterns:
            if re.search(pattern, question):
                count = len(df)
                return QueryResult.ok(
                    data=f"Total number of records: {count}",
                    engine="structured",
                    confidence=0.95,
                    metadata={"data": count},
                )

        return QueryResult.fail("NO_MATCH", "Not a count query.", engine="structured")

    def _find_column_name(self, search_name: str, columns: list[str]) -> Optional[str]:
        """Find the actual column name from a search string (case insensitive)."""
        search_lower = search_name.lower()

        for col in columns:
            if col.lower() == search_lower:
                return col

        for col in columns:
            if search_lower in col.lower() or col.lower() in search_lower:
                return col

        return None

    def _format_dataframe(self, df: pd.DataFrame, max_rows: int = 10) -> str:
        """Format DataFrame for display."""
        if df.empty:
            return "No data to display"

        header = "| " + " | ".join(df.columns) + " |"
        separator = "|" + "|".join(["-" * (len(col) + 2) for col in df.columns]) + "|"

        rows = [header, separator]

        for _, row in df.head(max_rows).iterrows():
            formatted_row = "| " + " | ".join([str(val) for val in row.values]) + " |"
            rows.append(formatted_row)

        if len(df) > max_rows:
            rows.append(f"| ... and {len(df) - max_rows} more rows ... |")

        return "\n".join(rows)
