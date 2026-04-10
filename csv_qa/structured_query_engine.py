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

    def _handle_aggregation_query(
        self, question: str, df: pd.DataFrame, columns: list[str]
    ) -> QueryResult:
        """Handle aggregation queries like 'max DecibelsA', 'average Location'."""
        agg_patterns = [
            (r"(?:what is|show|get)\s+(?:the\s+)?(max|maximum)\s+(\w+)", "max"),
            (r"(?:what is|show|get)\s+(?:the\s+)?(min|minimum)\s+(\w+)", "min"),
            (r"(?:what is|show|get)\s+(?:the\s+)?(avg|average|mean)\s+(\w+)", "mean"),
            (r"(?:what is|show|get)\s+(?:the\s+)?(sum|total)\s+(\w+)", "sum"),
            (r"(max|maximum)\s+(\w+)", "max"),
            (r"(min|minimum)\s+(\w+)", "min"),
            (r"(avg|average|mean)\s+(?:of\s+)?(\w+)", "mean"),
            (r"(sum|total)\s+(?:of\s+)?(\w+)", "sum"),
        ]

        for pattern, agg_func in agg_patterns:
            match = re.search(pattern, question)
            if not match:
                continue

            column_name = match.groups()[-1]
            actual_column = self._find_column_name(column_name, columns)
            if not actual_column:
                continue

            try:
                if not pd.api.types.is_numeric_dtype(df[actual_column]):
                    if agg_func in ["max", "min"]:
                        result = getattr(df[actual_column], agg_func)()
                        return QueryResult.ok(
                            data=f"The {agg_func} {actual_column} is: {result}",
                            engine="structured",
                            confidence=0.9,
                            metadata={"data": result},
                        )
                    continue

                if agg_func == "mean":
                    result = df[actual_column].mean()
                    return QueryResult.ok(
                        data=f"The average {actual_column} is: {result:.2f}",
                        engine="structured",
                        confidence=0.9,
                        metadata={"data": result},
                    )

                result = getattr(df[actual_column], agg_func)()
                return QueryResult.ok(
                    data=f"The {agg_func} {actual_column} is: {result}",
                    engine="structured",
                    confidence=0.9,
                    metadata={"data": result},
                )

            except Exception as e:
                logger.debug("Error in aggregation for column %s: %s", actual_column, e)
                continue

        return QueryResult.fail("NO_MATCH", "No aggregation pattern matched.", engine="structured")

    def _handle_filter_query(
        self, question: str, df: pd.DataFrame, columns: list[str]
    ) -> QueryResult:
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

    def _handle_column_query(
        self, question: str, df: pd.DataFrame, columns: list[str]
    ) -> QueryResult:
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

    def _handle_comparison_query(
        self, question: str, df: pd.DataFrame, columns: list[str]
    ) -> QueryResult:
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

    def _handle_statistics_query(
        self, question: str, df: pd.DataFrame, columns: list[str]
    ) -> QueryResult:
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

    def _handle_count_query(
        self, question: str, df: pd.DataFrame, columns: list[str]
    ) -> QueryResult:
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
