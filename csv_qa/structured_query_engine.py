"""
Structured Query Engine - handles direct DataFrame operations for precise queries
"""
import re
import pandas as pd
from typing import Dict, Any, Optional


class StructuredQueryEngine:
    """Handles structured queries that can be executed directly on DataFrames"""

    def __init__(self, debug_mode=False):
        self.debug_mode = debug_mode

    def execute_query(self, question: str, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Execute a structured query on the DataFrame

        Args:
            question: The user's question
            df: The pandas DataFrame to query

        Returns:
            Dict with 'success', 'result', and optionally 'error'
        """
        try:
            question_lower = question.lower()
            columns = df.columns.tolist()

            if self.debug_mode:
                print(f"Executing structured query: {question}")

            # Try different query patterns in order of specificity

            # 1. Aggregation queries (max, min, avg, sum, count)
            agg_result = self._handle_aggregation_query(
                question_lower, df, columns)
            if agg_result['success']:
                return agg_result

            # 2. Filtering queries (where conditions)
            filter_result = self._handle_filter_query(
                question_lower, df, columns)
            if filter_result['success']:
                return filter_result

            # 3. Column display queries
            column_result = self._handle_column_query(
                question_lower, df, columns)
            if column_result['success']:
                return column_result

            # 4. Comparison queries (above, below, greater than, etc.)
            comparison_result = self._handle_comparison_query(
                question_lower, df, columns)
            if comparison_result['success']:
                return comparison_result

            # 5. Basic statistics queries
            stats_result = self._handle_statistics_query(
                question_lower, df, columns)
            if stats_result['success']:
                return stats_result

            # 6. Count and unique value queries
            count_result = self._handle_count_query(
                question_lower, df, columns)
            if count_result['success']:
                return count_result

            return {'success': False, 'error': 'No matching structured query pattern found'}

        except Exception as e:
            if self.debug_mode:
                print(f"Error in structured query execution: {e}")
            return {'success': False, 'error': str(e)}

    def _handle_aggregation_query(self, question: str, df: pd.DataFrame, columns: list) -> Dict[str, Any]:
        """Handle aggregation queries like 'max DecibelsA', 'average Location'"""

        # Patterns for aggregation queries
        agg_patterns = [
            (r'(?:what is|show|get)\s+(?:the\s+)?(max|maximum)\s+(\w+)', 'max'),
            (r'(?:what is|show|get)\s+(?:the\s+)?(min|minimum)\s+(\w+)', 'min'),
            (r'(?:what is|show|get)\s+(?:the\s+)?(avg|average|mean)\s+(\w+)', 'mean'),
            (r'(?:what is|show|get)\s+(?:the\s+)?(sum|total)\s+(\w+)', 'sum'),
            (r'(max|maximum)\s+(\w+)', 'max'),
            (r'(min|minimum)\s+(\w+)', 'min'),
            (r'(avg|average|mean)\s+(?:of\s+)?(\w+)', 'mean'),
            (r'(sum|total)\s+(?:of\s+)?(\w+)', 'sum')
        ]

        for pattern, agg_func in agg_patterns:
            match = re.search(pattern, question)
            if match:
                # Extract column name (usually the last captured group)
                column_name = match.groups()[-1]

                # Find the actual column name (case insensitive)
                actual_column = self._find_column_name(column_name, columns)
                if not actual_column:
                    continue

                try:
                    # Check if column is numeric
                    if not pd.api.types.is_numeric_dtype(df[actual_column]):
                        # For non-numeric columns, only certain operations make sense
                        if agg_func in ['max', 'min']:
                            result = getattr(df[actual_column], agg_func)()
                            return {
                                'success': True,
                                'result': f"The {agg_func} {actual_column} is: {result}",
                                'data': result
                            }
                        else:
                            continue

                    # Perform aggregation
                    if agg_func == 'mean':
                        result = df[actual_column].mean()
                        return {
                            'success': True,
                            'result': f"The average {actual_column} is: {result:.2f}",
                            'data': result
                        }
                    else:
                        result = getattr(df[actual_column], agg_func)()
                        return {
                            'success': True,
                            'result': f"The {agg_func} {actual_column} is: {result}",
                            'data': result
                        }

                except Exception as e:
                    if self.debug_mode:
                        print(
                            f"Error in aggregation for column {actual_column}: {e}")
                    continue

        return {'success': False}

    def _handle_filter_query(self, question: str, df: pd.DataFrame, columns: list) -> Dict[str, Any]:
        """Handle filtering queries like 'show records where EventType is Festival'"""

        filter_patterns = [
            r'(?:show|list|find|get)\s+records\s+where\s+(\w+)\s+(?:is|equals?|=)\s+(\w+)',
            r'records\s+where\s+(\w+)\s+(?:is|equals?|=)\s+(\w+)',
            r'where\s+(\w+)\s+(?:is|equals?|=)\s+(\w+)'
        ]

        for pattern in filter_patterns:
            match = re.search(pattern, question)
            if match:
                column_name, value = match.groups()

                # Find the actual column name
                actual_column = self._find_column_name(column_name, columns)
                if not actual_column:
                    continue

                try:
                    # Apply filter
                    filtered_df = df[df[actual_column].astype(
                        str).str.lower() == value.lower()]

                    if filtered_df.empty:
                        return {
                            'success': True,
                            'result': f"No records found where {actual_column} is {value}",
                            'data': filtered_df
                        }

                    # Format result
                    result_text = f"Found {len(filtered_df)} records where {actual_column} is {value}:\n\n"
                    result_text += self._format_dataframe(filtered_df)

                    return {
                        'success': True,
                        'result': result_text,
                        'data': filtered_df
                    }

                except Exception as e:
                    if self.debug_mode:
                        print(
                            f"Error filtering by {actual_column} = {value}: {e}")
                    continue

        return {'success': False}

    def _handle_column_query(self, question: str, df: pd.DataFrame, columns: list) -> Dict[str, Any]:
        """Handle column display queries like 'show Location column'"""

        column_patterns = [
            r'(?:show|display|get)\s+(\w+)\s+(?:column|values?)',
            r'(?:unique|distinct)\s+(\w+)',
            r'(\w+)\s+(?:column|values?)'
        ]

        for pattern in column_patterns:
            match = re.search(pattern, question)
            if match:
                column_name = match.group(1)

                # Find the actual column name
                actual_column = self._find_column_name(column_name, columns)
                if not actual_column:
                    continue

                try:
                    if 'unique' in question or 'distinct' in question:
                        # Show unique values
                        unique_values = df[actual_column].unique()
                        result_text = f"Unique values in {actual_column}:\n"
                        result_text += "\n".join(
                            [f"- {val}" for val in unique_values[:20]])
                        if len(unique_values) > 20:
                            result_text += f"\n... and {len(unique_values) - 20} more values"

                        return {
                            'success': True,
                            'result': result_text,
                            'data': unique_values
                        }
                    else:
                        # Show all values in the column
                        values = df[actual_column].tolist()
                        result_text = f"Values in {actual_column} column:\n"
                        result_text += "\n".join(
                            [f"- {val}" for val in values[:20]])
                        if len(values) > 20:
                            result_text += f"\n... and {len(values) - 20} more values"

                        return {
                            'success': True,
                            'result': result_text,
                            'data': values
                        }

                except Exception as e:
                    if self.debug_mode:
                        print(f"Error displaying column {actual_column}: {e}")
                    continue

        return {'success': False}

    def _handle_comparison_query(self, question: str, df: pd.DataFrame, columns: list) -> Dict[str, Any]:
        """Handle comparison queries like 'DecibelsA above 80'"""

        comparison_patterns = [
            (r'(\w+)\s+(?:above|over|greater\s+than|>|>=)\s+(\d+(?:\.\d+)?)', '>'),
            (r'(\w+)\s+(?:below|under|less\s+than|<|<=)\s+(\d+(?:\.\d+)?)', '<'),
            (r'(\w+)\s+(?:equals?|=|==)\s+(\d+(?:\.\d+)?)', '=')
        ]

        for pattern, operator in comparison_patterns:
            match = re.search(pattern, question)
            if match:
                column_name, value_str = match.groups()

                # Find the actual column name
                actual_column = self._find_column_name(column_name, columns)
                if not actual_column:
                    continue

                try:
                    value = float(value_str)

                    # Check if column is numeric
                    if not pd.api.types.is_numeric_dtype(df[actual_column]):
                        continue

                    # Apply comparison
                    if operator == '>':
                        filtered_df = df[df[actual_column] > value]
                        op_text = f"greater than {value}"
                    elif operator == '<':
                        filtered_df = df[df[actual_column] < value]
                        op_text = f"less than {value}"
                    else:  # operator == '='
                        filtered_df = df[df[actual_column] == value]
                        op_text = f"equal to {value}"

                    if filtered_df.empty:
                        return {
                            'success': True,
                            'result': f"No records found where {actual_column} is {op_text}",
                            'data': filtered_df
                        }

                    result_text = f"Found {len(filtered_df)} records where {actual_column} is {op_text}:\n\n"
                    result_text += self._format_dataframe(filtered_df)

                    return {
                        'success': True,
                        'result': result_text,
                        'data': filtered_df
                    }

                except Exception as e:
                    if self.debug_mode:
                        print(
                            f"Error in comparison query for {actual_column}: {e}")
                    continue

        return {'success': False}

    def _handle_statistics_query(self, question: str, df: pd.DataFrame, columns: list) -> Dict[str, Any]:
        """Handle basic statistics queries"""

        if any(word in question for word in ['statistics', 'stats', 'describe']):
            numeric_columns = df.select_dtypes(
                include=['number']).columns.tolist()

            if not numeric_columns:
                return {
                    'success': True,
                    'result': "No numeric columns found for statistical analysis",
                    'data': None
                }

            stats_text = "📊 Basic Statistics:\n\n"
            for col in numeric_columns[:5]:  # Limit to first 5 numeric columns
                stats = df[col].describe()
                stats_text += f"{col}:\n"
                stats_text += f"  Count: {stats['count']:.0f}\n"
                stats_text += f"  Mean: {stats['mean']:.2f}\n"
                stats_text += f"  Std: {stats['std']:.2f}\n"
                stats_text += f"  Min: {stats['min']:.2f}\n"
                stats_text += f"  Max: {stats['max']:.2f}\n\n"

            return {
                'success': True,
                'result': stats_text,
                'data': df[numeric_columns].describe()
            }

        return {'success': False}

    def _handle_count_query(self, question: str, df: pd.DataFrame, columns: list) -> Dict[str, Any]:
        """Handle count queries like 'how many records', 'count records'"""

        count_patterns = [
            r'(?:how many|count)\s+records',
            r'number of records',
            r'total records'
        ]

        for pattern in count_patterns:
            if re.search(pattern, question):
                count = len(df)
                return {
                    'success': True,
                    'result': f"Total number of records: {count}",
                    'data': count
                }

        return {'success': False}

    def _find_column_name(self, search_name: str, columns: list) -> Optional[str]:
        """Find the actual column name from a search string (case insensitive)"""
        search_lower = search_name.lower()

        # Exact match first
        for col in columns:
            if col.lower() == search_lower:
                return col

        # Partial match
        for col in columns:
            if search_lower in col.lower() or col.lower() in search_lower:
                return col

        return None

    def _format_dataframe(self, df: pd.DataFrame, max_rows: int = 10) -> str:
        """Format DataFrame for display"""
        if df.empty:
            return "No data to display"

        # Create table header
        header = "| " + " | ".join(df.columns) + " |"
        separator = "|" + "|".join(["-" * (len(col) + 2)
                                   for col in df.columns]) + "|"

        rows = [header, separator]

        # Add data rows (limit to max_rows)
        for _, row in df.head(max_rows).iterrows():
            formatted_row = "| " + \
                " | ".join([str(val) for val in row.values]) + " |"
            rows.append(formatted_row)

        if len(df) > max_rows:
            rows.append(f"| ... and {len(df) - max_rows} more rows ... |")

        return "\n".join(rows)
