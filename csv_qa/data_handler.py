"""
CSV Data Handler module.
"""

import logging

import pandas as pd

from csv_qa.exceptions import DataLoadError

logger = logging.getLogger(__name__)


def build_schema_card(df: pd.DataFrame, max_categorical_values: int = 5) -> str:
    """Build a compact schema card describing the DataFrame.

    Produces one line per column with type, cardinality, and either a
    numeric range or the most common categorical values. Includes a
    single example row. Intended to be prepended to LLM contexts so the
    model has a stable anchor to the dataset shape without re-shipping
    raw rows on every call.
    """
    lines: list[str] = [f"Dataset: {len(df)} rows, {len(df.columns)} columns"]
    lines.append("Columns:")

    for col in df.columns:
        series = df[col]
        unique_count = series.nunique(dropna=True)
        if pd.api.types.is_numeric_dtype(series):
            col_type = "numeric"
            if unique_count > 0:
                detail = f"range {series.min()}..{series.max()}"
            else:
                detail = "empty"
        else:
            col_type = "categorical"
            top_values = series.dropna().astype(str).value_counts().head(max_categorical_values)
            if len(top_values) > 0:
                detail = "top=[" + ", ".join(top_values.index.tolist()) + "]"
            else:
                detail = "empty"
        lines.append(f"- {col} ({col_type}, {unique_count} unique, {detail})")

    if len(df) > 0:
        example = df.iloc[0]
        example_parts = [f"{col}={example[col]}" for col in df.columns]
        lines.append("Example row: " + " | ".join(example_parts))

    return "\n".join(lines)


class CSVDataHandler:
    """Handles loading and processing CSV data."""

    def __init__(self) -> None:
        """Initialize the CSV data handler."""
        self.csv_dataframe: pd.DataFrame | None = None
        self.csv_columns: list[str] = []
        self.current_csv_path: str | None = None
        self._cache_clear_callbacks: list = []
        self._schema_card: str | None = None

    def add_cache_clear_callback(self, callback) -> None:
        """Add a callback function to be called when cache should be cleared."""
        self._cache_clear_callbacks.append(callback)

    def _trigger_cache_clear(self) -> None:
        """Trigger all registered cache clear callbacks."""
        for callback in self._cache_clear_callbacks:
            try:
                callback()
            except Exception as e:
                logger.warning("Error in cache clear callback: %s", e)

    def load_csv(self, csv_path: str) -> pd.DataFrame:
        """Load CSV file and store dataframe and columns.

        Raises:
            DataLoadError: If the CSV file cannot be loaded.
        """
        try:
            if self.current_csv_path and self.current_csv_path != csv_path:
                logger.info(
                    "Loading new CSV file: %s (previous: %s)",
                    csv_path,
                    self.current_csv_path,
                )
                self._trigger_cache_clear()
            else:
                logger.info("Loading CSV file: %s", csv_path)

            self.csv_dataframe = pd.read_csv(csv_path)
            self.csv_columns = list(self.csv_dataframe.columns)
            self.current_csv_path = csv_path
            self._schema_card = None

            logger.debug("Loaded CSV with columns: %s", self.csv_columns)

            return self.csv_dataframe
        except Exception as e:
            raise DataLoadError(detail=str(e), path=csv_path) from e

    def get_columns(self) -> list[str]:
        """Get the CSV column names."""
        return self.csv_columns

    def get_dataframe(self) -> pd.DataFrame | None:
        """Get the CSV dataframe."""
        return self.csv_dataframe

    def is_loaded(self) -> bool:
        """Check if CSV data is loaded."""
        return self.csv_dataframe is not None

    def get_schema_card(self, max_categorical_values: int = 5) -> str:
        """Return a cached schema card for the loaded CSV."""
        if self.csv_dataframe is None:
            raise ValueError("CSV not loaded. Call load_csv first.")

        if self._schema_card is not None:
            return self._schema_card

        self._schema_card = build_schema_card(self.csv_dataframe, max_categorical_values)
        return self._schema_card

    def create_chunks(self) -> list[str]:
        """Create descriptive text chunks for each row in the CSV."""
        if self.csv_dataframe is None:
            raise ValueError("CSV not loaded. Call load_csv first.")

        chunks = []
        for idx, row in self.csv_dataframe.iterrows():
            chunk_parts = []
            for col in self.csv_columns:
                chunk_parts.append(f"{col}: {row[col]}")

            chunk = f"Row {idx}: " + " | ".join(chunk_parts)
            chunks.append(chunk)

        return chunks

    def find_rows_by_value(self, column: str, value, exact_match: bool = False) -> pd.DataFrame:
        """Find rows where the specified column matches the value.

        Args:
            column: Column name to search in.
            value: Value to search for.
            exact_match: If True, only return exact matches.

        Returns:
            DataFrame of matching rows.
        """
        if self.csv_dataframe is None:
            raise ValueError("CSV not loaded. Call load_csv first.")

        if isinstance(value, str):
            exact_matches = self.csv_dataframe[self.csv_dataframe[column].astype(str).str.lower() == value.lower()]

            if not exact_matches.empty or exact_match:
                return exact_matches

            contains_matches = self.csv_dataframe[
                self.csv_dataframe[column].astype(str).str.lower().str.contains(value.lower())
            ]
            return contains_matches

        return self.csv_dataframe[self.csv_dataframe[column] == value]

    def filter_by_comparison(self, column: str, operator: str, value: float) -> pd.DataFrame:
        """Filter the dataframe based on a numerical comparison.

        Args:
            column: Column name to filter on.
            operator: Comparison operator ('>', '<', or '=').
            value: Numerical value to compare against.

        Returns:
            DataFrame of matching rows.
        """
        if self.csv_dataframe is None:
            raise ValueError("CSV not loaded. Call load_csv first.")

        try:
            numeric_column = pd.to_numeric(self.csv_dataframe[column])

            if operator == ">":
                return self.csv_dataframe[numeric_column > value]
            elif operator == "<":
                return self.csv_dataframe[numeric_column < value]
            else:
                return self.csv_dataframe[numeric_column == value]
        except (ValueError, TypeError):
            return pd.DataFrame()

    def execute_query_plan(self, query_plan: dict) -> tuple[pd.DataFrame, str]:
        """Execute a query plan generated by the LLM.

        Args:
            query_plan: Dictionary with query plan details.

        Returns:
            Tuple of (DataFrame of results, description of operation).
        """
        if self.csv_dataframe is None:
            raise ValueError("CSV not loaded. Call load_csv first.")

        if not query_plan or not isinstance(query_plan, dict):
            return pd.DataFrame(), "Invalid query plan"

        operation = query_plan.get("operation", "").lower()
        columns = query_plan.get("columns", [])
        filters = query_plan.get("filters", [])
        groupby = query_plan.get("groupby", [])
        sort = query_plan.get("sort", {})
        limit = query_plan.get("limit", None)
        description = query_plan.get("description", "Query results")

        result_df = self.csv_dataframe.copy()

        # Apply filters
        if filters and isinstance(filters, list):
            for filter_cond in filters:
                if not isinstance(filter_cond, dict):
                    continue

                col = filter_cond.get("column")
                op = filter_cond.get("operator")
                val = filter_cond.get("value")

                if not (col and op and val is not None):
                    continue

                try:
                    if op in [">", "<", ">=", "<=", "=", "=="] and isinstance(val, (int, float)):
                        try:
                            numeric_col = pd.to_numeric(result_df[col])
                            if op == ">":
                                result_df = result_df[numeric_col > val]
                            elif op == "<":
                                result_df = result_df[numeric_col < val]
                            elif op == ">=":
                                result_df = result_df[numeric_col >= val]
                            elif op == "<=":
                                result_df = result_df[numeric_col <= val]
                            elif op in ["=", "=="]:
                                result_df = result_df[numeric_col == val]
                        except (ValueError, TypeError):
                            if op in ["=", "=="]:
                                result_df = result_df[result_df[col].astype(str).str.lower() == str(val).lower()]
                            elif op == "contains":
                                result_df = result_df[
                                    result_df[col].astype(str).str.lower().str.contains(str(val).lower())
                                ]
                    else:
                        if op in ["=", "=="]:
                            result_df = result_df[result_df[col].astype(str).str.lower() == str(val).lower()]
                        elif op == "contains":
                            result_df = result_df[result_df[col].astype(str).str.lower().str.contains(str(val).lower())]
                except Exception as e:
                    logger.warning("Error applying filter %s: %s", filter_cond, e)

        # Handle operations
        if operation == "list":
            if columns and len(columns) > 0:
                try:
                    result_df = result_df[columns]
                except KeyError:
                    valid_cols = [col for col in columns if col in result_df.columns]
                    if valid_cols:
                        result_df = result_df[valid_cols]

        elif operation == "count":
            if groupby:
                try:
                    result_df = result_df.groupby(groupby).size().reset_index(name="Count")
                except KeyError:
                    count = len(result_df)
                    result_df = pd.DataFrame({"Count": [count]})
            else:
                count = len(result_df)
                result_df = pd.DataFrame({"Count": [count]})

        elif operation == "aggregate":
            if groupby and columns:
                try:
                    agg_dict = {}
                    for col in columns:
                        if isinstance(col, dict):
                            agg_col = col.get("column")
                            agg_op = col.get("operation", "mean")
                            if agg_col:
                                agg_dict[agg_col] = agg_op
                        else:
                            agg_dict[col] = "mean"

                    if agg_dict:
                        result_df = result_df.groupby(groupby).agg(agg_dict).reset_index()
                except Exception as e:
                    logger.warning("Error in aggregation: %s", e)

        elif operation == "summarize":
            if columns:
                try:
                    valid_cols = [col for col in columns if col in result_df.columns]
                    if valid_cols:
                        result_df = result_df[valid_cols].describe().reset_index()
                except Exception as e:
                    logger.warning("Error in summarize: %s", e)

        # Apply sorting
        if sort:
            try:
                sort_col = sort.get("column")
                sort_order = sort.get("order", "asc")
                if sort_col and sort_col in result_df.columns:
                    ascending = sort_order.lower() != "desc"
                    result_df = result_df.sort_values(by=sort_col, ascending=ascending)
            except Exception as e:
                logger.warning("Error in sorting: %s", e)

        # Apply limit
        if limit and isinstance(limit, int) and limit > 0:
            result_df = result_df.head(limit)

        return result_df, description

    def get_sample_data(self, num_rows: int = 5) -> str:
        """Get sample data from the CSV for LLM context."""
        if self.csv_dataframe is None:
            raise ValueError("CSV not loaded. Call load_csv first.")

        sample = self.csv_dataframe.head(num_rows)
        return sample.to_string(index=False)

    def search_value_in_all_columns(
        self, value, logical_or: bool = False, previous_matches: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """Search for a value across all columns."""
        if self.csv_dataframe is None:
            raise ValueError("CSV not loaded. Call load_csv first.")

        value_str = str(value).lower()
        matches = []
        matched_indices: set = set()

        if logical_or and previous_matches is not None and not previous_matches.empty:
            for idx in previous_matches.index:
                matched_indices.add(idx)
                matches.append(previous_matches.loc[idx])

        for col in self.csv_columns:
            try:
                exact_matches = self.csv_dataframe[self.csv_dataframe[col].astype(str).str.lower() == value_str]
                if not exact_matches.empty:
                    for idx, row in exact_matches.iterrows():
                        if idx not in matched_indices:
                            matches.append(row)
                            matched_indices.add(idx)

                contains_matches = self.csv_dataframe[
                    self.csv_dataframe[col].astype(str).str.lower().str.contains(value_str)
                ]
                if not contains_matches.empty:
                    for idx, row in contains_matches.iterrows():
                        if idx not in matched_indices:
                            matches.append(row)
                            matched_indices.add(idx)
            except Exception as e:
                logger.debug("Error searching in column %s: %s", col, e)
                continue

        return pd.DataFrame(matches) if matches else pd.DataFrame()

    def search_multiple_values(self, values: list, logical_operator: str = "or") -> pd.DataFrame:
        """Search for multiple values across all columns with logical operators."""
        if not values:
            return pd.DataFrame()

        result_df = self.search_value_in_all_columns(values[0])

        for value in values[1:]:
            if logical_operator.lower() == "or":
                new_matches = self.search_value_in_all_columns(value, logical_or=True, previous_matches=result_df)
                result_df = new_matches
            elif logical_operator.lower() == "and":
                new_matches = self.search_value_in_all_columns(value)
                if not new_matches.empty and not result_df.empty:
                    result_df = result_df[result_df.index.isin(new_matches.index)]
                else:
                    result_df = pd.DataFrame()

        return result_df

    def analyze_data(
        self,
        filter_column: str | None = None,
        filter_value=None,
        target_columns: list[str] | None = None,
    ) -> dict:
        """Perform statistical analysis on the data."""
        if self.csv_dataframe is None:
            raise ValueError("CSV not loaded. Call load_csv first.")

        df = self.csv_dataframe

        if filter_column and filter_value:
            try:
                if isinstance(filter_value, str):
                    df = df[df[filter_column].astype(str).str.lower() == filter_value.lower()]
                else:
                    df = df[df[filter_column] == filter_value]
            except Exception as e:
                logger.warning("Error filtering data: %s", e)

        if df.empty:
            return {"error": "No data matching the filter criteria"}

        if not target_columns:
            target_columns = df.select_dtypes(include=["number"]).columns.tolist()
        else:
            target_columns = [
                col for col in target_columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
            ]

        if not target_columns:
            return {"error": "No numeric columns found for analysis"}

        results: dict = {
            "record_count": len(df),
            "columns": {},
            "categorical_counts": {},
            "outliers": {},
        }

        categorical_columns = [col for col in df.columns if col not in target_columns]

        for col in target_columns:
            try:
                col_stats = {
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "mean": float(df[col].mean()),
                    "median": float(df[col].median()),
                    "std": float(df[col].std()),
                }

                mean = col_stats["mean"]
                std = col_stats["std"]
                outliers = df[(df[col] < mean - 2 * std) | (df[col] > mean + 2 * std)]

                if not outliers.empty:
                    results["outliers"][col] = {
                        "count": len(outliers),
                        "values": outliers[col].tolist(),
                        "records": outliers.to_dict(orient="records")[:5],
                    }

                results["columns"][col] = col_stats
            except Exception as e:
                logger.debug("Error analyzing column %s: %s", col, e)
                continue

        for col in categorical_columns[:5]:
            try:
                value_counts = df[col].value_counts().head(10)
                results["categorical_counts"][col] = value_counts.to_dict()
            except Exception as e:
                logger.debug("Error analyzing categorical column %s: %s", col, e)
                continue

        return results
