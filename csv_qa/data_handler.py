"""
CSV Data Handler module
"""
import pandas as pd
import traceback


class CSVDataHandler:
    """
    Handles loading and processing CSV data
    """

    def __init__(self, debug_mode=False):
        """Initialize the CSV data handler"""
        self.csv_dataframe = None
        self.csv_columns = []
        self.debug_mode = debug_mode
        self.current_csv_path = None
        self._cache_clear_callbacks = []

    def add_cache_clear_callback(self, callback):
        """
        Add a callback function to be called when cache should be cleared

        Args:
            callback: Function to call when cache needs to be cleared
        """
        self._cache_clear_callbacks.append(callback)

    def _trigger_cache_clear(self):
        """
        Trigger all registered cache clear callbacks
        """
        for callback in self._cache_clear_callbacks:
            try:
                callback()
            except Exception as e:
                if self.debug_mode:
                    print(f"Error in cache clear callback: {e}")

    def load_csv(self, csv_path: str):
        """Load CSV file and store dataframe and columns"""
        try:
            # Check if we're loading a different CSV file
            if self.current_csv_path and self.current_csv_path != csv_path:
                if self.debug_mode:
                    print(
                        f"Loading new CSV file: {csv_path} (previous: {self.current_csv_path})")
                # Clear caches when loading a different CSV
                self._trigger_cache_clear()
            elif not self.current_csv_path:
                if self.debug_mode:
                    print(f"Loading CSV file for the first time: {csv_path}")

            self.csv_dataframe = pd.read_csv(csv_path)
            self.csv_columns = list(self.csv_dataframe.columns)
            self.current_csv_path = csv_path

            if self.debug_mode:
                print(f"Loaded CSV with columns: {self.csv_columns}")

            return self.csv_dataframe
        except Exception as e:
            print(f"Error loading CSV: {e}")
            traceback.print_exc()
            raise

    def get_columns(self):
        """Get the CSV column names"""
        return self.csv_columns

    def get_dataframe(self):
        """Get the CSV dataframe"""
        return self.csv_dataframe

    def is_loaded(self):
        """Check if CSV data is loaded"""
        return self.csv_dataframe is not None

    def create_chunks(self):
        """Create descriptive text chunks for each row in the CSV"""
        if self.csv_dataframe is None:
            raise ValueError("CSV not loaded. Call load_csv first.")

        chunks = []
        for idx, row in self.csv_dataframe.iterrows():
            # Create a descriptive text for each row including all columns
            chunk_parts = []
            for col in self.csv_columns:
                chunk_parts.append(f"{col}: {row[col]}")

            # Join all column descriptions with pipe separator
            chunk = f"Row {idx}: " + " | ".join(chunk_parts)
            chunks.append(chunk)

        return chunks

    def find_rows_by_value(self, column, value, exact_match=False):
        """Find rows where the specified column matches the value

        Args:
            column: Column name to search in
            value: Value to search for
            exact_match: If True, only return exact matches; if False, also include contains matches

        Returns:
            DataFrame of matching rows
        """
        if self.csv_dataframe is None:
            raise ValueError("CSV not loaded. Call load_csv first.")

        matches = None

        # Case-insensitive search if the value is a string
        if isinstance(value, str):
            # Try exact match first
            exact_matches = self.csv_dataframe[self.csv_dataframe[column].astype(
                str).str.lower() == value.lower()]

            # If exact match found or only exact matches requested, return those
            if not exact_matches.empty or exact_match:
                return exact_matches

            # Otherwise try contains match
            contains_matches = self.csv_dataframe[self.csv_dataframe[column].astype(
                str).str.lower().str.contains(value.lower())]
            return contains_matches
        else:
            # For non-string values, only do exact matching
            matches = self.csv_dataframe[self.csv_dataframe[column] == value]

        return matches

    def filter_by_comparison(self, column, operator, value):
        """Filter the dataframe based on a numerical comparison

        Args:
            column: Column name to filter on
            operator: Comparison operator ('>', '<', or '=')
            value: Numerical value to compare against

        Returns:
            DataFrame of matching rows
        """
        if self.csv_dataframe is None:
            raise ValueError("CSV not loaded. Call load_csv first.")

        # Convert column to numeric if possible
        try:
            # Try to convert the column to numeric values
            numeric_column = pd.to_numeric(self.csv_dataframe[column])

            # Apply the comparison operator
            if operator == '>':
                matches = self.csv_dataframe[numeric_column > value]
            elif operator == '<':
                matches = self.csv_dataframe[numeric_column < value]
            else:  # operator == '='
                matches = self.csv_dataframe[numeric_column == value]

            return matches
        except (ValueError, TypeError):
            # If column can't be converted to numeric, return empty DataFrame
            return pd.DataFrame()

    def execute_query_plan(self, query_plan: dict) -> tuple[pd.DataFrame, str]:
        """Execute a query plan generated by the LLM

        Args:
            query_plan: Dictionary with query plan details

        Returns:
            tuple: (DataFrame of results, description of operation)
        """
        if self.csv_dataframe is None:
            raise ValueError("CSV not loaded. Call load_csv first.")

        if not query_plan or not isinstance(query_plan, dict):
            return pd.DataFrame(), "Invalid query plan"

        operation = query_plan.get('operation', '').lower()
        columns = query_plan.get('columns', [])
        filters = query_plan.get('filters', [])
        groupby = query_plan.get('groupby', [])
        sort = query_plan.get('sort', {})
        limit = query_plan.get('limit', None)
        description = query_plan.get('description', 'Query results')

        # Start with the full dataframe
        result_df = self.csv_dataframe.copy()

        # Apply filters if any
        if filters:
            if isinstance(filters, list):
                for filter_cond in filters:
                    if isinstance(filter_cond, dict):
                        col = filter_cond.get('column')
                        op = filter_cond.get('operator')
                        val = filter_cond.get('value')

                        if col and op and val is not None:
                            try:
                                # Convert column to numeric if possible for numerical comparisons
                                if op in ['>', '<', '>=', '<=', '=', '=='] and isinstance(val, (int, float)):
                                    try:
                                        numeric_col = pd.to_numeric(
                                            result_df[col])
                                        if op == '>':
                                            result_df = result_df[numeric_col > val]
                                        elif op == '<':
                                            result_df = result_df[numeric_col < val]
                                        elif op == '>=':
                                            result_df = result_df[numeric_col >= val]
                                        elif op == '<=':
                                            result_df = result_df[numeric_col <= val]
                                        elif op in ['=', '==']:
                                            result_df = result_df[numeric_col == val]
                                    except (ValueError, TypeError):
                                        # If conversion fails, use string comparison
                                        if op in ['=', '==']:
                                            result_df = result_df[result_df[col].astype(
                                                str).str.lower() == str(val).lower()]
                                        elif op == 'contains':
                                            result_df = result_df[result_df[col].astype(
                                                str).str.lower().str.contains(str(val).lower())]
                                else:
                                    # String operations
                                    if op in ['=', '==']:
                                        result_df = result_df[result_df[col].astype(
                                            str).str.lower() == str(val).lower()]
                                    elif op == 'contains':
                                        result_df = result_df[result_df[col].astype(
                                            str).str.lower().str.contains(str(val).lower())]
                            except Exception as e:
                                print(
                                    f"Error applying filter {filter_cond}: {e}")

        # Handle different operations
        if operation == 'list':
            # List just returns the filtered data, optionally with specific columns
            # For 'where' queries, we want to return all columns of the filtered rows
            # Only filter columns if explicitly specified and non-empty
            if columns and len(columns) > 0:
                try:
                    result_df = result_df[columns]
                except KeyError:
                    # If some columns don't exist, just use what we have
                    valid_cols = [
                        col for col in columns if col in result_df.columns]
                    if valid_cols:
                        result_df = result_df[valid_cols]

        elif operation == 'count':
            # Count the number of rows, possibly grouped
            if groupby:
                try:
                    result_df = result_df.groupby(
                        groupby).size().reset_index(name='Count')
                except KeyError:
                    # If groupby columns don't exist, just count all rows
                    count = len(result_df)
                    result_df = pd.DataFrame({'Count': [count]})
            else:
                count = len(result_df)
                result_df = pd.DataFrame({'Count': [count]})

        elif operation == 'aggregate':
            # Perform aggregation operations
            if groupby and columns:
                try:
                    agg_dict = {}
                    for col in columns:
                        if isinstance(col, dict):
                            agg_col = col.get('column')
                            agg_op = col.get('operation', 'mean')
                            if agg_col:
                                agg_dict[agg_col] = agg_op
                        else:
                            agg_dict[col] = 'mean'  # Default to mean

                    if agg_dict:
                        result_df = result_df.groupby(
                            groupby).agg(agg_dict).reset_index()
                except Exception as e:
                    print(f"Error in aggregation: {e}")

        elif operation == 'summarize':
            # Provide summary statistics for columns
            if columns:
                try:
                    valid_cols = [
                        col for col in columns if col in result_df.columns]
                    if valid_cols:
                        result_df = result_df[valid_cols].describe(
                        ).reset_index()
                except Exception as e:
                    print(f"Error in summarize: {e}")

        # Apply sorting if specified
        if sort:
            try:
                sort_col = sort.get('column')
                sort_order = sort.get('order', 'asc')
                if sort_col and sort_col in result_df.columns:
                    ascending = sort_order.lower() != 'desc'
                    result_df = result_df.sort_values(
                        by=sort_col, ascending=ascending)
            except Exception as e:
                print(f"Error in sorting: {e}")

        # Apply limit if specified
        if limit and isinstance(limit, int) and limit > 0:
            result_df = result_df.head(limit)

        return result_df, description

    def get_sample_data(self, num_rows=5):
        """Get sample data from the CSV for LLM context

        Args:
            num_rows: Number of rows to include in the sample

        Returns:
            str: Formatted sample data
        """
        if self.csv_dataframe is None:
            raise ValueError("CSV not loaded. Call load_csv first.")

        sample = self.csv_dataframe.head(num_rows)
        return sample.to_string(index=False)

    def search_value_in_all_columns(self, value, logical_or=False, previous_matches=None):
        """Search for a value across all columns

        Args:
            value: Value to search for across all columns
            logical_or: If True, combine results with previous_matches using OR logic
            previous_matches: Optional DataFrame of previous matches to combine with

        Returns:
            DataFrame of matching rows
        """
        if self.csv_dataframe is None:
            raise ValueError("CSV not loaded. Call load_csv first.")

        value_str = str(value).lower()
        matches = []
        matched_indices = set()

        # If we have previous matches and using OR logic, initialize with those indices
        if logical_or and previous_matches is not None and not previous_matches.empty:
            for idx in previous_matches.index:
                matched_indices.add(idx)
                matches.append(previous_matches.loc[idx])

        for col in self.csv_columns:
            try:
                # Try exact matches first
                exact_matches = self.csv_dataframe[self.csv_dataframe[col].astype(
                    str).str.lower() == value_str]
                if not exact_matches.empty:
                    for idx, row in exact_matches.iterrows():
                        if idx not in matched_indices:
                            matches.append(row)
                            matched_indices.add(idx)

                # Then try substring matches
                contains_matches = self.csv_dataframe[self.csv_dataframe[col].astype(
                    str).str.lower().str.contains(value_str)]
                if not contains_matches.empty:
                    for idx, row in contains_matches.iterrows():
                        if idx not in matched_indices:
                            matches.append(row)
                            matched_indices.add(idx)
            except Exception as e:
                # Skip columns that can't be converted to string or have other issues
                if self.debug_mode:
                    print(f"Error searching in column {col}: {e}")
                continue

        return pd.DataFrame(matches) if matches else pd.DataFrame()

    def search_multiple_values(self, values, logical_operator="or"):
        """Search for multiple values across all columns with logical operators

        Args:
            values: List of values to search for
            logical_operator: 'or' or 'and' to combine results

        Returns:
            DataFrame of matching rows
        """
        if not values:
            return pd.DataFrame()

        # Start with the first value
        result_df = self.search_value_in_all_columns(values[0])

        # Process remaining values with the specified logical operator
        for value in values[1:]:
            if logical_operator.lower() == "or":
                # OR: Add new matches to existing results
                new_matches = self.search_value_in_all_columns(
                    value, logical_or=True, previous_matches=result_df)
                result_df = new_matches
            elif logical_operator.lower() == "and":
                # AND: Only keep rows that match both conditions
                new_matches = self.search_value_in_all_columns(value)
                if not new_matches.empty and not result_df.empty:
                    # Keep only rows that exist in both DataFrames
                    result_df = result_df[result_df.index.isin(
                        new_matches.index)]
                else:
                    # If either is empty, result is empty (AND logic)
                    result_df = pd.DataFrame()

        return result_df

    def analyze_data(self, filter_column=None, filter_value=None, target_columns=None):
        """Perform statistical analysis on the data

        Args:
            filter_column: Optional column to filter on
            filter_value: Optional value to filter by
            target_columns: Optional list of columns to analyze (defaults to all numeric columns)

        Returns:
            dict: Analysis results with statistics and outliers
        """
        if self.csv_dataframe is None:
            raise ValueError("CSV not loaded. Call load_csv first.")

        # Start with the full dataframe
        df = self.csv_dataframe

        # Apply filter if specified
        if filter_column and filter_value:
            try:
                if isinstance(filter_value, str):
                    df = df[df[filter_column].astype(
                        str).str.lower() == filter_value.lower()]
                else:
                    df = df[df[filter_column] == filter_value]
            except Exception as e:
                if self.debug_mode:
                    print(f"Error filtering data: {e}")
                # Continue with unfiltered data

        # If no records after filtering
        if df.empty:
            return {"error": "No data matching the filter criteria"}

        # Identify numeric columns if not specified
        if not target_columns:
            target_columns = df.select_dtypes(
                include=['number']).columns.tolist()
        else:
            # Filter to only include columns that exist and are numeric
            target_columns = [
                col for col in target_columns
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
            ]

        if not target_columns:
            return {"error": "No numeric columns found for analysis"}

        # Initialize results dictionary
        results = {
            "record_count": len(df),
            "columns": {},
            "categorical_counts": {},
            "outliers": {}
        }

        # Get categorical columns (non-numeric) for distribution analysis
        categorical_columns = [
            col for col in df.columns if col not in target_columns]

        # Analyze each numeric column
        for col in target_columns:
            try:
                col_stats = {
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "mean": float(df[col].mean()),
                    "median": float(df[col].median()),
                    "std": float(df[col].std())
                }

                # Identify outliers (values more than 2 standard deviations from mean)
                mean = col_stats["mean"]
                std = col_stats["std"]
                outliers = df[(df[col] < mean - 2*std) |
                              (df[col] > mean + 2*std)]

                if not outliers.empty:
                    results["outliers"][col] = {
                        "count": len(outliers),
                        "values": outliers[col].tolist(),
                        # Limit to 5 outlier records
                        "records": outliers.to_dict(orient="records")[:5]
                    }

                results["columns"][col] = col_stats
            except Exception as e:
                if self.debug_mode:
                    print(f"Error analyzing column {col}: {e}")
                continue

        # Get distribution of categorical columns
        for col in categorical_columns[:5]:  # Limit to 5 categorical columns
            try:
                value_counts = df[col].value_counts().head(10)  # Top 10 values
                results["categorical_counts"][col] = value_counts.to_dict()
            except Exception as e:
                if self.debug_mode:
                    print(f"Error analyzing categorical column {col}: {e}")
                continue

        return results
