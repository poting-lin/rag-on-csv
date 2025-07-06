"""
Enhanced Vector Search Engine - specifically designed for CSV data analysis
"""
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any, Optional
import traceback


class CSVAwareVectorSearch:
    """Enhanced vector search specifically designed for CSV data"""

    def __init__(self, debug_mode=False):
        self.debug_mode = debug_mode
        self.vectorizer = None
        self.vectors = None
        self.chunk_list = None
        self.chunk_metadata = None
        self.column_relationships = None

    def create_structured_chunks(self, df: pd.DataFrame) -> List[str]:
        """Create more meaningful chunks for CSV data"""
        chunks = []
        metadata = []

        if self.debug_mode:
            print(
                f"Creating structured chunks for DataFrame with {len(df)} rows and {len(df.columns)} columns")

        # 1. Column-specific chunks with rich descriptions
        for col in df.columns:
            col_info = self._analyze_column(df[col])
            chunk = f"Column {col}: {col_info['description']}"
            chunks.append(chunk)
            metadata.append({
                'type': 'column_description',
                'column': col,
                'data_type': col_info['data_type']
            })

        # 2. Row-based chunks with semantic context
        for idx, row in df.iterrows():
            row_desc = self._create_row_description(row, df.columns)
            chunks.append(f"Record {idx}: {row_desc}")
            metadata.append({
                'type': 'data_record',
                'row_index': idx,
                'columns': df.columns.tolist()
            })

            # Limit rows to prevent overwhelming the system
            if idx >= 100:  # Process first 100 rows for vector search
                break

        # 3. Statistical summary chunks for numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            stats = df[col].describe()
            chunk = (f"{col} statistics: "
                     f"min={stats['min']:.2f}, max={stats['max']:.2f}, "
                     f"mean={stats['mean']:.2f}, std={stats['std']:.2f}, "
                     f"median={stats['50%']:.2f}")
            chunks.append(chunk)
            metadata.append({
                'type': 'statistical_summary',
                'column': col,
                'stats': stats.to_dict()
            })

        # 4. Categorical distribution chunks
        categorical_cols = df.select_dtypes(
            include=['object', 'category']).columns
        for col in categorical_cols:
            if df[col].nunique() <= 20:  # Only for columns with reasonable number of unique values
                top_values = df[col].value_counts().head(10)
                chunk = f"{col} distribution: " + ", ".join([
                    f"{val}({count})" for val, count in top_values.items()
                ])
                chunks.append(chunk)
                metadata.append({
                    'type': 'categorical_distribution',
                    'column': col,
                    'distribution': top_values.to_dict()
                })

        # 5. Column relationship chunks
        relationships = self._analyze_column_relationships(df)
        for relationship in relationships:
            chunks.append(relationship)
            metadata.append({
                'type': 'column_relationship',
                'relationship': relationship
            })

        # 6. Data quality and pattern chunks
        quality_info = self._analyze_data_quality(df)
        for info in quality_info:
            chunks.append(info)
            metadata.append({
                'type': 'data_quality',
                'info': info
            })

        self.chunk_metadata = metadata

        if self.debug_mode:
            print(f"Created {len(chunks)} structured chunks")
            print(f"Chunk types: {set(m['type'] for m in metadata)}")

        return chunks

    def _analyze_column(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze a column to create meaningful descriptions"""
        col_type = series.dtype
        unique_count = series.nunique()
        null_count = series.isnull().sum()
        total_count = len(series)

        if pd.api.types.is_numeric_dtype(series):
            # Numeric column analysis
            stats = series.describe()
            description = (f"Numeric column with {unique_count} unique values, "
                           f"range {stats['min']:.2f} to {stats['max']:.2f}, "
                           f"average {stats['mean']:.2f}")

            # Check for potential outliers
            q1, q3 = stats['25%'], stats['75%']
            iqr = q3 - q1
            outliers = series[(series < q1 - 1.5*iqr) |
                              (series > q3 + 1.5*iqr)]
            if len(outliers) > 0:
                description += f", {len(outliers)} potential outliers detected"

            return {
                'description': description,
                'data_type': 'numeric',
                'stats': stats.to_dict()
            }

        elif pd.api.types.is_datetime64_any_dtype(series):
            # DateTime column analysis
            date_range = f"{series.min()} to {series.max()}"
            description = f"Date/time column spanning {date_range}"
            return {
                'description': description,
                'data_type': 'datetime',
                'range': date_range
            }

        else:
            # Categorical/text column analysis
            sample_values = series.dropna().unique()[:5]
            sample_str = ", ".join([str(val) for val in sample_values])

            description = (f"Categorical column with {unique_count} unique values "
                           f"({total_count - null_count} non-null), "
                           f"examples: {sample_str}")

            if len(sample_values) < unique_count:
                description += f" and {unique_count - len(sample_values)} more"

            return {
                'description': description,
                'data_type': 'categorical',
                'unique_count': unique_count,
                'sample_values': sample_values.tolist()
            }

    def _create_row_description(self, row: pd.Series, columns: List[str]) -> str:
        """Create natural language description of a row"""
        descriptions = []

        # Prioritize important-looking columns (shorter names, common types)
        important_cols = [col for col in columns if len(col) <= 15][:5]

        for col in important_cols:
            val = row[col]
            if pd.notna(val):
                descriptions.append(f"{col} is {val}")

        # If we don't have enough descriptions, add more columns
        if len(descriptions) < 3:
            for col in columns:
                if col not in important_cols:
                    val = row[col]
                    if pd.notna(val):
                        descriptions.append(f"{col} is {val}")
                    if len(descriptions) >= 5:  # Limit to 5 descriptions per row
                        break

        return ", ".join(descriptions)

    def _analyze_column_relationships(self, df: pd.DataFrame) -> List[str]:
        """Find correlations and relationships between columns"""
        relationships = []

        # Numeric correlations
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 1:
            correlations = df[numeric_cols].corr()

            for i, col1 in enumerate(numeric_cols):
                for j, col2 in enumerate(numeric_cols[i+1:], i+1):
                    corr = correlations.iloc[i, j]
                    if abs(corr) > 0.5:  # Strong correlation
                        rel_type = "positively correlated" if corr > 0 else "negatively correlated"
                        relationships.append(
                            f"Columns {col1} and {col2} are {rel_type} with correlation {corr:.2f}"
                        )

        # Categorical relationships (basic co-occurrence)
        categorical_cols = df.select_dtypes(
            include=['object', 'category']).columns
        if len(categorical_cols) > 1:
            for i, col1 in enumerate(categorical_cols):
                for col2 in categorical_cols[i+1:]:
                    if df[col1].nunique() <= 10 and df[col2].nunique() <= 10:
                        # Check if values tend to appear together
                        crosstab = pd.crosstab(df[col1], df[col2])
                        if crosstab.max().max() > len(df) * 0.3:  # If 30%+ of data shares same combination
                            relationships.append(
                                f"Columns {col1} and {col2} show co-occurrence patterns"
                            )

        return relationships[:5]  # Limit to top 5 relationships

    def _analyze_data_quality(self, df: pd.DataFrame) -> List[str]:
        """Analyze data quality and patterns"""
        quality_info = []

        # Missing value patterns
        missing_cols = df.columns[df.isnull().any()].tolist()
        if missing_cols:
            # Limit to top 3 columns with missing values
            for col in missing_cols[:3]:
                missing_count = df[col].isnull().sum()
                missing_pct = (missing_count / len(df)) * 100
                quality_info.append(
                    f"Column {col} has {missing_count} missing values ({missing_pct:.1f}%)"
                )

        # Duplicate rows
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            quality_info.append(
                f"Dataset contains {duplicate_count} duplicate rows")

        # Data type consistency
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check for mixed types in object columns
                sample = df[col].dropna().head(100)
                types = set(type(val).__name__ for val in sample)
                if len(types) > 1:
                    quality_info.append(
                        f"Column {col} contains mixed data types: {', '.join(types)}")

        return quality_info[:5]  # Limit to top 5 quality issues

    def build_vector_index(self, chunks: List[str]):
        """Build the vector index with enhanced features"""
        if not chunks:
            if self.debug_mode:
                print("No chunks provided for vector index")
            return

        try:
            # Enhanced TF-IDF with n-grams and better parameters
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2),  # Include bigrams
                min_df=1,
                max_df=0.8,
                lowercase=True,
                token_pattern=r'\b\w+\b'
            )

            # Build vectors
            self.vectors = self.vectorizer.fit_transform(chunks)
            self.chunk_list = chunks

            if self.debug_mode:
                print(f"Built vector index with {len(chunks)} chunks")
                print(f"Vector shape: {self.vectors.shape}")
                print(f"Vocabulary size: {len(self.vectorizer.vocabulary_)}")

        except Exception as e:
            if self.debug_mode:
                print(f"Error building vector index: {e}")
                traceback.print_exc()

    def retrieve_context(self, question: str, top_k: int = 5) -> str:
        """Retrieve the most relevant context with enhanced scoring"""
        if self.vectorizer is None or self.vectors is None or self.chunk_list is None:
            raise ValueError(
                "Vector index not built. Call build_vector_index first.")

        try:
            # Transform the question
            question_vector = self.vectorizer.transform([question])

            # Calculate similarities
            similarities = cosine_similarity(question_vector, self.vectors)[0]

            # Apply semantic boosting based on chunk types
            if self.chunk_metadata:
                similarities = self._apply_semantic_boosting(
                    question, similarities)

            # Get top matches
            top_indices = np.argsort(similarities)[-top_k:][::-1]

            # Filter out very low similarity scores
            filtered_indices = [
                idx for idx in top_indices if similarities[idx] > 0.1]

            if not filtered_indices:
                if self.debug_mode:
                    print("No relevant chunks found with sufficient similarity")
                return ""

            # Combine the top chunks with intelligent ordering
            context_chunks = []
            for idx in filtered_indices:
                chunk = self.chunk_list[idx]
                score = similarities[idx]

                # Add chunk with score information for debugging
                if self.debug_mode:
                    chunk_type = self.chunk_metadata[idx]['type'] if self.chunk_metadata else 'unknown'
                    print(
                        f"Selected chunk (score: {score:.3f}, type: {chunk_type}): {chunk[:100]}...")

                context_chunks.append(chunk)

            context = "\n\n".join(context_chunks)

            if self.debug_mode:
                print(f"Retrieved {len(context_chunks)} chunks as context")
                print(
                    f"Top similarity scores: {[similarities[i] for i in filtered_indices]}")

            return context

        except Exception as e:
            if self.debug_mode:
                print(f"Error in retrieve_context: {e}")
                traceback.print_exc()
            return ""

    def _apply_semantic_boosting(self, question: str, similarities: np.ndarray) -> np.ndarray:
        """Apply semantic boosting based on question type and chunk metadata"""
        if not self.chunk_metadata or len(self.chunk_metadata) != len(similarities):
            return similarities

        question_lower = question.lower()
        boosted_similarities = similarities.copy()

        # Boost factors for different scenarios
        for idx, metadata in enumerate(self.chunk_metadata):
            chunk_type = metadata['type']
            boost_factor = 1.0

            # Boost statistical summaries for analysis questions
            if chunk_type == 'statistical_summary':
                if any(word in question_lower for word in [
                    'average', 'mean', 'max', 'min', 'statistics', 'analyze'
                ]):
                    boost_factor = 1.3

            # Boost column descriptions for column-related questions
            elif chunk_type == 'column_description':
                if 'column' in metadata:
                    col_name = metadata['column'].lower()
                    if col_name in question_lower:
                        boost_factor = 1.5

            # Boost data records for specific value lookups
            elif chunk_type == 'data_record':
                if any(word in question_lower for word in [
                    'show', 'find', 'get', 'where', 'records'
                ]):
                    boost_factor = 1.2

            # Boost relationships for correlation questions
            elif chunk_type == 'column_relationship':
                if any(word in question_lower for word in [
                    'relationship', 'correlate', 'related', 'connection'
                ]):
                    boost_factor = 1.4

            # Boost categorical distributions for distribution questions
            elif chunk_type == 'categorical_distribution':
                if any(word in question_lower for word in [
                    'distribution', 'count', 'frequency', 'how many'
                ]):
                    boost_factor = 1.3

            # Apply boost
            boosted_similarities[idx] *= boost_factor

        return boosted_similarities

    def get_chunk_types_summary(self) -> Dict[str, int]:
        """Get summary of chunk types for debugging"""
        if not self.chunk_metadata:
            return {}

        summary = {}
        for metadata in self.chunk_metadata:
            chunk_type = metadata['type']
            summary[chunk_type] = summary.get(chunk_type, 0) + 1

        return summary

    def clear_cache(self):
        """Clear the vector cache"""
        self.vectorizer = None
        self.vectors = None
        self.chunk_list = None
        self.chunk_metadata = None
        self.column_relationships = None

        if self.debug_mode:
            print("Enhanced vector search cache cleared")
