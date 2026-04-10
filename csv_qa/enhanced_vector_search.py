"""
Enhanced Vector Search Engine - specifically designed for CSV data analysis.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .exceptions import VectorSearchError

logger = logging.getLogger(__name__)


class CSVAwareVectorSearch:
    """Enhanced vector search specifically designed for CSV data."""

    def __init__(self) -> None:
        """Initialize the enhanced vector search engine."""
        self.vectorizer: TfidfVectorizer | None = None
        self.vectors = None
        self.chunk_list: list[str] | None = None
        self.chunk_metadata: list[dict] | None = None
        self.column_relationships = None

    def create_structured_chunks(self, df: pd.DataFrame) -> list[str]:
        """Create more meaningful chunks for CSV data."""
        chunks: list[str] = []
        metadata: list[dict] = []

        logger.debug(
            "Creating structured chunks for DataFrame with %d rows and %d columns",
            len(df),
            len(df.columns),
        )

        # 1. Column-specific chunks
        for col in df.columns:
            col_info = self._analyze_column(df[col])
            chunk = f"Column {col}: {col_info['description']}"
            chunks.append(chunk)
            metadata.append(
                {
                    "type": "column_description",
                    "column": col,
                    "data_type": col_info["data_type"],
                }
            )

        # 2. Row-based chunks
        for idx, row in df.iterrows():
            row_desc = self._create_row_description(row, df.columns)
            chunks.append(f"Record {idx}: {row_desc}")
            metadata.append(
                {
                    "type": "data_record",
                    "row_index": idx,
                    "columns": df.columns.tolist(),
                }
            )
            if idx >= 100:
                break

        # 3. Statistical summary chunks
        numeric_cols = df.select_dtypes(include=["number"]).columns
        for col in numeric_cols:
            stats = df[col].describe()
            chunk = (
                f"{col} statistics: "
                f"min={stats['min']:.2f}, max={stats['max']:.2f}, "
                f"mean={stats['mean']:.2f}, std={stats['std']:.2f}, "
                f"median={stats['50%']:.2f}"
            )
            chunks.append(chunk)
            metadata.append(
                {
                    "type": "statistical_summary",
                    "column": col,
                    "stats": stats.to_dict(),
                }
            )

        # 4. Categorical distribution chunks
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns
        for col in categorical_cols:
            if df[col].nunique() <= 20:
                top_values = df[col].value_counts().head(10)
                chunk = f"{col} distribution: " + ", ".join([f"{val}({count})" for val, count in top_values.items()])
                chunks.append(chunk)
                metadata.append(
                    {
                        "type": "categorical_distribution",
                        "column": col,
                        "distribution": top_values.to_dict(),
                    }
                )

        # 5. Column relationship chunks
        relationships = self._analyze_column_relationships(df)
        for relationship in relationships:
            chunks.append(relationship)
            metadata.append({"type": "column_relationship", "relationship": relationship})

        # 6. Data quality chunks
        quality_info = self._analyze_data_quality(df)
        for info in quality_info:
            chunks.append(info)
            metadata.append({"type": "data_quality", "info": info})

        self.chunk_metadata = metadata

        logger.debug("Created %d structured chunks", len(chunks))
        logger.debug("Chunk types: %s", {m["type"] for m in metadata})

        return chunks

    def _analyze_column(self, series: pd.Series) -> dict[str, Any]:
        """Analyze a column to create meaningful descriptions."""
        unique_count = series.nunique()
        null_count = series.isnull().sum()
        total_count = len(series)

        if pd.api.types.is_numeric_dtype(series):
            stats = series.describe()
            description = (
                f"Numeric column with {unique_count} unique values, "
                f"range {stats['min']:.2f} to {stats['max']:.2f}, "
                f"average {stats['mean']:.2f}"
            )

            q1, q3 = stats["25%"], stats["75%"]
            iqr = q3 - q1
            outliers = series[(series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)]
            if len(outliers) > 0:
                description += f", {len(outliers)} potential outliers detected"

            return {"description": description, "data_type": "numeric", "stats": stats.to_dict()}

        elif pd.api.types.is_datetime64_any_dtype(series):
            date_range = f"{series.min()} to {series.max()}"
            description = f"Date/time column spanning {date_range}"
            return {"description": description, "data_type": "datetime", "range": date_range}

        else:
            sample_values = series.dropna().unique()[:5]
            sample_str = ", ".join([str(val) for val in sample_values])

            description = (
                f"Categorical column with {unique_count} unique values "
                f"({total_count - null_count} non-null), "
                f"examples: {sample_str}"
            )

            if len(sample_values) < unique_count:
                description += f" and {unique_count - len(sample_values)} more"

            return {
                "description": description,
                "data_type": "categorical",
                "unique_count": unique_count,
                "sample_values": sample_values.tolist(),
            }

    def _create_row_description(self, row: pd.Series, columns) -> str:
        """Create natural language description of a row."""
        descriptions: list[str] = []
        important_cols = [col for col in columns if len(col) <= 15][:5]

        for col in important_cols:
            val = row[col]
            if pd.notna(val):
                descriptions.append(f"{col} is {val}")

        if len(descriptions) < 3:
            for col in columns:
                if col not in important_cols:
                    val = row[col]
                    if pd.notna(val):
                        descriptions.append(f"{col} is {val}")
                    if len(descriptions) >= 5:
                        break

        return ", ".join(descriptions)

    def _analyze_column_relationships(self, df: pd.DataFrame) -> list[str]:
        """Find correlations and relationships between columns."""
        relationships: list[str] = []

        numeric_cols = df.select_dtypes(include=["number"]).columns
        if len(numeric_cols) > 1:
            correlations = df[numeric_cols].corr()

            for i, col1 in enumerate(numeric_cols):
                for j, col2 in enumerate(numeric_cols[i + 1 :], i + 1):
                    corr = correlations.iloc[i, j]
                    if abs(corr) > 0.5:
                        rel_type = "positively correlated" if corr > 0 else "negatively correlated"
                        relationships.append(f"Columns {col1} and {col2} are {rel_type} with correlation {corr:.2f}")

        categorical_cols = df.select_dtypes(include=["object", "category"]).columns
        if len(categorical_cols) > 1:
            for i, col1 in enumerate(categorical_cols):
                for col2 in categorical_cols[i + 1 :]:
                    if df[col1].nunique() <= 10 and df[col2].nunique() <= 10:
                        crosstab = pd.crosstab(df[col1], df[col2])
                        if crosstab.max().max() > len(df) * 0.3:
                            relationships.append(f"Columns {col1} and {col2} show co-occurrence patterns")

        return relationships[:5]

    def _analyze_data_quality(self, df: pd.DataFrame) -> list[str]:
        """Analyze data quality and patterns."""
        quality_info: list[str] = []

        missing_cols = df.columns[df.isnull().any()].tolist()
        if missing_cols:
            for col in missing_cols[:3]:
                missing_count = df[col].isnull().sum()
                missing_pct = (missing_count / len(df)) * 100
                quality_info.append(f"Column {col} has {missing_count} missing values ({missing_pct:.1f}%)")

        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            quality_info.append(f"Dataset contains {duplicate_count} duplicate rows")

        for col in df.columns:
            if df[col].dtype == "object":
                sample = df[col].dropna().head(100)
                types = set(type(val).__name__ for val in sample)
                if len(types) > 1:
                    quality_info.append(f"Column {col} contains mixed data types: {', '.join(types)}")

        return quality_info[:5]

    def build_vector_index(self, chunks: list[str]) -> None:
        """Build the vector index with enhanced features.

        Raises:
            VectorSearchError: If vector index cannot be built.
        """
        if not chunks:
            logger.debug("No chunks provided for vector index")
            return

        try:
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words="english",
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.8,
                lowercase=True,
                token_pattern=r"\b\w+\b",
            )

            self.vectors = self.vectorizer.fit_transform(chunks)
            self.chunk_list = chunks

            logger.debug("Built vector index with %d chunks", len(chunks))
            logger.debug("Vector shape: %s", self.vectors.shape)
            logger.debug("Vocabulary size: %d", len(self.vectorizer.vocabulary_))

        except Exception as e:
            logger.error("Error building vector index: %s", e, exc_info=True)
            raise VectorSearchError(detail=str(e)) from e

    def retrieve_context(self, question: str, top_k: int = 5) -> str:
        """Retrieve the most relevant context with enhanced scoring.

        Raises:
            VectorSearchError: If retrieval fails unexpectedly.
        """
        if self.vectorizer is None or self.vectors is None or self.chunk_list is None:
            raise ValueError("Vector index not built. Call build_vector_index first.")

        try:
            question_vector = self.vectorizer.transform([question])
            similarities = cosine_similarity(question_vector, self.vectors)[0]

            if self.chunk_metadata:
                similarities = self._apply_semantic_boosting(question, similarities)

            top_indices = np.argsort(similarities)[-top_k:][::-1]
            filtered_indices = [idx for idx in top_indices if similarities[idx] > 0.1]

            if not filtered_indices:
                logger.debug("No relevant chunks found with sufficient similarity")
                return ""

            context_chunks: list[str] = []
            for idx in filtered_indices:
                chunk = self.chunk_list[idx]
                score = similarities[idx]

                if self.chunk_metadata:
                    chunk_type = self.chunk_metadata[idx]["type"]
                    logger.debug(
                        "Selected chunk (score: %.3f, type: %s): %s...",
                        score,
                        chunk_type,
                        chunk[:100],
                    )

                context_chunks.append(chunk)

            context = "\n\n".join(context_chunks)

            logger.debug("Retrieved %d chunks as context", len(context_chunks))
            logger.debug(
                "Top similarity scores: %s",
                [similarities[i] for i in filtered_indices],
            )

            return context

        except Exception as e:
            logger.error("Error in retrieve_context: %s", e, exc_info=True)
            raise VectorSearchError(detail=str(e)) from e

    def _apply_semantic_boosting(self, question: str, similarities: np.ndarray) -> np.ndarray:
        """Apply semantic boosting based on question type and chunk metadata."""
        if not self.chunk_metadata or len(self.chunk_metadata) != len(similarities):
            return similarities

        question_lower = question.lower()
        boosted_similarities = similarities.copy()

        for idx, metadata in enumerate(self.chunk_metadata):
            chunk_type = metadata["type"]
            boost_factor = 1.0

            if chunk_type == "statistical_summary":
                if any(word in question_lower for word in ["average", "mean", "max", "min", "statistics", "analyze"]):
                    boost_factor = 1.3

            elif chunk_type == "column_description":
                if "column" in metadata:
                    col_name = metadata["column"].lower()
                    if col_name in question_lower:
                        boost_factor = 1.5

            elif chunk_type == "data_record":
                if any(word in question_lower for word in ["show", "find", "get", "where", "records"]):
                    boost_factor = 1.2

            elif chunk_type == "column_relationship":
                if any(word in question_lower for word in ["relationship", "correlate", "related", "connection"]):
                    boost_factor = 1.4

            elif chunk_type == "categorical_distribution":
                if any(word in question_lower for word in ["distribution", "count", "frequency", "how many"]):
                    boost_factor = 1.3

            boosted_similarities[idx] *= boost_factor

        return boosted_similarities

    def get_chunk_types_summary(self) -> dict[str, int]:
        """Get summary of chunk types for debugging."""
        if not self.chunk_metadata:
            return {}

        summary: dict[str, int] = {}
        for metadata in self.chunk_metadata:
            chunk_type = metadata["type"]
            summary[chunk_type] = summary.get(chunk_type, 0) + 1

        return summary

    def clear_cache(self) -> None:
        """Clear the vector cache."""
        self.vectorizer = None
        self.vectors = None
        self.chunk_list = None
        self.chunk_metadata = None
        self.column_relationships = None

        logger.debug("Enhanced vector search cache cleared")
