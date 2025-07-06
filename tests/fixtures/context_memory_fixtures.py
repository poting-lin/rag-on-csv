"""
Fixtures and utilities for Context Memory integration tests.
"""
import pytest
from csv_qa.question_answerer import CSVQuestionAnswerer


@pytest.fixture
def context_memory_qa():
    """Create a CSV Question Answerer with context memory enabled for testing."""
    return CSVQuestionAnswerer(
        model_name="llama3.2:1b",
        debug_mode=True,
        enable_context_memory=True
    )


class ContextMemoryTestHelper:
    """Helper class for context memory integration tests."""

    @staticmethod
    def verify_aggregation_result(answer, aggregation_type, column_name=None, should_not_contain=None):
        """Verify aggregation operation results."""
        answer_lower = answer.lower()

        # Check for aggregation type
        aggregation_keywords = {
            'max': ['max', 'maximum', 'highest', 'largest'],
            'min': ['min', 'minimum', 'lowest', 'smallest'],
            'avg': ['avg', 'average', 'mean'],
            'sum': ['sum', 'total'],
            'count': ['count', 'number']
        }

        keywords = aggregation_keywords.get(
            aggregation_type, [aggregation_type])
        assert any(keyword in answer_lower for keyword in keywords), \
            f"Answer should contain aggregation keyword for '{aggregation_type}'"

        # Check for column name if provided
        if column_name:
            assert column_name.lower(
            ) in answer_lower, f"Answer should mention column '{column_name}'"

        # Check for strings that should not be present
        if should_not_contain:
            for unwanted in should_not_contain:
                assert unwanted not in answer, f"Answer should not contain '{unwanted}'"


@pytest.fixture
def context_test_helper():
    """Provide the ContextMemoryTestHelper instance."""
    return ContextMemoryTestHelper()
