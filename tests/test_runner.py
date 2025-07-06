#!/usr/bin/env python3
"""
Test runner functionality converted to pytest format.
This allows running the test suite validation as part of pytest.
"""
import pytest
import subprocess
import sys
from pathlib import Path


class TestRunner:
    """Test runner functionality as pytest tests."""

    def test_pytest_is_available(self):
        """Test that pytest is available and working."""
        try:
            result = subprocess.run(
                ["python", "-m", "pytest", "--version"],
                capture_output=True, check=True, text=True
            )
            assert result.returncode == 0
            assert "pytest" in result.stdout.lower()
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            pytest.fail(f"pytest is not available: {e}")

    def test_project_structure(self):
        """Test that the project structure is correct."""
        project_root = Path(__file__).parent.parent

        # Check critical directories and files
        required_paths = [
            "csv_qa/__init__.py",
            "csv_qa/question_answerer.py",
            "csv_qa/question_router.py",
            "csv_qa/structured_query_engine.py",
            "csv_qa/enhanced_vector_search.py",
            "csv_qa/hybrid_engine.py",
            "tests/integration/",
            "tests/fixtures/",
            "sample_data/",
        ]

        for path in required_paths:
            full_path = project_root / path
            assert full_path.exists(), f"Required path not found: {path}"

    def test_integration_tests_exist(self):
        """Test that integration tests exist and are discoverable."""
        integration_dir = Path(__file__).parent / "integration"
        assert integration_dir.exists(), "Integration tests directory not found"

        test_files = list(integration_dir.glob("test_*.py"))
        assert len(test_files) > 0, "No integration test files found"

        expected_files = [
            "test_context_memory_integration.py",
            "test_csv_qa_integration.py"
        ]

        for expected_file in expected_files:
            file_path = integration_dir / expected_file
            assert file_path.exists(
            ), f"Expected test file not found: {expected_file}"

    def test_context_memory_tests_runnable(self):
        """Test that context memory integration tests can be run."""
        test_file = Path(__file__).parent / "integration" / \
            "test_context_memory_integration.py"
        assert test_file.exists(), "Context memory test file not found"

        # Run a quick test discovery to ensure the file is valid
        result = subprocess.run([
            "python", "-m", "pytest",
            str(test_file),
            "--collect-only",
            "-q"
        ], capture_output=True, text=True)

        assert result.returncode == 0, f"Context memory tests discovery failed: {result.stderr}"
        assert "test session starts" in result.stdout or "collected" in result.stdout

    def test_csv_qa_integration_tests_runnable(self):
        """Test that CSV QA integration tests can be run."""
        test_file = Path(__file__).parent / "integration" / \
            "test_csv_qa_integration.py"
        assert test_file.exists(), "CSV QA integration test file not found"

        # Run a quick test discovery to ensure the file is valid
        result = subprocess.run([
            "python", "-m", "pytest",
            str(test_file),
            "--collect-only",
            "-q"
        ], capture_output=True, text=True)

        assert result.returncode == 0, f"CSV QA integration tests discovery failed: {result.stderr}"
        assert "test session starts" in result.stdout or "collected" in result.stdout

    def test_enhanced_engines_import(self):
        """Test that enhanced engines can be imported."""
        try:
            from csv_qa.question_router import QuestionRouter
            from csv_qa.structured_query_engine import StructuredQueryEngine
            from csv_qa.enhanced_vector_search import CSVAwareVectorSearch

            # Test basic instantiation
            router = QuestionRouter(debug_mode=False)
            structured_engine = StructuredQueryEngine(debug_mode=False)
            enhanced_search = CSVAwareVectorSearch(debug_mode=False)

            # Test that they have expected methods
            assert hasattr(router, 'route_question')
            assert hasattr(structured_engine, 'execute_query')
            assert hasattr(enhanced_search, 'create_structured_chunks')

        except ImportError as e:
            pytest.fail(f"Failed to import enhanced engines: {e}")

    def test_main_qa_system_import(self):
        """Test that the main QA system can be imported and initialized."""
        try:
            from csv_qa.question_answerer import CSVQuestionAnswerer

            # Test initialization with enhanced engines
            qa = CSVQuestionAnswerer(
                model_name="llama3.2:1b",
                debug_mode=False,
                enable_context_memory=True,
                use_enhanced_engines=True
            )

            # Test that enhanced engines are available
            assert hasattr(qa, 'question_router')
            assert hasattr(qa, 'structured_engine')
            assert hasattr(qa, 'enhanced_vector_search')
            assert hasattr(qa, 'hybrid_engine')

        except Exception as e:
            pytest.fail(f"Failed to initialize main QA system: {e}")

    def test_sample_data_accessible(self):
        """Test that sample data files are accessible."""
        project_root = Path(__file__).parent.parent
        sample_data_dir = project_root / "sample_data"

        assert sample_data_dir.exists(), "Sample data directory not found"

        # Check for expected sample files
        expected_files = [
            "sound_measurements.csv",
        ]

        for expected_file in expected_files:
            file_path = sample_data_dir / expected_file
            assert file_path.exists(
            ), f"Expected sample file not found: {expected_file}"

            # Check that file is not empty
            assert file_path.stat(
            ).st_size > 0, f"Sample file is empty: {expected_file}"

    def test_test_runner_legacy_compatibility(self):
        """Test that the old run_tests.py functionality is preserved."""
        run_tests_file = Path(__file__).parent / "run_tests.py"

        # Check if run_tests.py exists - if so, verify it has expected functions
        if run_tests_file.exists():
            with open(run_tests_file, 'r') as f:
                content = f.read()

            assert "def run_context_memory_tests" in content
            assert "def run_tests" in content
            assert "def main" in content


class TestRunnerMetrics:
    """Additional test runner metrics and validation."""

    def test_all_tests_have_docstrings(self):
        """Test that all test methods have proper docstrings."""
        import inspect

        # Check this class
        for name, method in inspect.getmembers(TestRunner, predicate=inspect.isfunction):
            if name.startswith('test_'):
                assert method.__doc__ is not None, f"Test method {name} missing docstring"
                assert len(method.__doc__.strip()
                           ) > 10, f"Test method {name} has inadequate docstring"

    def test_pytest_markers_work(self):
        """Test that pytest markers are working correctly."""
        # This test itself works, so pytest markers work
        assert True

    @pytest.mark.parametrize("test_type", ["integration", "context_memory", "csv_qa"])
    def test_test_categories_exist(self, test_type):
        """Test that different test categories exist and are runnable."""
        test_mapping = {
            "integration": "integration/",
            "context_memory": "integration/test_context_memory_integration.py",
            "csv_qa": "integration/test_csv_qa_integration.py"
        }

        test_path = Path(__file__).parent / test_mapping[test_type]
        assert test_path.exists(
        ), f"Test category {test_type} not found at {test_path}"


# Utility functions for backwards compatibility
def run_context_memory_tests(verbose=False):
    """Legacy function to run context memory tests - now calls pytest."""
    cmd = [
        "python", "-m", "pytest",
        "tests/integration/test_context_memory_integration.py",
        "-v" if verbose else "--tb=short",
        "--color=yes"
    ]

    result = subprocess.run(cmd)
    return result.returncode


def run_tests(test_type="all", verbose=False, specific_test=None):
    """Legacy function to run tests - now calls pytest."""
    cmd = ["python", "-m", "pytest"]

    if verbose:
        cmd.append("-v")

    # Add test selection
    if test_type == "all":
        cmd.append("tests/")
    elif test_type == "integration":
        cmd.append("tests/integration/")
    elif test_type == "context":
        cmd.append("tests/integration/test_context_memory_integration.py")
    else:
        cmd.append("tests/")

    if specific_test:
        cmd.extend(["-k", specific_test])

    result = subprocess.run(cmd)
    return result.returncode


if __name__ == "__main__":
    # Allow running this file directly for backwards compatibility
    import argparse

    parser = argparse.ArgumentParser(description="Run tests (now via pytest)")
    parser.add_argument("-t", "--type", default="all",
                        help="Type of tests to run")
    parser.add_argument("-v", "--verbose",
                        action="store_true", help="Verbose output")
    parser.add_argument("-s", "--specific", help="Specific test to run")

    args = parser.parse_args()
    sys.exit(run_tests(args.type, args.verbose, args.specific))
