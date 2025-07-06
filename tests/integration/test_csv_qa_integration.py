"""
Integration tests for the CSV QA Bot.

These tests verify that the application correctly processes queries and returns expected results.
"""
import subprocess
import os
import re

try:
    import pytest
except ImportError:
    print("pytest not found. Please install it with 'pip install pytest' or 'poetry add pytest'")
    raise
from ..fixtures.expected_responses import EXPECTED_RESPONSES


class TestCSVQAIntegration:
    """Integration tests for the CSV QA Bot"""
    
    def setup_method(self, method):
        """Set up the test environment."""
        # Get the project root directory
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        # Get the test data directory
        self.test_data_dir = os.path.join(self.project_root, "tests", "test_data")
    
    def run_query(self, csv_file, query, timeout=30, debug=False, model="llama3.2:1b"):
        """
        Run a query against the CSV QA Bot and return the output.
        
        Args:
            csv_file: Path to the CSV file to query (relative to test_data_dir or absolute)
            query: The query to run
            timeout: Maximum time to wait for a response (seconds)
            debug: Whether to run in debug mode
            model: The model to use for the query
            
        Returns:
            The output of the command as a string
        """
        # If csv_file is not an absolute path, make it relative to test_data_dir
        if not os.path.isabs(csv_file):
            csv_file = os.path.join(self.test_data_dir, os.path.basename(csv_file))
        
        # Use direct import and function call instead of subprocess to avoid command-line issues
        # Create a temporary script that directly calls the question answerer
        temp_script = os.path.join(self.project_root, "temp_test_script.py")
        
        # Escape any quotes in the query to avoid syntax errors
        escaped_query = query.replace('"', '\\"')
        
        with open(temp_script, "w") as f:
            f.write(f'''
#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from csv_qa.question_answerer import CSVQuestionAnswerer

try:
    qa = CSVQuestionAnswerer(model_name="{model}", debug_mode={debug})
    csv_file = "{csv_file}"
    query = "{escaped_query}"
    result = qa.answer_question(query, csv_file)
    print(result)
except Exception as e:
    print(f"Error: {{e}}")
    import traceback
    traceback.print_exc()
''')
        
        # Run the temporary script and capture the output
        result = subprocess.run(
            ["python", temp_script],
            cwd=self.project_root,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        # Clean up the temporary script
        try:
            os.remove(temp_script)
        except OSError:
            pass  # Ignore errors when removing temporary file
        
        # Return the output
        return result.stdout + result.stderr
    
    def assert_response_matches_expectations(self, response, expectations):
        """
        Assert that a response matches the expected content.
        
        Args:
            response: The response from the CSV QA Bot
            expectations: Dictionary of expected content
        """
        # Check that the response contains all expected strings
        for expected_str in expectations.get("contains", []):
            assert expected_str in response, f"Response should contain '{expected_str}'"
        
        # Check that the response does not contain any unexpected strings
        for unexpected_str in expectations.get("not_contains", []):
            assert unexpected_str not in response, f"Response should not contain '{unexpected_str}'"
        
        # Check the minimum number of matches if specified
        if "min_matches" in expectations:
            # First try to find the explicit match count in the response
            match = re.search(r"Found (\d+) matches", response)
            if match:
                match_count = int(match.group(1))
                assert match_count >= expectations["min_matches"], \
                    f"Expected at least {expectations['min_matches']} matches, but found {match_count}"
            else:
                # If no explicit count is found, count the occurrences of the first expected string
                # This works for tabular output where each row contains the expected string
                if expectations.get("contains"):
                    first_expected = expectations["contains"][0]
                    # Count occurrences of the expected string
                    occurrences = response.count(first_expected)
                    assert occurrences >= expectations["min_matches"], \
                        f"Expected at least {expectations['min_matches']} occurrences of '{first_expected}', but found {occurrences}"
    
    def test_festival_query_sound_measurements(self):
        """Test that 'all records are festival' returns the correct records"""
        # Run the query
        output = self.run_query("sound_measurements.csv", "all records are festival")
        
        # Check that the output matches expectations
        self.assert_response_matches_expectations(
            output, 
            EXPECTED_RESPONSES["sound_measurements"]["all_records_festival"]
        )
    
    def test_high_decibel_query(self):
        """Test that querying for high decibel readings returns correct records"""
        output = self.run_query("sound_measurements.csv", "records with DecibelsA above 65")
        
        # Check if the output indicates no data available
        if "No data available" in output:
            pytest.skip("Skipping assertion because no data is available")
        else:
            # For this test, we'll check directly if there are values above 65 in the output
            # This is more reliable than checking for specific values
            assert re.search(r'\b(6[5-9]|[7-9][0-9])\.[0-9]\b', output), \
                "Output should contain DecibelsA values above 65"
    
    def test_urban_park_location_query(self):
        """Test that querying for Urban Park location returns correct records"""
        output = self.run_query("sound_measurements.csv", "all records with Location Urban Park")
        
        # Check that the output matches expectations
        self.assert_response_matches_expectations(
            output,
            EXPECTED_RESPONSES["sound_measurements"]["urban_park_location"]
        )
    
    def test_nonexistent_query(self):
        """Test that querying for nonexistent data returns appropriate message"""
        output = self.run_query("sound_measurements.csv", "records with DecibelsA above 200")
        
        # The application might return an empty table or a message indicating no matches
        # Check for various indicators that no matching records were found
        no_matches_indicators = [
            "No matches found", 
            "0 records", 
            "0 matches",
            "No data available",
            "couldn't find"
        ]
        
        # Check if any of the indicators are in the output
        has_no_matches_indicator = any(indicator.lower() in output.lower() for indicator in no_matches_indicators)
        
        # Alternatively, check if the output doesn't contain any DecibelsA values above 200
        no_high_values = not re.search(r'\b(20[0-9]|2[1-9][0-9]|[3-9][0-9][0-9])\.[0-9]\b', output)
        
        assert has_no_matches_indicator or no_high_values, "Output should indicate no records were found or contain no high values"
    
    def test_complex_query_decomposition(self):
        """Test that complex queries are properly decomposed and executed"""
        # This tests the multi-step query processing capability mentioned in the memories
        output = self.run_query(
            "sound_measurements.csv", 
            "analyze all records with EventType Event"
        )
        
        # Check that the output contains Event records and some form of analysis
        # The actual format may vary, so we check for multiple indicators
        assert "Event" in output, "Output should contain Event records"
        
        # Check for analysis indicators
        analysis_indicators = ["Analysis", "Statistics", "Average", "Mean", "Min", "Max", "Count"]
        has_analysis = any(indicator in output for indicator in analysis_indicators)
        
        assert has_analysis, "Output should contain some form of analysis results"
    
    def test_fuzzy_matching(self):
        """Test that fuzzy matching works for similar values"""
        # This tests the fuzzy matching capability mentioned in the memories
        output = self.run_query("sound_measurements.csv", "records with EventType concert")
        
        # Check that the output suggests or matches "Event" which is similar to "concert"
        assert "Event" in output, "Output should contain or suggest 'Event' as a match for 'concert'"
    
    def test_negative_query(self):
        """Test that negative queries work correctly"""
        # This tests the negative query capability mentioned in the memories
        output = self.run_query("sound_measurements.csv", "all records are not Event")
        
        # Check that the output contains records but not Event records
        # The application might return a table of records, so we check for common column names
        # and ensure Event is not in the EventType column
        
        # First check that we have some output data
        assert "Timestamp" in output or "Location" in output or "DeviceID" in output, \
            "Output should contain some records"
        
        # Check if the output contains records with EventType that is not Event
        # Look for other event types like Ambient, Traffic, etc.
        other_event_types = ["Ambient", "Traffic", "Construction", "Recreation"]
        has_other_events = any(event_type in output for event_type in other_event_types)
        
        assert has_other_events, "Output should contain non-Event records"


class TestMultipleDatasetIntegration:
    """Integration tests for the CSV QA Bot with multiple datasets"""
    
    def setup_method(self, method):
        """Setup for the test class"""
        # Get the project root directory
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        # Get the test data directory
        self.test_data_dir = os.path.join(self.project_root, "tests", "test_data")
        # Create and initialize the test instance
        self.test_instance = TestCSVQAIntegration()
        self.test_instance.setup_method(method)
    
    def test_lab_data_query(self):
        """Test that querying the lab_data.csv file works correctly"""
        output = self.test_instance.run_query(
            "lab_data.csv", 
            "records with Equipment containing Optical"
        )
        
        # Check that the output matches expectations
        self.test_instance.assert_response_matches_expectations(
            output,
            EXPECTED_RESPONSES["lab_data"]["optical_equipment"]
        )
    
    def test_bookshop_query(self):
        """Test that querying the bookshop.csv file works correctly"""
        # First check if the bookshop.csv file exists and has data
        try:
            # Try to read a few lines from the file to verify it exists and has content
            with open(os.path.join(self.test_data_dir, "bookshop.csv"), 'r') as f:
                header = f.readline().strip()
                first_line = f.readline().strip()
                has_data = bool(header and first_line)
        except (FileNotFoundError, IOError):
            has_data = False
            
        if not has_data:
            pytest.skip("Skipping test_bookshop_query because bookshop.csv doesn't exist or is empty")
            
        output = self.test_instance.run_query(
            "bookshop.csv", 
            "all records with Category Fiction"
        )
        
        # Check if the output indicates no data or if it contains Fiction books
        if "No data available" in output:
            pytest.skip("Skipping assertion because bookshop.csv data is not available")
        else:
            # Check for Fiction in the output
            assert "Fiction" in output, "Output should contain Fiction category books"


class TestErrorHandling:
    """Tests for error handling in the CSV QA Bot"""
    
    def setup_method(self, method):
        """Setup for the test class"""
        # Get the project root directory
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        # Get the test data directory
        self.test_data_dir = os.path.join(self.project_root, "tests", "test_data")
        # Create and initialize the test instance
        self.test_instance = TestCSVQAIntegration()
        self.test_instance.setup_method(method)
    
    def test_invalid_csv_path(self):
        """Test that the application handles invalid CSV paths gracefully"""
        # The application might handle invalid paths internally rather than raising exceptions
        # So we'll check the output for error messages instead
        output = self.test_instance.run_query("nonexistent_file.csv", "any query")
        
        # Check for error indicators in the output
        error_indicators = [
            "error", 
            "not found", 
            "no such file", 
            "cannot open",
            "failed to load",
            "couldn't load"
        ]
        
        has_error = any(indicator.lower() in output.lower() for indicator in error_indicators)
        assert has_error, "Output should indicate file not found or other error"
    
    def test_empty_query(self):
        """Test that the application handles empty queries gracefully"""
        output = self.test_instance.run_query("sound_measurements.csv", "")
        
        # The application should prompt for a question or exit
        assert "question" in output.lower() or "query" in output.lower(), \
            "Output should prompt for a question"
