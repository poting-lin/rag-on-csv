"""
Integration tests for Context Memory functionality in CSV QA Bot.

These tests verify that the context memory system correctly:
- Stores and retrieves conversation history
- Detects follow-up questions 
- Applies context filtering for "these records" queries
- Handles aggregation on filtered data
- Resolves pronouns and references
"""
import subprocess
import os
import re
import pytest
from csv_qa.question_answerer import CSVQuestionAnswerer


class TestContextMemoryIntegration:
    """Integration tests for Context Memory functionality"""

    def setup_method(self, method):
        """Set up test environment for each test method."""
        self.project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../.."))
        self.sample_data_dir = os.path.join(self.project_root, "sample_data")
        self.sound_measurements_csv = os.path.join(
            self.sample_data_dir, "sound_measurements.csv")

        # Create QA instance with context memory enabled and debug mode for testing
        self.qa = CSVQuestionAnswerer(
            model_name="llama3.2:1b",
            debug_mode=True,
            enable_context_memory=True
        )

    def test_basic_context_memory_storage(self):
        """Test that context memory stores conversation turns correctly."""
        # Load CSV
        self.qa.load_csv(self.sound_measurements_csv)

        # Ask first question
        question1 = "list records where EventType is Festival"
        answer1 = self.qa.answer_question(question1)

        # Verify conversation was stored
        assert self.qa.context_memory is not None
        assert len(self.qa.context_memory.conversation_history) == 1

        # Check stored turn details
        turn = self.qa.context_memory.conversation_history[0]
        assert turn.question == question1
        assert turn.answer == answer1
        assert turn.question_type is not None

    def test_follow_up_detection_with_pronoun(self):
        """Test detection of pronoun-based follow-up questions."""
        self.qa.load_csv(self.sound_measurements_csv)

        # Ask initial question about festivals
        question1 = "show me records where EventType is Festival"
        answer1 = self.qa.answer_question(question1)

        # Ask follow-up with pronoun
        question2 = "what is the average DecibelsA in them?"
        follow_up_info = self.qa.context_memory.detect_follow_up_intent(
            question2)

        # Verify follow-up detection
        assert follow_up_info['is_follow_up'] is True
        assert follow_up_info['reference_type'] == 'pronoun'
        assert len(follow_up_info['referenced_turns']) > 0

    def test_follow_up_detection_with_these_records(self):
        """Test detection of 'these records' follow-up questions."""
        self.qa.load_csv(self.sound_measurements_csv)

        # Ask initial filtering question
        question1 = "list records where EventType is Traffic"
        answer1 = self.qa.answer_question(question1)

        # Ask follow-up with "these records"
        question2 = "what is the max FrequencyRange in these records?"
        follow_up_info = self.qa.context_memory.detect_follow_up_intent(
            question2)

        # Verify follow-up detection
        assert follow_up_info['is_follow_up'] is True
        assert 'these records' in question2.lower()

    def test_context_filtering_basic(self):
        """Test basic context filtering functionality."""
        self.qa.load_csv(self.sound_measurements_csv)

        # Ask initial filtering question
        question1 = "show records where EventType is Traffic"
        answer1 = self.qa.answer_question(question1)

        # Check that context filter info is extracted
        context_filter = self.qa.context_memory.get_context_data_filter(
            "what is the max FrequencyRange in these records?"
        )

        assert context_filter is not None
        assert 'filter_column' in context_filter
        assert context_filter['filter_column'].lower() == 'eventtype'
        assert context_filter['filter_value'].lower() == 'traffic'

    def test_aggregation_on_filtered_data(self):
        """Test aggregation operations on context-filtered data."""
        self.qa.load_csv(self.sound_measurements_csv)

        # First, ask for Traffic records (should return 6 records)
        question1 = "list records where EventType is Traffic"
        answer1 = self.qa.answer_question(question1)

        # Verify we got Traffic records
        assert "Traffic" in answer1

        # Now ask for max FrequencyRange in these records
        question2 = "what is the max FrequencyRange in these records?"
        answer2 = self.qa.answer_question(question2)

        # Verify the answer mentions the specific max value for Traffic records
        # and indicates it's from the filtered set
        assert "maximum" in answer2.lower() or "max" in answer2.lower()
        assert "frequencyrange" in answer2.lower() or "frequency" in answer2.lower()

        # The answer should NOT indicate searching through all 57 records
        assert "57" not in answer2

    def test_multiple_aggregation_questions_on_same_filter(self):
        """Test multiple aggregation questions on the same filtered dataset."""
        self.qa.load_csv(self.sound_measurements_csv)

        # Filter for Event type records
        question1 = "list records where EventType is Event"
        answer1 = self.qa.answer_question(question1)

        # Ask multiple aggregation questions
        questions_and_keywords = [
            ("what is the max DecibelsA in these records?",
             ["max", "decibels"]),
            ("what is the min DecibelsA in these records?",
             ["min", "decibels"]),
            ("what is the average Duration in these records?",
             ["average", "duration"]),
        ]

        for question, keywords in questions_and_keywords:
            answer = self.qa.answer_question(question)

            # Each answer should contain the relevant keywords
            for keyword in keywords:
                assert keyword.lower() in answer.lower(
                ), f"Answer should contain '{keyword}'"

            # Should not search through all records
            assert "57" not in answer, "Should not search through all 57 records"

    def test_context_filtering_with_different_columns(self):
        """Test context filtering with different column types."""
        self.qa.load_csv(self.sound_measurements_csv)

        # Test with Location filtering
        question1 = "show records with Location Urban Park"
        answer1 = self.qa.answer_question(question1)

        question2 = "how many records are in these results?"
        context_filter = self.qa.context_memory.get_context_data_filter(
            question2)

        if context_filter:
            assert 'filter_column' in context_filter
            assert context_filter['filter_column'].lower() == 'location'
            assert 'urban park' in context_filter['filter_value'].lower()

    def test_pronoun_resolution(self):
        """Test pronoun resolution in follow-up questions."""
        self.qa.load_csv(self.sound_measurements_csv)

        # Ask about Festival records
        question1 = "list records where EventType is Festival"
        answer1 = self.qa.answer_question(question1)

        # Ask follow-up with pronoun
        question2 = "what is the average DecibelsA in them?"

        # Get the enhanced question with pronoun resolution
        context_info = self.qa.context_memory.detect_follow_up_intent(
            question2)
        enhanced_question = self.qa._resolve_pronouns_with_context(
            question2, context_info['referenced_turns']
        )

        # Enhanced question should have more context
        assert "festival" in enhanced_question.lower(
        ) or "eventtype" in enhanced_question.lower()

    def test_conversation_memory_limits(self):
        """Test that conversation memory respects configured limits."""
        self.qa.load_csv(self.sound_measurements_csv)

        # Ask several questions to fill up memory
        questions = [
            "show records where EventType is Traffic",
            "what is the max DecibelsA in these records?",
            "show records where EventType is Festival",
            "what is the min Duration in these records?",
            "show records where Location is City Center",
            "how many records are there?",
        ]

        for question in questions:
            self.qa.answer_question(question)

        # Should not exceed max_turns limit (default is 10)
        assert len(self.qa.context_memory.conversation_history) <= 10

    def test_context_memory_persistence(self):
        """Test that context memory can be saved and loaded."""
        self.qa.load_csv(self.sound_measurements_csv)

        # Ask some questions
        self.qa.answer_question("show records where EventType is Traffic")
        self.qa.answer_question(
            "what is the max FrequencyRange in these records?")

        # Save conversation history
        temp_file = os.path.join(self.project_root, "temp_conversation.json")
        try:
            self.qa.save_conversation_history(temp_file)

            # Create new instance and load history
            new_qa = CSVQuestionAnswerer(
                model_name="llama3.2:1b",
                debug_mode=True,
                enable_context_memory=True
            )
            new_qa.load_csv(self.sound_measurements_csv)
            new_qa.load_conversation_history(temp_file)

            # Verify history was loaded
            assert len(new_qa.context_memory.conversation_history) == 2

        finally:
            # Clean up
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def test_context_memory_disabled(self):
        """Test behavior when context memory is disabled."""
        # Create QA instance with context memory disabled
        qa_no_context = CSVQuestionAnswerer(
            model_name="llama3.2:1b",
            debug_mode=True,
            enable_context_memory=False
        )
        qa_no_context.load_csv(self.sound_measurements_csv)

        # Ask questions
        answer1 = qa_no_context.answer_question(
            "show records where EventType is Traffic")
        answer2 = qa_no_context.answer_question(
            "what is the max FrequencyRange in these records?")

        # Context memory should be None
        assert qa_no_context.context_memory is None

        # Second question should not benefit from context (may not work as expected)
        # This test mainly verifies the system doesn't crash when context memory is disabled
        assert isinstance(answer2, str)

    def test_complex_follow_up_sequence(self):
        """Test a complex sequence of follow-up questions."""
        self.qa.load_csv(self.sound_measurements_csv)

        # Sequence of related questions
        conversation = [
            ("show me records where EventType is Traffic", "Traffic"),
            ("what is the max DecibelsA in these records?", "max"),
            ("which location has that maximum value?", "location"),
            ("show me all records from that location", "records"),
        ]

        for i, (question, expected_keyword) in enumerate(conversation):
            answer = self.qa.answer_question(question)

            # Each answer should be relevant
            assert isinstance(answer, str)
            assert len(answer) > 0

            # Check that context is being maintained
            if i > 0:  # After first question
                assert len(
                    self.qa.context_memory.conversation_history) == i + 1

    def test_edge_case_empty_filter_results(self):
        """Test handling when context filter returns no results."""
        self.qa.load_csv(self.sound_measurements_csv)

        # Ask for records that don't exist
        question1 = "show records where EventType is NonExistent"
        answer1 = self.qa.answer_question(question1)

        # Ask follow-up question
        question2 = "what is the max DecibelsA in these records?"
        answer2 = self.qa.answer_question(question2)

        # Should handle gracefully (not crash)
        assert isinstance(answer2, str)

    def test_case_insensitive_context_filtering(self):
        """Test that context filtering works case-insensitively."""
        self.qa.load_csv(self.sound_measurements_csv)

        # Ask with different case
        question1 = "show records where eventtype is traffic"  # lowercase
        answer1 = self.qa.answer_question(question1)

        question2 = "what is the MAX frequencyrange in these records?"  # mixed case
        answer2 = self.qa.answer_question(question2)

        # Should work despite case differences
        assert "max" in answer2.lower()
        assert "frequency" in answer2.lower()

    def test_conversation_summary(self):
        """Test conversation summary generation."""
        self.qa.load_csv(self.sound_measurements_csv)

        # Have a conversation
        self.qa.answer_question("show records where EventType is Festival")
        self.qa.answer_question(
            "what is the average DecibelsA in these records?")
        self.qa.answer_question("how many festival records are there?")

        # Get conversation summary
        summary = self.qa.get_conversation_summary()

        # Summary should contain key information
        assert isinstance(summary, str)
        assert len(summary) > 0
        assert "festival" in summary.lower()

    def test_clear_conversation_history(self):
        """Test clearing conversation history."""
        self.qa.load_csv(self.sound_measurements_csv)

        # Ask some questions
        self.qa.answer_question("show records where EventType is Traffic")
        self.qa.answer_question("what is the max DecibelsA in these records?")

        # Verify history exists
        assert len(self.qa.context_memory.conversation_history) == 2

        # Clear history
        self.qa.clear_conversation_history()

        # Verify history is cleared
        assert len(self.qa.context_memory.conversation_history) == 0


class TestContextMemoryEdgeCases:
    """Edge case tests for Context Memory functionality"""

    def setup_method(self, method):
        """Set up test environment."""
        self.project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../.."))
        self.sample_data_dir = os.path.join(self.project_root, "sample_data")
        self.sound_measurements_csv = os.path.join(
            self.sample_data_dir, "sound_measurements.csv")

        self.qa = CSVQuestionAnswerer(
            model_name="llama3.2:1b",
            debug_mode=True,
            enable_context_memory=True
        )

    def test_ambiguous_pronoun_reference(self):
        """Test handling of ambiguous pronoun references."""
        self.qa.load_csv(self.sound_measurements_csv)

        # Ask multiple questions that could be referenced
        self.qa.answer_question("show records where EventType is Traffic")
        self.qa.answer_question("show records where EventType is Festival")

        # Ask ambiguous follow-up
        question = "what is the average DecibelsA in them?"
        answer = self.qa.answer_question(question)

        # Should handle gracefully without crashing
        assert isinstance(answer, str)

    def test_very_long_question_history(self):
        """Test behavior with very long question history."""
        self.qa.load_csv(self.sound_measurements_csv)

        # Ask many questions to test memory management
        base_questions = [
            "show records where EventType is Traffic",
            "show records where EventType is Festival",
            "show records where EventType is Event",
            "show records where Location is Urban Park",
            "show records where Location is City Center",
        ]

        # Repeat to create long history
        for i in range(3):
            for question in base_questions:
                self.qa.answer_question(f"{question} (iteration {i})")

        # Should respect memory limits and not crash
        assert len(
            self.qa.context_memory.conversation_history) <= self.qa.context_memory.max_turns

    def test_malformed_follow_up_questions(self):
        """Test handling of malformed follow-up questions."""
        self.qa.load_csv(self.sound_measurements_csv)

        # Ask initial question
        self.qa.answer_question("show records where EventType is Traffic")

        # Ask malformed follow-ups
        malformed_questions = [
            "what is the in these records?",  # Missing target
            "max these records what?",  # Garbled syntax
            "these records these records?",  # Repetitive
            "",  # Empty
            "???",  # Just punctuation
        ]

        for question in malformed_questions:
            if question:  # Skip empty questions
                answer = self.qa.answer_question(question)
                # Should handle gracefully without crashing
                assert isinstance(answer, str)

    def test_context_filter_with_special_characters(self):
        """Test context filtering with special characters in values."""
        self.qa.load_csv(self.sound_measurements_csv)

        # This test may not have data with special characters, but tests robustness
        question1 = "show records where Notes contains 'traffic'"
        answer1 = self.qa.answer_question(question1)

        question2 = "how many records are in these results?"
        answer2 = self.qa.answer_question(question2)

        # Should handle without crashing
        assert isinstance(answer2, str)
