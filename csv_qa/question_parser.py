"""
Question Parser module for extracting information from user questions
"""
import re
from difflib import get_close_matches


class QuestionParser:
    """
    Parses user questions to extract target column, ID column, and ID value
    """

    def __init__(self, csv_columns, debug_mode=False):
        """Initialize the question parser with CSV column names"""
        self.csv_columns = csv_columns
        self.debug_mode = debug_mode

        # Define known commands for spell checking
        self.known_commands = [
            "summarize", "list", "count", "filter", "show", "find",
            "get", "display", "aggregate", "average", "mean", "sum", "min", "max"
        ]

        # Help query patterns
        self.help_patterns = [
            r"what (?:kind of |types of )?questions can I ask",
            r"what can I ask",
            r"help me",
            r"show me example questions",
            r"give me examples",
            r"how to use"
        ]

    def is_relevant_question(self, question: str) -> bool:
        """Check if the question is relevant to the CSV data"""
        # For now, assume all questions are relevant
        # In a real application, you might want to filter out irrelevant questions
        return True

    def extract_question_info(self, question: str):
        """
        Extract target column, ID column, and ID value from the question

        Args:
            question: Question to extract info from

        Returns:
            dict: A dictionary containing extracted information with keys:
                target_column: Column to get value from
                id_column: Column to use for identification
                id_value: Value to search for in id_column
                comparison_info: Information about numerical comparison if any
                command: Command type if detected
                is_help_query: Boolean indicating if this is a help query
                command_correction: Suggested correction for command typos
        """
        if not question:
            return {}

        question = question.lower()
        target_column = None
        id_column = None
        id_value = None
        comparison_info = None
        command_info = None

        # Check for help queries first
        for pattern in self.help_patterns:
            if re.search(pattern, question.lower()):
                if self.debug_mode:
                    print(f"Detected help query: {question}")
                return {
                    'command': 'help',
                    'operation': 'suggest_questions',
                    'is_help_query': True
                }

        # Check for special commands like summarize, list, count
        command_patterns = [
            r"summarize",
            r"list",
            r"count"
        ]

        # First check for exact matches
        for pattern in command_patterns:
            match = re.search(pattern, question)
            if match:
                command_info = {
                    'command': pattern
                }
                break

        # If no exact match, check for possible typos
        if not command_info:
            # Extract words from the question
            words = question.lower().split()

            # Check each word against known commands for possible typos
            for word in words:
                # Skip very short words
                if len(word) < 4:
                    continue

                # Find closest matches
                close_matches = get_close_matches(
                    word, self.known_commands, n=1, cutoff=0.75)

                if close_matches:
                    closest_match = close_matches[0]
                    if self.debug_mode:
                        print(
                            f"Detected possible typo: '{word}' -> '{closest_match}'")

                    # Check if the closest match is one of our command patterns
                    if closest_match in [p.strip() for p in command_patterns] or closest_match in ["summarize", "list", "count"]:
                        if self.debug_mode:
                            print(
                                f"Detected command typo: '{word}' -> '{closest_match}'")
                        command_info = {
                            'command': closest_match,
                            'original': word,
                            'auto_correct': True  # Mark for auto-correction without asking
                        }
                        return {'command_correction': closest_match}
                        break

        # Check for numerical comparison patterns like "price above 10" or "books that price > 11"
        for col in self.csv_columns:
            col_lower = col.lower()

            # Pattern for "column above/below/greater than/less than value"
            comparison_patterns = [
                # Above patterns
                rf"(?:.*\s)?{col_lower}\s+(?:above|over|greater than|more than|>|>=|higher than)\s+(\d+(?:\.\d+)?)",
                rf"(?:.*\s)?{col_lower}\s+(?:is|are)\s+(?:above|over|greater than|more than|>|>=|higher than)\s+(\d+(?:\.\d+)?)",

                # Below patterns
                rf"(?:.*\s)?{col_lower}\s+(?:below|under|less than|<|<=|lower than)\s+(\d+(?:\.\d+)?)",
                rf"(?:.*\s)?{col_lower}\s+(?:is|are)\s+(?:below|under|less than|<|<=|lower than)\s+(\d+(?:\.\d+)?)",

                # Equal patterns
                rf"(?:.*\s)?{col_lower}\s+(?:equal to|equals|=|==)\s+(\d+(?:\.\d+)?)",
                rf"(?:.*\s)?{col_lower}\s+(?:is|are)\s+(\d+(?:\.\d+)?)"
            ]

            for i, pattern in enumerate(comparison_patterns):
                match = re.search(pattern, question)
                if match:
                    value = float(match.group(1))

                    # Determine the operator based on the pattern index
                    if i < 2:
                        operator = ">"
                    elif i < 4:
                        operator = "<"
                    else:
                        operator = "="

                    comparison_info = {
                        'column': col,
                        'operator': operator,
                        'value': value
                    }

                    # If we find a column name in the beginning of the question, it might be the target
                    words = question.split()
                    if words and any(col_name.lower() in words[0] for col_name in self.csv_columns):
                        target_column = col

                    break

            if comparison_info:
                break

        # Try to find target column (what the user is asking about)
        for col in self.csv_columns:
            col_lower = col.lower()

            # Check for phrases like "what is the <column>" or "tell me the <column>"
            target_patterns = [
                rf"what(?:'s| is| are) (?:the |)({col_lower})",
                rf"(?:tell|show|give) me (?:the |)({col_lower})",
                rf"(?:find|get|retrieve) (?:the |)({col_lower})"
            ]

            for pattern in target_patterns:
                match = re.search(pattern, question)
                if match:
                    target_column = col
                    break

            if target_column:
                break

        # If no target column found, check if any column is mentioned
        if not target_column:
            for col in self.csv_columns:
                col_lower = col.lower()
                if col_lower in question:
                    # The column is mentioned but not as a target
                    # It might be used as an identifier
                    id_column = col

                    # Try to find the value for this column in the question
                    # This is a simplified approach - in a real app, you'd use NER or more sophisticated techniques
                    words = question.split()
                    col_index = -1

                    # Find the position of the column name in the question
                    for i, word in enumerate(words):
                        if col_lower in word:
                            col_index = i
                            break

                    # Look for the value after the column name
                    if col_index >= 0 and col_index < len(words) - 1:
                        # Simple heuristic: take the next word as the value
                        # In a real app, you'd use more sophisticated techniques
                        potential_value = words[col_index + 1]

                        # Remove any punctuation
                        potential_value = re.sub(
                            r'[^\w\s]', '', potential_value)

                        if potential_value:
                            id_value = potential_value
                            break

        # Check for direct column values like "lab 1" or "researcher John"
        if not target_column and not id_column:
            for col in self.csv_columns:
                col_lower = col.lower()
                # Check if the question starts with a column name
                if question.strip().lower().startswith(col_lower):
                    id_column = col
                    # Extract the value after the column name
                    value_part = question.strip()[len(col_lower):].strip()
                    if value_part:
                        id_value = value_part
                        break

        # If we still don't have an ID value, check if the question contains just a simple ID or value
        if not id_value and len(question.strip().split()) <= 2:
            # The question might be just an ID or value
            id_value = question.strip()

            # Try to determine which column this value might belong to
            # For now, we'll use the first column as the default ID column if none was found
            if not id_column and self.csv_columns:
                id_column = self.csv_columns[0]

        # Enhance command detection with more context
        if command_info:
            # Check for column names after the command
            words = question.split()
            command_word_index = -1

            for i, word in enumerate(words):
                if command_info['command'] in word:
                    command_word_index = i
                    break

            if command_word_index >= 0 and command_word_index < len(words) - 1:
                # Look for column names after the command word
                command_columns = []
                for col in self.csv_columns:
                    col_lower = col.lower()
                    for i in range(command_word_index + 1, len(words)):
                        if col_lower in words[i]:
                            command_columns.append(col)

                if command_columns:
                    command_info['columns'] = command_columns

            # Set operation type based on command
            if command_info['command'] == 'summarize':
                command_info['operation'] = 'summarize'
            elif command_info['command'] == 'list':
                command_info['operation'] = 'list'
            elif command_info['command'] == 'count':
                command_info['operation'] = 'count'

        # Build the result dictionary
        result = {}
        if target_column:
            result['target_column'] = target_column
        if id_column:
            result['id_column'] = id_column
        if id_value:
            result['id_value'] = id_value
        if comparison_info:
            result['comparison_info'] = comparison_info
        if command_info:
            # Merge command_info dictionary into the result
            result.update(command_info)

        if self.debug_mode:
            print(f"Extracted from question: {result}")

        return result
