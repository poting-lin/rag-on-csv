"""
Question Router module - intelligently routes questions to appropriate engines
"""
import re
from typing import Dict, List


class QuestionRouter:
    """Routes questions to the most appropriate processing engine"""

    def __init__(self, debug_mode=False):
        self.debug_mode = debug_mode

    def route_question(self, question: str, csv_columns: List[str]) -> str:
        """
        Route questions to the most appropriate engine

        Args:
            question: The user's question
            csv_columns: List of available CSV columns

        Returns:
            str: Engine type ('structured', 'semantic', or 'hybrid')
        """
        if self.debug_mode:
            print(f"Routing question: {question}")

        # Route to structured engine for precise queries
        if self._is_structured_query(question, csv_columns):
            if self.debug_mode:
                print("Routed to: structured engine")
            return 'structured'

        # Route to semantic engine for exploratory questions
        elif self._is_semantic_query(question):
            if self.debug_mode:
                print("Routed to: semantic engine")
            return 'semantic'

        # Use hybrid for complex queries
        else:
            if self.debug_mode:
                print("Routed to: hybrid engine")
            return 'hybrid'

    def _is_structured_query(self, question: str, columns: List[str]) -> bool:
        """Check if question can be answered with direct DataFrame operations"""
        question_lower = question.lower()

        # Patterns that indicate structured queries
        structured_patterns = [
            # Exact filtering: "show records where EventType is Festival"
            r'(?:show|list|find|get)\s+records\s+where\s+(\w+)\s+(?:is|equals?|=)\s+(\w+)',

            # Aggregations: "what is the max DecibelsA?"
            r'(?:what is|show|get)\s+(?:the\s+)?(?:max|maximum|min|minimum|avg|average|mean|sum|total|count)\s+(\w+)',

            # Comparisons: "DecibelsA above 80"
            r'(\w+)\s+(?:above|below|over|under|greater|less|>|<|>=|<=)\s+(\d+)',

            # Direct column access: "show Location column"
            r'(?:show|display|get)\s+(\w+)\s+(?:column|values?)',

            # Count queries: "how many records", "count records"
            r'(?:how many|count)\s+records',

            # Simple lookups: "EventType values", "unique Location"
            r'(?:unique|distinct)\s+(\w+)',

            # Basic statistics: "average of DecibelsA"
            r'(?:average|mean|median)\s+(?:of\s+)?(\w+)'
        ]

        # Check pattern matches
        for pattern in structured_patterns:
            if re.search(pattern, question_lower):
                if self.debug_mode:
                    print(f"Matched structured pattern: {pattern}")
                return True

        # Check if question mentions specific columns and has simple structure
        mentioned_columns = [
            col for col in columns if col.lower() in question_lower]
        if mentioned_columns:
            # Simple queries with column mentions
            simple_indicators = [
                'show', 'display', 'get', 'find', 'list', 'what is',
                'max', 'min', 'average', 'count', 'sum', 'total'
            ]
            if any(indicator in question_lower for indicator in simple_indicators):
                if self.debug_mode:
                    print(
                        f"Mentioned columns: {mentioned_columns} with simple indicators")
                return True

        # Check for direct value lookups
        if any(word in question_lower for word in ['where', 'equals', 'is', '=']):
            if mentioned_columns:
                if self.debug_mode:
                    print("Direct value lookup detected")
                return True

        return False

    def _is_semantic_query(self, question: str) -> bool:
        """Check if question needs semantic understanding"""
        question_lower = question.lower()

        semantic_indicators = [
            # Exploratory analysis
            'unusual', 'abnormal', 'interesting', 'patterns', 'insights',
            'overview', 'summary', 'explain', 'describe', 'analyze',
            'correlations', 'relationships', 'trends', 'anomalies',

            # Complex analysis requests
            'distribution', 'outliers', 'statistical', 'statistics',
            'comparison', 'compare', 'differences', 'similarities',

            # Interpretive questions
            'why', 'how', 'what does', 'what can you tell me',
            'interpret', 'meaning', 'significance',

            # Open-ended exploration
            'explore', 'investigate', 'discover', 'reveal',
            'highlight', 'notable', 'remarkable'
        ]

        has_semantic_indicators = any(
            indicator in question_lower for indicator in semantic_indicators)

        if has_semantic_indicators:
            if self.debug_mode:
                matching_indicators = [
                    ind for ind in semantic_indicators if ind in question_lower]
                print(f"Semantic indicators found: {matching_indicators}")

        return has_semantic_indicators

    def get_query_complexity(self, question: str) -> str:
        """Determine the complexity level of the query"""
        question_lower = question.lower()

        # Simple queries - single operation
        if any(word in question_lower for word in ['show', 'get', 'what is']):
            return 'simple'

        # Medium complexity - filtering or aggregation
        elif any(word in question_lower for word in ['where', 'filter', 'group', 'average', 'count']):
            return 'medium'

        # Complex queries - multiple operations or analysis
        elif any(word in question_lower for word in ['analyze', 'compare', 'correlate', 'trends']):
            return 'complex'

        return 'medium'  # Default

    def extract_query_intent(self, question: str, columns: List[str]) -> Dict:
        """Extract the intent and components from the question"""
        question_lower = question.lower()

        intent = {
            'type': self.route_question(question, columns),
            'complexity': self.get_query_complexity(question),
            'mentioned_columns': [col for col in columns if col.lower() in question_lower],
            'operations': [],
            'filters': [],
            'aggregations': []
        }

        # Extract operations
        operations = {
            'show': ['show', 'display', 'list', 'get'],
            'filter': ['where', 'filter', 'with'],
            'aggregate': ['count', 'sum', 'average', 'max', 'min'],
            'analyze': ['analyze', 'summarize', 'overview'],
            'compare': ['compare', 'correlate', 'relationship']
        }

        for op_type, keywords in operations.items():
            if any(keyword in question_lower for keyword in keywords):
                intent['operations'].append(op_type)

        # Extract filters (simple pattern matching)
        filter_match = re.search(
            r'where\s+(\w+)\s+(?:is|equals?|=)\s+(\w+)', question_lower)
        if filter_match:
            intent['filters'].append({
                'column': filter_match.group(1),
                'operator': '=',
                'value': filter_match.group(2)
            })

        # Extract aggregations
        agg_match = re.search(
            r'(max|min|average|mean|sum|count)\s+(\w+)', question_lower)
        if agg_match:
            intent['aggregations'].append({
                'function': agg_match.group(1),
                'column': agg_match.group(2)
            })

        if self.debug_mode:
            print(f"Extracted intent: {intent}")

        return intent
