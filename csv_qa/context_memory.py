"""
Context Memory module for maintaining conversation history and context awareness
"""
import re
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class ConversationTurn:
    """Represents a single question-answer turn in the conversation"""
    question: str
    answer: str
    timestamp: datetime
    question_type: str  # 'lookup', 'analysis', 'filter', 'summary', etc.
    entities_mentioned: List[str]  # columns, values mentioned
    result_count: int  # number of records returned
    confidence_score: float  # how confident we are in the answer
    metadata: Dict[str, Any]  # additional context
    # Store actual query results for context
    result_data: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        # Don't serialize result_data to avoid large JSON files
        if 'result_data' in data:
            data.pop('result_data')
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationTurn':
        """Create from dictionary"""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        # result_data won't be restored from file (intentionally)
        data['result_data'] = None
        return cls(**data)


class ConversationContext:
    """Manages conversation context and memory"""

    def __init__(self, max_turns: int = 10, max_age_minutes: int = 30, debug_mode: bool = False):
        """
        Initialize conversation context

        Args:
            max_turns: Maximum number of turns to keep in memory
            max_age_minutes: Maximum age of turns to keep (in minutes)
            debug_mode: Enable debug logging
        """
        self.max_turns = max_turns
        self.max_age_minutes = max_age_minutes
        self.debug_mode = debug_mode

        self.conversation_history: List[ConversationTurn] = []
        self.session_start = datetime.now()
        # What the user is currently focused on
        self.current_focus: Optional[str] = None
        self.mentioned_entities: Dict[str, int] = {}  # Entity -> mention count

        # For semantic similarity of questions
        self.vectorizer = None
        self.question_vectors = None

    def add_turn(self, question: str, answer: str, question_type: str = "unknown",
                 entities_mentioned: List[str] = None, result_count: int = 0,
                 confidence_score: float = 1.0, metadata: Dict[str, Any] = None,
                 result_data: Optional[Dict[str, Any]] = None):
        """Add a new conversation turn"""

        if entities_mentioned is None:
            entities_mentioned = []
        if metadata is None:
            metadata = {}

        turn = ConversationTurn(
            question=question,
            answer=answer,
            timestamp=datetime.now(),
            question_type=question_type,
            entities_mentioned=entities_mentioned,
            result_count=result_count,
            confidence_score=confidence_score,
            metadata=metadata,
            result_data=result_data
        )

        self.conversation_history.append(turn)

        # Update entity mentions
        for entity in entities_mentioned:
            self.mentioned_entities[entity] = self.mentioned_entities.get(
                entity, 0) + 1

        # Update current focus based on the question
        self._update_current_focus(question, entities_mentioned)

        # Clean up old turns
        self._cleanup_old_turns()

        # Rebuild vectors for similarity search
        self._rebuild_question_vectors()

        if self.debug_mode:
            print(
                f"Added turn: {question[:50]}... -> {len(self.conversation_history)} total turns")

    def get_relevant_context(self, current_question: str, max_context_turns: int = 3) -> List[ConversationTurn]:
        """
        Get the most relevant previous turns for the current question

        Args:
            current_question: The current question being asked
            max_context_turns: Maximum number of previous turns to return

        Returns:
            List of relevant conversation turns, ordered by relevance
        """
        if not self.conversation_history:
            return []

        relevant_turns = []

        # 1. Check for explicit references (pronouns, "that", "it", "them", etc.)
        if self._has_context_references(current_question):
            # Include recent turns that might be referenced
            relevant_turns.extend(
                self.conversation_history[-2:])  # Last 2 turns

        # 2. Check for semantic similarity with previous questions
        similar_turns = self._find_similar_questions(
            current_question, max_context_turns)
        relevant_turns.extend(similar_turns)

        # 3. Include turns that mention entities in the current question
        entity_related_turns = self._find_entity_related_turns(
            current_question, max_context_turns)
        relevant_turns.extend(entity_related_turns)

        # 4. Include recent turns if focus hasn't changed much
        if self._is_continuing_conversation(current_question):
            relevant_turns.extend(self.conversation_history[-2:])

        # Remove duplicates while preserving order
        seen = set()
        unique_turns = []
        for turn in relevant_turns:
            turn_id = id(turn)
            if turn_id not in seen:
                seen.add(turn_id)
                unique_turns.append(turn)

        # Sort by relevance score and recency
        scored_turns = []
        for turn in unique_turns:
            score = self._calculate_relevance_score(turn, current_question)
            scored_turns.append((score, turn))

        scored_turns.sort(reverse=True, key=lambda x: x[0])

        # Return top results
        return [turn for _, turn in scored_turns[:max_context_turns]]

    def get_context_summary(self, relevant_turns: List[ConversationTurn]) -> str:
        """Generate a context summary from relevant turns"""
        if not relevant_turns:
            return ""

        context_parts = []
        context_parts.append("Previous conversation context:")

        for i, turn in enumerate(relevant_turns, 1):
            # Format timestamp
            time_ago = datetime.now() - turn.timestamp
            if time_ago.seconds < 60:
                time_str = "just now"
            elif time_ago.seconds < 3600:
                time_str = f"{time_ago.seconds // 60} minutes ago"
            else:
                time_str = f"{time_ago.seconds // 3600} hours ago"

            context_parts.append(
                f"\n{i}. User asked ({time_str}): {turn.question}")

            # Include a summary of the answer
            answer_summary = turn.answer[:200] + \
                "..." if len(turn.answer) > 200 else turn.answer
            context_parts.append(f"   Answer: {answer_summary}")

            if turn.result_count > 0:
                context_parts.append(
                    f"   (Returned {turn.result_count} records)")

        return "\n".join(context_parts)

    def get_context_data_filter(self, current_question: str) -> Optional[Dict[str, Any]]:
        """
        Get filtering context data for follow-up questions that reference previous results

        Returns:
            Dictionary containing filter information if this is a follow-up question
            that should be applied to a subset of data, None otherwise
        """
        if not self.conversation_history:
            return None

        # Check if this question references previous results
        if not self._has_context_references(current_question):
            return None

        # Look for the most recent turn that returned actual data
        for turn in reversed(self.conversation_history):
            if turn.result_data and turn.result_count > 0:
                # Check if this turn had filtering that we should preserve
                if 'filter_applied' in turn.result_data:
                    if self.debug_mode:
                        print(
                            f"Found context data filter from previous turn: {turn.question[:50]}...")
                    return turn.result_data
                break

        return None

    def detect_follow_up_intent(self, current_question: str) -> Dict[str, Any]:
        """
        Detect if the current question is a follow-up to previous questions

        Returns:
            Dictionary with follow-up information:
            - is_follow_up: boolean
            - reference_type: 'pronoun', 'implicit', 'continuation'
            - referenced_turns: List of relevant previous turns
            - suggested_context: Context to include in the query
        """
        result = {
            'is_follow_up': False,
            'reference_type': None,
            'referenced_turns': [],
            'suggested_context': ""
        }

        if not self.conversation_history:
            return result

        # Check for explicit references
        if self._has_context_references(current_question):
            result['is_follow_up'] = True
            result['reference_type'] = 'pronoun'
            # Recent context
            result['referenced_turns'] = self.conversation_history[-2:]

        # Check for implicit continuation
        elif self._is_continuing_conversation(current_question):
            result['is_follow_up'] = True
            result['reference_type'] = 'continuation'
            result['referenced_turns'] = self.conversation_history[-1:]  # Last turn

        # Check for similar questions (refinement)
        similar_turns = self._find_similar_questions(current_question, 2)
        if similar_turns:
            result['is_follow_up'] = True
            if not result['reference_type']:
                result['reference_type'] = 'refinement'
            result['referenced_turns'].extend(similar_turns)

        # Generate context summary
        if result['referenced_turns']:
            result['suggested_context'] = self.get_context_summary(
                result['referenced_turns'])

        return result

    def get_conversation_summary(self) -> str:
        """Get a summary of the entire conversation"""
        if not self.conversation_history:
            return "No conversation history."

        total_turns = len(self.conversation_history)
        session_duration = datetime.now() - self.session_start

        # Count question types
        type_counts = {}
        for turn in self.conversation_history:
            type_counts[turn.question_type] = type_counts.get(
                turn.question_type, 0) + 1

        # Most mentioned entities
        top_entities = sorted(self.mentioned_entities.items(),
                              key=lambda x: x[1], reverse=True)[:5]

        summary = [
            "Conversation Summary:",
            f"- Duration: {session_duration.seconds // 60} minutes",
            f"- Total questions: {total_turns}",
            f"- Question types: {', '.join([f'{k}({v})' for k, v in type_counts.items()])}",
            f"- Current focus: {self.current_focus or 'None'}",
        ]

        if top_entities:
            summary.append(
                f"- Most discussed: {', '.join([f'{e}({c})' for e, c in top_entities])}")

        return "\n".join(summary)

    def clear_context(self):
        """Clear all conversation context"""
        self.conversation_history.clear()
        self.mentioned_entities.clear()
        self.current_focus = None
        self.session_start = datetime.now()
        self.vectorizer = None
        self.question_vectors = None

        if self.debug_mode:
            print("Cleared conversation context")

    def save_to_file(self, filepath: str):
        """Save conversation history to a JSON file"""
        data = {
            'session_start': self.session_start.isoformat(),
            'current_focus': self.current_focus,
            'mentioned_entities': self.mentioned_entities,
            'conversation_history': [turn.to_dict() for turn in self.conversation_history]
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def load_from_file(self, filepath: str):
        """Load conversation history from a JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)

        self.session_start = datetime.fromisoformat(data['session_start'])
        self.current_focus = data.get('current_focus')
        self.mentioned_entities = data.get('mentioned_entities', {})

        self.conversation_history = [
            ConversationTurn.from_dict(turn_data)
            for turn_data in data.get('conversation_history', [])
        ]

        self._rebuild_question_vectors()

    # Private helper methods

    def _has_context_references(self, question: str) -> bool:
        """Check if the question has pronouns or references to previous context"""
        context_patterns = [
            r'\b(it|they|them|that|those|this|these)\b',
            r'\b(the same|similar|like that|like those)\b',
            r'\b(also|too|as well)\b',
            r'\b(what about|how about)\b',
            r'\b(and|but|however|instead)\b'
        ]

        question_lower = question.lower()
        return any(re.search(pattern, question_lower) for pattern in context_patterns)

    def _is_continuing_conversation(self, question: str) -> bool:
        """Check if the question continues the current conversation topic"""
        if not self.current_focus:
            return False

        # Check if current focus entities are mentioned
        question_lower = question.lower()
        focus_words = self.current_focus.lower().split()

        return any(word in question_lower for word in focus_words if len(word) > 3)

    def _find_similar_questions(self, current_question: str, max_results: int = 3) -> List[ConversationTurn]:
        """Find questions similar to the current one using TF-IDF similarity"""
        if not self.conversation_history or self.question_vectors is None:
            return []

        try:
            # Transform current question
            current_vector = self.vectorizer.transform([current_question])

            # Calculate similarities
            similarities = cosine_similarity(
                current_vector, self.question_vectors)[0]

            # Get top similar questions (excluding perfect matches)
            similar_indices = []
            for i, sim in enumerate(similarities):
                # Exclude too similar (duplicates) and too different
                if 0.3 < sim < 0.95:
                    similar_indices.append((i, sim))

            # Sort by similarity and take top results
            similar_indices.sort(key=lambda x: x[1], reverse=True)

            return [self.conversation_history[i] for i, _ in similar_indices[:max_results]]

        except Exception as e:
            if self.debug_mode:
                print(f"Error finding similar questions: {e}")
            return []

    def _find_entity_related_turns(self, current_question: str, max_results: int = 3) -> List[ConversationTurn]:
        """Find turns that mention entities from the current question"""
        # Extract potential entities from current question (simple approach)
        words = re.findall(
            r'\b[A-Z][a-zA-Z]+\b|\b\w+[_]\w+\b', current_question)

        related_turns = []
        for turn in self.conversation_history:
            overlap = set(words) & set(turn.entities_mentioned)
            if overlap:
                related_turns.append((len(overlap), turn))

        # Sort by overlap count and recency
        related_turns.sort(key=lambda x: (x[0], x[1].timestamp), reverse=True)

        return [turn for _, turn in related_turns[:max_results]]

    def _calculate_relevance_score(self, turn: ConversationTurn, current_question: str) -> float:
        """Calculate relevance score for a conversation turn"""
        score = 0.0

        # Recency bonus (more recent = higher score)
        age_minutes = (datetime.now() - turn.timestamp).seconds / 60
        recency_score = max(0, 1 - (age_minutes / self.max_age_minutes))
        score += recency_score * 0.3

        # Confidence bonus
        score += turn.confidence_score * 0.2

        # Entity overlap bonus
        current_words = set(current_question.lower().split())
        turn_entities = set(entity.lower()
                            for entity in turn.entities_mentioned)
        overlap = len(current_words & turn_entities)
        if turn_entities:
            overlap_ratio = overlap / len(turn_entities)
            score += overlap_ratio * 0.3

        # Question type consistency bonus
        if turn.question_type in ['filter', 'lookup'] and any(
            word in current_question.lower()
            for word in ['where', 'show', 'find', 'get', 'list']
        ):
            score += 0.2

        return score

    def _update_current_focus(self, question: str, entities_mentioned: List[str]):
        """Update the current conversation focus"""
        if entities_mentioned:
            # Focus on the most important entity
            self.current_focus = max(
                entities_mentioned, key=lambda e: self.mentioned_entities.get(e, 0))
        elif not self.current_focus:
            # Extract potential focus from question
            words = question.split()
            capitalized_words = [
                w for w in words if w[0].isupper() and len(w) > 3]
            if capitalized_words:
                self.current_focus = capitalized_words[0]

    def _cleanup_old_turns(self):
        """Remove old conversation turns based on age and count limits"""
        cutoff_time = datetime.now() - timedelta(minutes=self.max_age_minutes)

        # Remove old turns
        self.conversation_history = [
            turn for turn in self.conversation_history
            if turn.timestamp > cutoff_time
        ]

        # Keep only the most recent turns if we exceed the limit
        if len(self.conversation_history) > self.max_turns:
            self.conversation_history = self.conversation_history[-self.max_turns:]

        # Clean up entity mentions for removed turns
        if len(self.conversation_history) < len(self.mentioned_entities):
            current_entities = set()
            for turn in self.conversation_history:
                current_entities.update(turn.entities_mentioned)

            # Remove entities that are no longer mentioned
            self.mentioned_entities = {
                entity: count for entity, count in self.mentioned_entities.items()
                if entity in current_entities
            }

    def _rebuild_question_vectors(self):
        """Rebuild TF-IDF vectors for question similarity search"""
        if not self.conversation_history:
            self.vectorizer = None
            self.question_vectors = None
            return

        questions = [turn.question for turn in self.conversation_history]

        if len(questions) == 1:
            # Need at least 2 questions for TF-IDF
            self.vectorizer = None
            self.question_vectors = None
            return

        try:
            self.vectorizer = TfidfVectorizer(
                stop_words='english', max_features=1000)
            self.question_vectors = self.vectorizer.fit_transform(questions)
        except Exception as e:
            if self.debug_mode:
                print(f"Error building question vectors: {e}")
            self.vectorizer = None
            self.question_vectors = None
