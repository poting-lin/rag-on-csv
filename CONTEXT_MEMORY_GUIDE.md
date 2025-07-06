# Context Memory Implementation Guide

This guide explains how to implement and use Context Memory in the RAG on CSV system to enable conversational AI capabilities with follow-up questions and conversation history.

## Overview

Context Memory allows the CSV Q&A Bot to:
- Remember previous questions and answers
- Detect follow-up questions with pronouns or implicit references
- Resolve pronouns using conversation context
- Maintain conversation focus and topic tracking
- Provide conversation summaries and history management

## Architecture

### Core Components

1. **ConversationContext** (`csv_qa/context_memory.py`)
   - Manages conversation history storage
   - Detects follow-up questions and context references
   - Provides semantic similarity search for questions
   - Handles conversation persistence

2. **ConversationTurn** (Data Structure)
   - Stores individual Q&A interactions
   - Tracks metadata like question type, entities, confidence
   - Enables serialization for persistence

3. **Enhanced CSVQuestionAnswerer** 
   - Integrates context memory into question answering
   - Resolves pronouns and context references
   - Stores conversation turns automatically

## Implementation Details

### 1. Context Memory Module

```python
# Key features of ConversationContext class:

class ConversationContext:
    def __init__(self, max_turns=10, max_age_minutes=30, debug_mode=False):
        # Configurable memory limits and debugging
        
    def add_turn(self, question, answer, question_type, entities_mentioned, ...):
        # Store new conversation turn
        
    def get_relevant_context(self, current_question, max_context_turns=3):
        # Find relevant previous turns for current question
        
    def detect_follow_up_intent(self, current_question):
        # Detect if question is a follow-up to previous questions
        
    def get_context_summary(self, relevant_turns):
        # Generate human-readable context summary
```

### 2. Follow-up Detection

The system detects follow-up questions through multiple strategies:

**Pronoun Detection:**
```python
# Patterns that indicate context references
context_patterns = [
    r'\b(it|they|them|that|those|this|these)\b',
    r'\b(the same|similar|like that|like those)\b',
    r'\b(also|too|as well)\b',
    r'\b(what about|how about)\b',
    r'\b(and|but|however|instead)\b'
]
```

**Semantic Similarity:**
- Uses TF-IDF vectorization to find similar previous questions
- Configurable similarity thresholds (0.3 < similarity < 0.95)

**Entity Tracking:**
- Tracks mentioned columns, values, and proper nouns
- Identifies when current questions reference previously mentioned entities

### 3. Pronoun Resolution

Simple but effective pronoun resolution:

```python
def _resolve_pronouns_with_context(self, question, referenced_turns):
    if latest_turn.entities_mentioned:
        main_entity = latest_turn.entities_mentioned[0]
        
        # Replace common pronouns
        resolved_question = re.sub(r'\bit\b', main_entity, question, flags=re.IGNORECASE)
        resolved_question = re.sub(r'\bthat\b', main_entity, resolved_question, flags=re.IGNORECASE)
        resolved_question = re.sub(r'\bthose\b', main_entity, resolved_question, flags=re.IGNORECASE)
    
    return resolved_question
```

### 4. Question Classification

Automatic classification of question types for better context tracking:

```python
def _classify_question_type(self, question):
    question_lower = question.lower()
    
    if any(word in question_lower for word in ['summarize', 'summary']):
        return 'summary'
    elif any(word in question_lower for word in ['list', 'show', 'display']):
        return 'list'
    elif any(word in question_lower for word in ['count', 'how many']):
        return 'count'
    # ... more classifications
```

## Usage Examples

### Basic Usage

```python
# Initialize with context memory enabled
qa = CSVQuestionAnswerer(
    model_name="llama3.2:1b",
    debug_mode=True,
    enable_context_memory=True
)

# Load CSV data
qa.load_csv("sample_data/lab_data.csv")

# Have a conversation with follow-up questions
qa.answer_question("list all researchers")
qa.answer_question("what labs do they work in?")  # Uses context to resolve "they"
qa.answer_question("show me the equipment")
qa.answer_question("how many of them are there?")  # Refers to equipment from previous question
```

### Advanced Features

```python
# Get conversation summary
summary = qa.get_conversation_summary()
print(summary)

# Clear conversation history
qa.clear_conversation_history()

# Save/load conversation history
qa.save_conversation_history("conversation.json")
qa.load_conversation_history("conversation.json")
```

### Command Line Usage

```bash
# Enable context memory (default)
python main.py -f data.csv

# Disable context memory
python main.py -f data.csv --no-context

# Save conversation history on exit
python main.py -f data.csv --save-history session.json

# Load previous conversation
python main.py -f data.csv --load-history session.json

# Special commands during conversation:
# /summary - Show conversation summary
# /clear - Clear conversation history
# /q - Show example questions
```

## Configuration Options

### Context Memory Parameters

```python
ConversationContext(
    max_turns=10,           # Maximum conversation turns to remember
    max_age_minutes=30,     # Maximum age of turns to keep
    debug_mode=False        # Enable detailed logging
)
```

### Question Answerer Parameters

```python
CSVQuestionAnswerer(
    model_name="llama3.2:1b",
    debug_mode=False,
    enable_context_memory=True  # Enable/disable context memory
)
```

## Performance Considerations

### Memory Management
- Automatic cleanup of old conversation turns
- Configurable limits on history size and age
- Efficient vector operations using scikit-learn

### Computational Efficiency
- TF-IDF vectors rebuilt only when necessary
- Context retrieval limited to relevant turns only
- Lazy evaluation of similarity computations

### Scalability
- Memory usage scales linearly with conversation length
- Processing time increases minimally with context size
- Suitable for extended interactive sessions

## Testing and Debugging

### Debug Mode Features
```python
# Enable debug mode to see:
# - Context detection decisions
# - Pronoun resolution steps
# - Relevance scoring details
# - Memory cleanup operations

qa = CSVQuestionAnswerer(debug_mode=True, enable_context_memory=True)
```

### Demo Script
Run the provided demo script to see context memory in action:

```bash
python context_memory_demo.py
```

The demo includes:
- Automated conversation simulation
- Feature-specific demonstrations
- Interactive testing environment

## Error Handling

The system gracefully handles:
- Missing context when no previous turns exist
- Invalid pronoun references
- Memory cleanup failures
- File I/O errors for persistence

## Integration with Existing Features

Context Memory integrates seamlessly with:
- **Interactive Mode**: Suggestions work with context-aware questions
- **Fuzzy Matching**: Context helps disambiguate similar values
- **Vector Search**: Context enhances semantic search relevance
- **Command Processing**: Special commands work alongside context memory

## Future Enhancements

Potential improvements to consider:
- **Multi-turn Reasoning**: Chain multiple context turns for complex questions
- **Entity Linking**: More sophisticated entity resolution
- **Context Ranking**: Advanced relevance scoring algorithms
- **Cross-session Memory**: Persistent memory across application restarts
- **User Modeling**: Adapt to individual user conversation patterns

## Best Practices

### For Users
- Use pronouns naturally ("it", "they", "that")
- Ask follow-up questions in context
- Use "/summary" to check conversation state
- Clear history with "/clear" when changing topics

### For Developers
- Enable debug mode during development
- Test with various conversation patterns
- Monitor memory usage in long conversations
- Validate context detection accuracy

### For Deployment
- Configure appropriate memory limits
- Set up conversation history persistence
- Monitor performance with real user conversations
- Implement fallback behavior for context failures

## Conclusion

Context Memory transforms the CSV Q&A Bot from a stateless question-answerer into a conversational AI assistant. By remembering previous interactions and understanding follow-up questions, it provides a more natural and efficient user experience for data exploration and analysis.

The implementation is designed to be:
- **Robust**: Handles edge cases and errors gracefully
- **Efficient**: Minimal performance impact on question answering
- **Extensible**: Easy to add new context detection strategies
- **User-friendly**: Natural conversation flow with useful debugging features 