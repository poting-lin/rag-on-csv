# RAG on CSV

A Retrieval-Augmented Generation (RAG) system designed specifically for **small LLMs** (1B-8B parameters) to answer questions about CSV data. 

## Features

- **Web Interface**: Streamlit-based UI with drag & drop file upload, real-time configuration, and interactive chat
- **Terminal Interface**: Command-line interface for automation and advanced usage
- **Multi-Engine Question Processing**:
  - **Structured Engine**: Direct pandas operations for aggregations, filtering, and comparisons (23x faster)
  - **Semantic Engine**: Enhanced vector search with CSV-aware chunking
  - **Hybrid Engine**: Combines multiple approaches with confidence scoring
- **Intelligent Question Router**: Automatically determines the best engine for each question type
- **Enhanced Vector Search**: CSV-specific chunking with column descriptions, statistics, and relationships
- **Context Memory**: Conversational AI that remembers previous queries and can answer follow-up questions
- **Advanced Analytics**: Statistical analysis, outlier detection, and data quality assessment
- **Smart Suggestions**: AI-generated suggested questions tailored to your data
- **Interactive Mode**: Fuzzy matching suggestions for typos and similar values


## Prerequisites

- Python 3.8+ (Python 3.11 recommended)
- [Poetry](https://python-poetry.org/docs/#installation) for dependency management
- [Ollama](https://ollama.ai/) running locally with the llama3.2:1b model (or another compatible model)

## Installation

1. Clone this repository:
2. Set up Python environment with pyenv (if using):
   ```bash
   pyenv local 3.11.7
   ```

3. Configure Poetry to use the correct Python version:
   ```bash
   poetry env use python3.11
   ```

4. Install dependencies using Poetry:
   ```bash
   poetry install
   ```

5. Make sure Ollama is running:
   ```bash
   # Either run the desktop app or use the command line
   ollama serve
   ```

6. Pull the required model:
   ```bash
   ollama pull llama3.2:1b
   ```

## Quick Start

### Web Interface (Recommended)

1. **Install dependencies:**
   ```bash
   poetry install
   ```

2. **Make sure Ollama is running:**
   ```bash
   # Start Ollama service
   ollama serve
   
   # In another terminal, pull a model
   ollama pull llama3.2:1b
   ```

3. **Launch the web interface:**
   ```bash
   poetry run streamlit run streamlit_app.py
   ```

4. **Open your browser:**
   - Streamlit will automatically open your default browser
   - Or manually go to: **http://localhost:8501**

5. **Start using the app:**
   - Upload a CSV file or select sample data from the sidebar
   - Configure your preferred settings (model, features)
   - Ask questions about your data!

### Terminal Interface

```bash
poetry install
poetry run python main.py
```

## Multi-Engine Architecture

The system now features an intelligent multi-engine architecture that automatically routes questions to the most appropriate processing engine:

### **Question Router**
Analyzes questions and routes them based on patterns:
- **Structured patterns**: `"show records where X = Y"` → Structured Engine
- **Aggregation patterns**: `"what is the max/min/avg of X"` → Structured Engine  
- **Semantic patterns**: `"analyze the data trends"` → Semantic Engine
- **Complex patterns**: `"find anomalies in sales data"` → Hybrid Engine


### **Enhanced Vector Search Engine**  
CSV-aware semantic search with rich context:
- **Column descriptions**: Detailed metadata about each column
- **Statistical summaries**: Mean, median, range, distribution info
- **Categorical distributions**: Unique values and frequencies
- **Relationship analysis**: Cross-column correlations and dependencies
- **Data quality info**: Missing values, outliers, data types

### **Hybrid Engine**
Combines multiple approaches with confidence scoring:
- **Multi-engine coordination**: Runs structured + semantic engines in parallel
- **Confidence scoring**: Chooses best result based on confidence metrics
- **Fallback logic**: Falls back to alternative engines if primary fails
- **Result synthesis**: Combines insights from multiple engines

### **Example Question Routing:**

```bash
"show records where EventType is Traffic" 
→ Structured Engine (pattern match) → ⚡ 10ms response

"what patterns do you see in the sound measurements?"
→ Semantic Engine (analysis request) → 🔍 Enhanced context

"find unusual records in sales data and explain why"  
→ Hybrid Engine (complex analysis) → 🔀 Multi-engine approach
```

## Usage

You can use the CSV Q&A Bot in two ways:

### Web Interface (Streamlit)

Launch the interactive web UI for the best user experience:

```bash
# Install dependencies
poetry install

# Start the web interface (default port 8501)
poetry run streamlit run streamlit_app.py

# Or with custom port
poetry run streamlit run streamlit_app.py --server.port 8502

# Run in headless mode (no auto-open browser)
poetry run streamlit run streamlit_app.py --server.headless true
```

**Enhanced Features:**
- **File Upload**: Drag & drop CSV files or use sample datasets
- **Dynamic Model Selection**: Automatically detects all downloaded Ollama models on your system
- **Interactive Chat**: Conversational Q&A with enhanced context memory
- **Smart Suggestions**: AI-generated questions tailored to your specific data
- **Data Preview**: View CSV structure and sample data with enhanced analytics
- **Export**: Download conversation history
- **Real-time Configuration**: Toggle enhanced engines and features
- **Engine Status**: See which engine handled each question

**How to use:**
1. **Configure settings** in the sidebar - the app automatically detects your downloaded Ollama models
2. **Upload your CSV** or select from sample datasets
3. **View the enhanced data preview** with statistics and insights
4. **Click suggested questions** or type your own in the text area
5. **Ask follow-up questions** - the system remembers context with enhanced memory
6. **Monitor engine routing** - see which engine processed each question
7. **Download your conversation** history when done

**Enhanced Sample Questions to Try:**
- **Structured**: "Show me all records where Location is Urban Park" (Fast)
- **Aggregation**: "What is the average DecibelsA by Location?" (Fast)
- **Semantic**: "Analyze patterns in the sound measurement data" (Rich context)
- **Hybrid**: "Find unusual noise patterns and explain what might cause them" (Multi-engine)

### Terminal Mode

For command-line usage and automation with enhanced performance:

```bash
# Install dependencies
poetry install

# Run the application with enhanced engines (default)
poetry run python main.py

# Run with a specific CSV file
poetry run python main.py -f path/to/your/file.csv

# Run with debug mode (shows engine routing)
poetry run python main.py -d

# Run with a specific Ollama model
poetry run python main.py -m llama3.2:1b

# Run with interactive mode (enables fuzzy matching suggestions)
poetry run python main.py -i

# Disable enhanced engines (use original vector search only)
poetry run python main.py --disable-enhanced
```

### Special Commands

The application supports enhanced commands and features:

- **`/q` Command**: Type `/q` at any time to see AI-generated example questions tailored to your specific CSV data

```
❓ Enter your question (press Enter to quit):
> /q

🔍 Here are some questions you can ask about this CSV:
🧠 AI-Generated suggestions based on your data structure:

⚡ Fast Structured Queries:
- "show records where Location is Urban Park" 
- "count records where EventType is Traffic"
- "list records where DecibelsA > 70"

📊 Statistical Analysis:
- "what is the average DecibelsA by Location?"
- "show max FrequencyRange for each EventType"

🔍 Semantic Analysis:
- "analyze patterns in the sound measurements"
- "what insights can you provide about this data?"

🔀 Advanced Hybrid Queries:
- "find unusual noise patterns and explain them"
- "identify anomalies in the sound data"
```

- **`/summary` Command**: View enhanced conversation summary with engine usage statistics
- **`/clear` Command**: Clear conversation history 
- **`/engines` Command**: View engine routing statistics and performance metrics
- **Help Queries**: Natural language help requests show enhanced capabilities
- **Automatic Command Correction**: Improved typo correction with context awareness

### Enhanced Interactive Mode

The `-i` flag enables enhanced interactive mode with improved fuzzy matching:

```
❓ Enter your question (press Enter to quit):
> hiroo

🤖 Enhanced Answer: I couldn't find 'hiroo' in the CSV data. 
📊 Engine: Hybrid (Structured + Semantic search)
💡 Did you mean 'Dr. Hiro Tanaka'? 
🎯 Confidence: 85% match

Would you like to use the suggested value? (yes/no): yes

🔍 Routing to Structured Engine for exact match...
⚡ Processing time: 8ms

🤖 Answer: Found 1 match for Researcher Dr. Hiro Tanaka:

Match 1:
- SampleID: A006
- Instrument: Atomic Absorption AA-7000
- Researcher: Dr. Hiro Tanaka
- Lab: Lab 1
📊 Query processed by: Structured Engine
----------------------------------------
```

### Enhanced Command Line Arguments

- `-f, --file`: Path to CSV file (default: lab_data.csv)
- `-d, --debug`: Enable debug mode for verbose output with engine routing info
- `-m, --model`: Ollama model to use (default: llama3.2:1b)
- `-i, --interactive`: Enable enhanced interactive mode with improved fuzzy matching
- `--no-context`: Disable context memory (conversational AI features)
- `--disable-enhanced`: Disable enhanced engines (use original vector search only)
- `--save-history`: Save conversation history to file on exit
- `--load-history`: Load conversation history from file on startup


### Sample Data

The project includes sample CSV files to test with. The enhanced system automatically adapts to any CSV structure and provides AI-generated suggestions based on your specific data.

## Troubleshooting

### Common Issues

- **Poetry installation problems**: If you encounter issues with Poetry, make sure your Python environment is properly set up. Use `poetry env use python3.11` to specify the Python version.

- **Ollama connection errors**: Ensure Ollama is running with `ollama serve` or through the desktop app before starting the application.

- **CSV file not found**: Make sure to provide the correct path to your CSV file with the `-f` option or place your default CSV file in the project directory.

### Enhanced Engine Issues

- **Engine routing problems**: 
  ```bash
  # Enable debug mode to see engine routing decisions
  python main.py -d
  
  # Check engine status in interactive mode
  > /engines
  ```

- **Performance issues**:
  ```bash
  # Disable enhanced engines to use original vector search
  python main.py --disable-enhanced
  
  # Check if structured engine is being used for appropriate queries
  python main.py -d  # Watch for "Routed to: structured engine"
  ```

- **Enhanced features not working**:
  - Verify all enhanced engine imports work: `python -c "from csv_qa.question_router import QuestionRouter"`
  - Run enhanced engine tests: `python -m pytest tests/test_runner.py::TestRunner::test_enhanced_engines_import -v`
  - Check for dependency issues: `poetry install`

### Streamlit-Specific Issues

- **Streamlit won't start**: 
  ```bash
  # Check if port is already in use
  lsof -i :8501
  
  # Use a different port
  poetry run streamlit run streamlit_app.py --server.port 8502
  ```

- **Enhanced features not visible in UI**: 
  - Refresh the browser page
  - Clear Streamlit cache: Remove `.streamlit/` directory
  - Check console for JavaScript errors

- **Model not responding**:
  - Verify Ollama is running: `ollama list`
  - Check if the selected model is available: `ollama pull llama3.2:1b`
  - Try switching to a different model in the sidebar
  - If no models appear in the dropdown, follow the download instructions shown in the app

- **Memory issues with large CSV files**:
  - The enhanced system is more memory efficient but still loads entire CSV
  - For files >100MB, structured engine provides better performance
  - Consider filtering data before upload or use terminal interface