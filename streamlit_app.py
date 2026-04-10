#!/usr/bin/env python3
"""
Streamlit Web UI for CSV Question Answering Bot.

A web interface for the RAG-based CSV question answering system with:
- File upload functionality
- Interactive question answering
- Context memory (conversational AI)
- Model selection
- Conversation history
- Suggested questions
- Download results
"""

import json
import logging
import os
import tempfile
import time
from enum import Enum

import requests as http_requests

import pandas as pd
import streamlit as st

from csv_qa.config import POPULAR_MODELS
from csv_qa.question_answerer import CSVQuestionAnswerer
from csv_qa.exceptions import (
    CSVQAError,
    DataLoadError,
    OllamaConnectionError,
    OllamaTimeoutError,
    OllamaResponseError,
    QueryEngineError,
)

logger = logging.getLogger(__name__)


def configure_logging(debug_mode: bool = False) -> None:
    """Configure logging for the Streamlit app."""
    level = logging.DEBUG if debug_mode else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("csv_qa").setLevel(level)


class OllamaStatus(Enum):
    """Ollama service readiness status."""

    OLLAMA_DOWN = "ollama_down"
    MODEL_DOWNLOADING = "model_downloading"
    READY = "ready"


def check_ollama_ready() -> OllamaStatus:
    """Check if Ollama is running and the model is available.

    Returns:
        OllamaStatus indicating the current state.
    """
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    model_name = os.environ.get("OLLAMA_MODEL", "llama3.2:1b")
    try:
        resp = http_requests.get(f"{base_url}/api/tags", timeout=3)
        resp.raise_for_status()
        models = resp.json().get("models", [])
        for model in models:
            if model.get("name", "").startswith(model_name):
                return OllamaStatus.READY
        return OllamaStatus.MODEL_DOWNLOADING
    except Exception:
        return OllamaStatus.OLLAMA_DOWN


def get_ollama_models() -> list[str]:
    """Get list of available Ollama models via HTTP API.

    Uses the Ollama REST API instead of the CLI binary, so it works both
    locally and inside Docker (where the ollama binary is not installed).
    """
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    try:
        resp = http_requests.get(f"{base_url}/api/tags", timeout=5)
        resp.raise_for_status()
        models = []
        for model in resp.json().get("models", []):
            name = model.get("name", "")
            if name:
                models.append(name)
        return models
    except Exception:
        return []


def pull_ollama_model(model_name: str) -> bool:
    """Pull a model from Ollama, showing download progress in Streamlit.

    Args:
        model_name: The model name to pull (e.g. 'llama3.2:3b').

    Returns:
        True if the pull succeeded, False otherwise.
    """
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    try:
        resp = http_requests.post(
            f"{base_url}/api/pull",
            json={"name": model_name},
            stream=True,
            timeout=600,
        )
        resp.raise_for_status()

        progress_bar = st.progress(0, text=f"Downloading {model_name}...")
        for line in resp.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            status = data.get("status", "")
            total = data.get("total", 0)
            completed = data.get("completed", 0)
            if total > 0:
                pct = completed / total
                progress_bar.progress(pct, text=f"{status}: {completed / 1e9:.1f} / {total / 1e9:.1f} GB")
            else:
                progress_bar.progress(0, text=status)
        progress_bar.progress(1.0, text=f"{model_name} downloaded successfully!")
        return True
    except Exception as e:
        logger.error("Failed to pull model %s: %s", model_name, e)
        st.error(f"Failed to download {model_name}: {e}")
        return False


def initialize_session_state() -> None:
    """Initialize session state variables."""
    if "qa_instance" not in st.session_state:
        st.session_state.qa_instance = None
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    if "csv_columns" not in st.session_state:
        st.session_state.csv_columns = []
    if "current_csv_file" not in st.session_state:
        st.session_state.current_csv_file = None
    if "suggested_questions" not in st.session_state:
        st.session_state.suggested_questions = []


def create_qa_instance(model_name: str, enable_context_memory: bool) -> CSVQuestionAnswerer | None:
    """Create and configure QA instance."""
    try:
        return CSVQuestionAnswerer(
            model_name=model_name,
            enable_context_memory=enable_context_memory,
        )
    except CSVQAError as e:
        st.error(f"Failed to initialize QA system: {e.user_message}")
        return None
    except Exception as e:
        logger.error("Unexpected error initializing QA system", exc_info=True)
        st.error(f"Failed to initialize QA system: {e}")
        return None


def load_csv_file(uploaded_file, qa_instance: CSVQuestionAnswerer):
    """Load CSV file and return column information."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        columns = qa_instance.load_csv(tmp_file_path)
        df = pd.read_csv(tmp_file_path)
        return tmp_file_path, columns, df
    except DataLoadError as e:
        st.error(f"Failed to load CSV: {e.detail}")
        return None, None, None
    except Exception as e:
        logger.error("Unexpected error loading CSV", exc_info=True)
        st.error(f"Error loading CSV file: {e}")
        return None, None, None


def display_conversation_history() -> None:
    """Display conversation history in a nice format."""
    if st.session_state.conversation_history:
        st.subheader("Conversation History")
        for i, (question, answer) in enumerate(st.session_state.conversation_history):
            question_preview = f"Q{i + 1}: {question[:50]}..." if len(question) > 50 else f"Q{i + 1}: {question}"
            with st.expander(question_preview, expanded=False):
                st.write("**Question:**", question)
                st.write("**Answer:**", answer)
    else:
        st.info("No conversation history yet. Ask a question to get started!")


def display_suggested_questions(suggested_questions_text: str) -> None:
    """Display suggested questions in a user-friendly format."""
    if not suggested_questions_text:
        return

    st.subheader("Suggested Questions")

    lines = suggested_questions_text.split("\n")
    questions = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith("🔍") and not line.startswith("📊") and not line.startswith("-"):
            if line.startswith("- "):
                line = line[2:]
            if line:
                questions.append(line)

    cols = st.columns(2)
    for i, question in enumerate(questions[:6]):
        col = cols[i % 2]
        with col:
            if st.button(question, key=f"suggested_q_{i}", use_container_width=True):
                st.session_state.current_question = question
                st.rerun()


def process_question(
    question: str,
    qa_instance: CSVQuestionAnswerer,
    csv_file_path: str,
    interactive_mode: bool,
) -> tuple:
    """Process a question and return the answer."""
    try:
        if interactive_mode:
            result = qa_instance.process_question_with_suggestions(csv_file_path, question)
            if isinstance(result, tuple) and len(result) == 3:
                answer, suggested_values, is_suggestion = result
                return answer, suggested_values, is_suggestion
            return result, [], False
        else:
            answer = qa_instance.answer_question(question, csv_file_path)
            if isinstance(answer, tuple) and len(answer) == 3:
                suggestion_text, suggested_values, is_suggestion = answer
                return suggestion_text, suggested_values, is_suggestion
            return answer, [], False

    except OllamaConnectionError as e:
        st.error(e.user_message)
        return e.user_message, [], False
    except OllamaTimeoutError as e:
        st.warning(e.user_message)
        return e.user_message, [], False
    except OllamaResponseError as e:
        st.error(e.user_message)
        return e.user_message, [], False
    except QueryEngineError as e:
        st.error(f"Query engine error: {e.user_message}")
        return e.user_message, [], False
    except CSVQAError as e:
        st.error(f"Error: {e.user_message}")
        return e.user_message, [], False
    except Exception:
        logger.error("Unhandled exception processing question", exc_info=True)
        error_msg = "An unexpected error occurred. Enable debug mode for details."
        st.error(error_msg)
        return error_msg, [], False


def main() -> None:
    """Main Streamlit application."""
    st.set_page_config(
        page_title="CSV Q&A Bot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    initialize_session_state()

    # Fix selectbox dropdown clipping inside sidebar expanders
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] [data-testid="stExpander"] {
            overflow: visible;
        }
        [data-testid="stSidebar"] {
            overflow: visible;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Check Ollama readiness (useful in Docker where model may still be downloading)
    status = check_ollama_ready()
    if status != OllamaStatus.READY:
        st.title("CSV Question Answering Bot")
        if status == OllamaStatus.OLLAMA_DOWN:
            st.status("Waiting for Ollama service to start...", state="running")
        else:
            model_name = os.environ.get("OLLAMA_MODEL", "llama3.2:1b")
            st.status(
                f"Downloading model {model_name}... This may take a few minutes on first run.",
                state="running",
            )
        time.sleep(3)
        st.rerun()
        return

    st.title("CSV Question Answering Bot")
    st.markdown("Ask natural language questions about your CSV data!")

    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")

        available_models = get_ollama_models()

        if available_models:
            model_name = st.selectbox(
                "Ollama Model",
                available_models,
                index=0,
                help="Select from models available in your Ollama instance",
            )
        else:
            st.warning("No models available yet. Download one below.")
            model_name = "llama3.2:1b"

        # Download new model section
        downloadable = [m for m in POPULAR_MODELS if m not in available_models]
        if downloadable:
            with st.expander("Download New Model"):
                selected_model = st.selectbox(
                    "Choose a model to download",
                    downloadable,
                    index=0,
                    help="Select a model and click Pull to download it",
                )
                if st.button("Pull", use_container_width=True):
                    if pull_ollama_model(selected_model):
                        time.sleep(1)
                        st.rerun()

        st.subheader("Features")
        enable_context_memory = st.checkbox(
            "Context Memory",
            value=True,
            help="Enable conversational AI - remembers previous questions and context",
        )

        interactive_mode = st.checkbox(
            "Interactive Suggestions",
            value=True,
            help="Enable fuzzy matching and suggestions for typos or similar values",
        )

        debug_mode = st.checkbox(
            "Debug Mode",
            value=False,
            help="Enable verbose debug output in logs",
        )

        # Configure logging based on debug toggle
        configure_logging(debug_mode)

        st.subheader("CSV File")
        uploaded_file = st.file_uploader(
            "Upload CSV File",
            type=["csv"],
            help="Upload your CSV file to analyze",
        )

        if not uploaded_file:
            st.markdown("**Or use sample data:**")
            sample_options = {
                "Lab Data": "sample_data/lab_data.csv",
                "Bookshop Data": "sample_data/bookshop.csv",
                "Sound Measurements": "sample_data/sound_measurements.csv",
            }

            selected_sample = st.selectbox("Choose sample dataset", ["None"] + list(sample_options.keys()))

            if selected_sample != "None" and os.path.exists(sample_options[selected_sample]):
                uploaded_file = sample_options[selected_sample]
                st.success(f"Using {selected_sample}")

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        if (
            st.session_state.qa_instance is None
            or getattr(st.session_state, "current_model", None) != model_name
            or getattr(st.session_state, "current_context", None) != enable_context_memory
        ):
            with st.spinner("Initializing Q&A system..."):
                st.session_state.qa_instance = create_qa_instance(model_name, enable_context_memory)
                st.session_state.current_model = model_name
                st.session_state.current_context = enable_context_memory

        if uploaded_file and st.session_state.qa_instance:
            if uploaded_file != st.session_state.current_csv_file:
                with st.spinner("Loading CSV file..."):
                    if isinstance(uploaded_file, str):
                        csv_file_path = uploaded_file
                        try:
                            st.session_state.csv_columns = st.session_state.qa_instance.load_csv(csv_file_path)
                            df = pd.read_csv(csv_file_path)
                        except DataLoadError as e:
                            st.error(f"Failed to load CSV: {e.detail}")
                            csv_file_path = None
                            df = None
                    else:
                        csv_file_path, columns, df = load_csv_file(uploaded_file, st.session_state.qa_instance)
                        st.session_state.csv_columns = columns

                    if csv_file_path and df is not None:
                        st.session_state.current_csv_file = uploaded_file
                        st.session_state.current_csv_path = csv_file_path
                        st.success(f"CSV loaded successfully! Found {len(df)} rows and {len(df.columns)} columns.")

                        with st.spinner("Generating suggested questions..."):
                            suggested_questions = st.session_state.qa_instance.generate_suggested_questions(
                                csv_file_path
                            )
                            st.session_state.suggested_questions = suggested_questions

        # Show CSV preview
        if hasattr(st.session_state, "current_csv_path") and st.session_state.current_csv_path:
            try:
                df = pd.read_csv(st.session_state.current_csv_path)

                st.subheader("CSV Preview")
                st.dataframe(df.head(10), use_container_width=True)

                col_info1, col_info2, col_info3 = st.columns(3)
                with col_info1:
                    st.metric("Rows", len(df))
                with col_info2:
                    st.metric("Columns", len(df.columns))
                with col_info3:
                    st.metric(
                        "Memory Usage",
                        f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB",
                    )

                with st.expander("Column Information"):
                    col_info_df = pd.DataFrame(
                        {
                            "Column": df.columns,
                            "Type": df.dtypes.astype(str),
                            "Non-Null Count": df.count(),
                            "Sample Values": [
                                ", ".join(df[col].dropna().astype(str).unique()[:3]) for col in df.columns
                            ],
                        }
                    )
                    st.dataframe(col_info_df, use_container_width=True)

            except Exception as e:
                logger.error("Error displaying CSV preview", exc_info=True)
                st.error(f"Error displaying CSV preview: {e}")

        # Question input
        if hasattr(st.session_state, "current_csv_path") and st.session_state.current_csv_path:
            st.subheader("Ask a Question")

            current_question = ""
            if hasattr(st.session_state, "current_question"):
                current_question = st.session_state.current_question
                delattr(st.session_state, "current_question")

            question = st.text_area(
                "Enter your question about the CSV data:",
                value=current_question,
                height=100,
                placeholder="e.g., 'How many records are there?', 'Show me records where Location is Urban Park'",
            )

            with st.expander("Special Commands"):
                st.markdown(
                    """
                - `/summary` - View conversation summary (context memory mode)
                - `/clear` - Clear conversation history (context memory mode)
                - `/q` or `/help` or `what questions can I ask` - Show suggested questions
                """
                )

            col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])

            with col_btn1:
                ask_button = st.button("Ask Question", type="primary", disabled=not question.strip())

            with col_btn2:
                if enable_context_memory:
                    clear_history = st.button("Clear History")
                    if clear_history:
                        st.session_state.conversation_history = []
                        if st.session_state.qa_instance:
                            st.session_state.qa_instance.clear_conversation_history()
                        st.success("Conversation history cleared!")
                        st.rerun()

            if ask_button and question.strip():
                with st.spinner("Thinking..."):
                    if question.strip() == "/summary" and enable_context_memory:
                        answer = st.session_state.qa_instance.get_conversation_summary()
                        st.session_state.conversation_history.append((question, answer))
                        st.success("**Conversation Summary:**")
                        st.write(answer)

                    elif question.strip() == "/clear" and enable_context_memory:
                        st.session_state.qa_instance.clear_conversation_history()
                        st.session_state.conversation_history = []
                        answer = "Conversation history cleared."
                        st.success("**Answer:**")
                        st.write(answer)

                    elif question.strip().lower() in ["help", "/q"] or any(
                        help_query in question.lower()
                        for help_query in [
                            "what questions",
                            "example",
                            "examples",
                            "what can i ask",
                        ]
                    ):
                        answer = st.session_state.suggested_questions
                        if answer:
                            st.session_state.conversation_history.append((question, answer))
                            st.success("**Suggested Questions:**")
                            st.write(answer)
                        else:
                            fallback_answer = """
**Here are some general questions you can ask about CSV data:**

**Basic Questions:**
- How many records are there?
- List all records
- Count records
- Summarize the data

**Filtering Questions:**
- Show me records where [column] is [value]
- List records with [specific criteria]
- Find records containing [text]

**Analysis Questions:**
- What is the average [column name]?
- What is the maximum/minimum [column name]?
- Group by [column name]

*Note: Load a CSV file to get specific suggested questions for your data.*
                            """
                            st.session_state.conversation_history.append((question, fallback_answer))
                            st.success("**General Help:**")
                            st.write(fallback_answer)

                    else:
                        answer, suggested_values, is_suggestion = process_question(
                            question,
                            st.session_state.qa_instance,
                            st.session_state.current_csv_path,
                            interactive_mode,
                        )

                        st.session_state.conversation_history.append((question, answer))

                        if is_suggestion and suggested_values and interactive_mode:
                            st.warning(f"**Suggestion**: {answer}")

                            if len(suggested_values) == 1:
                                if st.button(f"Use suggestion: {suggested_values[0]}"):
                                    with st.spinner("Processing suggestion..."):
                                        corrected_answer, _, _ = process_question(
                                            suggested_values[0],
                                            st.session_state.qa_instance,
                                            st.session_state.current_csv_path,
                                            False,
                                        )
                                        st.session_state.conversation_history.append(
                                            (suggested_values[0], corrected_answer)
                                        )
                                        st.rerun()
                            else:
                                st.write("**Available suggestions:**")
                                for i, suggestion in enumerate(suggested_values):
                                    if st.button(f"{suggestion}", key=f"suggestion_{i}"):
                                        with st.spinner("Processing suggestion..."):
                                            corrected_answer, _, _ = process_question(
                                                suggestion,
                                                st.session_state.qa_instance,
                                                st.session_state.current_csv_path,
                                                False,
                                            )
                                            st.session_state.conversation_history.append((suggestion, corrected_answer))
                                            st.rerun()
                        else:
                            st.success("**Answer:**")
                            st.write(answer)

        else:
            st.info("Please upload a CSV file or select sample data to get started!")

    with col2:
        if st.session_state.suggested_questions:
            display_suggested_questions(st.session_state.suggested_questions)

        if enable_context_memory:
            display_conversation_history()

        st.subheader("System Status")
        status_container = st.container()

        with status_container:
            if st.session_state.qa_instance:
                st.success("QA System: Ready")
            else:
                st.error("QA System: Not initialized")

            if hasattr(st.session_state, "current_csv_path"):
                st.success("CSV File: Loaded")
            else:
                st.warning("CSV File: Not loaded")

            st.info(f"Model: {model_name}")
            st.info(f"Context Memory: {'On' if enable_context_memory else 'Off'}")
            st.info(f"Interactive Mode: {'On' if interactive_mode else 'Off'}")

        if enable_context_memory and st.session_state.conversation_history:
            st.subheader("Export")

            conversation_text = ""
            for i, (q, a) in enumerate(st.session_state.conversation_history):
                conversation_text += f"Q{i + 1}: {q}\nA{i + 1}: {a}\n\n"

            st.download_button(
                label="Download Conversation",
                data=conversation_text,
                file_name="conversation_history.txt",
                mime="text/plain",
            )


if __name__ == "__main__":
    main()
