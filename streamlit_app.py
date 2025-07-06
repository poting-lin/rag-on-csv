#!/usr/bin/env python3
"""
Streamlit Web UI for CSV Question Answering Bot

A web interface for the RAG-based CSV question answering system with:
- File upload functionality
- Interactive question answering
- Context memory (conversational AI)
- Model selection
- Conversation history
- Suggested questions
- Download results
"""

import streamlit as st
import pandas as pd
import os
import tempfile
import subprocess
from csv_qa.question_answerer import CSVQuestionAnswerer


def get_ollama_models():
    """Get list of available Ollama models."""
    try:
        # Try to get models using ollama list command
        result = subprocess.run(
            ['ollama', 'list'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            models = []
            for line in lines[1:]:  # Skip header line
                if line.strip():
                    # Extract model name (first column)
                    model_name = line.split()[0]
                    if model_name and ':' in model_name:
                        models.append(model_name)
            return models
        else:
            return []
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
        return []


def initialize_session_state():
    """Initialize session state variables."""
    if 'qa_instance' not in st.session_state:
        st.session_state.qa_instance = None
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'csv_columns' not in st.session_state:
        st.session_state.csv_columns = []
    if 'current_csv_file' not in st.session_state:
        st.session_state.current_csv_file = None
    if 'suggested_questions' not in st.session_state:
        st.session_state.suggested_questions = []


def create_qa_instance(model_name, debug_mode, enable_context_memory):
    """Create and configure QA instance."""
    try:
        qa = CSVQuestionAnswerer(
            model_name=model_name,
            debug_mode=debug_mode,
            enable_context_memory=enable_context_memory
        )
        return qa
    except Exception as e:
        st.error(f"Failed to initialize QA system: {str(e)}")
        return None


def load_csv_file(uploaded_file, qa_instance):
    """Load CSV file and return column information."""
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        # Load CSV with QA instance
        columns = qa_instance.load_csv(tmp_file_path)

        # Also load for display
        df = pd.read_csv(tmp_file_path)

        return tmp_file_path, columns, df
    except Exception as e:
        st.error(f"Error loading CSV file: {str(e)}")
        return None, None, None


def display_conversation_history():
    """Display conversation history in a nice format."""
    if st.session_state.conversation_history:
        st.subheader("💬 Conversation History")

        for i, (question, answer) in enumerate(st.session_state.conversation_history):
            question_preview = f"Q{i+1}: {question[:50]}..." if len(
                question) > 50 else f"Q{i+1}: {question}"
            with st.expander(question_preview, expanded=False):
                st.write("**Question:**", question)
                st.write("**Answer:**", answer)
    else:
        st.info("No conversation history yet. Ask a question to get started!")


def display_suggested_questions(suggested_questions_text):
    """Display suggested questions in a user-friendly format."""
    if suggested_questions_text:
        st.subheader("💡 Suggested Questions")

        # Parse the suggested questions from the text
        lines = suggested_questions_text.split('\n')
        questions = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('🔍') and not line.startswith('📊') and not line.startswith('-'):
                # Remove bullet points and clean up
                if line.startswith('- '):
                    line = line[2:]
                if line:
                    questions.append(line)

        # Display as clickable buttons
        cols = st.columns(2)
        for i, question in enumerate(questions[:6]):  # Limit to 6 questions
            col = cols[i % 2]
            with col:
                if st.button(question, key=f"suggested_q_{i}", use_container_width=True):
                    st.session_state.current_question = question
                    st.rerun()


def process_question(question, qa_instance, csv_file_path, interactive_mode):
    """Process a question and return the answer."""
    try:
        if interactive_mode:
            result = qa_instance.process_question_with_suggestions(
                csv_file_path, question)

            # Handle tuple responses (suggestions)
            if isinstance(result, tuple) and len(result) == 3:
                answer, suggested_values, is_suggestion = result
                return answer, suggested_values, is_suggestion
            else:
                return result, [], False
        else:
            answer = qa_instance.answer_question(question, csv_file_path)

            # Handle tuple responses (suggestions) in non-interactive mode
            if isinstance(answer, tuple) and len(answer) == 3:
                suggestion_text, suggested_values, is_suggestion = answer
                return suggestion_text, suggested_values, is_suggestion
            else:
                return answer, [], False

    except Exception as e:
        error_msg = f"Error processing question: {str(e)}"
        st.error(error_msg)
        return error_msg, [], False


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="CSV Q&A Bot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize session state
    initialize_session_state()

    # Header
    st.title("🤖 CSV Question Answering Bot")
    st.markdown("Ask natural language questions about your CSV data!")

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Model selection
        available_models = get_ollama_models()

        if available_models:
            model_name = st.selectbox(
                "🧠 Ollama Model",
                available_models,
                index=0,
                help="Select the Ollama model to use for question answering"
            )
        else:
            st.error("🚨 No Ollama models found!")
            st.markdown("""
            **To download models, run these commands in your terminal:**
            ```bash
            # Download recommended models
            ollama pull llama3.2:1b      # Small, fast model
            ollama pull llama3.2:3b      # Better quality
            ollama pull llama3.1:8b      # High quality
            
            # Or other models
            ollama pull mistral
            ollama pull codellama
            ```
            
            **Then refresh this page.**
            """)

            # Provide fallback model selection
            st.warning("Using fallback model selection:")
            fallback_models = [
                "llama3.2:1b", "llama3.2:3b", "llama3.1:8b",
                "mistral", "codellama"
            ]
            model_name = st.selectbox(
                "🧠 Fallback Model (may not work)",
                fallback_models,
                index=0,
                help="These models may not be available on your system"
            )

        # Feature toggles
        st.subheader("🎛️ Features")
        enable_context_memory = st.checkbox(
            "Context Memory",
            value=True,
            help="Enable conversational AI - remembers previous questions and context"
        )

        interactive_mode = st.checkbox(
            "Interactive Suggestions",
            value=True,
            help="Enable fuzzy matching and suggestions for typos or similar values"
        )

        debug_mode = st.checkbox(
            "Debug Mode",
            value=False,
            help="Enable verbose debug output"
        )

        # CSV file upload
        st.subheader("📁 CSV File")
        uploaded_file = st.file_uploader(
            "Upload CSV File",
            type=['csv'],
            help="Upload your CSV file to analyze"
        )

        # Sample data option
        if not uploaded_file:
            st.markdown("**Or use sample data:**")
            sample_options = {
                "Lab Data": "sample_data/lab_data.csv",
                "Bookshop Data": "sample_data/bookshop.csv",
                "Sound Measurements": "sample_data/sound_measurements.csv"
            }

            selected_sample = st.selectbox(
                "Choose sample dataset",
                ["None"] + list(sample_options.keys())
            )

            if selected_sample != "None" and os.path.exists(sample_options[selected_sample]):
                uploaded_file = sample_options[selected_sample]
                st.success(f"Using {selected_sample}")

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        # Initialize QA system when parameters change
        if (st.session_state.qa_instance is None or
            getattr(st.session_state, 'current_model', None) != model_name or
            getattr(st.session_state, 'current_debug', None) != debug_mode or
                getattr(st.session_state, 'current_context', None) != enable_context_memory):

            with st.spinner("Initializing Q&A system..."):
                st.session_state.qa_instance = create_qa_instance(
                    model_name, debug_mode, enable_context_memory)
                st.session_state.current_model = model_name
                st.session_state.current_debug = debug_mode
                st.session_state.current_context = enable_context_memory

        # Load CSV file when uploaded or changed
        if uploaded_file and st.session_state.qa_instance:
            if uploaded_file != st.session_state.current_csv_file:
                with st.spinner("Loading CSV file..."):
                    if isinstance(uploaded_file, str):
                        # Sample file path
                        csv_file_path = uploaded_file
                        st.session_state.csv_columns = st.session_state.qa_instance.load_csv(
                            csv_file_path)
                        df = pd.read_csv(csv_file_path)
                    else:
                        # Uploaded file
                        csv_file_path, columns, df = load_csv_file(
                            uploaded_file, st.session_state.qa_instance)
                        st.session_state.csv_columns = columns

                    if csv_file_path:
                        st.session_state.current_csv_file = uploaded_file
                        st.session_state.current_csv_path = csv_file_path
                        st.success(
                            f"✅ CSV loaded successfully! Found {len(df)} rows and {len(df.columns)} columns.")

                        # Generate suggested questions
                        with st.spinner("Generating suggested questions..."):
                            suggested_questions = st.session_state.qa_instance.generate_suggested_questions(
                                csv_file_path)
                            st.session_state.suggested_questions = suggested_questions

        # Show CSV preview
        if hasattr(st.session_state, 'current_csv_path') and st.session_state.current_csv_path:
            try:
                df = pd.read_csv(st.session_state.current_csv_path)

                st.subheader("📊 CSV Preview")
                st.dataframe(df.head(10), use_container_width=True)

                # CSV info
                col_info1, col_info2, col_info3 = st.columns(3)
                with col_info1:
                    st.metric("Rows", len(df))
                with col_info2:
                    st.metric("Columns", len(df.columns))
                with col_info3:
                    st.metric(
                        "Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")

                # Column information
                with st.expander("📋 Column Information"):
                    col_info_df = pd.DataFrame({
                        'Column': df.columns,
                        'Type': df.dtypes.astype(str),
                        'Non-Null Count': df.count(),
                        'Sample Values': [', '.join(df[col].dropna().astype(str).unique()[:3])
                                          for col in df.columns]
                    })
                    st.dataframe(col_info_df, use_container_width=True)

            except Exception as e:
                st.error(f"Error displaying CSV preview: {str(e)}")

        # Question input
        if hasattr(st.session_state, 'current_csv_path') and st.session_state.current_csv_path:
            st.subheader("❓ Ask a Question")

            # Check if we have a suggested question to use
            current_question = ""
            if hasattr(st.session_state, 'current_question'):
                current_question = st.session_state.current_question
                delattr(st.session_state, 'current_question')

            question = st.text_area(
                "Enter your question about the CSV data:",
                value=current_question,
                height=100,
                placeholder="e.g., 'How many records are there?', 'Show me records where Location is Urban Park'"
            )

            # Special commands info
            with st.expander("💡 Special Commands"):
                st.markdown("""
                - `/summary` - View conversation summary (context memory mode)
                - `/clear` - Clear conversation history (context memory mode)
                - `/q` or `/help` or `what questions can I ask` - Show suggested questions
                """)

            col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])

            with col_btn1:
                ask_button = st.button(
                    "🔍 Ask Question", type="primary", disabled=not question.strip())

            with col_btn2:
                if enable_context_memory:
                    clear_history = st.button("🗑️ Clear History")
                    if clear_history:
                        st.session_state.conversation_history = []
                        if st.session_state.qa_instance:
                            st.session_state.qa_instance.clear_conversation_history()
                        st.success("Conversation history cleared!")
                        st.rerun()

            # Process question
            if ask_button and question.strip():
                with st.spinner("🤔 Thinking..."):

                    # Handle special commands
                    if question.strip() == "/summary" and enable_context_memory:
                        answer = st.session_state.qa_instance.get_conversation_summary()
                        st.session_state.conversation_history.append(
                            (question, answer))
                        st.success("✅ **Conversation Summary:**")
                        st.write(answer)

                    elif question.strip() == "/clear" and enable_context_memory:
                        st.session_state.qa_instance.clear_conversation_history()
                        st.session_state.conversation_history = []
                        answer = "🗑️ Conversation history cleared."
                        st.success("✅ **Answer:**")
                        st.write(answer)

                    elif (question.strip().lower() in ["help", "/q"] or
                          any(help_query in question.lower()
                              for help_query in ["what questions", "example", "examples", "what can i ask"])):
                        answer = st.session_state.suggested_questions
                        if answer:
                            st.session_state.conversation_history.append(
                                (question, answer))
                            st.success("✅ **Suggested Questions:**")
                            st.write(answer)
                        else:
                            # If no suggested questions are available
                            fallback_answer = """
🔍 **Here are some general questions you can ask about CSV data:**

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
                            st.session_state.conversation_history.append(
                                (question, fallback_answer))
                            st.success("✅ **General Help:**")
                            st.write(fallback_answer)

                    else:
                        # Process regular question
                        answer, suggested_values, is_suggestion = process_question(
                            question,
                            st.session_state.qa_instance,
                            st.session_state.current_csv_path,
                            interactive_mode
                        )

                        # Add to conversation history
                        st.session_state.conversation_history.append(
                            (question, answer))

                        # Handle suggestions in interactive mode
                        if is_suggestion and suggested_values and interactive_mode:
                            st.warning(f"💡 **Suggestion**: {answer}")

                            if len(suggested_values) == 1:
                                if st.button(f"✅ Use suggestion: {suggested_values[0]}"):
                                    with st.spinner("Processing suggestion..."):
                                        corrected_answer, _, _ = process_question(
                                            suggested_values[0],
                                            st.session_state.qa_instance,
                                            st.session_state.current_csv_path,
                                            False
                                        )
                                        st.session_state.conversation_history.append(
                                            (suggested_values[0], corrected_answer))
                                        st.rerun()
                            else:
                                st.write("**Available suggestions:**")
                                for i, suggestion in enumerate(suggested_values):
                                    if st.button(f"✅ {suggestion}", key=f"suggestion_{i}"):
                                        with st.spinner("Processing suggestion..."):
                                            corrected_answer, _, _ = process_question(
                                                suggestion,
                                                st.session_state.qa_instance,
                                                st.session_state.current_csv_path,
                                                False
                                            )
                                            st.session_state.conversation_history.append(
                                                (suggestion, corrected_answer))
                                            st.rerun()
                        else:
                            st.success("✅ **Answer:**")
                            st.write(answer)

        else:
            st.info(
                "👆 Please upload a CSV file or select sample data to get started!")

    with col2:
        # Display suggested questions
        if st.session_state.suggested_questions:
            display_suggested_questions(st.session_state.suggested_questions)

        # Display conversation history
        if enable_context_memory:
            display_conversation_history()

        # System status
        st.subheader("📊 System Status")
        status_container = st.container()

        with status_container:
            if st.session_state.qa_instance:
                st.success("✅ QA System: Ready")
            else:
                st.error("❌ QA System: Not initialized")

            if hasattr(st.session_state, 'current_csv_path'):
                st.success("✅ CSV File: Loaded")
            else:
                st.warning("⚠️ CSV File: Not loaded")

            st.info(f"🧠 Model: {model_name}")
            st.info(
                f"💬 Context Memory: {'On' if enable_context_memory else 'Off'}")
            st.info(
                f"🔧 Interactive Mode: {'On' if interactive_mode else 'Off'}")

        # Download conversation history
        if enable_context_memory and st.session_state.conversation_history:
            st.subheader("💾 Export")

            # Prepare conversation for download
            conversation_text = ""
            for i, (q, a) in enumerate(st.session_state.conversation_history):
                conversation_text += f"Q{i+1}: {q}\nA{i+1}: {a}\n\n"

            st.download_button(
                label="📥 Download Conversation",
                data=conversation_text,
                file_name="conversation_history.txt",
                mime="text/plain"
            )


if __name__ == "__main__":
    main()
