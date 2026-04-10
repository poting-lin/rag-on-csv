import argparse
import logging

from csv_qa.exceptions import (
    CSVQAError,
    DataLoadError,
    OllamaConnectionError,
    OllamaTimeoutError,
    OllamaResponseError,
    QueryEngineError,
)
from csv_qa.question_answerer import CSVQuestionAnswerer

logger = logging.getLogger(__name__)


def configure_logging(debug: bool) -> None:
    """Configure logging level based on debug flag."""
    level = logging.DEBUG if debug else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def main() -> None:
    """Main entry point for the CSV Question Answering Bot."""
    parser = argparse.ArgumentParser(description="CSV Question Answering Bot")
    parser.add_argument("-f", "--file", default="sample_data/lab_data.csv", help="Path to CSV file")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("-m", "--model", default="llama3.2:1b", help="Ollama model to use")
    parser.add_argument("-i", "--interactive", action="store_true", help="Enable interactive mode with suggestions")
    parser.add_argument("--no-context", action="store_true", help="Disable context memory")
    parser.add_argument("--save-history", help="Save conversation history to file on exit")
    parser.add_argument("--load-history", help="Load conversation history from file on startup")
    args = parser.parse_args()

    configure_logging(args.debug)

    csv_file = args.file
    model = args.model
    interactive_mode = args.interactive
    enable_context_memory = not args.no_context
    save_history_file = args.save_history
    load_history_file = args.load_history

    logger.debug("Using CSV file: %s", csv_file)
    logger.debug("Using model: %s", model)

    print("\n" + "=" * 40)
    print("CSV Question Answering Bot")
    print("=" * 40)
    print(f"Loaded CSV file: {csv_file}")
    print(f"Using model: {model}")
    if interactive_mode:
        print("Interactive mode enabled - will suggest similar values when exact matches aren't found")
    if enable_context_memory:
        print("Context memory enabled - remembers conversation history")
    print("=" * 40)
    print("\nTip: Type '/q' to see example questions you can ask")

    try:
        qa = CSVQuestionAnswerer(
            model_name=model,
            enable_context_memory=enable_context_memory,
        )
    except OllamaConnectionError as e:
        print(f"\nCannot connect to Ollama: {e.user_message}")
        logger.error("Ollama connection failed: %s", e.detail)
        return
    except CSVQAError as e:
        print(f"\nFailed to initialize: {e.user_message}")
        logger.error("Initialization error [%s]: %s", e.error_code, e.detail)
        return
    except Exception:
        print("\nUnexpected error during initialization. Check logs for details.")
        logger.exception("Unexpected initialization error")
        return

    if load_history_file:
        try:
            qa.load_conversation_history(load_history_file)
        except Exception:
            logger.warning("Failed to load conversation history from %s", load_history_file, exc_info=True)
            print(f"Warning: Could not load history from {load_history_file}")

    try:
        columns = qa.load_csv(csv_file)
        logger.debug("CSV columns: %s", columns)
    except DataLoadError as e:
        print(f"\n{e.user_message}")
        logger.error("Data load error [%s]: %s", e.error_code, e.detail)
        return
    except Exception:
        print("\nUnexpected error loading CSV file. Check logs for details.")
        logger.exception("Unexpected error loading CSV")
        return

    try:
        _run_interactive_loop(qa, csv_file, interactive_mode, enable_context_memory)
    except KeyboardInterrupt:
        print("\nInterrupted. Exiting.")

    if save_history_file and enable_context_memory:
        try:
            qa.save_conversation_history(save_history_file)
            print(f"Conversation history saved to {save_history_file}")
        except Exception:
            logger.warning("Failed to save conversation history", exc_info=True)
            print("Warning: Could not save conversation history.")


def _run_interactive_loop(
    qa: CSVQuestionAnswerer,
    csv_file: str,
    interactive_mode: bool,
    enable_context_memory: bool,
) -> None:
    """Run the main question-answer loop."""
    while True:
        q = input("\nEnter your question (press Enter to quit):\n> ")
        if not q.strip():
            print("Exiting.")
            break

        # Handle slash commands
        if _handle_slash_command(q, qa, csv_file, enable_context_memory):
            continue

        try:
            if interactive_mode:
                _handle_interactive_question(qa, csv_file, q)
            else:
                _handle_standard_question(qa, csv_file, q)
        except OllamaConnectionError as e:
            print(f"\nConnection error: {e.user_message}")
            logger.error("Ollama connection lost: %s", e.detail)
        except OllamaTimeoutError as e:
            print(f"\nTimeout: {e.user_message}")
            logger.error("Ollama timeout: %s", e.detail)
        except OllamaResponseError as e:
            print(f"\nResponse error: {e.user_message}")
            logger.error("Ollama response error: %s", e.detail)
        except QueryEngineError as e:
            print(f"\nQuery error: {e.user_message}")
            logger.error("Query engine error [%s]: %s", e.error_code, e.detail)
        except CSVQAError as e:
            print(f"\nError: {e.user_message}")
            logger.error("CSV QA error [%s]: %s", e.error_code, e.detail)
        except Exception:
            print("\nUnexpected error processing your question. Check logs for details.")
            logger.exception("Unexpected error processing question")


def _handle_slash_command(
    q: str,
    qa: CSVQuestionAnswerer,
    csv_file: str,
    enable_context_memory: bool,
) -> bool:
    """Handle slash commands. Returns True if a command was handled."""
    stripped = q.strip()

    if stripped == "/q" or any(
        kw in q.lower() for kw in ["help", "what questions", "example", "examples", "what can i ask"]
    ):
        print("\nHere are some questions you can ask about this CSV:")
        suggested_questions = qa.generate_suggested_questions(csv_file)
        print(suggested_questions)
        print("\n" + "-" * 40)
        return True

    if stripped == "/summary":
        if enable_context_memory:
            print("\nConversation Summary:")
            print(qa.get_conversation_summary())
        else:
            print("Context memory is disabled.")
        print("\n" + "-" * 40)
        return True

    if stripped == "/clear":
        if enable_context_memory:
            qa.clear_conversation_history()
            print("Conversation history cleared.")
        else:
            print("Context memory is disabled.")
        print("\n" + "-" * 40)
        return True

    return False


def _handle_interactive_question(qa: CSVQuestionAnswerer, csv_file: str, question: str) -> None:
    """Handle a question in interactive mode with suggestions."""
    result = qa.process_question_with_suggestions(csv_file, question)

    if isinstance(result, tuple) and len(result) == 3:
        answer, suggested_values, is_suggestion = result
    else:
        answer, suggested_values, is_suggestion = result, [], False

    print("\nAnswer:", answer)
    print("-" * 40)

    if not (is_suggestion and suggested_values):
        return

    if any(cmd in answer.lower() for cmd in ["summarize", "list", "count"]):
        confirmation = input("Would you like to use the corrected command? (yes/no): ").strip().lower()
        if confirmation in ("yes", "y") and len(suggested_values) == 1:
            corrected_command = suggested_values[0]
            print(f"\nExecuting command: {corrected_command}")
            answer = qa.answer_question(corrected_command, csv_file)
            if isinstance(answer, tuple):
                print("\nAnswer:", answer[0])
            else:
                print("\nAnswer:", answer)
            print("-" * 40)
    else:
        confirmation = input("Would you like to use one of the suggested values? (yes/no): ").strip().lower()
        if confirmation not in ("yes", "y"):
            return

        if len(suggested_values) == 1:
            new_question = suggested_values[0]
            print(f"\nSearching for: {new_question}")
            answer = qa.answer_question(new_question, csv_file)
            print("\nAnswer:", answer)
            print("-" * 40)
        else:
            print("\nWhich suggestion would you like to use?")
            for i, suggestion in enumerate(suggested_values, 1):
                print(f"{i}. {suggestion}")

            choice = input("Enter the number of your choice (or press Enter to skip): ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(suggested_values):
                new_question = suggested_values[int(choice) - 1]
                print(f"\nSearching for: {new_question}")
                answer = qa.answer_question(new_question, csv_file)
                print("\nAnswer:", answer)
                print("-" * 40)


def _handle_standard_question(qa: CSVQuestionAnswerer, csv_file: str, question: str) -> None:
    """Handle a question in standard (non-interactive) mode."""
    answer = qa.answer_question(question, csv_file)

    if isinstance(answer, tuple) and len(answer) == 3:
        suggestion_text, _suggested_values, _is_suggestion = answer
        print("\nAnswer:", suggestion_text)
    else:
        print("\nAnswer:", answer)

    print("-" * 40)


if __name__ == "__main__":
    main()
