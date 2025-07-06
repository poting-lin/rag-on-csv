import argparse
import traceback
from csv_qa.question_answerer import CSVQuestionAnswerer


def main():
    """Main entry point for the CSV Question Answering Bot"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="CSV Question Answering Bot")
    parser.add_argument(
        "-f", "--file", default="sample_data/lab_data.csv", help="Path to CSV file")
    parser.add_argument("-d", "--debug", action="store_true",
                        help="Enable debug mode")
    parser.add_argument("-m", "--model", default="llama3.2:1b",
                        help="Ollama model to use")
    parser.add_argument("-i", "--interactive", action="store_true",
                        help="Enable interactive mode with suggestions")
    parser.add_argument("--no-context", action="store_true",
                        help="Disable context memory")
    parser.add_argument(
        "--save-history", help="Save conversation history to file on exit")
    parser.add_argument(
        "--load-history", help="Load conversation history from file on startup")
    args = parser.parse_args()

    try:
        csv_file = args.file
        model = args.model
        debug_mode = args.debug
        interactive_mode = args.interactive
        enable_context_memory = not args.no_context
        save_history_file = args.save_history
        load_history_file = args.load_history

        if debug_mode:
            print(f"Using CSV file: {csv_file}")
            print(f"Using model: {model}")
        print("\n" + "=" * 40)
        print("🤖 CSV Question Answering Bot")
        print("=" * 40)
        print(f"📊 Loaded CSV file: {csv_file}")
        print(f"🧠 Using model: {model}")
        if interactive_mode:
            print(
                "💬 Interactive mode enabled - will suggest similar values when exact matches aren't found")
        if enable_context_memory:
            print("🧠 Context memory enabled - remembers conversation history")
        print("=" * 40)
        print("\n💡 Tip: Type '/q' to see example questions you can ask")

        qa = CSVQuestionAnswerer(
            model_name=model,
            debug_mode=debug_mode,
            enable_context_memory=enable_context_memory
        )

        if load_history_file:
            qa.load_conversation_history(load_history_file)

        columns = qa.load_csv(csv_file)
        if debug_mode:
            print(f"CSV columns: {columns}")

        while True:
            try:
                q = input("\n❓ Enter your question (press Enter to quit):\n> ")
                if not q.strip():
                    print("👋 Exiting.")
                    break

                if q.strip() == "/q" or any(help_query in q.lower() for help_query in [
                        "help", "what questions", "example", "examples", "what can i ask"
                ]):
                    print("\n🔍 Here are some questions you can ask about this CSV:")
                    suggested_questions = qa.generate_suggested_questions(
                        csv_file)
                    print(suggested_questions)
                    print("\n" + "-" * 40)
                    continue
                elif q.strip() == "/summary":
                    if enable_context_memory:
                        print("\n📊 Conversation Summary:")
                        print(qa.get_conversation_summary())
                    else:
                        print("Context memory is disabled.")
                    print("\n" + "-" * 40)
                    continue
                elif q.strip() == "/clear":
                    if enable_context_memory:
                        qa.clear_conversation_history()
                        print("🗑️ Conversation history cleared.")
                    else:
                        print("Context memory is disabled.")
                    print("\n" + "-" * 40)
                    continue

                if interactive_mode:
                    result = qa.process_question_with_suggestions(csv_file, q)

                    if isinstance(result, tuple) and len(result) == 3:
                        answer, suggested_values, is_suggestion = result
                        print("\n🤖 Answer:", answer)
                    else:
                        answer = result
                        suggested_values = []
                        is_suggestion = False
                        print("\n🤖 Answer:", answer)

                    print("-" * 40)

                    if is_suggestion and suggested_values:
                        if any(cmd in answer.lower() for cmd in ["summarize", "list", "count"]):
                            confirmation = input(
                                "Would you like to use the corrected command? (yes/no): ").strip().lower()
                            if confirmation in ["yes", "y"]:
                                if len(suggested_values) == 1:
                                    corrected_command = suggested_values[0]
                                    print(
                                        f"\nExecuting command: {corrected_command}")
                                    answer = qa.answer_question(
                                        corrected_command, csv_file)

                                    if isinstance(answer, tuple):
                                        print("\n🤖 Answer:", answer[0])
                                    else:
                                        print("\n🤖 Answer:", answer)
                                    print("-" * 40)
                        else:
                            confirmation = input(
                                "Would you like to use one of the suggested values? (yes/no): ").strip().lower()
                            if confirmation in ["yes", "y"]:
                                if len(suggested_values) == 1:
                                    new_question = suggested_values[0]
                                    print(f"\nSearching for: {new_question}")
                                    answer = qa.answer_question(
                                        new_question, csv_file)
                                    print("\n🤖 Answer:", answer)
                                    print("-" * 40)
                                else:
                                    print(
                                        "\nWhich suggestion would you like to use?")
                                    for i, suggestion in enumerate(suggested_values, 1):
                                        print(f"{i}. {suggestion}")

                                    choice = input(
                                        "Enter the number of your choice (or press Enter to skip): ").strip()
                                    if choice.isdigit() and 1 <= int(choice) <= len(suggested_values):
                                        new_question = suggested_values[int(
                                            choice) - 1]
                                        print(
                                            f"\nSearching for: {new_question}")
                                        answer = qa.answer_question(
                                            new_question, csv_file)
                                        print("\n🤖 Answer:", answer)
                                        print("-" * 40)
                else:
                    answer = qa.answer_question(q, csv_file)

                    if isinstance(answer, tuple) and len(answer) == 3:
                        suggestion_text, suggested_values, is_suggestion = answer
                        print("\n🤖 Answer:", suggestion_text)
                    else:
                        print("\n🤖 Answer:", answer)

                    print("-" * 40)

            except KeyboardInterrupt:
                print("\n👋 Interrupted. Exiting.")
                break
            except Exception as e:
                print(f"❌ Error processing question: {e}")
                traceback.print_exc()

        if save_history_file and enable_context_memory:
            qa.save_conversation_history(save_history_file)
            print(f"💾 Conversation history saved to {save_history_file}")

    except Exception as e:
        print(f"❌ Fatal error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
