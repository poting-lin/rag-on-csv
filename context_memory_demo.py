#!/usr/bin/env python3
"""
Context Memory Demo for CSV Q&A Bot

This script demonstrates how the context memory feature enables follow-up questions
and maintains conversation history in the CSV Q&A system.
"""

from csv_qa.question_answerer import CSVQuestionAnswerer


def demo_context_memory():
    """Demonstrate context memory capabilities"""
    print("=" * 50)
    print("🧠 Context Memory Demo for CSV Q&A Bot")
    print("=" * 50)

    # Initialize with context memory enabled
    qa = CSVQuestionAnswerer(
        model_name="llama3.2:1b",
        debug_mode=True,  # Enable to see context detection
        enable_context_memory=True
    )

    # Load sample data
    csv_file = "sample_data/lab_data.csv"
    qa.load_csv(csv_file)

    print(f"\n📊 Loaded CSV: {csv_file}")
    print("Now let's have a conversation with follow-up questions...")

    # Simulate a conversation with context
    conversation_pairs = [
        ("list all researchers", "Initial question - no context needed"),
        ("what about their labs?", "Follow-up using 'what about' - should use context"),
        ("show me the equipment", "New topic - equipment"),
        ("how many of them are there?",
         "Follow-up using pronoun 'them' - should refer to equipment"),
        ("what instruments does Dr. Smith use?", "Specific researcher question"),
        ("what about Dr. Chen?", "Follow-up about another researcher"),
        ("summarize the data", "General summary request"),
        ("show me more details about that",
         "Follow-up using 'that' - should refer to previous summary")
    ]

    for i, (question, description) in enumerate(conversation_pairs, 1):
        print("\n" + "─" * 40)
        print(f"Question {i}: {question}")
        print(f"Context: {description}")
        print("─" * 40)

        # Get answer
        answer = qa.answer_question(question, csv_file)

        # Display answer
        if isinstance(answer, tuple):
            print(f"🤖 Answer: {answer[0]}")
            if len(answer) > 1 and answer[1]:
                print(f"💡 Suggestions: {answer[1]}")
        else:
            print(f"🤖 Answer: {answer}")

            # Show conversation summary every few turns
        if i % 3 == 0:
            print(f"\n📊 Conversation Summary (after {i} questions):")
            print(qa.get_conversation_summary())

    print("\n" + "=" * 50)
    print("📊 Final Conversation Summary:")
    print(qa.get_conversation_summary())
    print("=" * 50)


def demo_context_features():
    """Demonstrate specific context memory features"""
    print("\n" + "=" * 50)
    print("🔧 Context Memory Features Demo")
    print("=" * 50)

    qa = CSVQuestionAnswerer(enable_context_memory=True, debug_mode=True)
    qa.load_csv("sample_data/lab_data.csv")

    # Feature 1: Pronoun Resolution
    print("\n1️⃣ Pronoun Resolution:")
    qa.answer_question("show me all researchers")
    answer = qa.answer_question("what labs do they work in?")
    print(f"Response to 'what labs do they work in?': {answer}")

    # Feature 2: Context Continuation
    print("\n2️⃣ Context Continuation:")
    qa.answer_question("list equipment in Lab 1")
    answer = qa.answer_question("what about Lab 2?")
    print(f"Response to 'what about Lab 2?': {answer}")

    # Feature 3: Similar Question Detection
    print("\n3️⃣ Similar Question Detection:")
    qa.answer_question("show all researchers")
    answer = qa.answer_question("list all researchers")  # Similar to previous
    print(f"Response to similar question: {answer}")

    # Feature 4: Session Persistence
    print("\n4️⃣ Session Persistence:")
    print("Saving conversation history...")
    qa.save_conversation_history("demo_conversation.json")

    # Clear and reload
    qa.clear_conversation_history()
    print("History cleared. Loading back...")
    qa.load_conversation_history("demo_conversation.json")
    print(qa.get_conversation_summary())


def interactive_demo():
    """Interactive demo where user can try context memory"""
    print("\n" + "=" * 50)
    print("🎮 Interactive Context Memory Demo")
    print("=" * 50)
    print("Try asking follow-up questions using:")
    print("• Pronouns: 'it', 'they', 'them', 'that'")
    print("• Continuation: 'what about...', 'and...'")
    print("• Commands: '/summary', '/clear', '/q'")
    print("• Type 'quit' to exit")
    print("=" * 50)

    qa = CSVQuestionAnswerer(enable_context_memory=True, debug_mode=False)
    qa.load_csv("sample_data/lab_data.csv")

    while True:
        try:
            question = input("\n❓ Your question: ").strip()

            if question.lower() in ['quit', 'exit', '']:
                break
            elif question == '/summary':
                print("📊 Conversation Summary:")
                print(qa.get_conversation_summary())
                continue
            elif question == '/clear':
                qa.clear_conversation_history()
                print("🗑️ Conversation history cleared.")
                continue
            elif question == '/q':
                print("🔍 Example questions:")
                print(qa.generate_suggested_questions())
                continue

            answer = qa.answer_question(question)

            if isinstance(answer, tuple):
                print(f"🤖 Answer: {answer[0]}")
                if len(answer) > 1 and answer[1]:
                    print(f"💡 Suggestions: {answer[1]}")
            else:
                print(f"🤖 Answer: {answer}")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")

    print("\n👋 Demo finished!")


if __name__ == "__main__":
    # Run all demos
    demo_context_memory()
    demo_context_features()

    # Ask if user wants interactive demo
    response = input(
        "\nWould you like to try the interactive demo? (y/n): ").strip().lower()
    if response in ['y', 'yes']:
        interactive_demo()

    print("\n🎉 Context Memory Demo Complete!")
    print("Features demonstrated:")
    print("✅ Follow-up question detection")
    print("✅ Pronoun resolution")
    print("✅ Context continuation")
    print("✅ Conversation history")
    print("✅ Session persistence")
