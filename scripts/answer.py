from __future__ import annotations

import argparse
from src.rag.pipeline import answer_question


def _print_sources(sources: list[dict]) -> None:
    if sources:
        print("\nSources:")
        for i, c in enumerate(sources, start=1):
            print(f"[{i}] {c.get('title')}")
            print(c.get("url"))
            print()


def interactive_loop(top_k: int) -> None:
    history: list[dict] = []
    print("Medical RAG — interactive mode. Type 'exit' or 'quit' to stop.\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if not question or question.lower() in ("exit", "quit", "q"):
            break
        answer, sources, history = answer_question(question, top_k=top_k, history=history)
        print(f"\nAssistant: {answer}\n")
        _print_sources(sources)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question", type=str, nargs="?", default=None)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--interactive", "-i", action="store_true")
    args = parser.parse_args()

    if args.interactive:
        interactive_loop(args.top_k)
        return

    if not args.question:
        parser.error("Provide a question or use --interactive / -i for multi-turn mode.")

    answer, used_sources, _ = answer_question(args.question, top_k=args.top_k)
    print(answer)
    _print_sources(used_sources)


if __name__ == "__main__":
    main()
