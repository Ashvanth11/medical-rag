"""
Evaluation benchmark for the medical RAG pipeline.

Usage:
    python -m scripts.evaluate
    python -m scripts.evaluate --output data/eval/results.json
    python -m scripts.evaluate --top_k 3

Metrics per question:
    retrieval_hit   - at least one used_source URL matches an expected URL
    citation_present - answer contains at least one [N] citation
    keyword_hit     - answer contains at least one expected keyword (case-insensitive)
    answered        - answer is not the DONT_KNOW fallback

Aggregate metrics:
    retrieval_recall  = mean(retrieval_hit)
    citation_rate     = mean(citation_present)
    keyword_accuracy  = mean(keyword_hit)
    answer_rate       = mean(answered)
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

# Ensure project root is on the path when run as a module
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.pipeline import DONT_KNOW, answer_question

CITATION_RE = re.compile(r"\[\d+\]")


def load_qa_pairs(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def evaluate_one(pair: dict, top_k: int) -> dict:
    question = pair["question"]
    expected_urls = set(pair.get("expected_source_urls", []))
    expected_keywords = [kw.lower() for kw in pair.get("expected_answer_keywords", [])]

    t0 = time.time()
    answer, used_sources, _ = answer_question(question, top_k=top_k)
    latency = round(time.time() - t0, 2)

    answered = DONT_KNOW not in answer
    citation_present = bool(CITATION_RE.search(answer))

    used_urls = {s.get("url", "") for s in used_sources}
    retrieval_hit = bool(expected_urls & used_urls)

    answer_lower = answer.lower()
    keyword_hit = any(kw in answer_lower for kw in expected_keywords)

    return {
        "id": pair.get("id", ""),
        "topic": pair.get("topic", ""),
        "question": question,
        "answered": answered,
        "retrieval_hit": retrieval_hit,
        "citation_present": citation_present,
        "keyword_hit": keyword_hit,
        "latency_s": latency,
        "answer_preview": answer[:120].replace("\n", " "),
        "used_urls": sorted(used_urls),
        "expected_urls": sorted(expected_urls),
    }


def print_results_table(results: list[dict]) -> None:
    col_w = {"id": 16, "topic": 22, "ans": 4, "ret": 4, "cit": 4, "kw": 4, "lat": 6}
    header = (
        f"{'ID':<{col_w['id']}} {'TOPIC':<{col_w['topic']}} "
        f"{'ANS':>{col_w['ans']}} {'RET':>{col_w['ret']}} "
        f"{'CIT':>{col_w['cit']}} {'KW':>{col_w['kw']}} {'LAT(s)':>{col_w['lat']}}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for r in results:
        ans = "Y" if r["answered"] else "N"
        ret = "Y" if r["retrieval_hit"] else "N"
        cit = "Y" if r["citation_present"] else "N"
        kw = "Y" if r["keyword_hit"] else "N"
        print(
            f"{r['id']:<{col_w['id']}} {r['topic']:<{col_w['topic']}} "
            f"{ans:>{col_w['ans']}} {ret:>{col_w['ret']}} "
            f"{cit:>{col_w['cit']}} {kw:>{col_w['kw']}} {r['latency_s']:>{col_w['lat']}}"
        )
    print(sep)


def print_aggregate(results: list[dict]) -> dict:
    n = len(results)
    if n == 0:
        print("No results.")
        return {}

    def pct(key):
        return round(sum(r[key] for r in results) / n * 100, 1)

    agg = {
        "n": n,
        "answer_rate": pct("answered"),
        "retrieval_recall": pct("retrieval_hit"),
        "citation_rate": pct("citation_present"),
        "keyword_accuracy": pct("keyword_hit"),
        "avg_latency_s": round(sum(r["latency_s"] for r in results) / n, 2),
    }

    print("\nAggregate Results")
    print(f"  Questions evaluated : {agg['n']}")
    print(f"  Answer rate         : {agg['answer_rate']}%  (pipeline returned an answer, not DONT_KNOW)")
    print(f"  Retrieval recall    : {agg['retrieval_recall']}%  (correct source in used_sources)")
    print(f"  Citation rate       : {agg['citation_rate']}%  (answer contains [N] citation)")
    print(f"  Keyword accuracy    : {agg['keyword_accuracy']}%  (answer contains expected keyword)")
    print(f"  Avg latency         : {agg['avg_latency_s']}s")
    return agg


def main():
    parser = argparse.ArgumentParser(description="Evaluate the medical RAG pipeline.")
    parser.add_argument(
        "--qa_pairs",
        default="data/eval/qa_pairs.json",
        help="Path to the Q&A pairs JSON file (default: data/eval/qa_pairs.json)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write full results JSON (e.g. data/eval/results.json)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of chunks to retrieve per question (default: 5)",
    )
    parser.add_argument(
        "--topic",
        default=None,
        help="Only evaluate questions for a specific topic (e.g. diabetes)",
    )
    args = parser.parse_args()

    qa_pairs = load_qa_pairs(args.qa_pairs)
    if args.topic:
        qa_pairs = [p for p in qa_pairs if p.get("topic") == args.topic]
        if not qa_pairs:
            print(f"No questions found for topic '{args.topic}'.")
            sys.exit(1)

    print(f"Evaluating {len(qa_pairs)} questions (top_k={args.top_k})...\n")

    results = []
    for i, pair in enumerate(qa_pairs, 1):
        print(f"  [{i}/{len(qa_pairs)}] {pair['question'][:70]}...", end=" ", flush=True)
        result = evaluate_one(pair, top_k=args.top_k)
        results.append(result)
        flags = "".join([
            "A" if result["answered"] else ".",
            "R" if result["retrieval_hit"] else ".",
            "C" if result["citation_present"] else ".",
            "K" if result["keyword_hit"] else ".",
        ])
        print(f"[{flags}] {result['latency_s']}s")

    print()
    print_results_table(results)
    agg = print_aggregate(results)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({"aggregate": agg, "results": results}, f, indent=2)
        print(f"\nFull results written to {args.output}")

    # Exit with non-zero if retrieval recall is below 50%
    if agg.get("retrieval_recall", 100) < 50:
        sys.exit(1)


if __name__ == "__main__":
    main()
