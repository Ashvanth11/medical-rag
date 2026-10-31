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
    answered        - pipeline trace identifies an answer rather than a fallback

Aggregate metrics:
    retrieval_recall  = mean(retrieval_hit)
    citation_rate     = mean(citation_present)
    keyword_accuracy  = mean(keyword_hit)
    answer_rate       = mean(answered)
"""

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from datetime import datetime, timezone

# Ensure project root is on the path when run as a module
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.pipeline import answer_question
from src.rag.generate import MODEL

CITATION_RE = re.compile(r"\[\d+\]")


def load_qa_pairs(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def evaluate_one(pair: dict, top_k: int) -> dict:
    question = pair["question"]
    expected_urls = set(pair.get("expected_source_urls", []))
    expected_keywords = [kw.lower() for kw in pair.get("expected_answer_keywords", [])]

    t0 = time.time()
    trace = {}
    answer, used_sources, _ = answer_question(question, top_k=top_k, trace=trace)
    latency = round(time.time() - t0, 2)

    answered = trace["status"] == "answered"
    citation_present = bool(CITATION_RE.search(answer))

    used_urls = {s.get("url", "") for s in used_sources}
    retrieval_hit = bool(expected_urls & used_urls)

    answer_lower = answer.lower()
    keyword_hit = any(kw in answer_lower for kw in expected_keywords)

    return {
        "id": pair.get("id", ""),
        "topic": pair.get("topic", ""),
        "question": question,
        "answer": answer,
        "trace": trace,
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
    print(f"  Answer rate         : {agg['answer_rate']}%  (pipeline returned an answer rather than a fallback)")
    print(f"  Retrieval recall    : {agg['retrieval_recall']}%  (correct source in used_sources)")
    print(f"  Citation rate       : {agg['citation_rate']}%  (answer contains [N] citation)")
    print(f"  Keyword match rate  : {agg['keyword_accuracy']}%  (answer contains expected keyword)")
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
    parser.add_argument("--ids", nargs="+", help="Evaluate only these question IDs")
    parser.add_argument("--seed-capture", help="Reuse matching full answers from a previous capture")
    args = parser.parse_args()

    qa_pairs = load_qa_pairs(args.qa_pairs)
    if args.ids:
        available = {p["id"] for p in qa_pairs}
        if set(args.ids) - available:
            parser.error("Unknown question IDs")
        qa_pairs = [p for p in qa_pairs if p["id"] in args.ids]
    if args.topic:
        qa_pairs = [p for p in qa_pairs if p.get("topic") == args.topic]
        if not qa_pairs:
            print(f"No questions found for topic '{args.topic}'.")
            sys.exit(1)

    print(f"Evaluating {len(qa_pairs)} questions (top_k={args.top_k})...\n")

    results = []
    metadata = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "generator_model": MODEL, "temperature": 0, "top_k": args.top_k,
        "qa_pairs": args.qa_pairs,
    }
    reused = {}
    if args.seed_capture:
        seed_bytes = Path(args.seed_capture).read_bytes()
        seed = json.loads(seed_bytes)
        for field in ("generator_model", "temperature", "top_k", "qa_pairs"):
            if seed.get("metadata", {}).get(field) != metadata[field]:
                parser.error(f"Seed capture differs in {field}")
        pairs_by_id = {p["id"]: p for p in qa_pairs}
        for row in seed["results"]:
            if row["id"] not in pairs_by_id:
                continue
            pair = pairs_by_id[row["id"]]
            if row["question"] != pair["question"] or "answer" not in row or "trace" not in row:
                parser.error("Seed answer does not match the current question or lacks trace data")
            if row["expected_urls"] != sorted(set(pair.get("expected_source_urls", []))):
                parser.error("Seed expected URLs differ")
            expected_keywords = [kw.lower() for kw in pair.get("expected_answer_keywords", [])]
            row["keyword_hit"] = any(kw in row["answer"].lower() for kw in expected_keywords)
            reused[row["id"]] = row
        metadata["seed_capture_sha256"] = hashlib.sha256(seed_bytes).hexdigest()
        metadata["seed_capture_metadata"] = seed["metadata"]
        metadata["reused_ids"] = list(reused)

    def save(aggregate):
        if args.output:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = out_path.with_suffix(out_path.suffix + ".tmp")
            temporary.write_text(json.dumps({
                "metadata": metadata, "aggregate": aggregate, "results": results,
            }, indent=2))
            temporary.replace(out_path)

    for i, pair in enumerate(qa_pairs, 1):
        print(f"  [{i}/{len(qa_pairs)}] {pair['question'][:70]}...", end=" ", flush=True)
        result = reused.get(pair["id"])
        if result is None:
            result = evaluate_one(pair, top_k=args.top_k)
        else:
            print("[reused]", end=" ")
        results.append(result)
        save({})
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
        save(agg)
        print(f"\nFull results written to {args.output}")

    # Exit with non-zero if retrieval recall is below 50%
    if agg.get("retrieval_recall", 100) < 50:
        sys.exit(1)


if __name__ == "__main__":
    main()
