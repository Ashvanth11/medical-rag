from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List

from src.rag.chunking import ChunkConfig, make_chunks_from_doc


# -------------------------
# Paths
# -------------------------

CLEANED_DIR = Path("data/docs/cleaned")
OUTPUT_DIR = Path("data/docs/chunks")

CHUNKS_JSONL = OUTPUT_DIR / "medical_chunks.jsonl"
STATS_JSON = OUTPUT_DIR / "chunk_stats.json"


# -------------------------
# Helpers
# -------------------------

def load_cleaned_docs() -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    for path in sorted(CLEANED_DIR.glob("*.json")):
        with path.open("r", encoding="utf-8") as f:
            docs.append(json.load(f))
    return docs


# -------------------------
# Main runner
# -------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cfg = ChunkConfig(
        target_tokens=350,
        overlap_tokens=60,
        min_tokens=80, # was 120
        max_tokens=450,
    )

    docs = load_cleaned_docs()
    all_chunks: List[Dict[str, Any]] = []

    for doc in docs:
        chunks = make_chunks_from_doc(doc, cfg)
        all_chunks.extend(chunks)

    # Write chunks as JSONL
    with CHUNKS_JSONL.open("w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    # Compute stats
    token_counts = [c["token_est"] for c in all_chunks]
    char_counts = [c["char_len"] for c in all_chunks]

    stats = {
        "docs_in": len(docs),
        "chunks_out": len(all_chunks),
        "token_est_min": min(token_counts) if token_counts else 0,
        "token_est_max": max(token_counts) if token_counts else 0,
        "token_est_avg": round(sum(token_counts) / len(token_counts), 2) if token_counts else 0,
        "char_len_min": min(char_counts) if char_counts else 0,
        "char_len_max": max(char_counts) if char_counts else 0,
        "char_len_avg": round(sum(char_counts) / len(char_counts), 2) if char_counts else 0,
        "chunk_config": cfg.__dict__,
    }

    with STATS_JSON.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    # Print for immediate feedback
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
