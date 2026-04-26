from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Any

from tqdm import tqdm

from src.utils.io import read_json, write_json, ensure_dir

ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = ROOT / "data" / "docs"
CLEAN_DIR = DOCS_DIR / "cleaned"


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    lines = [ln.strip() for ln in text.split("\n")]
    cleaned = []
    for ln in lines:
        if not ln:
            cleaned.append("")
            continue

        # Drop very short lines that are often nav junk
        if len(ln) < 30 and not re.search(r"[.!?]$", ln):
            continue

        # Drop common boilerplate fragments
        lower = ln.lower()
        if "about cdc" in lower or "contact us" in lower or "800-232-4636" in lower:
            continue

        cleaned.append(ln)

    out = "\n".join(cleaned)
    out = re.sub(r"\n{3,}", "\n\n", out).strip()
    return out


def run() -> None:
    ensure_dir(CLEAN_DIR)
    for path in tqdm(sorted(DOCS_DIR.glob("*.json")), desc="Cleaning docs"):
        doc: Dict[str, Any] = read_json(path)
        if doc.get("error"):
            # keep error docs as-is for debugging
            write_json(CLEAN_DIR / path.name, doc)
            continue

        text = doc.get("text", "")
        doc["text"] = clean_text(text)
        doc["char_len"] = len(doc["text"])
        write_json(CLEAN_DIR / path.name, doc)


if __name__ == "__main__":
    run()