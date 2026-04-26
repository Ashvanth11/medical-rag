from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any

from src.utils.io import read_json

ROOT = Path(__file__).resolve().parents[2]
CLEAN_DIR = ROOT / "data" / "docs" / "cleaned"


def run(n: int = 5, min_chars: int = 800) -> None:
    paths = sorted(CLEAN_DIR.glob("*.json"))
    print(f"Found {len(paths)} cleaned docs")

    bad: List[str] = []
    for p in paths[:n]:
        doc: Dict[str, Any] = read_json(p)
        title = doc.get("title", "")
        url = doc.get("url", "")
        text = doc.get("text", "")
        char_len = len(text)

        print("\n" + "=" * 80)
        print(f"Title: {title}")
        print(f"URL:   {url}")
        print(f"Chars: {char_len}")
        print(text[:600])

        if char_len < min_chars:
            bad.append(url)

    if bad:
        print("\n" + "=" * 80)
        print("FLAGGED: These pages extracted too little text. Open them and verify:")
        for u in bad:
            print(f"- {u}")


if __name__ == "__main__":
    run()