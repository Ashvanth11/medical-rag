from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

import requests
import trafilatura
from bs4 import BeautifulSoup
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]

from src.ingest.cdc_urls import MEDICAL_URLS
from src.utils.io import write_json, write_text, ensure_dir
RAW_HTML_DIR = ROOT / "data" / "raw_html"
DOCS_DIR = ROOT / "data" / "docs"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; medical-rag-cdc/1.0; +https://example.com)"
}


def detect_source(url: str) -> str:
    if "nhlbi.nih.gov" in url:
        return "NHLBI"
    elif "nia.nih.gov" in url:
        return "NIA"
    return "CDC"


def safe_id(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]

# TODO: NHLBI pages (e.g. /health/blood-cholesterol/living-with) may not match these selectors;
# add class_="content-wrapper" or id="main-content" if re-scraping yields empty docs.

def extract_with_bs4(html: str) -> Tuple[str, str]:
    soup = BeautifulSoup(html, "lxml")
    title = (soup.title.get_text(" ", strip=True) if soup.title else "").strip()

    # Remove obvious noise
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.decompose()

    # CDC-specific content containers (try in order)
    main = (
        soup.find("main")
        or soup.find("article")
        or soup.find("div", id="content")
        or soup.find("div", class_="cdc-page-content")
    )

    if main:
        text = main.get_text("\n", strip=True)
    else:
        text = soup.get_text("\n", strip=True)

    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return title, text



def fetch_html(url: str, timeout: int = 30) -> str:
    r = requests.get(url, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    return r.text


def extract_main_text(html: str) -> Tuple[str, str]:
    # Try trafilatura first
    extracted = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=True,
        favor_precision=True,
    )
    if extracted and len(extracted.strip()) > 400:
        # Title from BS4, body from trafilatura
        soup = BeautifulSoup(html, "lxml")
        title = (soup.title.get_text(" ", strip=True) if soup.title else "").strip()
        return title, extracted.strip()

    # Fallback to bs4
    return extract_with_bs4(html)


def run() -> None:
    ensure_dir(RAW_HTML_DIR)
    ensure_dir(DOCS_DIR)

    retrieved_at = datetime.now(timezone.utc).isoformat()

    for url in tqdm(MEDICAL_URLS, desc="Scraping medical pages"):
        doc_id = safe_id(url)
        html_path = RAW_HTML_DIR / f"{doc_id}.html"
        json_path = DOCS_DIR / f"{doc_id}.json"
        source = detect_source(url)

        try:
            html = fetch_html(url)
            write_text(html_path, html)

            title, text = extract_main_text(html)

            doc: Dict[str, str] = {
                "id": doc_id,
                "url": url,
                "title": title,
                "retrieved_at": retrieved_at,
                "source": source,
                "text": text,
            }
            write_json(json_path, doc)
        except Exception as e:
            # Save an error stub so you can inspect failures
            write_json(
                json_path,
                {
                    "id": doc_id,
                    "url": url,
                    "title": "",
                    "retrieved_at": retrieved_at,
                    "source": source,
                    "text": "",
                    "error": repr(e),
                },
            )


if __name__ == "__main__":
    run()
