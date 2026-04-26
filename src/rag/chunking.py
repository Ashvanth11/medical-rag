from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple


# -------------------------
# Configuration
# -------------------------

@dataclass(frozen=True)
class ChunkConfig:
    target_tokens: int = 350
    overlap_tokens: int = 60
    min_tokens: int = 120
    max_tokens: int = 450


# -------------------------
# Regex utilities
# -------------------------

# Split on sentence endings followed by whitespace
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

# # Split on blank lines (paragraph breaks)
# _PARAGRAPH_SPLIT = re.compile(r"\n{2,}")


def _is_heading(line: str) -> bool:
    return (
        line.isupper()
        and len(line.split()) >= 2
        and len(line.split()) <= 10
    )


# -------------------------
# Token estimation
# -------------------------

def _token_estimate(text: str) -> int:
    """
    Lightweight token estimate using whitespace.
    Stable and sufficient for chunk sizing.
    """
    return len(text.split())


# -------------------------
# Text splitting helpers
# -------------------------

def _split_into_paragraphs(text: str) -> List[str]:
    blocks = []
    current = []

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            if current:
                blocks.append(" ".join(current).strip())
                current = []
            continue

        if _is_heading(line) and current:
            blocks.append(" ".join(current).strip())
            current = [line]
        else:
            current.append(line)

    if current:
        blocks.append(" ".join(current).strip())

    return blocks



def _split_into_sentences(paragraph: str) -> List[str]:
    sentences = [
        s.strip()
        for s in _SENTENCE_SPLIT.split(paragraph)
        if s.strip()
    ]
    return sentences if sentences else [paragraph.strip()]


# -------------------------
# Sentence packing logic
# -------------------------

def _pack_sentences(
    sentences: List[str],
    cfg: ChunkConfig
) -> List[Tuple[str, int]]:
    """
    Pack sentences into chunks near target_tokens with overlap.
    Returns (chunk_text, token_estimate).
    """
    chunks: List[Tuple[str, int]] = []
    current: List[str] = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = _token_estimate(sentence)

        # Hard fallback for extremely long sentences
        if sentence_tokens > cfg.max_tokens:
            words = sentence.split()
            start = 0
            while start < len(words):
                end = min(start + cfg.target_tokens, len(words))
                piece = " ".join(words[start:end]).strip()
                chunks.append((piece, _token_estimate(piece)))
                start = max(end - cfg.overlap_tokens, end)
            continue

        if current_tokens + sentence_tokens <= cfg.target_tokens:
            current.append(sentence)
            current_tokens += sentence_tokens
        else:
            chunk_text = " ".join(current).strip()
            if chunk_text:
                chunks.append((chunk_text, current_tokens))

            # Build overlap from previous chunk
            if cfg.overlap_tokens > 0 and chunk_text:
                prev_words = chunk_text.split()
                overlap_words = (
                    prev_words[-cfg.overlap_tokens:]
                    if len(prev_words) > cfg.overlap_tokens
                    else prev_words
                )
                current = [" ".join(overlap_words), sentence]
                current_tokens = (
                    _token_estimate(current[0]) + sentence_tokens
                )
            else:
                current = [sentence]
                current_tokens = sentence_tokens

    if current:
        chunk_text = " ".join(current).strip()
        if chunk_text:
            chunks.append((chunk_text, current_tokens))

    return chunks


# -------------------------
# Public chunking API
# -------------------------

def chunk_text(text: str, cfg: ChunkConfig) -> List[str]:
    """
    Convert raw document text into a list of semantic chunks.
    """
    paragraphs = _split_into_paragraphs(text)
    packed_chunks: List[str] = []

    for paragraph in paragraphs:
        sentences = _split_into_sentences(paragraph)
        for chunk_text, _ in _pack_sentences(sentences, cfg):
            packed_chunks.append(chunk_text)

    # Merge tiny tail chunks
    merged: List[str] = []

    for chunk in packed_chunks:
        chunk_tokens = _token_estimate(chunk)

        if merged:
            prev_tokens = _token_estimate(merged[-1])

            # Merge only if small AND safe to merge
            if (
                chunk_tokens < cfg.min_tokens
                and prev_tokens + chunk_tokens <= cfg.max_tokens
            ):
                merged[-1] = (merged[-1] + "\n" + chunk).strip()
                continue

        merged.append(chunk)
        
    return merged



def make_chunks_from_doc(
    doc: Dict[str, Any],
    cfg: ChunkConfig
) -> List[Dict[str, Any]]:
    """
    Produce chunk records with metadata from a cleaned medical document.
    """
    text = (doc.get("text") or "").strip()
    if not text:
        return []

    chunks = chunk_text(text, cfg)
    records: List[Dict[str, Any]] = []

    cursor = 0
    for idx, chunk in enumerate(chunks):
        pos = text.find(chunk, cursor)
        if pos == -1:
            pos = cursor

        start_char = pos
        end_char = pos + len(chunk)
        cursor = max(cursor, end_char)

        records.append({
            "doc_id": doc.get("id"),
            "chunk_id": f"{doc.get('id')}::c{idx:04d}",
            "title": doc.get("title"),
            "url": doc.get("url"),
            "source": doc.get("source"),
            "text": chunk,
            "char_len": len(chunk),
            "token_est": _token_estimate(chunk),
            "start_char": start_char,
            "end_char": end_char,
        })

    return records
