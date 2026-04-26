from __future__ import annotations

import re

from src.rag.prompting import SYSTEM_PROMPT, build_user_prompt
from src.rag.generate import generate_answer
from src.rag.retrieve import retrieve_top_k
from src.rag.classify import classify_query, QueryClass

DONT_KNOW = "I don't know based on the available medical content."

OUT_OF_SCOPE_OFFTOPIC = (
    "This assistant is designed to answer questions grounded in indexed medical guidelines "
    "— topics like diabetes, heart disease, cancer screening, vaccinations, and preventive care. "
    "Your question appears to be outside that scope. "
    "For general knowledge, please use a general-purpose search engine or assistant."
)

OUT_OF_SCOPE_MEDICAL = (
    "I couldn't find information on that in the medical guidelines I have access to. "
    "This may be a specialized or emerging topic not yet covered in the indexed content. "
    "For personal medical advice or complex clinical questions, "
    "please consult a qualified healthcare provider."
)
MAX_HISTORY_TURNS = 10


def answer_question(
    question: str,
    top_k: int = 5,
    history: list[dict] | None = None,
) -> tuple[str, list[dict], list[dict]]:
    trimmed = (history or [])[-(MAX_HISTORY_TURNS * 2):]

    chunks, scores = retrieve_top_k(question, top_k=top_k)

    # Deduplicate chunks by URL, keeping the highest-scoring one per URL
    seen_urls: set[str] = set()
    deduped_chunks, deduped_scores = [], []
    for chunk, score in zip(chunks, scores):
        url = chunk.get("url")
        if url not in seen_urls:
            seen_urls.add(url)
            deduped_chunks.append(chunk)
            deduped_scores.append(score)
    chunks, scores = deduped_chunks, deduped_scores

    new_turn = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": DONT_KNOW},
    ]

    # Simple retrieval confidence gate
    if not chunks or max(scores) < 0.25:
        qclass = classify_query(question)
        fallback = OUT_OF_SCOPE_OFFTOPIC if qclass == QueryClass.OFF_TOPIC else OUT_OF_SCOPE_MEDICAL
        new_turn[1]["content"] = fallback
        return fallback, [], trimmed + new_turn

    user_prompt = build_user_prompt(question, chunks)
    answer = generate_answer(SYSTEM_PROMPT, user_prompt, trimmed)

    # Hard fallback logic
    has_citation = "[" in answer
    has_fallback = DONT_KNOW in answer

    if has_citation and has_fallback:
        answer = answer.replace(DONT_KNOW, "").strip()

    if not has_citation:
        qclass = classify_query(question)
        answer = OUT_OF_SCOPE_OFFTOPIC if qclass == QueryClass.OFF_TOPIC else OUT_OF_SCOPE_MEDICAL

    used = set(int(n) for n in re.findall(r"\[(\d+)\]", answer))
    used_sources = [c for i, c in enumerate(chunks, start=1) if i in used]

    new_turn[1]["content"] = answer
    updated_history = trimmed + new_turn

    return answer, used_sources, updated_history
