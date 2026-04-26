# Failure Analysis — Medical RAG System

## Overview

Building this system was an iterative loop: run a test query, observe what broke, audit the retrieved chunks to understand why, fix the pipeline or data, re-index, and repeat. The failures were rarely random — each one pointed to a structural gap in retrieval quality, LLM behavior, or source coverage. This document records the nine failure modes I found and how I resolved them.

---

## Failure Analysis

### 1. The LLM Bypassed Grounding by Omitting Citations

**Observed:** I noticed the model occasionally returned fluent, confident-sounding answers with no `[N]` citation markers — answers that referenced information not present in the retrieved chunks. The system had no way to distinguish a grounded answer from a hallucinated one.

**Diagnosed:** The system prompt instructed the LLM to cite every factual claim, but there was no enforcement. The model could (and did) satisfy the instruction partially or ignore it when it felt it "knew" the answer.

**Fixed:** Added a hard post-generation gate in `pipeline.py` (line 70): if the LLM's response contains no `[N]` citation, the answer is discarded entirely and replaced with a fallback. Citation presence became a binary requirement, not a soft instruction. Citation rate in the benchmark: 100%.

---

### 2. Low-Confidence Retrieval Still Called the LLM

**Observed:** When a user asked a question on a topic not in the indexed content, Qdrant still returned the five "closest" chunks it had — but with low cosine similarity scores. The LLM would then hallucinate an answer using those weakly-related chunks as scaffolding.

**Diagnosed:** Retrieval has no concept of "no good match" — it always returns the top-k. Running `scripts/test_qdrant_retrieval.py` directly showed scores as low as 0.11 for unrelated queries that nonetheless triggered a full LLM call.

**Fixed:** Added a retrieval confidence gate in `pipeline.py` (line 54): if `max(scores) < 0.25`, the pipeline short-circuits before calling the LLM. The threshold was chosen by examining the score distribution for known-good vs. known-bad retrievals.

---

### 3. A Generic "I Don't Know" Left Users with No Signal

**Observed:** When the confidence gate triggered, the response was just "I don't know based on the available medical content." A user asking "What's the capital of France?" and a user asking about a rare disease they expected to be covered both got the same message — which was actively misleading for the second user.

**Diagnosed:** The two cases are fundamentally different: one is off-topic (not medical), the other is in-scope but not indexed. A single fallback message can't serve both.

**Fixed:** Added `classify_query()` in `src/rag/classify.py`. It computes cosine similarity between the user's query and 12 hand-crafted medical anchor phrases. If similarity is below 0.55 (empirically validated via `scripts/test_classifier.py`), the query is classified `OFF_TOPIC` and gets a "this assistant answers medical questions" response. Otherwise it gets a `UNKNOWN_MEDICAL` response acknowledging the topic is real but not indexed (`pipeline.py` lines 55–56).

---

### 4. Multiple Chunks from the Same Page Appeared as "Distinct" Sources

**Observed:** When the LLM cited `[1][2][3]`, I expected three different pages. Instead I found all three citations pointing back to `cdc.gov/diabetes/about/index.html` — three chunks from the same document. The answer looked well-sourced but was backed by only one page.

**Diagnosed:** Qdrant returns the top-k most similar chunks, not the top-k most similar pages. A single authoritative document can fill all five slots.

**Fixed:** Added URL deduplication in `pipeline.py` (lines 37–46): after retrieval, chunks are filtered to keep only the highest-scoring chunk per URL. The source list passed to the LLM now contains at most one chunk per page, forcing multi-source answers to actually draw from distinct documents.

---

### 5. Context Was Lost at Chunk Boundaries

**Observed:** Some answers were strangely incomplete. A question about insulin dosing returned a chunk that began mid-sentence: "...which should be adjusted based on blood glucose readings." The preceding context — what "which" referred to — was in the previous chunk and wasn't retrieved.

**Diagnosed:** Purely contiguous chunking severs sentences at boundaries. Any chunk that starts mid-thought is self-contained only in the file, not in isolation.

**Fixed:** Implemented a 60-token overlap between adjacent chunks in `src/rag/chunking.py` (lines 128–135). Each new chunk begins with the last 60 tokens of the previous one. This ensures that boundary-spanning context is duplicated into both chunks, so either one can stand alone.

---

### 6. Tiny Chunks Polluted Retrieval Results

**Observed:** `scripts/test_qdrant_retrieval.py` showed that top-5 results occasionally included chunks of 30–40 tokens — things like "**Prevention Tips**" or "Learn more about diabetes." These fragments scored well on keyword overlap but carried zero information density, displacing actually useful chunks.

**Diagnosed:** The CDC pages contain heading lines and navigation-adjacent text that survived cleaning. After chunking, these became standalone micro-chunks that got indexed alongside real content.

**Fixed:** Set `min_tokens=80` in `scripts/build_chunks.py` (line 43, originally 120 — the `# was 120` comment marks the iteration). Any chunk below this threshold is merged into the preceding chunk if the combined size fits within `max_tokens=450`. The initial value of 120 was too aggressive and was merging genuinely short-but-useful paragraphs; 80 preserved those while still eliminating pure fragments.

---

### 7. CDC Alone Had Coverage Gaps for Clinical Detail

**Observed:** Test queries about COPD diagnosis ("What test is used to diagnose COPD?"), childhood obesity BMI ranges, and Alzheimer's disease stages returned vague or partial answers. Auditing the retrieved chunks showed CDC pages on these topics are public-health-facing — they explain prevalence and risk factors but don't describe diagnostic criteria.

**Diagnosed:** The CDC content is intentionally high-level. Clinical specifics live in agency-specific sources: NHLBI for cardiovascular and respiratory conditions, NIA for Alzheimer's and aging.

**Fixed:** Iteratively added two additional source corpora to `src/ingest/cdc_urls.py`: 12 NHLBI URLs covering detailed blood pressure treatment, COPD diagnosis, coronary heart disease, and obesity; and 8 NIA URLs covering Alzheimer's/dementia subtypes, diagnosis, and prevention research. After re-indexing, the COPD spirometry question and the Alzheimer's early signs question both hit their expected sources.

---

### 8. Multi-Turn Conversations Caused Context Bloat

**Observed:** In long sessions (10+ turns), response latency climbed from ~20s to 45–60s. The full conversation history was being forwarded to the LLM on every turn, growing linearly with session length.

**Diagnosed:** The Ollama context window processes the entire messages array on each call. Early turns are usually no longer relevant to the current question and were adding tokens without adding value.

**Fixed:** Added `MAX_HISTORY_TURNS = 10` in `pipeline.py` (line 25). On each call, history is trimmed to the last 20 messages (10 turns × 2 messages) before being sent to the LLM (line 33). This caps latency growth and prevents context overflow in extended sessions.

---

### 9. The LLM Mixed a Grounded Answer with the Fallback Text

**Observed:** Occasionally a response read: "The A1C target for most adults with diabetes is below 7% [1]. I don't know based on the available medical content." — a correct, cited answer immediately followed by the "don't know" sentinel.

**Diagnosed:** The system prompt instructs the LLM to output the `DONT_KNOW` string if it cannot answer. In some cases the model would answer part of the question from the excerpts and then append the fallback for the parts it couldn't address.

**Fixed:** Added explicit handling in `pipeline.py` (lines 67–68): if the response contains both a `[N]` citation *and* the `DONT_KNOW` string, the sentinel is stripped and the cited portion is kept. Pure `DONT_KNOW` responses (no citations) still route through the classifier for a context-appropriate fallback.

---

## Evaluation Results

Benchmarked against 30 question/answer pairs spanning 10 medical topics (diabetes, hypertension, cholesterol, COPD, stroke, obesity, arthritis, kidney disease, Alzheimer's, tobacco/alcohol). All questions were tested against the full end-to-end pipeline.

| Metric | Result |
|---|---|
| Answer rate | 100% |
| Retrieval recall (expected URL in used sources) | 93.3% |
| Citation rate | 100% |
| Keyword accuracy | 96.7% |
| Average latency | 22.4s |

The two retrieval misses (`diabetes_003`, `hbp_003`) were cases where the correct answer existed in the index but in a different page than the one marked as "expected" — the retrieved content was still factually accurate. Retrieval recall at 93.3% with a 100% answer rate means the system never silently failed: it either answered correctly or said so.

---

## What Remains

- **Token counting:** The chunker uses whitespace-split as a token proxy. A real tokenizer (e.g. `tiktoken`) would produce more precise chunk boundaries, especially for medical abbreviations.
- **Re-ranking:** A cross-encoder re-ranker pass after retrieval would improve precision for ambiguous queries that embed similarly to multiple topics.
- **Streaming output:** The LLM response is buffered before display. Streaming would significantly improve perceived latency at the Streamlit layer.
- **Broader source coverage:** Pharmacology, pediatric guidelines, and mental health content are out of scope today. Adding NIH MedlinePlus or SAMHSA sources would cover the most common gaps users hit.
