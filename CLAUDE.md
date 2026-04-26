# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

All commands must be run from the project root with the virtualenv activated (`.venv/bin/activate`).

**Start Qdrant (required before running anything that queries the vector DB):**
```bash
docker compose up -d
```

**Run the Streamlit app:**
```bash
streamlit run app.py
```

**CLI query (no UI):**
```bash
python -m scripts.answer "What is the A1C target for adults with diabetes?"
python -m scripts.answer "Your question here" --top_k 3
```

**Re-run the ingestion pipeline (in order):**
```bash
# 1. Scrape CDC pages → data/raw_html/ and data/docs/
python -m src.ingest.scrape_cdc

# 2. Clean raw docs → data/docs/cleaned/
python -m src.ingest.clean_cdc

# 3. Chunk cleaned docs → data/docs/chunks/cdc_chunks.jsonl
python -m scripts.build_chunks

# 4. Embed chunks and index into Qdrant
python -m scripts.build_embeddings_qdrant
```

**Test retrieval directly against Qdrant:**
```bash
python -m scripts.test_qdrant_retrieval
```

## Architecture

### Data flow
```
CDC URLs (src/ingest/cdc_urls.py)
  → scrape_cdc.py        → data/raw_html/*.html + data/docs/*.json
  → clean_cdc.py         → data/docs/cleaned/*.json
  → build_chunks.py      → data/docs/chunks/cdc_chunks.jsonl
  → build_embeddings_qdrant.py → Qdrant collection "cdc_guidelines"
```

At query time:
```
User question
  → src/rag/retrieve.py   (BAAI/bge-small-en-v1.5 embedding → Qdrant cosine search)
  → src/rag/pipeline.py   (confidence gate: drops if max score < 0.25)
  → src/rag/prompting.py  (builds numbered [1][2]... context block)
  → src/rag/generate.py   (Ollama llama3.1 at localhost:11434, temperature=0)
  → citation enforcement  (answer discarded if it contains no [N] citations)
```

### Key design decisions

- **Doc IDs** are `sha256(url)[:16]` — stable and collision-resistant across re-scrapes.
- **Chunking** (`src/rag/chunking.py`) is sentence-aware with overlap (60 tokens). Token estimates use whitespace-split, not a tokenizer — good enough for sizing.
- **Retrieval confidence gate** in `pipeline.py`: if the top Qdrant score is below 0.25, the pipeline short-circuits and returns a "don't know" response without calling the LLM.
- **Citation enforcement**: the pipeline strips the answer and returns "don't know" if the LLM response contains no `[N]` citations. This prevents hallucinated answers that bypass grounding.
- **LLM**: currently hardcoded to Ollama `llama3.1`. `openai` and `google-generativeai` are installed but not wired in — `src/rag/generate.py` is the single place to swap the backend.
- **Qdrant** runs locally via Docker with data persisted to `qdrant_data/` (mounted volume). Collection name is `cdc_guidelines`, vector dimension is 384 (BGE-small).

### Source locations
- CDC URLs to scrape: `src/ingest/cdc_urls.py`
- Chunking logic and `ChunkConfig`: `src/rag/chunking.py`
- System prompt and user prompt builder: `src/rag/prompting.py`
- Retrieval (embedding model + Qdrant client): `src/rag/retrieve.py`
- LLM call: `src/rag/generate.py`
- Full RAG pipeline (orchestration + citation logic): `src/rag/pipeline.py`
- Streamlit UI: `app.py`
