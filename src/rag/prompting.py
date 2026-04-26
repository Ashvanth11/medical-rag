SYSTEM_PROMPT = """You are a medical question-answering assistant.

CRITICAL RULES:
- Answer ONLY using the provided medical excerpts.
- Do NOT use outside knowledge.
- If the excerpts contain information that directly answers the question, provide the answer using only that information.
- You do NOT need to provide a complete or comprehensive explanation unless it appears in the excerpts.
- Only respond with:
  I don't know based on the available medical content.
  if the excerpts contain NO information relevant to the question.
- Do NOT add commentary, notes, or disclaimers.
- Every factual sentence MUST include a citation like [1], [2], etc.
- Do NOT include any text after your final answer.

"""


def build_user_prompt(question: str, retrieved_chunks: list[str]) -> str:
    formatted_chunks = []

    for i, chunk in enumerate(retrieved_chunks, start=1):
        formatted_chunks.append(f"[{i}]\n{chunk}\n")

    context_block = "\n".join(formatted_chunks)

    return f"""Question:
{question}

Medical Excerpts:
{context_block}

Answer the question using only the medical excerpts.
Include citations after every factual sentence.
If the excerpts contain no relevant information, say you don't know.
"""
