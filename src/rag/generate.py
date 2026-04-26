from __future__ import annotations

import os
import requests

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")
MODEL = "llama3.1"


def generate_answer(system_prompt: str, user_prompt: str, history: list[dict] | None = None) -> str:
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history or [])
    messages.append({"role": "user", "content": user_prompt})

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.0
            }
        }
    )

    response.raise_for_status()
    return response.json()["message"]["content"].strip()
