
import requests

from app.config import OLLAMA_URL, OLLAMA_MODEL


def ask_llm(prompt: str) -> str:

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0
            }
        },
        timeout=120
    )

    response.raise_for_status()

    data = response.json()

    return data["response"]
