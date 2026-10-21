# llm.py

from dotenv import load_dotenv
from openai import OpenAI
import os
import re

load_dotenv("../.env")

API_KEY = os.getenv("NVIDIA_API_KEY")

_client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=API_KEY
)

MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"

NOT_FOUND_RESPONSE = "I don't have enough information to answer that"

# --------------------------------------------------
# THINKING BLOCK STRIPPER
# --------------------------------------------------

def _extract_final_answer(raw: str) -> str:
    if not raw:
        return ""

    # Strip <think>...</think> blocks
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    lines = cleaned.split("\n")

    # Find last numbered reasoning line e.g. "1.  **...**" or "2.  Check..."
    last_reasoning_line = -1
    for i, line in enumerate(lines):
        if re.match(r"^\d+\.\s", line.strip()):
            last_reasoning_line = i

    if last_reasoning_line >= 0:
        rest = lines[last_reasoning_line + 1:]
        final_lines = []
        in_answer = False
        for line in rest:
            s = line.strip()
            if not s and not in_answer:
                continue
            if re.match(r"^[-*•]", s) and not in_answer:
                continue
            if line.startswith("   ") and not in_answer:
                continue
            in_answer = True
            final_lines.append(line)
        if final_lines:
            cleaned = "\n".join(final_lines).strip()

    # Strip preamble phrases
    preamble_patterns = [
        r"^here'?s? (?:a |my )?(?:thinking|reasoning|analysis|thought).*?:\s*",
        r"^let me (?:think|analyze|reason).*?:\s*",
        r"^thinking.*?:\s*",
    ]
    for pattern in preamble_patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE | re.DOTALL).strip()

    # If still long, take last short paragraph
    if len(cleaned) > 300:
        paragraphs = [p.strip() for p in cleaned.split("\n\n") if p.strip()]
        if paragraphs:
            last_para = paragraphs[-1]
            if len(last_para) < 200:
                cleaned = last_para

    return cleaned.strip()


NOT_FOUND_PATTERNS = [
    r"information not found",
    r"\bnot found\b",
    r"i (?:don't|do not|couldn't|could not) (?:find|have|know|answer)",
    r"(?:no|not enough) information",
    r"(?:the )?memories do not contain",
    r"(?:the )?provided memories do not",
    r"(?:cannot|can't) answer",
    r"(?:no|nothing) (?:relevant|matching|related)",
    r"no (?:direct )?answer",
]


def _is_not_found(text: str) -> bool:
    t = text.lower()
    for pattern in NOT_FOUND_PATTERNS:
        if re.search(pattern, t):
            return True
    return False


# --------------------------------------------------
# RAG SYSTEM PROMPT
# --------------------------------------------------

RAG_SYSTEM_PROMPT = (
    "You are the SENSEA memory assistant.\n"
    "You will be given retrieved memories and a user question.\n\n"
    "STRICT RULES:\n"
    "1. Answer ONLY using the exact content of the provided retrieved memories.\n"
    "2. Do NOT use any outside knowledge.\n"
    "3. Do NOT invent, guess, or fabricate any information.\n"
    "4. Do NOT infer topics not explicitly mentioned in the memories.\n"
    "5. If the memories do not contain a direct answer, respond with exactly:\n"
    "   I don't have enough information to answer that\n"
    "6. Do NOT output your reasoning, thinking steps, or analysis.\n"
    "7. Output ONLY the final answer — one sentence maximum.\n"
    "8. A memory about neural networks does NOT answer questions about "
    "quantum physics, biology, history, or any other unrelated topic.\n"
    "9. A memory about a lesson does NOT answer a question about an exam "
    "unless the memory explicitly mentions an exam.\n\n"
    "OUTPUT FORMAT:\n"
    "One sentence answer from memory content, or: I don't have enough information to answer that\n"
    "No preamble. No explanation. No reasoning. Just the answer."
)


def generate_answer(question: str, context: str) -> str:
    if not context:
        return NOT_FOUND_RESPONSE

    user_content = (
        f"{context}\n"
        f"User question: {question}\n\n"
        f"Reply in ONE sentence only. No thinking. No steps. Just the answer."
    )

    try:
        response = _client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": RAG_SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ],
            max_tokens=80,
            temperature=0.0,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        raw = response.choices[0].message.content.strip()

        if not raw or len(raw) < 2:
            return NOT_FOUND_RESPONSE

        answer = _extract_final_answer(raw)

        if not answer:
            return NOT_FOUND_RESPONSE

        if _is_not_found(answer):
            return NOT_FOUND_RESPONSE

        return answer

    except Exception:
        return NOT_FOUND_RESPONSE


FALLBACK_SYSTEM_PROMPT = (
    "You are the SENSEA assistant. "
    "Answer the user's question helpfully in one sentence."
)


def generate_fallback(question: str, mode: str) -> str:
    try:
        response = _client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": FALLBACK_SYSTEM_PROMPT},
                {"role": "user", "content": question}
            ],
            max_tokens=80,
            temperature=0.5,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return NOT_FOUND_RESPONSE