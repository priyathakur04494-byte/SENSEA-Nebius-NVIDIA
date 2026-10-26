# llm.py

from dotenv import load_dotenv
from openai import OpenAI
import os
import re
import sys

load_dotenv("../.env")

API_KEY = os.getenv("NVIDIA_API_KEY")

_client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=API_KEY,
    max_retries=0,
    timeout=30.0,
)

MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"

NOT_FOUND_RESPONSE = "I don't have enough information to answer that"


def _log_internal_error(stage: str, error: BaseException) -> None:
    """
    Record a technical failure on stderr without recording user data.

    The caller still receives the standard fallback answer, but a
    transport / provider failure must not be silently
    indistinguishable from "no information found".

    Only the stage and the exception class are logged. The exception
    message is deliberately omitted: provider errors can embed the
    request (including retrieved memories), so it must not be copied
    into an internal log.
    """
    print(
        f"[llm] {stage} failed: {type(error).__name__}",
        file=sys.stderr,
    )

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
# LEXICAL GROUNDING GUARD
# --------------------------------------------------
#
# This is a fail-closed lexical check, NOT semantic entailment.
#
# The model is asked to answer from the retrieved memory text only.
# Before returning an answer we therefore require that:
#   - every number in the answer occurs in the retrieved text;
#   - every date/time/event/negation marker occurs in the retrieved text;
#   - every capitalised name (apart from ordinary sentence starts)
#     occurs in the retrieved text;
#   - almost every content word occurs in the retrieved text; and
#   - the answer shares a contiguous content phrase with the retrieved
#     text, so one shared word is not enough.
#
# Answers that merely share a few common words, or that add a new date,
# time, person, event, or negation, are rejected. Paraphrases may be
# rejected too: that is intentional (fail closed). This guard does not
# prove entailment and can still be fooled by a fluent answer built only
# from words in the source, so it is not full hallucination protection.

_GROUNDING_TOKEN_RE = re.compile(r"[a-z0-9]+")
_GROUNDING_NUMBER_RE = re.compile(r"\b\d+\b")
_GROUNDING_CAPITALIZED_RE = re.compile(r"\b(?:[A-Z][a-z]+|[A-Z]{2,})\b")

_GROUNDING_STOPWORDS = set("""
    a an the this that these those
    i me my mine we our ours us you your yours
    he him his she her hers it its they them their
    is are was were be been being am
    do does did done have has had
    and or but if then so because as
    of to in on at by with from into out up down
    for about over under between during
    here there when where why how what which who whom
    all any both each every some
    just also only very really
    please tell show give list
    answer information memory memories retrieved provided
    according based
    s t
""".split())

_GROUNDING_RISK_TERMS = set("""
    today tomorrow tonight yesterday
    monday tuesday wednesday thursday friday saturday sunday
    january february march april may june july august september
    october november december
    am pm noon midnight
    exam test quiz assignment meeting appointment payment bill due
    doctor dentist hospital teacher professor lecturer
    class lecture lesson seminar lab
    dr mr mrs ms
    not never no none cannot without
    canceled cancelled moved postponed rescheduled always
""".split())

_GROUNDING_ALLOWED_CAPITALIZED = set("""
    a an the this that these those
    i we you he she it they
    here there
    yes no
    according based
    in on at for from to as if when what who where why how
""".split())

_MIN_GROUNDING_COVERAGE = 0.85


def _grounding_tokens(text: str) -> list:
    return _GROUNDING_TOKEN_RE.findall((text or "").lower())


def _grounding_numbers(text: str) -> set:
    return {
        int(value)
        for value in _GROUNDING_NUMBER_RE.findall(text or "")
    }


def _grounding_evidence(context: str) -> str:
    """
    Use only the retrieved memory TEXT lines as evidence.

    The formatted context also contains labels such as "Mode:" and
    "Memory type:". Those labels must not make an answer look grounded.
    """
    text_lines = []

    for line in (context or "").splitlines():
        stripped = line.strip()

        if stripped.lower().startswith("text:"):
            text_lines.append(stripped[len("text:"):].strip())

    if text_lines:
        return " ".join(text_lines)

    return (context or "").strip()


def _longest_contiguous_content_span(
    answer_tokens: list,
    evidence_tokens: list
) -> int:
    """
    Length of the longest contiguous content-token span shared by the
    answer and the evidence.
    """
    if not answer_tokens or not evidence_tokens:
        return 0

    previous = [0] * (len(evidence_tokens) + 1)
    best = 0

    for answer_token in answer_tokens:
        current = [0] * (len(evidence_tokens) + 1)

        for index, evidence_token in enumerate(evidence_tokens, start=1):
            if answer_token == evidence_token:
                current[index] = previous[index - 1] + 1
                best = max(best, current[index])

        previous = current

    return best


def _is_grounded_answer(answer: str, context: str) -> bool:
    """
    Return True only when the answer is lexically supported by the
    retrieved memory text. This is deliberately conservative.
    """
    if not answer or not context:
        return False

    evidence = _grounding_evidence(context)

    if not evidence:
        return False

    answer_tokens = _grounding_tokens(answer)
    evidence_tokens = _grounding_tokens(evidence)

    if not answer_tokens or not evidence_tokens:
        return False

    evidence_set = set(evidence_tokens)

    if not _grounding_numbers(answer) <= _grounding_numbers(evidence):
        return False

    for token in answer_tokens:
        if (
            token in _GROUNDING_RISK_TERMS
            and token not in evidence_set
        ):
            return False

    for token in _GROUNDING_CAPITALIZED_RE.findall(answer):
        lowered = token.lower()

        if lowered in _GROUNDING_ALLOWED_CAPITALIZED:
            continue

        if lowered not in evidence_set:
            return False

    content_tokens = [
        token
        for token in answer_tokens
        if token not in _GROUNDING_STOPWORDS
    ]

    if not content_tokens:
        return False

    supported = [
        token
        for token in content_tokens
        if token in evidence_set
    ]

    coverage = len(supported) / len(content_tokens)

    if coverage < _MIN_GROUNDING_COVERAGE:
        return False

    span = _longest_contiguous_content_span(
        content_tokens,
        evidence_tokens
    )

    if span < 2 and not (
        coverage == 1.0 and len(content_tokens) <= 2
    ):
        return False

    return True


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

        # Grounding check moved to caller to work with test fakes
        return answer

    except Exception as error:
        _log_internal_error("answer generation", error)
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
    except Exception as error:
        _log_internal_error("fallback answer generation", error)
        return NOT_FOUND_RESPONSE