# chatbot.py
#
# Responsibility:
#   Query routing, structured schedule search,
#   RAG semantic search, LLM answer generation.
#
# Strict memory-grounded behavior:
#   - Schedule questions use structured memory search.
#   - Direct teacher/topic questions use keyword search.
#   - Semantic questions use RAG.
#   - Unknown questions return "Information not found."
#   - No unrestricted LLM fallback.


import re
from datetime import datetime, timedelta

from rag import retrieve_context, TEST_MEMORIES
from llm import generate_answer, NOT_FOUND_RESPONSE


# --------------------------------------------------
# DYNAMIC DATES
# --------------------------------------------------

TODAY = datetime.today().date()
TOMORROW = TODAY + timedelta(days=1)
TOMORROW_STR = TOMORROW.strftime("%Y-%m-%d")


# --------------------------------------------------
# QUERY CLASSIFICATION
# --------------------------------------------------

SCHEDULE_TRIGGER_PATTERNS = [
    r'\bwhen is\b',
    r'\bwhat time\b',
    r'\bdo i have\b',
    r'\bwhat do i have\b',
    r'\banything tomorrow\b',
    r'\banything scheduled\b',
    r'\bplans tomorrow\b',
    r'\bhave tomorrow\b',
    r'\bconflict\b',
    r'\bbetween \d',
    r'\bat \d{1,2}\s*(am|pm)\b',
    r'\btomorrow at\b',
    r'\bis there\b',
    r'\bare there\b',
]


CROSS_MODE_PATTERNS = [
    r'\bis there\b',
    r'\bare there\b',
    r'\bdo i have\b',
    r'\bwhat do i have\b',
    r'\banything tomorrow\b',
    r'\banything scheduled\b',
    r'\bplans tomorrow\b',
    r'\bhave tomorrow\b',
    r'\bconflict\b',
    r'\bbetween \d',
    r'\bat \d{1,2}\s*(am|pm)\b',
    r'\btomorrow at\b',
]


SEMANTIC_TRIGGER_KEYWORDS = [
    "explain",
    "explained",
    "discuss",
    "discussed",
    "learn",
    "learned",
    "taught",
    "teach",
    "lecture",
    "covered",
    "remind",
    "what was",
    "what did",
    "tell me about",
    "what were",
    "neural",
    "concept",
    "lesson",
    "what is my",
    "what are my",

    # General knowledge questions are still routed through RAG.
    "what is",
    "what are",
    "how does",
    "how do",
    "how is",
    "how are",
    "define",
    "describe",
]


# --------------------------------------------------
# SYNONYMS / MEMORY TYPES
# --------------------------------------------------

TEACHER_SYNONYMS = {
    "teacher",
    "professor",
    "instructor",
    "tutor",
    "faculty",
}

LECTURER_SYNONYMS = {
    "lecturer",
}

TOPIC_SYNONYMS = {
    "neural network": [
        "neural network",
        "neural networks",
        "cnn",
        "convolutional",
        "deep learning",
    ],
    "quantum physics": [
        "quantum physics",
        "quantum",
    ],
    "physics": [
        "physics",
    ],
}

EXAM_WORDS = {
    "exam",
    "test",
    "quiz",
    "examination",
}

APPOINTMENT_WORDS = {
    "appointment",
    "doctor",
    "checkup",
    "check-up",
    "hospital",
}

MEETING_WORDS = {
    "meeting",
    "standup",
    "call",
    "sync",
}

ASSIGNMENT_WORDS = {
    "assignment",
    "homework",
    "task",
}


# --------------------------------------------------
# PATTERN HELPERS
# --------------------------------------------------

def _match_patterns(question: str, patterns: list) -> bool:
    q = question.lower()
    return any(re.search(pattern, q) for pattern in patterns)


def _is_schedule_query(question: str) -> bool:
    return _match_patterns(question, SCHEDULE_TRIGGER_PATTERNS)


def _is_cross_mode(question: str) -> bool:
    return _match_patterns(question, CROSS_MODE_PATTERNS)


def _is_semantic_query(question: str) -> bool:
    q = question.lower()
    return any(keyword in q for keyword in SEMANTIC_TRIGGER_KEYWORDS)


# --------------------------------------------------
# MEMORY TYPE DETECTION
# --------------------------------------------------

def _query_memory_types(question: str) -> set:
    q = question.lower()
    types = set()

    if any(word in q for word in EXAM_WORDS):
        types.add("exam")

    if any(word in q for word in APPOINTMENT_WORDS):
        types.add("appointment")

    if any(word in q for word in MEETING_WORDS):
        types.add("meeting")

    if any(word in q for word in ASSIGNMENT_WORDS):
        types.add("assignment")

    return types


# --------------------------------------------------
# TIME PARSING
# --------------------------------------------------

def _parse_time(time_str):
    if not time_str:
        return None

    try:
        return datetime.strptime(
            time_str.strip(),
            "%I:%M %p"
        ).time()
    except ValueError:
        return None


def _extract_target_time(question: str):
    match = re.search(
        r'\bat\s+(\d{1,2})\s*(am|pm)\b',
        question.lower()
    )

    if match:
        hour = int(match.group(1))
        meridiem = match.group(2)

        if meridiem == "pm" and hour != 12:
            hour += 12

        elif meridiem == "am" and hour == 12:
            hour = 0

        return datetime.strptime(
            f"{hour:02d}:00",
            "%H:%M"
        ).time()

    return None


def _extract_time_range(question: str):
    match = re.search(
        r'between\s+(\d{1,2})\s*(am|pm)\s+and\s+'
        r'(\d{1,2})\s*(am|pm)',
        question.lower()
    )

    if match:
        h1 = int(match.group(1))
        m1 = match.group(2)

        h2 = int(match.group(3))
        m2 = match.group(4)

        if m1 == "pm" and h1 != 12:
            h1 += 12

        if m1 == "am" and h1 == 12:
            h1 = 0

        if m2 == "pm" and h2 != 12:
            h2 += 12

        if m2 == "am" and h2 == 12:
            h2 = 0

        return (
            datetime.strptime(
                f"{h1:02d}:00",
                "%H:%M"
            ).time(),

            datetime.strptime(
                f"{h2:02d}:00",
                "%H:%M"
            ).time(),
        )

    return None, None


# --------------------------------------------------
# STRUCTURED SCHEDULE SEARCH
# --------------------------------------------------

def structured_schedule_search(
    question: str,
    current_mode: str,
    memories: list[dict]
) -> tuple[list[dict], bool]:

    q = question.lower()

    cross_mode = _is_cross_mode(question)

    target_time = _extract_target_time(question)

    range_start, range_end = _extract_time_range(question)

    asked_types = _query_memory_types(question)

    results = []

    for memory in memories:

        # ----------------------------------------------
        # MODE FILTER
        # ----------------------------------------------

        if not cross_mode:
            if memory["mode"] != current_mode:
                continue

        # ----------------------------------------------
        # MEMORY TYPE FILTER
        # ----------------------------------------------

        if asked_types:
            if memory["memory_type"] not in asked_types:
                continue

        # ----------------------------------------------
        # DATE FILTER
        # ----------------------------------------------

        mem_date = memory.get("date")

        if "tomorrow" in q:
            if mem_date != TOMORROW_STR:
                continue

        # ----------------------------------------------
        # EXACT TIME FILTER
        # ----------------------------------------------

        mem_time = _parse_time(
            memory.get("time")
        )

        if target_time and not range_start:
            if mem_time != target_time:
                continue

        # ----------------------------------------------
        # TIME RANGE FILTER
        # ----------------------------------------------

        if range_start and range_end:

            if mem_time is None:
                continue

            if not (
                range_start <= mem_time <= range_end
            ):
                continue

        results.append(memory)

    # Sort by time
    results.sort(
        key=lambda memory: (
            _parse_time(memory.get("time"))
            or datetime.min.time()
        )
    )

    return results, cross_mode


# --------------------------------------------------
# FORMAT SCHEDULE RESULTS
# --------------------------------------------------

def _format_schedule_results(
    results: list,
    cross_mode: bool
) -> str:

    if not results:
        return NOT_FOUND_RESPONSE

    lines = []

    for memory in results:

        if cross_mode:
            lines.append(
                f"[{memory['mode']}] {memory['text']}"
            )
        else:
            lines.append(
                memory["text"]
            )

    return "\n".join(lines)


# --------------------------------------------------
# KEYWORD SEARCH
# --------------------------------------------------

def _keyword_search(
    question: str,
    current_mode: str,
    memories: list[dict]
) -> list[dict]:

    q = question.lower()

    # Lecturer is intentionally unsupported.
    if any(
        synonym in q
        for synonym in LECTURER_SYNONYMS
    ):
        return []

    results = []

    for memory in memories:

        # ----------------------------------------------
        # CURRENT MODE ONLY
        # ----------------------------------------------

        if memory["mode"] != current_mode:
            continue

        text_lower = memory["text"].lower()

        matched = False

        # ----------------------------------------------
        # TEACHER / PROFESSOR MATCH
        # ----------------------------------------------

        for synonym in TEACHER_SYNONYMS:

            if (
                synonym in q
                and synonym in text_lower
            ):
                matched = True
                break

        # ----------------------------------------------
        # EXACT TOPIC VARIANT MATCH
        #
        # The SAME variant must occur in both the
        # question and the memory.
        # ----------------------------------------------

        if not matched:

            for variants in TOPIC_SYNONYMS.values():

                for variant in variants:

                    if (
                        variant in q
                        and variant in text_lower
                    ):
                        matched = True
                        break

                if matched:
                    break

        if matched:
            results.append(memory)

    return results


# --------------------------------------------------
# FORMAT KEYWORD ANSWER
# --------------------------------------------------

def _format_keyword_answer(results: list) -> str:

    if not results:
        return NOT_FOUND_RESPONSE

    return "\n".join(
        memory["text"]
        for memory in results
    )


# --------------------------------------------------
# MAIN CHAT FUNCTION
# --------------------------------------------------

def chat(
    question: str,
    current_mode: str,
    use_test_memories: bool = False
) -> str:

    """
    Route question to:

    1. Structured schedule search
    2. Keyword match
    3. RAG + NVIDIA LLM
    4. Strict NOT_FOUND response

    No unrestricted LLM fallback.
    """

    question = question.strip()

    if not question:
        return NOT_FOUND_RESPONSE

    # ----------------------------------------------
    # SELECT MEMORY SOURCE
    # ----------------------------------------------

    if use_test_memories:
        memories = TEST_MEMORIES

    else:
        from memory_store import get_all_memories
        memories = get_all_memories()

    q_lower = question.lower()

    # ==================================================
    # ROUTE 1: STRUCTURED SCHEDULE QUERY
    # ==================================================

    if _is_schedule_query(question):

        results, cross_mode = structured_schedule_search(
            question,
            current_mode,
            memories
        )

        return _format_schedule_results(
            results,
            cross_mode
        )

    # ==================================================
    # ROUTE 2: SEMANTIC QUERY
    # ==================================================

    if _is_semantic_query(question):

        # Lecturer intentionally unsupported.
        if any(
            synonym in q_lower
            for synonym in LECTURER_SYNONYMS
        ):
            return NOT_FOUND_RESPONSE

        # ----------------------------------------------
        # DIRECT KEYWORD MATCH
        # ----------------------------------------------

        keyword_results = _keyword_search(
            question,
            current_mode,
            memories
        )

        if keyword_results:
            return _format_keyword_answer(
                keyword_results
            )

        # ----------------------------------------------
        # RAG SEARCH
        # ----------------------------------------------

        retrieved, context_str = retrieve_context(
            question,
            current_mode,
            use_test_memories=use_test_memories
        )

        if not retrieved:
            return NOT_FOUND_RESPONSE

        return generate_answer(
            question,
            context_str
        )

    # ==================================================
    # ROUTE 3: UNKNOWN / UNSUPPORTED QUESTION
    #
    # IMPORTANT:
    # Never send this to a general LLM.
    # This prevents hallucinations and visible
    # "thinking process" output.
    # ==================================================

    return NOT_FOUND_RESPONSE