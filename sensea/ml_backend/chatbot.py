# chatbot.py
#
# Responsibility:
#   Query routing, structured schedule search,
#   RAG semantic search, LLM answer generation.
#
# Strict memory-grounded behavior:
#   - Schedule questions use structured memory search across modes.
#   - Direct teacher/topic questions use keyword search.
#   - Semantic questions use RAG.
#   - Unknown questions return "Information not found."
#   - No unrestricted LLM fallback.


import re
import sys
from datetime import datetime, timedelta

from rag import retrieve_context, TEST_MEMORIES
from llm import NOT_FOUND_RESPONSE

import llm


# --------------------------------------------------
# DYNAMIC DATES
# --------------------------------------------------

TODAY = datetime.today().date()
TOMORROW = TODAY + timedelta(days=1)
TOMORROW_STR = TOMORROW.strftime("%Y-%m-%d")
TODAY_STR = TODAY.strftime("%Y-%m-%d")


# --------------------------------------------------
# ERROR LOGGING
# --------------------------------------------------

def _log_internal_error(stage: str, error: BaseException) -> None:
    """
    Record a technical failure without recording user data.

    The user still receives the standard fallback answer, but a
    technical failure must not be indistinguishable from "no
    information found", so it is written to stderr.

    Only the stage and the exception class are logged. The exception
    message is deliberately omitted: provider and library exceptions can
    embed prompt or memory text, so it must not be copied into an
    internal log.
    """
    print(
        f"[chatbot] {stage} failed: {type(error).__name__}",
        file=sys.stderr,
    )


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
    # A time range must look like a time. A bare "between 2 ..." also
    # occurs in ordinary questions ("the difference between 2 neural
    # network types") and must not trigger the cross-mode calendar.
    r'\bbetween \d{1,2}(?::\d{2})?\s*(?:am|pm)\b',
    r'\bat \d{1,2}\s*(am|pm)\b',
    r'\btomorrow at\b',
    # "Is there / are there" is a schedule trigger only when the question
    # actually names a schedule item. A bare "is there anything about
    # neural networks?" is an ordinary conversation question and must not
    # be answered from the cross-mode calendar.
    r'\b(?:is|are) there\b[^?.!]*\b(?:events?|appointments?|meetings?'
    r'|exams?|tests?|classes?|lectures?|plans?|calendar|scheduled|schedule)\b',
]


# --------------------------------------------------
# DATE PARSING
# --------------------------------------------------
#
# The schedule filter used to recognise ONE date: the literal word
# "tomorrow". "today", "yesterday" and explicit dates ("September 9",
# "2026-10-30") were ignored, so those questions answered with events
# from the wrong day.
#
# A question that names a date is now filtered to EXACTLY that date.

MONTH_NUMBERS = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}

RELATIVE_DAYS = {
    "today": 0,
    "tomorrow": 1,
    "yesterday": -1,
}

_MONTH_PATTERN = "|".join(
    sorted(MONTH_NUMBERS, key=len, reverse=True)
)

_ISO_DATE_RE = re.compile(
    r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b"
)

_MONTH_DAY_RE = re.compile(
    rf"\b({_MONTH_PATTERN})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?\b",
    re.IGNORECASE,
)

_DAY_MONTH_RE = re.compile(
    rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTH_PATTERN})\b",
    re.IGNORECASE,
)

_RELATIVE_DAY_RE = re.compile(
    r"\b(today|tomorrow|yesterday)\b",
    re.IGNORECASE,
)


def _format_date(year: int, month: int, day: int):
    try:
        return datetime(year, month, day).strftime("%Y-%m-%d")
    except ValueError:
        return None


def _extract_requested_date(question: str):
    """
    Return the date ("YYYY-MM-DD") the question asks about, or None.

    Handles:
      - relative words: today / tomorrow / yesterday
      - explicit ISO dates: 2026-10-30, 2026/10/30
      - written dates: "September 9", "9 September", "Sept 9"
    """
    q = question.lower()

    match = _ISO_DATE_RE.search(q)
    if match:
        year, month, day = (
            int(part) for part in match.groups()
        )
        return _format_date(year, month, day)

    match = _RELATIVE_DAY_RE.search(q)
    if match:
        offset = RELATIVE_DAYS[match.group(1).lower()]
        return (TODAY + timedelta(days=offset)).strftime("%Y-%m-%d")

    match = _MONTH_DAY_RE.search(q)
    if match:
        month = MONTH_NUMBERS[match.group(1).lower()]
        day = int(match.group(2))
        return _format_date(TODAY.year, month, day)

    match = _DAY_MONTH_RE.search(q)
    if match:
        day = int(match.group(1))
        month = MONTH_NUMBERS[match.group(2).lower()]
        return _format_date(TODAY.year, month, day)

    return None


# --------------------------------------------------
# CROSS-MODE SCHEDULE LOOKUP
# --------------------------------------------------
#
# The user's own schedule spans every mode: an appointment is stored
# by a hospital conversation, a meeting by a meeting conversation, an
# exam by a classroom conversation. A SCHEDULE question is therefore
# answered from all four modes:
#
#   "What do I have tomorrow?"
#   "What do I have between 2 PM and 4 PM?"
#   "Do I have anything scheduled tomorrow?"
#   "When is my doctor appointment?"     (named event, no date)
#   "When is my meeting?"                (named event, no date)
#   "Do I have any pending payment tomorrow?"
#
# The question still decides WHAT is returned:
#
#   - a named event keeps the memory type filter and must mention the
#     event the question names, so "When is my physics exam?" never
#     answers with a maths test just because both are exam memories,
#   - a named date is always applied as an exact filter, so a "today"
#     question can only ever return memories dated today and a date
#     without events returns the standard fallback instead of another
#     day's events,
#   - an exact time / time range keeps its filter.
#
# Ordinary conversation is NOT affected. It never reaches the schedule
# route, so it is still answered from the active mode only:
#
#   "What did my doctor tell me?"          -> active mode only
#   "What did my teacher explain?"         -> active mode only
#   "What did we learn about neural networks?" -> active mode only

ENUMERATION_PATTERNS = [
    r'\bwhat do i have\b',
    r'\bdo i have\s+(?:any|anything|some)\b',
    # Bare "anything" is intentionally absent: on its own it is not a
    # request for the calendar and caused ordinary questions such as
    # "Is there anything about neural networks?" to enumerate every
    # timed event across modes.
    r'\bplans?\b',
    r'\bconflict\b',
    r'\bschedul(?:e|ed|es|ing)\b',
    r'\bevents?\b',
    r'\ball (?:my|the)\b',
    r'\beverything\b',
]

TIME_QUALIFIER_PATTERNS = [
    r'\bat \d{1,2}\s*(am|pm)\b',
    r'\bbetween \d{1,2}(?::\d{2})?\s*(?:am|pm)\b',
    r'\bthis (?:morning|afternoon|evening|night)\b',
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
# EVENT NAME DETECTION
# --------------------------------------------------
#
# The memory type is not specific enough on its own: "exam" covers both
# a maths test and a physics exam. A schedule question that names the
# event ("When is my physics exam?") must therefore match the named
# event as well as the memory type.
#
# The words that only describe the question or the calendar entry are
# ignored, and so are the interchangeable type labels (exam/test,
# meeting, assignment). The appointment words "doctor", "hospital" and
# "checkup" are kept: they name a specific appointment, not just its
# type. Every other word is an event word.

_EVENT_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _normalize_event_token(token: str) -> str:
    """
    Fold a simple plural so "math" and "maths" compare equal.
    """
    if (
        len(token) > 3
        and token.endswith("s")
        and not token.endswith("ss")
    ):
        return token[:-1]

    return token


def _event_tokens(text: str) -> set:
    return {
        _normalize_event_token(token)
        for token in _EVENT_TOKEN_RE.findall(
            (text or "").lower()
        )
    }


TYPE_LABEL_WORDS = {
    token
    for type_word in (
        EXAM_WORDS
        | MEETING_WORDS
        | ASSIGNMENT_WORDS
        | {"appointment"}
    )
    for token in re.findall(
        r"[a-z0-9]+",
        type_word.lower()
    )
}

# The appointment words "doctor", "hospital" and "checkup" are NOT type
# labels: they name a specific appointment and stay event words. These
# groups let "checkup" find a "Doctor appointment" without letting a
# doctor appointment find a dentist appointment.
EVENT_NAME_SYNONYMS = [
    {
        "doctor",
        "physician",
        "medical",
        "clinic",
        "hospital",
        "checkup",
        "check",
    },
    {
        "dentist",
        "dental",
    },
]

_EVENT_STOPWORD_WORDS = """
    when what which who whose where why how
    is are was were be been being
    do does did done have has had
    i me my mine we our ours us you your yours he she it its
    they them their
    a an the this that these those there here
    any anything something some all both each every
    no not yes please tell show give list
    about for of to in on at by with from into out up down
    over under and or if then so just also only exactly
    between before after until during around near
    next last upcoming pending due
    scheduled schedule event plan reminder time conflict calendar
    day week month year today tomorrow tonight yesterday
    morning afternoon evening night am pm noon midnight
    o clock hour minute
    first second third final midterm term paper module
    subject unit course class lecture lab lesson seminar
""".split()

EVENT_STOPWORDS = {
    _normalize_event_token(word)
    for word in _EVENT_STOPWORD_WORDS
}


def _extract_event_terms(question: str) -> set:
    """
    Return the words that name the requested event.

    "When is my physics exam?" -> {"physic"}
    "When is my team meeting?" -> {"team"}
    "When is my exam?"         -> set()
    """
    return {
        token
        for token in _event_tokens(question)
        if len(token) > 1
        and not token.isdigit()
        and token not in EVENT_STOPWORDS
        and token not in TYPE_LABEL_WORDS
    }


def _terms_match(memory_tokens: set, question_terms: set, groups) -> bool:
    for group in groups:
        group_tokens = set()

        for word in group:
            group_tokens |= _event_tokens(word)

        if (
            question_terms & group_tokens
            and group_tokens & memory_tokens
        ):
            return True

    return False


def _memory_matches_event_terms(memory: dict, terms: set) -> bool:
    """
    True when a memory mentions one of the requested event words.

    The mode name counts as part of the event, so "Do I have a hospital
    appointment?" still matches an appointment stored in the hospital
    mode even when its text says only "Doctor appointment".
    """
    if not terms:
        return True

    text_tokens = _event_tokens(memory.get("text", ""))

    mode_tokens = _event_tokens(memory.get("mode", ""))

    if terms & (text_tokens | mode_tokens):
        return True

    # Synonyms only ever match the memory text. Otherwise the mode name
    # ("hospital") would make every appointment in that mode match every
    # medical synonym ("checkup").
    return _terms_match(
        text_tokens,
        terms,
        list(TOPIC_SYNONYMS.values()) + EVENT_NAME_SYNONYMS
    )


# --------------------------------------------------
# PAYMENT REMINDERS
# --------------------------------------------------
#
# A payment reminder is stored with the memory type that is already
# established for reminders: "reminder". That type carries no payment
# flag, so a payment reminder is told apart from another reminder by
# its text. A memory that is already typed as a payment ("payment") is
# matched on its type and needs no payment word.
#
# A payment question therefore only matches reminder memories whose
# text mentions a payment, plus memories typed as a payment. That
# keeps unrelated reminders (for example "Reminder to buy groceries.")
# out of a payment answer, and a reminder that is not in the store is
# never invented.

PAYMENT_MEMORY_TYPES = {
    "reminder",
    "payment",
}

PAYMENT_PATTERNS = [
    r'\bpayments?\b',
    r'\bbills?\b',
    r'\binvoices?\b',
    r'\bfees?\b',
    r'\bdues\b',
    r'\brent\b',
    r'\binstallments?\b',
]


# --------------------------------------------------
# PATTERN HELPERS
# --------------------------------------------------

def _match_patterns(question: str, patterns: list) -> bool:
    q = question.lower()
    return any(re.search(pattern, q) for pattern in patterns)


def _is_schedule_query(question: str) -> bool:
    return _match_patterns(question, SCHEDULE_TRIGGER_PATTERNS)


def _is_enumeration(question: str) -> bool:
    """
    True for a question that enumerates the whole schedule instead of
    asking about one named kind of event.
    """
    return _match_patterns(question, ENUMERATION_PATTERNS)


def _has_time_qualifier(question: str) -> bool:
    """
    True for a question that pins the lookup to a time of day.
    """
    return _match_patterns(question, TIME_QUALIFIER_PATTERNS)


def _mentions_payment(text: str) -> bool:
    """
    True when a text talks about a payment.
    """
    return _match_patterns(text or "", PAYMENT_PATTERNS)


def _is_payment_query(question: str) -> bool:
    """
    True for a question that asks about a payment reminder.

    A payment word alone is not enough: "What did Bill explain?" is a
    conversation question, not a payment lookup. Only a question that
    also asks for the schedule (a schedule trigger, a date or a time
    of day) is treated as a payment lookup.
    """
    if not _mentions_payment(question):
        return False

    if _is_schedule_query(question):
        return True

    if _extract_requested_date(question):
        return True

    if _has_time_qualifier(question):
        return True

    return False


def _is_cross_mode(question: str, asked_types: set = None) -> bool:
    """
    Decide whether the schedule search may look beyond the active mode.

    The schedule of the user is one calendar that spans the General,
    Classroom, Meeting and Hospital modes, so a schedule question may
    look beyond the active mode when it is

      - scoped by a date ("... tomorrow", "... on September 9"),
      - scoped by a time of day ("... at 3 PM", "... between 2 PM and 4 PM"),
      - asking for one named kind of event ("When is my doctor
        appointment?", "When is my meeting?"), or
      - enumerating the schedule ("Do I have anything scheduled?").

    The question still selects WHAT is returned: the memory type
    filter, the date filter and the time filters are applied exactly
    as they are inside the active mode.

    This is only about the schedule route. Ordinary conversation
    questions never reach it and stay inside the active mode.
    """
    if _extract_requested_date(question):
        return True

    if _has_time_qualifier(question):
        return True

    if asked_types:
        return True

    if _is_enumeration(question):
        return True

    return False


def _is_semantic_query(question: str) -> bool:
    q = question.lower()
    return any(keyword in q for keyword in SEMANTIC_TRIGGER_KEYWORDS)


def _is_topic_scoped_schedule_question(question: str) -> bool:
    """
    True when a would-be schedule question actually asks about a named
    topic but does not name a supported schedule type.

    Examples:
      "Do I have any notes about neural networks?"  -> True
      "What do I have about neural networks?"       -> True
      "Do I have anything scheduled?"               -> False
      "Do I have a maths exam tomorrow?"            -> False (exam type)

    Such questions belong to the conversation routes. Letting them reach
    the calendar route used to enumerate every timed event across modes,
    leaking other modes' schedule items for an ordinary topic question.
    """
    return (
        bool(_extract_event_terms(question))
        and not _query_memory_types(question)
    )


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

    asked_types = _query_memory_types(question)

    event_terms = _extract_event_terms(question)

    requested_date = _extract_requested_date(question)

    enumeration = _is_enumeration(question)

    payment_query = _is_payment_query(question)

    # A payment reminder is a scheduled item of the user, so it may
    # be looked up in every mode.
    cross_mode = (
        _is_cross_mode(question, asked_types)
        or payment_query
    )

    target_time = _extract_target_time(question)

    range_start, range_end = _extract_time_range(question)

    results = []

    for memory in memories:

        # ----------------------------------------------
        # MODE FILTER
        # ----------------------------------------------

        if not cross_mode:
            if memory.get("mode") != current_mode:
                continue

        # ----------------------------------------------
        # MEMORY TYPE FILTER
        # ----------------------------------------------

        if payment_query:

            # ------------------------------------------
            # PAYMENT FILTER
            #
            # Only a reminder that actually mentions a
            # payment can answer a payment question.
            # An unrelated reminder is never returned,
            # and a reminder is never invented.
            #
            # A memory that is already typed as a
            # payment does not need the text check.
            # ------------------------------------------

            if memory.get("memory_type") not in PAYMENT_MEMORY_TYPES:
                continue

            if (
                memory.get("memory_type") != "payment"
                and not _mentions_payment(memory.get("text", ""))
            ):
                continue

        elif asked_types:
            if memory.get("memory_type") not in asked_types:
                continue

            # ------------------------------------------
            # EVENT NAME FILTER
            #
            # The memory type alone is not specific
            # enough: a maths test and a physics exam
            # are both "exam" memories. When the
            # question names the event, the memory must
            # mention it too.
            # ------------------------------------------

            if not _memory_matches_event_terms(memory, event_terms):
                continue

        # ----------------------------------------------
        # DATE FILTER
        #
        # A question that names a date is filtered to EXACTLY that
        # date. Memories without a date never match a dated
        # question.
        # ----------------------------------------------

        if requested_date:
            if memory.get("date") != requested_date:
                continue

        # ----------------------------------------------
        # SCHEDULED EVENT FILTER
        #
        # "Do I have anything scheduled today?" asks for calendar
        # entries, not for a record of what was said. A memory
        # without a time is not a scheduled event, so a lesson or a
        # conversation record is never enumerated (cross-mode or not).
        #
        # A payment reminder is the exception: a payment that is
        # due on a day has no time of day.
        # ----------------------------------------------

        if enumeration and not payment_query:
            if not memory.get("time"):
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
                f"[{memory.get('mode')}] {memory.get('text', '')}"
            )
        else:
            lines.append(
                memory.get("text", "")
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

        if memory.get("mode") != current_mode:
            continue

        text_lower = memory.get("text", "").lower()

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
        memory.get("text", "")
        for memory in results
    )


# --------------------------------------------------
# MAIN CHAT FUNCTION
# --------------------------------------------------

def chat(
    question: str,
    current_mode: str,
    use_test_memories: bool = False,
    user_id: str | None = None
) -> str:

    """
    Route question to:

    1. Structured schedule search
    2. Keyword match
    3. RAG + NVIDIA LLM
    4. Strict NOT_FOUND response

    No unrestricted LLM fallback.

    user_id is an optional memory owner scope. It must come from a
    trusted authentication layer, never from the question or another
    arbitrary client field. None preserves the legacy unscoped
    prototype behaviour; it does NOT provide multi-user isolation.
    The use_test_memories fixture path is unscoped by design.
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
        try:
            from memory_store import get_all_memories

            # Keep the legacy no-argument call for unscoped callers and
            # existing test doubles; add the scope only when supplied.
            if user_id is None:
                memories = get_all_memories()
            else:
                memories = get_all_memories(user_id=user_id)
        except Exception as error:
            _log_internal_error("memory store read", error)
            return NOT_FOUND_RESPONSE

    q_lower = question.lower()

    # ==================================================
    # ROUTE 1: STRUCTURED SCHEDULE QUERY
    #
    # A payment reminder is part of the schedule, so a payment
    # question is routed here as well.
    #
    # A question that names a topic without naming a supported
    # schedule type ("Do I have any notes about neural networks?")
    # is a conversation question even if it starts with a schedule
    # phrase. It must not enumerate the calendar across modes.
    # ==================================================

    payment_query = _is_payment_query(question)

    is_schedule_route = _is_schedule_query(question) or payment_query

    if (
        is_schedule_route
        and not payment_query
        and _is_topic_scoped_schedule_question(question)
    ):
        is_schedule_route = False

    if is_schedule_route:

        try:
            results, cross_mode = structured_schedule_search(
                question,
                current_mode,
                memories
            )
        except Exception as error:
            _log_internal_error("schedule search", error)
            return NOT_FOUND_RESPONSE

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

        try:
            keyword_results = _keyword_search(
                question,
                current_mode,
                memories
            )
        except Exception as error:
            _log_internal_error("keyword search", error)
            return NOT_FOUND_RESPONSE

        if keyword_results:
            return _format_keyword_answer(
                keyword_results
            )

        # ----------------------------------------------
        # RAG SEARCH
        # ----------------------------------------------

        try:
            if user_id is None:
                retrieved, context_str = retrieve_context(
                    question,
                    current_mode,
                    use_test_memories=use_test_memories
                )
            else:
                retrieved, context_str = retrieve_context(
                    question,
                    current_mode,
                    use_test_memories=use_test_memories,
                    user_id=user_id
                )
        except Exception as error:
            _log_internal_error("memory retrieval", error)
            return NOT_FOUND_RESPONSE

        if not retrieved:
            return NOT_FOUND_RESPONSE

        try:
            generated = llm.generate_answer(
                question,
                context_str
            )
        except Exception as error:
            _log_internal_error("answer generation", error)
            return NOT_FOUND_RESPONSE

        if not generated:
            return NOT_FOUND_RESPONSE

        if llm._is_not_found(generated):
            return NOT_FOUND_RESPONSE

        if not llm._is_grounded_answer(generated, context_str):
            return NOT_FOUND_RESPONSE

        return generated

    # ==================================================
    # ROUTE 3: UNKNOWN / UNSUPPORTED QUESTION
    #
    # IMPORTANT:
    # Never send this to a general LLM.
    # This prevents hallucinations and visible
    # "thinking process" output.
    # ==================================================

    return NOT_FOUND_RESPONSE