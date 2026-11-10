# test_chatbot_safety.py
#
# Offline safety and regression tests for the chatbot.
#
# Every LLM call is routed to tests/offline_llm.py. The real OpenAI
# client is replaced before any test runs, so this file cannot consume
# NVIDIA/Nebius credits. Any accidental direct client use fails loudly.
#
# Run from the backend directory:
#   PYTHONPATH="$PWD" ../../rag_test_env/bin/python tests/test_chatbot_safety.py

import contextlib
import io
import os
import sys
from datetime import datetime, timedelta

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

from offline_llm import (
    NetworkBlocked,
    ORIGINAL_CLIENT,
    ORIGINAL_GENERATE_ANSWER,
    install_offline_llm,
)

FAKE_LLM = install_offline_llm()

import chatbot
import llm
import memory_store
from chatbot import chat
from llm import NOT_FOUND_RESPONSE
from rag import TEST_MEMORIES


TODAY = datetime.today().date()
TOMORROW = TODAY + timedelta(days=1)
TOMORROW_STR = TOMORROW.strftime("%Y-%m-%d")
TODAY_STR = TODAY.strftime("%Y-%m-%d")
FALLBACK = "I don't have enough information to answer that"


# --------------------------------------------------
# TEST RUNNER
# --------------------------------------------------

passed = 0
failed = 0
results_log = []


def test(name: str, ok: bool, notes: str = ""):
    global passed, failed

    status = "PASS" if ok else "FAIL"

    if ok:
        passed += 1
    else:
        failed += 1

    results_log.append(f"[{status}] {name}")

    if not ok:
        results_log.append(f"        -> {notes}")


def answer(question: str, mode: str = "classroom",
           use_test_memories: bool = True) -> str:
    return chat(question, mode, use_test_memories=use_test_memories)


@contextlib.contextmanager
def patched_retrieve(retrieved: list, context: str):
    """Make the retrieval step deterministic without changing production code."""
    original = chatbot.retrieve_context

    chatbot.retrieve_context = (
        lambda query, current_mode, use_test_memories=False: (
            retrieved, context
        )
    )

    try:
        yield
    finally:
        chatbot.retrieve_context = original


def load_store(memories: list) -> None:
    memory_store.clear_memories()
    memory_store.add_memories(memories)


# ==================================================
# SECTION A: FALLBACK AND EMPTY INPUT
# ==================================================

test(
    "A1: the exact fallback text is unchanged",
    NOT_FOUND_RESPONSE == FALLBACK,
    f"NOT_FOUND_RESPONSE={NOT_FOUND_RESPONSE!r}"
)

FAKE_LLM.reset()
empty_answer = answer("   ")

test(
    "A2: an empty question returns the exact fallback",
    empty_answer == FALLBACK,
    f"got {empty_answer!r}"
)

test(
    "A3: an empty question never calls the LLM",
    FAKE_LLM.call_count == 0,
    f"calls={FAKE_LLM.call_count}"
)


# ==================================================
# SECTION B: SCHEDULE FILTERS AND FALLBACKS
# ==================================================

tomorrow_answer = answer("What do I have tomorrow?")

test(
    "B1: tomorrow returns only the three tomorrow events",
    "maths test" in tomorrow_answer.lower()
    and "team meeting" in tomorrow_answer.lower()
    and "doctor appointment" in tomorrow_answer.lower()
    and "neural network" not in tomorrow_answer.lower()
    and "assignment" not in tomorrow_answer.lower(),
    f"got {tomorrow_answer!r}"
)

exact_time_answer = answer("What do I have tomorrow at 3 PM?")

test(
    "B2: an exact time returns only the matching event",
    "maths test" in exact_time_answer.lower()
    and "team meeting" not in exact_time_answer.lower()
    and "doctor appointment" not in exact_time_answer.lower(),
    f"got {exact_time_answer!r}"
)

range_answer = answer("What do I have tomorrow between 2 PM and 4 PM?")

test(
    "B3: a time range keeps only in-range events",
    "team meeting" in range_answer.lower()
    and "maths test" in range_answer.lower()
    and "doctor appointment" in range_answer.lower()
    and "neural network" not in range_answer.lower(),
    f"got {range_answer!r}"
)

test(
    "B4: a requested date with no events returns the exact fallback",
    answer("Do I have anything scheduled today?") == FALLBACK,
    f"got {answer('Do I have anything scheduled today?')!r}"
)

test(
    "B5: a named event that is not stored returns the exact fallback",
    answer("When is my physics exam?") == FALLBACK,
    f"got {answer('When is my physics exam?')!r}"
)

test(
    "B6: a named event matches its own event, not a same-type event",
    "maths test" in answer("When is my exam?").lower()
    and "physics" not in answer("When is my exam?").lower(),
    f"got {answer('When is my exam?')!r}"
)


# ==================================================
# SECTION C: MODE ISOLATION AND INTENDED CROSS-MODE
# ==================================================

CUSTOM_MEMORIES = [
    {
        "mode": "classroom",
        "text": "Teacher explained convolutional neural networks.",
        "memory_type": "lesson",
        "importance": "medium",
        "date": TODAY_STR,
        "time": None,
    },
    {
        "mode": "classroom",
        "text": "Maths test is tomorrow at 3 PM.",
        "memory_type": "exam",
        "importance": "high",
        "date": TOMORROW_STR,
        "time": "03:00 PM",
    },
    {
        "mode": "hospital",
        "text": "Doctor appointment is tomorrow at 4 PM.",
        "memory_type": "appointment",
        "importance": "high",
        "date": TOMORROW_STR,
        "time": "04:00 PM",
    },
    {
        "mode": "meeting",
        "text": "Team meeting is tomorrow at 2 PM.",
        "memory_type": "meeting",
        "importance": "high",
        "date": TOMORROW_STR,
        "time": "02:00 PM",
    },
    {
        "mode": "general",
        "text": "Reminder to buy groceries.",
        "memory_type": "reminder",
        "importance": "low",
        "date": TOMORROW_STR,
        "time": None,
    },
]

load_store(CUSTOM_MEMORIES)

FAKE_LLM.reset()
conversation_hospital = answer(
    "What did the teacher explain?", "hospital", use_test_memories=False
)

test(
    "C1: a conversation question in hospital does not leak classroom text",
    "Teacher explained convolutional neural networks." not in conversation_hospital
    and all(
        "Teacher explained convolutional neural networks." not in call["context"]
        for call in FAKE_LLM.calls
    ),
    f"got {conversation_hospital!r}, calls={FAKE_LLM.calls!r}"
)

FAKE_LLM.reset()
conversation_classroom = answer(
    "What did my doctor tell me?", "classroom", use_test_memories=False
)

test(
    "C2: a conversation question in classroom does not leak hospital text",
    "Doctor appointment is tomorrow at 4 PM." not in conversation_classroom
    and all(
        "Doctor appointment is tomorrow at 4 PM." not in call["context"]
        for call in FAKE_LLM.calls
    ),
    f"got {conversation_classroom!r}, calls={FAKE_LLM.calls!r}"
)

FAKE_LLM.reset()
conversation_by_topic = answer(
    "What did we learn about neural networks?",
    "hospital",
    use_test_memories=False,
)

test(
    "C3: a topic question in hospital does not leak classroom text",
    "Teacher explained convolutional neural networks." not in conversation_by_topic
    and all(
        "Teacher explained convolutional neural networks." not in call["context"]
        for call in FAKE_LLM.calls
    ),
    f"got {conversation_by_topic!r}, calls={FAKE_LLM.calls!r}"
)

schedule_cross_mode = answer(
    "When is my doctor appointment?", "classroom", use_test_memories=False
)

test(
    "C4: an intended named schedule query may use another mode",
    "doctor appointment" in schedule_cross_mode.lower()
    and schedule_cross_mode != FALLBACK,
    f"got {schedule_cross_mode!r}"
)

enumerated = answer(
    "What do I have tomorrow?", "classroom", use_test_memories=False
)

test(
    "C5: an intended date query may enumerate the cross-mode schedule",
    "maths test" in enumerated.lower()
    and "team meeting" in enumerated.lower()
    and "doctor appointment" in enumerated.lower(),
    f"got {enumerated!r}"
)

FAKE_LLM.reset()
broad_conversation = answer(
    "Is there anything about neural networks?",
    "classroom",
    use_test_memories=False,
)

test(
    "C6: broad 'is there anything' does not enumerate other modes",
    "Teacher explained convolutional neural networks." in broad_conversation
    and "Team meeting" not in broad_conversation
    and "Doctor appointment" not in broad_conversation
    and all(
        "Team meeting" not in call["context"]
        and "Doctor appointment" not in call["context"]
        for call in FAKE_LLM.calls
    ),
    f"got {broad_conversation!r}, calls={FAKE_LLM.calls!r}"
)

FAKE_LLM.reset()
notes_conversation = answer(
    "Are there any notes about neural networks?",
    "classroom",
    use_test_memories=False,
)

test(
    "C7: broad 'are there' does not enumerate other modes",
    "Teacher explained convolutional neural networks." in notes_conversation
    and "Team meeting" not in notes_conversation
    and "Doctor appointment" not in notes_conversation,
    f"got {notes_conversation!r}"
)

test(
    "C8: a specific 'is there' schedule question still works",
    "maths test" in answer(
        "Is there an exam tomorrow?", "hospital", use_test_memories=False
    ).lower(),
    f"got {answer('Is there an exam tomorrow?', 'hospital', use_test_memories=False)!r}"
)

test(
    "C9: a specific 'are there' schedule question still works",
    "team meeting" in answer(
        "Are there any events tomorrow?", "classroom", use_test_memories=False
    ).lower(),
    f"got {answer('Are there any events tomorrow?', 'classroom', use_test_memories=False)!r}"
)

FAKE_LLM.reset()
between_conversation = answer(
    "What is the difference between 2 neural network types?",
    "classroom",
    use_test_memories=False,
)

test(
    "C10: a bare 'between <number>' question does not cross modes",
    "Teacher explained convolutional neural networks." in between_conversation
    and "Team meeting" not in between_conversation
    and "Doctor appointment" not in between_conversation,
    f"got {between_conversation!r}"
)

test(
    "C11: a bare 'between <number>' topic question does not cross modes",
    "Teacher explained convolutional neural networks." in answer(
        "Explain the difference between 2 lessons about neural networks.",
        "classroom",
        use_test_memories=False,
    )
    and "Team meeting" not in answer(
        "Explain the difference between 2 lessons about neural networks.",
        "classroom",
        use_test_memories=False,
    )
    and "Doctor appointment" not in answer(
        "Explain the difference between 2 lessons about neural networks.",
        "classroom",
        use_test_memories=False,
    ),
    "an ordinary comparison question must not enumerate the calendar"
)

test(
    "C12: a real time range still crosses modes as intended",
    "team meeting" in answer(
        "What do I have between 2 PM and 4 PM?",
        "classroom",
        use_test_memories=False,
    ).lower(),
    f"got {answer('What do I have between 2 PM and 4 PM?', 'classroom', use_test_memories=False)!r}"
)

TOPIC_LEAK_QUESTIONS = [
    "Do I have any notes about neural networks?",
    "Do I have any understanding of neural networks?",
    "What do I have about neural networks?",
]

for index, topic_question in enumerate(TOPIC_LEAK_QUESTIONS, start=13):
    FAKE_LLM.reset()
    topic_answer = answer(
        topic_question, "classroom", use_test_memories=False
    )

    test(
        f"C{index}: topic question does not enumerate other modes: "
        f"{topic_question}",
        "Teacher explained convolutional neural networks." in topic_answer
        and "Team meeting" not in topic_answer
        and "Doctor appointment" not in topic_answer
        and all(
            "Team meeting" not in call["context"]
            and "Doctor appointment" not in call["context"]
            for call in FAKE_LLM.calls
        ),
        f"got {topic_answer!r}, calls={FAKE_LLM.calls!r}"
    )

test(
    "C16: a pure schedule enumeration is still preserved",
    all(
        phrase in answer(
            "Do I have anything scheduled?", "classroom", use_test_memories=False
        ).lower()
        for phrase in ("maths test", "team meeting", "doctor appointment")
    ),
    f"got {answer('Do I have anything scheduled?', 'classroom', use_test_memories=False)!r}"
)

test(
    "C17: a non-topic conflict query is still a schedule query",
    "maths test" in answer(
        "Do I have a conflict tomorrow?", "classroom", use_test_memories=False
    ).lower(),
    f"got {answer('Do I have a conflict tomorrow?', 'classroom', use_test_memories=False)!r}"
)

test(
    "C18: a typed event query is still a schedule query",
    "maths test" in answer(
        "Do I have a maths exam tomorrow?",
        "hospital",
        use_test_memories=False,
    ).lower(),
    f"got {answer('Do I have a maths exam tomorrow?', 'hospital', use_test_memories=False)!r}"
)


# ==================================================
# SECTION D: LLM ROUTING, TIMEOUT, RETRIES
# ==================================================

D1_RETRIEVED = [{"memory": TEST_MEMORIES[3], "score": 0.9}]
D1_CONTEXT = (
    "Retrieved memories:\n\n"
    "1.\n"
    "   Mode: classroom\n"
    "   Memory type: lesson\n"
    "   Text: Teacher explained convolutional neural networks.\n"
)

FAKE_LLM.reset()

with patched_retrieve([], ""):
    no_context_answer = answer(
        "What is the meaning of life?", "classroom", use_test_memories=True
    )

test(
    "D1: no retrieved context returns the fallback",
    no_context_answer == FALLBACK,
    f"got {no_context_answer!r}"
)

test(
    "D2: no retrieved context never calls the LLM",
    FAKE_LLM.call_count == 0,
    f"calls={FAKE_LLM.call_count}"
)

FAKE_LLM.reset()

with patched_retrieve(D1_RETRIEVED, D1_CONTEXT):
    rag_answer = answer(
        "What concept was emphasized?", "classroom", use_test_memories=True
    )

test(
    "D3: a valid RAG query invokes the fake LLM exactly once",
    FAKE_LLM.call_count == 1,
    f"calls={FAKE_LLM.call_count}"
)

test(
    "D4: the fake LLM receives the question and retrieved context",
    FAKE_LLM.call_count == 1
    and FAKE_LLM.calls[0]["question"] == "What concept was emphasized?"
    and FAKE_LLM.calls[0]["context"] == D1_CONTEXT,
    f"calls={FAKE_LLM.calls!r}"
)

test(
    "D5: the answer comes from the fake LLM, not a live API",
    rag_answer == (FAKE_LLM.last_response if FAKE_LLM.last_response is not None else FAKE_LLM.response),
    f"got {rag_answer!r}, fake={FAKE_LLM.response!r}, last={FAKE_LLM.last_response!r}"
)

test(
    "D6: API retries are disabled",
    ORIGINAL_CLIENT.max_retries == 0,
    f"max_retries={ORIGINAL_CLIENT.max_retries!r}"
)

test(
    "D7: API timeout is bounded to 30 seconds",
    float(ORIGINAL_CLIENT.timeout) == 30.0,
    f"timeout={ORIGINAL_CLIENT.timeout!r}"
)

try:
    llm._client.chat.completions.create(
        model="offline-test",
        messages=[],
    )
    guarded = False
except NetworkBlocked:
    guarded = True

test(
    "D8: direct real-client use during tests fails loudly",
    guarded,
    "the installed test client did not block a direct call"
)


# ==================================================
# SECTION E: ERROR HANDLING AND LOG PRIVACY
# ==================================================

PRIVATE_SECRET = (
    "PRIVATE_MEMORY_TEXT_9f3a: Teacher explained convolutional "
    "neural networks."
)

FAKE_LLM.reset()
FAKE_LLM.answer_exception = RuntimeError(PRIVATE_SECRET)

stderr = io.StringIO()

with patched_retrieve(D1_RETRIEVED, D1_CONTEXT):
    with contextlib.redirect_stderr(stderr):
        llm_failure_answer = answer(
            "What concept was emphasized?", "classroom", use_test_memories=True
        )

logged = stderr.getvalue()
FAKE_LLM.reset()

test(
    "E1: an LLM exception returns the exact fallback",
    llm_failure_answer == FALLBACK,
    f"got {llm_failure_answer!r}"
)

test(
    "E2: an LLM exception is logged with stage and type",
    "answer generation failed" in logged
    and "RuntimeError" in logged,
    f"stderr={logged!r}"
)

test(
    "E3: the log does not contain private memory text",
    PRIVATE_SECRET not in logged
    and "Teacher explained" not in logged
    and "neural networks" not in logged,
    f"stderr={logged!r}"
)

STORE_SECRET = "API_KEY=sk-must-never-be-logged; PRIVATE_STORE_TEXT"
QUESTION_SECRET = "UNIQUE_QUESTION_SECRET_42"

original_get_all = memory_store.get_all_memories


def broken_store():
    raise RuntimeError(STORE_SECRET)


memory_store.get_all_memories = broken_store

stderr = io.StringIO()

try:
    with contextlib.redirect_stderr(stderr):
        store_failure_answer = chat(
            f"Do I have a meeting tomorrow? {QUESTION_SECRET}",
            "classroom",
            use_test_memories=False,
        )
finally:
    memory_store.get_all_memories = original_get_all

logged = stderr.getvalue()

test(
    "E4: a store failure returns the exact fallback",
    store_failure_answer == FALLBACK,
    f"got {store_failure_answer!r}"
)

test(
    "E5: a store failure is logged with stage and type",
    "memory store read" in logged
    and "RuntimeError" in logged,
    f"stderr={logged!r}"
)

test(
    "E6: the log does not contain secret or question text",
    STORE_SECRET not in logged
    and QUESTION_SECRET not in logged
    and "sk-must-never-be-logged" not in logged,
    f"stderr={logged!r}"
)


class _RaisingCompletions:
    def create(self, *args, **kwargs):
        raise RuntimeError(PRIVATE_SECRET)


class _RaisingChat:
    def __init__(self):
        self.completions = _RaisingCompletions()


class _RaisingClient:
    def __init__(self):
        self.chat = _RaisingChat()


# Exercise the production llm.generate_answer error handler directly,
# swapping only its client for a raising stub. No network is involved.
original_client = llm._client
llm._client = _RaisingClient()

stderr = io.StringIO()

try:
    with contextlib.redirect_stderr(stderr):
        direct_llm_answer = ORIGINAL_GENERATE_ANSWER(
            "What concept was emphasized?", D1_CONTEXT
        )
finally:
    llm._client = original_client

logged = stderr.getvalue()

test(
    "E7: llm.generate_answer returns the exact fallback on a client error",
    direct_llm_answer == FALLBACK,
    f"got {direct_llm_answer!r}"
)

test(
    "E8: the llm client error log omits private payload text",
    "answer generation failed" in logged
    and "RuntimeError" in logged
    and PRIVATE_SECRET not in logged,
    f"stderr={logged!r}"
)

test(
    "E9: llm.generate_answer returns the fallback for empty context",
    ORIGINAL_GENERATE_ANSWER("What concept was emphasized?", "") == FALLBACK,
    "empty context must not reach the client"
)


# ==================================================
# SECTION F: ANSWER GROUNDING
# ==================================================
#
# A lexical grounding guard (in llm.generate_answer) now validates
# that an answer is supported by the retrieved memory text. It is a
# fail-closed check (numbers/dates/names/negations/content coverage and
# a contiguous content span), not semantic entailment, and cannot
# guarantee protection against every hallucination. The guard runs
# after prompt instructions and fallback-pattern filtering and does
# not call a second LLM or make any network requests.

FAKE_LLM.reset()
# D1_CONTEXT contains only "Teacher explained convolutional neural
# networks." No physics exam exists there. An ungrounded fabricat-
# ion must be rejected in favour of the exact fallback.
FAKE_LLM.response = "Your physics exam is tomorrow at 9 AM."

with patched_retrieve(D1_RETRIEVED, D1_CONTEXT):
    ungrounded_answer = answer(
        "What concept was emphasized?", "classroom", use_test_memories=True
    )

FAKE_LLM.reset()

test(
    "F1: an ungrounded fabricated answer is rejected",
    ungrounded_answer == FALLBACK
    and "physics" not in D1_CONTEXT.lower(),
    f"got {ungrounded_answer!r}, llm_calls={FAKE_LLM.call_count}"
)

FAKE_LLM.reset()
# The fake LLM derives a grounded answer from the context by default,
# so a valid memory-based answer must still be accepted.
with patched_retrieve(D1_RETRIEVED, D1_CONTEXT):
    grounded_answer = answer(
        "What concept was emphasized?", "classroom", use_test_memories=True
    )

FAKE_LLM.reset()

test(
    "F2: a grounded answer derived from retrieved context is accepted",
    grounded_answer == "Teacher explained convolutional neural networks."
    and "convolutional" in grounded_answer.lower(),
    f"got {grounded_answer!r}"
)

FAKE_LLM.reset()
FAKE_LLM.response = "The maths test is on 2099-12-31 at 9 AM."

with patched_retrieve(D1_RETRIEVED, D1_CONTEXT):
    added_date_answer = answer(
        "What concept was emphasized?", "classroom", use_test_memories=True
    )

FAKE_LLM.reset()

test(
    "F3: an answer that adds unsupported dates or times is rejected",
    added_date_answer == FALLBACK,
    f"got {added_date_answer!r}"
)

FAKE_LLM.reset()
FAKE_LLM.response = "Dr Smith said the exam was cancelled."

with patched_retrieve(D1_RETRIEVED, D1_CONTEXT):
    added_person_answer = answer(
        "What concept was emphasized?", "classroom", use_test_memories=True
    )

FAKE_LLM.reset()

test(
    "F4: an answer that adds unsupported people or events is rejected",
    added_person_answer == FALLBACK,
    f"got {added_person_answer!r}"
)

FAKE_LLM.reset()
with patched_retrieve([], ""):
    empty_context_answer = answer(
        "What concept was emphasized?", "classroom", use_test_memories=True
    )

test(
    "F5: empty or irrelevant retrieved context returns the exact fallback",
    empty_context_answer == FALLBACK,
    f"got {empty_context_answer!r}"
)

FAKE_LLM.reset()
FAKE_LLM.response = "The class was held in Room 5B."

with patched_retrieve(D1_RETRIEVED, D1_CONTEXT):
    irrelevant_detail_answer = answer(
        "What concept was emphasized?", "classroom", use_test_memories=True
    )

FAKE_LLM.reset()

test(
    "F6: details not present in the retrieved context are rejected",
    irrelevant_detail_answer == FALLBACK,
    f"got {irrelevant_detail_answer!r}"
)

FAKE_LLM.reset()
FAKE_LLM.response = FALLBACK

with patched_retrieve(D1_RETRIEVED, D1_CONTEXT):
    fallback_passthrough = answer(
        "What concept was emphasized?", "classroom", use_test_memories=True
    )

FAKE_LLM.reset()

test(
    "F7: a valid fallback response is preserved exactly",
    fallback_passthrough == FALLBACK,
    f"got {fallback_passthrough!r}"
)


# ==================================================
# SECTION G: UNSUPPORTED QUESTIONS AND INTERFACES
# ==================================================

FAKE_LLM.reset()
unrelated_answer = answer("What is the capital of France?")

test(
    "G1: an unrelated question returns the exact fallback",
    unrelated_answer == FALLBACK,
    f"got {unrelated_answer!r}"
)

test(
    "G2: an unrelated question does not call the LLM",
    FAKE_LLM.call_count == 0,
    f"calls={FAKE_LLM.call_count}"
)

FAKE_LLM.reset()
# User isolation is not implemented in the Python stack today (no
# trustworthy user_id). The scoped API is present in memory_store/rag/
# chatbot but must not be treated as enforced unless wired to auth.
FAKE_LLM.response = "Teacher explained convolutional neural networks."

load_store([
    {
        "mode": "classroom",
        "text": "Teacher explained convolutional neural networks.",
        "memory_type": "lesson",
        "importance": "medium",
        "date": TODAY_STR,
        "time": None,
    },
])

memory_store.clear_memories(user_id="user_a")
memory_store.add_memories(
    [
        {
            "mode": "classroom",
            "text": "User A secret plan for tomorrow.",
            "memory_type": "meeting",
            "importance": "high",
            "date": TOMORROW_STR,
            "time": "09:00 AM",
        }
    ],
    user_id="user_a",
)

memory_store.add_memories(
    [
        {
            "mode": "classroom",
            "text": "User B private appointment at 5 PM.",
            "memory_type": "appointment",
            "importance": "high",
            "date": TOMORROW_STR,
            "time": "05:00 PM",
        }
    ],
    user_id="user_b",
)

user_a_memories = memory_store.get_all_memories(user_id="user_a")
user_b_memories = memory_store.get_all_memories(user_id="user_b")
all_memories = memory_store.get_all_memories()

test(
    "U1: explicit user_id scopes memory reads",
    len(user_a_memories) == 1
    and len(user_b_memories) == 1
    and user_a_memories[0]["text"] == "User A secret plan for tomorrow."
    and user_b_memories[0]["text"] == "User B private appointment at 5 PM.",
    f"a={len(user_a_memories)} b={len(user_b_memories)} all={len(all_memories)}"
)

test(
    "U2: user-scoped retrieval does not return the other user's memory",
    memory_store.get_memories_by_mode("classroom", user_id="user_a")
    == user_a_memories
    and memory_store.get_entries_by_mode("classroom", user_id="user_a")[0][
        "memory"
    ]["text"]
    == "User A secret plan for tomorrow."
    and all(
        entry["memory"]["text"] != "User B private appointment at 5 PM."
        for entry in memory_store.get_entries_by_mode(
            "classroom", user_id="user_a"
        )
    ),
    "cross-user leakage in scoped access"
)

memory_store.clear_memories(user_id="user_a")

test(
    "U3: user-scoped clear does not affect another user's memories",
    len(memory_store.get_all_memories(user_id="user_a")) == 0
    and len(memory_store.get_all_memories(user_id="user_b")) == 1
    and memory_store.get_all_memories(user_id="user_b")[0]["text"]
    == "User B private appointment at 5 PM.",
    "user_a clear removed user_b data"
)

# The legacy unscoped API remains available but provides NO isolation.
# This must not be misread as a security guarantee.
memory_store.clear_memories()
load_store(TEST_MEMORIES)
FAKE_LLM.reset()

FAKE_LLM.reset()
unsupported_answer = answer("What is my physics exam?")

test(
    "G3: an unsupported answer does not invent information",
    unsupported_answer == FALLBACK,
    f"got {unsupported_answer!r}"
)

test(
    "G4: an unsupported answer does not call the LLM",
    FAKE_LLM.call_count == 0,
    f"calls={FAKE_LLM.call_count}"
)

test(
    "G5: the expected interfaces remain callable",
    callable(chat)
    and callable(llm.generate_answer)
    and callable(llm.generate_fallback)
    and isinstance(unsupported_answer, str),
    "one of the expected interfaces is missing"
)


# ==================================================
# RESULTS SUMMARY
# ==================================================

# Restore the real functions so the process leaves no surprising global
# state, even though the test process is about to exit.

memory_store.clear_memories()

print("\n" + "=" * 60)
print("CHATBOT OFFLINE SAFETY TEST RESULTS")
print("=" * 60)

for line in results_log:
    print(line)

print("\n" + "=" * 60)
print(f"Total: {passed + failed} | Passed: {passed} | Failed: {failed}")
print(f"Fake LLM answer calls across the run: {FAKE_LLM.total_calls}")
print("=" * 60)

if failed:
    print("\nSome tests failed.")
    sys.exit(1)

print("\nAll offline safety tests passed!")
