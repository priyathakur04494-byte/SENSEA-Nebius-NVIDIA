# test_chatbot.py
#
# Regression tests for the chatbot: date handling, cross-mode
# isolation and error boundaries.
#
# Run with: python test_chatbot.py

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

from offline_llm import install_offline_llm

FAKE_LLM = install_offline_llm()

import chatbot
import llm
import memory_store
from chatbot import chat
from llm import NOT_FOUND_RESPONSE

TODAY = datetime.today().date()
TOMORROW = TODAY + timedelta(days=1)
YESTERDAY = TODAY - timedelta(days=1)


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
        results_log.append(f"        → {notes}")


def answer(question: str, mode: str = "classroom") -> str:
    return chat(question, mode, use_test_memories=True)


# ==================================================
# SECTION A: FALLBACK TEXT
# ==================================================

test(
    "A1: the fallback text is unchanged",
    NOT_FOUND_RESPONSE == "I don't have enough information to answer that",
    f"NOT_FOUND_RESPONSE={NOT_FOUND_RESPONSE!r}"
)

test(
    "A2: an empty question returns the fallback",
    answer("   ") == NOT_FOUND_RESPONSE,
    f"got {answer('   ')!r}"
)

# ==================================================
# SECTION B: DATE EXTRACTION
# ==================================================

test(
    "B1: 'tomorrow' is parsed",
    chatbot._extract_requested_date(
        "What do I have tomorrow?"
    ) == TOMORROW.strftime("%Y-%m-%d"),
    f"got {chatbot._extract_requested_date('What do I have tomorrow?')}"
)

test(
    "B2: 'today' is parsed",
    chatbot._extract_requested_date(
        "What do I have today?"
    ) == TODAY.strftime("%Y-%m-%d"),
    f"got {chatbot._extract_requested_date('What do I have today?')}"
)

test(
    "B3: 'yesterday' is parsed",
    chatbot._extract_requested_date(
        "What did I have yesterday?"
    ) == YESTERDAY.strftime("%Y-%m-%d"),
    f"got {chatbot._extract_requested_date('What did I have yesterday?')}"
)

test(
    "B4: an ISO date is parsed",
    chatbot._extract_requested_date(
        "Do I have a meeting on 2026-10-30?"
    ) == "2026-10-30",
    f"got {chatbot._extract_requested_date('Do I have a meeting on 2026-10-30?')}"
)

test(
    "B5: a written date is parsed (month name first)",
    chatbot._extract_requested_date(
        "Do I have a meeting on September 9?"
    ) == f"{TODAY.year}-09-09",
    f"got {chatbot._extract_requested_date('Do I have a meeting on September 9?')}"
)

test(
    "B6: a written date is parsed (day first)",
    chatbot._extract_requested_date(
        "Do I have a meeting on 9 September?"
    ) == f"{TODAY.year}-09-09",
    f"got {chatbot._extract_requested_date('Do I have a meeting on 9 September?')}"
)

test(
    "B7: an abbreviated month is parsed",
    chatbot._extract_requested_date(
        "Do I have a meeting on Sept 9?"
    ) == f"{TODAY.year}-09-09",
    f"got {chatbot._extract_requested_date('Do I have a meeting on Sept 9?')}"
)

test(
    "B8: a question without a date has no date",
    chatbot._extract_requested_date(
        "What did the teacher explain?"
    ) is None,
    f"got {chatbot._extract_requested_date('What did the teacher explain?')}"
)

# ==================================================
# SECTION C: DATE FILTER
# ==================================================

tomorrow_answer = answer("What do I have tomorrow?")

test(
    "C1: 'tomorrow' returns the tomorrow events",
    "maths test" in tomorrow_answer.lower()
    and "team meeting" in tomorrow_answer.lower()
    and "doctor appointment" in tomorrow_answer.lower(),
    f"got {tomorrow_answer!r}"
)

test(
    "C2: 'tomorrow' does not return today's lessons",
    "neural network" not in tomorrow_answer.lower(),
    f"got {tomorrow_answer!r}"
)

test(
    "C3: 'tomorrow' does not return the undated assignment",
    "assignment" not in tomorrow_answer.lower(),
    f"got {tomorrow_answer!r}"
)

test(
    "C4: 'today' does not return tomorrow's events",
    answer("Do I have anything scheduled today?") == NOT_FOUND_RESPONSE,
    f"got {answer('Do I have anything scheduled today?')!r}"
)

test(
    "C5: an explicit date that has no event returns the fallback",
    answer(
        "Do I have a hospital appointment on "
        f"{(TODAY + timedelta(days=10)).strftime('%Y-%m-%d')}?"
    ) == NOT_FOUND_RESPONSE,
    "a date with no event must not answer with another day's event"
)

test(
    "C6: an explicit date that has an event is answered",
    "doctor appointment" in answer(
        "Do I have a hospital appointment on "
        f"{TOMORROW.strftime('%Y-%m-%d')}?"
    ).lower(),
    "the event of the requested date must be returned"
)

test(
    "C7: 'yesterday' does not leak tomorrow's events",
    answer("Do I have anything scheduled yesterday?") == NOT_FOUND_RESPONSE,
    f"got {answer('Do I have anything scheduled yesterday?')!r}"
)

# ==================================================
# SECTION D: CROSS-MODE SCHEDULE AND MODE ISOLATION
# ==================================================
#
# A SCHEDULE question may be answered from every mode, because the
# schedule of the user is one calendar. An ordinary CONVERSATION
# question is still answered from the active mode only.

test(
    "D1: a named schedule event is found in another mode",
    "doctor appointment" in answer(
        "When is my doctor appointment?", "classroom"
    ).lower(),
    "a schedule question may be answered from another mode"
)

test(
    "D2: the active mode still answers its own event",
    "maths test" in answer("When is my exam?", "classroom").lower(),
    "classroom mode must answer its own exam"
)

test(
    "D3: a date question may look across modes",
    "doctor appointment" in answer(
        "Do I have anything scheduled tomorrow?", "classroom"
    ).lower(),
    "a date scoped question may cross modes"
)

test(
    "D4: a time question may look across modes",
    "team meeting" in answer(
        "What do I have between 2 PM and 4 PM?", "meeting"
    ).lower(),
    "a time scoped question may cross modes"
)

test(
    "D5: an unrelated question never crosses modes",
    answer("What is the capital of France?", "hospital")
    == NOT_FOUND_RESPONSE,
    "an unrelated question must stay in the mode"
)

test(
    "D6: a conversation question never crosses modes",
    "Doctor appointment is tomorrow at 4 PM." not in answer(
        "What did my doctor tell me?", "classroom"
    ),
    "a conversation question must not reach another mode"
)

test(
    "D7: a payment query without a matching reminder returns the fallback",
    answer("Do I have any pending payment tomorrow?", "classroom")
    == NOT_FOUND_RESPONSE,
    "no payment reminder is stored, so the standard fallback is returned"
)

test(
    "D8: a named exam does not match a different exam of the same type",
    answer("When is my physics exam?", "classroom") == NOT_FOUND_RESPONSE,
    "the store holds a maths test, not a physics exam"
)

# ==================================================
# SECTION E: ERROR BOUNDARIES
# ==================================================

original_get_entries = memory_store.get_entries_by_mode


def broken_store(mode):
    raise RuntimeError("simulated store failure")


memory_store.get_entries_by_mode = broken_store

stderr = io.StringIO()

try:
    with contextlib.redirect_stderr(stderr):
        out = chat(
            "What did the teacher explain?",
            "classroom",
            use_test_memories=False
        )
    logged = stderr.getvalue()

except Exception as error:  # pragma: no cover - must not happen
    out = f"CRASHED {type(error).__name__}: {error}"
    logged = stderr.getvalue()

finally:
    memory_store.get_entries_by_mode = original_get_entries

test(
    "E1: a store failure returns the fallback instead of crashing",
    out == NOT_FOUND_RESPONSE,
    f"got {out!r}"
)

test(
    "E2: a store failure is logged, not hidden",
    "failed:" in logged
    and "RuntimeError" in logged,
    f"stderr={logged!r}"
)

test(
    "E2b: an exception message is not copied into the log",
    "simulated store failure" not in logged,
    f"stderr={logged!r}"
)

original_generate = llm.generate_answer


def broken_llm(question, context):
    raise RuntimeError("simulated LLM failure")


llm.generate_answer = broken_llm

stderr = io.StringIO()

try:
    with contextlib.redirect_stderr(stderr):
        out = chat(
            "What was explained in the class about the topics?",
            "classroom",
            use_test_memories=True
        )
    logged = stderr.getvalue()

except Exception as error:  # pragma: no cover - must not happen
    out = f"CRASHED {type(error).__name__}: {error}"
    logged = stderr.getvalue()

finally:
    llm.generate_answer = original_generate

test(
    "E3: an LLM failure returns the fallback instead of crashing",
    out == NOT_FOUND_RESPONSE,
    f"got {out!r}"
)

test(
    "E4: an LLM failure is logged, not hidden",
    "answer generation failed" in logged
    and "RuntimeError" in logged,
    f"stderr={logged!r}"
)

test(
    "E5: the failure log does not contain memory text",
    "neural network" not in logged,
    f"stderr={logged!r}"
)

# ==================================================
# RESULTS SUMMARY
# ==================================================

print("\n" + "=" * 60)
print("CHATBOT TEST RESULTS")
print("=" * 60)

for line in results_log:
    print(line)

print("\n" + "=" * 60)
print(f"Total: {passed + failed} | Passed: {passed} | Failed: {failed}")
print("=" * 60)

if failed:
    print("\nSome tests failed.")
    sys.exit(1)

print("\nAll tests passed!")
