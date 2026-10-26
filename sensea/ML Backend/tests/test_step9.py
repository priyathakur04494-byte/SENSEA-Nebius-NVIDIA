# test_step9.py
# Step 9 tests + full Step 7/8 regression suite.
# Run: python test_step9.py

import sys

from offline_llm import install_offline_llm

FAKE_LLM = install_offline_llm()

from normalizer import normalize_memory_item, normalize_extractor_output
from memory_store import (
    add_memory, add_memories, get_all_memories,
    get_memories_by_mode, clear_memories, memory_count
)
from rag import semantic_search, retrieve_context, MIN_SIMILARITY
from chatbot import chat

# --------------------------------------------------
# TEST RUNNER
# --------------------------------------------------

passed = 0
failed = 0
results_log = []


def test(name, result, expected_contains=None, must_not_contain=None, expected_exact=None, expected_equals=None):
    global passed, failed
    ok = True
    notes = []
    result_str = str(result).lower()

    if expected_contains:
        for phrase in expected_contains:
            if phrase.lower() not in result_str:
                ok = False
                notes.append(f"MISSING: '{phrase}'")

    if must_not_contain:
        for phrase in must_not_contain:
            if phrase.lower() in result_str:
                ok = False
                notes.append(f"MUST NOT CONTAIN: '{phrase}'")

    if expected_exact is not None:
        if str(expected_exact).lower() not in result_str:
            ok = False
            notes.append(f"EXPECTED EXACT: '{expected_exact}'")

    if expected_equals is not None:
        if str(result) != str(expected_equals):
            ok = False
            notes.append(f"EXPECTED EQUALS: {expected_equals!r}")

    status = "PASS" if ok else "FAIL"
    if ok:
        passed += 1
    else:
        failed += 1

    results_log.append(f"[{status}] {name}")
    if not ok:
        for note in notes:
            results_log.append(f"        → {note}")
        results_log.append(f"        → Got: {result!r}")


# ==================================================
# SECTION A–M: NORMALIZER TESTS
# ==================================================

print("Running Step 9 normalizer tests...\n")

# A: Extractor JSON accepted
extractor_json = {
    "simplified_text": "The professor explained CNNs.",
    "summary": "Professor explained CNNs.",
    "important_information": [
        {
            "text": "Professor explained convolutional neural networks.",
            "memory_type": "lesson",
            "importance": "medium",
            "date": "2026-09-05",
            "time": None
        }
    ]
}
result_a = normalize_extractor_output(extractor_json, "classroom")
test("A: Extractor JSON accepted", str(len(result_a)), expected_exact="1")

# B: Mode attached correctly
test("B: Mode attached correctly", result_a[0].get("mode", ""), expected_contains=["classroom"])

# C: Text preserved
test("C: Text preserved", result_a[0].get("text", ""),
     expected_contains=["Professor explained convolutional neural networks."])

# D: Memory type preserved
test("D: Memory type preserved", result_a[0].get("memory_type", ""),
     expected_contains=["lesson"])

# E: Importance preserved
test("E: Importance preserved", result_a[0].get("importance", ""),
     expected_contains=["medium"])

# F: Date preserved
test("F: Date preserved", result_a[0].get("date", ""),
     expected_contains=["2026-09-05"])

# G: Time preserved (None → None)
test("G: Time=None preserved", str(result_a[0].get("time")), expected_contains=["none"])

# H: Missing date handled safely
item_no_date = {"text": "No date item.", "memory_type": "lesson", "importance": "low", "date": None, "time": None}
mem_no_date = normalize_memory_item(item_no_date, "classroom")
test("H: Missing date handled safely", str(mem_no_date.get("date")), expected_contains=["none"])

# I: Missing time handled safely
item_no_time = {"text": "No time item.", "memory_type": "lesson", "importance": "low", "date": "2026-09-05", "time": None}
mem_no_time = normalize_memory_item(item_no_time, "classroom")
test("I: Missing time handled safely", str(mem_no_time.get("time")), expected_contains=["none"])

# J: Classroom memory stored correctly
clear_memories()
classroom_mem = {
    "mode": "classroom", "text": "Teacher explained CNNs.",
    "memory_type": "lesson", "importance": "medium",
    "date": "2026-09-05", "time": None
}
add_memory(classroom_mem)
test("J: Classroom memory stored", str(memory_count()), expected_exact="1")
test("J2: Classroom memory retrievable by mode",
     str(len(get_memories_by_mode("classroom"))), expected_exact="1")

# K: Meeting memory stored correctly
meeting_mem = {
    "mode": "meeting", "text": "Team meeting tomorrow at 2 PM.",
    "memory_type": "meeting", "importance": "high",
    "date": "2026-09-07", "time": "02:00 PM"
}
add_memory(meeting_mem)
test("K: Meeting memory stored", str(memory_count()), expected_exact="2")

# L: Hospital memory stored correctly
hospital_mem = {
    "mode": "hospital", "text": "Doctor appointment tomorrow at 4 PM.",
    "memory_type": "appointment", "importance": "high",
    "date": "2026-09-07", "time": "04:00 PM"
}
add_memory(hospital_mem)
test("L: Hospital memory stored", str(memory_count()), expected_exact="3")

# M: General memory stored correctly
general_mem = {
    "mode": "general", "text": "Reminder to buy groceries.",
    "memory_type": "reminder", "importance": "low",
    "date": None, "time": None
}
add_memory(general_mem)
test("M: General memory stored", str(memory_count()), expected_exact="4")

# ==================================================
# SECTION N–S: RAG WITH REAL MEMORIES
# ==================================================

print("Running Step 9 RAG + memory store tests...\n")

# Setup: clear and load test memories into memory_store
clear_memories()
from datetime import datetime, timedelta
TODAY = datetime.today().date()
TOMORROW = TODAY + timedelta(days=1)
TOMORROW_STR = TOMORROW.strftime("%Y-%m-%d")
TODAY_STR = TODAY.strftime("%Y-%m-%d")

real_memories = [
    {
        "mode": "classroom",
        "text": "Maths test is tomorrow at 3 PM.",
        "memory_type": "exam",
        "importance": "high",
        "date": TOMORROW_STR,
        "time": "03:00 PM"
    },
    {
        "mode": "meeting",
        "text": "Team meeting is tomorrow at 2 PM.",
        "memory_type": "meeting",
        "importance": "high",
        "date": TOMORROW_STR,
        "time": "02:00 PM"
    },
    {
        "mode": "hospital",
        "text": "Doctor appointment is tomorrow at 4 PM.",
        "memory_type": "appointment",
        "importance": "high",
        "date": TOMORROW_STR,
        "time": "04:00 PM"
    },
    {
        "mode": "classroom",
        "text": "Teacher explained convolutional neural networks.",
        "memory_type": "lesson",
        "importance": "medium",
        "date": TODAY_STR,
        "time": None
    },
    {
        "mode": "classroom",
        "text": "The professor explained deep learning concepts.",
        "memory_type": "lesson",
        "importance": "medium",
        "date": TODAY_STR,
        "time": None
    },
    {
        "mode": "classroom",
        "text": "Assignment on neural networks is due next week.",
        "memory_type": "assignment",
        "importance": "medium",
        "date": None,
        "time": None
    }
]
add_memories(real_memories)

# N: RAG retrieves real normalized memory
retrieved_n = semantic_search(
    "What did we learn about neural networks?",
    "classroom",
    use_test_memories=False
)
test("N: RAG retrieves real normalized memory",
     " ".join(r["memory"]["text"] for r in retrieved_n),
     expected_contains=["neural"])

# O: Mode filtering — hospital cannot retrieve classroom
retrieved_o = semantic_search(
    "What did we learn about neural networks?",
    "hospital",
    use_test_memories=False
)
test("O: Mode filtering — hospital blocks classroom", str(len(retrieved_o)), expected_exact="0")

# P: Irrelevant memories rejected by threshold
retrieved_p = semantic_search(
    "What is the capital of France?",
    "classroom",
    use_test_memories=False
)
test("P: Irrelevant memories rejected by threshold", str(len(retrieved_p)), expected_exact="0")

# Q: Unknown information returns not found
answer_q = chat("What is my physics exam?", "classroom", use_test_memories=False)
test("Q: Unknown info returns not found", answer_q, expected_contains=["I don't have enough information to answer that"])

# R: Cross-mode named schedule lookup — classroom finds the hospital appointment
#
# A schedule question is answered from every mode: the user's schedule
# is one calendar. The question still decides what is returned, so the
# hospital appointment is returned for an appointment question and
# nothing else is.
answer_r = chat("When is my doctor appointment?", "classroom", use_test_memories=False)
test("R: Cross-mode schedule — classroom finds the hospital appointment", answer_r,
     expected_contains=["doctor appointment"],
     must_not_contain=["I don't have enough information to answer that"])

# S: Hospital cannot access classroom memory
answer_s = chat("What did the professor explain?", "hospital", use_test_memories=False)
test("S: Hospital cannot access classroom memory", answer_s,
     expected_contains=["I don't have enough information to answer that"])

# T: Structured schedule search with real memories
answer_t = chat("When is my exam?", "classroom", use_test_memories=False)
test("T: Structured schedule search — classroom exam", answer_t,
     expected_contains=["maths test"])

# U: Cross-mode schedule search with real memories
answer_u = chat("What do I have tomorrow between 2 PM and 4 PM?", "classroom",
                use_test_memories=False)
test("U: Cross-mode schedule — all three events", answer_u,
     expected_contains=["team meeting", "maths test", "doctor appointment"])

# ==================================================
# SECTION V: STEP 7/8 REGRESSION — TEST_MEMORIES PATH
# ==================================================

print("Running Step 7/8 regression tests (TEST_MEMORIES path)...\n")

# V1–V10: Step 7 core regression
answer_v1 = chat("When is my exam?", "classroom", use_test_memories=True)
test("V1: [Reg] Classroom — when is my exam", answer_v1, expected_contains=["maths test"])

answer_v2 = chat("What did the teacher explain?", "classroom", use_test_memories=True)
test("V2: [Reg] Classroom — teacher explained", answer_v2,
     must_not_contain=["I don't have enough information to answer that"])

answer_v3 = chat("What did the professor explain?", "classroom", use_test_memories=True)
test("V3: [Reg] Classroom — professor explained", answer_v3,
     must_not_contain=["I don't have enough information to answer that"])

answer_v4 = chat("What did the lecturer explain?", "classroom", use_test_memories=True)
test("V4: [Reg] Classroom — lecturer not found", answer_v4,
     expected_contains=["I don't have enough information to answer that"])

answer_v5 = chat("What is my physics exam?", "classroom", use_test_memories=True)
test("V5: [Reg] Classroom — physics exam not found", answer_v5,
     expected_contains=["I don't have enough information to answer that"])

answer_v6 = chat("When is my appointment?", "hospital", use_test_memories=True)
test("V6: [Reg] Hospital — when is my appointment", answer_v6,
     expected_contains=["doctor appointment"])

answer_v7 = chat("When is my exam?", "hospital", use_test_memories=True)
test("V7: [Reg] Cross-mode schedule — hospital finds the classroom exam", answer_v7,
     expected_contains=["maths test"],
     must_not_contain=["I don't have enough information to answer that"])

answer_v8 = chat("What do I have tomorrow at 3 PM?", "meeting", use_test_memories=True)
test("V8: [Reg] Cross-mode exact time — 3 PM", answer_v8, expected_contains=["maths test"])

answer_v9 = chat("What do I have tomorrow between 2 PM and 4 PM?", "hospital",
                 use_test_memories=True)
test("V9: [Reg] Cross-mode time range", answer_v9,
     expected_contains=["team meeting", "maths test", "doctor appointment"])

answer_v10 = chat("What is my physics exam?", "meeting", use_test_memories=True)
test("V10: [Reg] Unknown — meeting mode", answer_v10,
     expected_contains=["I don't have enough information to answer that"])

# Step 8 regression (TEST_MEMORIES path)
retrieved_w1 = semantic_search(
    "What did we learn about neural network training?",
    "classroom", use_test_memories=True
)
test("W1: [Reg] Semantic — classroom neural networks",
     " ".join(r["memory"]["text"] for r in retrieved_w1),
     expected_contains=["neural"])

retrieved_w2 = semantic_search(
    "What did we learn about neural network training?",
    "hospital", use_test_memories=True
)
test("W2: [Reg] Mode filter — hospital blocks classroom", str(len(retrieved_w2)),
     expected_exact="0")

retrieved_w3 = semantic_search(
    "What is the capital of France?",
    "hospital", use_test_memories=True
)
test("W3: [Reg] Threshold — unrelated rejected", str(len(retrieved_w3)), expected_exact="0")

answer_w4 = chat("What did we learn about neural networks?", "classroom",
                 use_test_memories=True)
test("W4: [Reg] Generation route (offline) — classroom neural networks", answer_w4,
     must_not_contain=["I don't have enough information to answer that"])

answer_w5 = chat("What is my physics exam?", "classroom", use_test_memories=True)
test("W5: [Reg] Not-found — physics exam", answer_w5,
     expected_contains=["I don't have enough information to answer that"])

answer_w6 = chat("What did we learn about neural networks?", "hospital",
                 use_test_memories=True)
test("W6: [Reg] Mode protection — hospital", answer_w6,
     expected_contains=["I don't have enough information to answer that"])

answer_w7 = chat("When is my doctor appointment?", "classroom", use_test_memories=True)
test("W7: [Reg] Cross-mode schedule — classroom finds the hospital appointment", answer_w7,
     expected_contains=["doctor appointment"],
     must_not_contain=["I don't have enough information to answer that"])

answer_w8 = chat("What do I have tomorrow?", "classroom", use_test_memories=True)
test("W8: [Reg] Cross-mode schedule", answer_w8,
     expected_contains=["meeting", "maths", "doctor"])

answer_w9 = chat("What do I have tomorrow at 3 PM?", "classroom", use_test_memories=True)
test("W9: [Reg] Exact time — 3 PM", answer_w9, expected_contains=["maths test"])

answer_w10 = chat("What do I have tomorrow between 2 PM and 4 PM?", "classroom",
                  use_test_memories=True)
test("W10: [Reg] Time range — 2–4 PM", answer_w10,
     expected_contains=["team meeting", "maths test", "doctor appointment"])

# ==================================================
# NORMALIZER INTEGRATION TEST
# ==================================================

print("Running normalizer integration test...\n")

clear_memories()

extractor_output = {
    "simplified_text": "The professor explained convolutional neural networks.",
    "summary": "Professor explained CNNs.",
    "important_information": [
        {
            "text": "Professor explained convolutional neural networks.",
            "memory_type": "lesson",
            "importance": "medium",
            "date": TODAY_STR,
            "time": None
        }
    ]
}

normalized = normalize_extractor_output(extractor_output, "classroom")
added = add_memories(normalized)

test("X1: Normalizer → memory_store integration",
     str(memory_count()), expected_exact=str(added))

answer_x2 = chat("What did the professor explain?", "classroom", use_test_memories=False)
test("X2: Normalizer → RAG → answer (professor)",
     answer_x2, must_not_contain=["I don't have enough information to answer that"])

# ==================================================
# SECTION Y: CROSS-MODE SCHEDULE + PAYMENT REGRESSION
# ==================================================
#
# A SCHEDULE question may be answered from every SENSEA mode, because
# the schedule of the user is one calendar:
#
#   "When is my doctor appointment?"  -> the hospital appointment
#   "When is my meeting?"             -> the meeting
#   "What do I have tomorrow?"        -> every scheduled event
#
# The date, exact time and time range filters still apply, and an
# ordinary conversation question is still answered from the active
# mode only.
#
# A PAYMENT reminder is stored with the established memory type
# "reminder", so it is told apart from another reminder by its text.

print("Running Step 9 cross-mode schedule + payment tests...\n")

FALLBACK = "I don't have enough information to answer that"

YESTERDAY_STR = (TODAY - timedelta(days=1)).strftime("%Y-%m-%d")
OTHER_DAY_STR = (TODAY + timedelta(days=10)).strftime("%Y-%m-%d")

APPOINTMENT_TEXT = "Doctor appointment is tomorrow at 4 PM."
MEETING_TEXT = "Team meeting is tomorrow at 2 PM."
EXAM_TEXT = "Maths test is tomorrow at 3 PM."
LESSON_TEXT = "Teacher explained convolutional neural networks."
GROCERY_TEXT = "Reminder to buy groceries."
PAYMENT_TEXT = "Payment reminder: hospital bill payment is due tomorrow."

schedule_memories = [
    {
        "mode": "classroom",
        "text": EXAM_TEXT,
        "memory_type": "exam",
        "importance": "high",
        "date": TOMORROW_STR,
        "time": "03:00 PM"
    },
    {
        "mode": "meeting",
        "text": MEETING_TEXT,
        "memory_type": "meeting",
        "importance": "high",
        "date": TOMORROW_STR,
        "time": "02:00 PM"
    },
    {
        "mode": "hospital",
        "text": APPOINTMENT_TEXT,
        "memory_type": "appointment",
        "importance": "high",
        "date": TOMORROW_STR,
        "time": "04:00 PM"
    },
    {
        "mode": "classroom",
        "text": LESSON_TEXT,
        "memory_type": "lesson",
        "importance": "medium",
        "date": TODAY_STR,
        "time": None
    },
    {
        # An unrelated reminder on the same day: it must never be
        # returned as a payment.
        "mode": "general",
        "text": GROCERY_TEXT,
        "memory_type": "reminder",
        "importance": "low",
        "date": TOMORROW_STR,
        "time": None
    }
]

payment_memory = {
    "mode": "general",
    "text": PAYMENT_TEXT,
    "memory_type": "reminder",
    "importance": "high",
    "date": TOMORROW_STR,
    "time": None
}

clear_memories()
add_memories(schedule_memories + [payment_memory])

# Y1: A named appointment is found in another mode
answer_y1 = chat("When is my doctor appointment?", "classroom", use_test_memories=False)
test("Y1: Named appointment lookup from another mode", answer_y1,
     expected_contains=["doctor appointment"],
     must_not_contain=[FALLBACK, "team meeting", "maths test"])

# Y2: A named meeting is found in another mode
answer_y2 = chat("When is my meeting?", "general", use_test_memories=False)
test("Y2: Named meeting lookup from another mode", answer_y2,
     expected_contains=["team meeting"],
     must_not_contain=[FALLBACK, "doctor appointment", "maths test"])

# Y3: The schedule is enumerated across modes
answer_y3 = chat("What do I have tomorrow?", "classroom", use_test_memories=False)
test("Y3: Cross-mode schedule enumeration", answer_y3,
     expected_contains=["maths test", "team meeting", "doctor appointment"],
     must_not_contain=[FALLBACK])

# Y4: A named date is an exact filter
answer_y4a = chat(
    f"Do I have a hospital appointment on {OTHER_DAY_STR}?", "classroom",
    use_test_memories=False
)
test("Y4a: A date without an event returns the fallback", answer_y4a,
     expected_contains=[FALLBACK])

answer_y4b = chat(
    f"Do I have a hospital appointment on {TOMORROW_STR}?", "classroom",
    use_test_memories=False
)
test("Y4b: A named date returns only that day's event", answer_y4b,
     expected_contains=["doctor appointment"],
     must_not_contain=[FALLBACK, "maths test", "team meeting"])

answer_y4c = chat(
    f"Do I have anything scheduled on {YESTERDAY_STR}?", "classroom",
    use_test_memories=False
)
test("Y4c: Another date never leaks tomorrow's events", answer_y4c,
     expected_contains=[FALLBACK])

# Y5: The exact time and the time range filters still apply
answer_y5a = chat("What do I have tomorrow at 3 PM?", "meeting",
                  use_test_memories=False)
test("Y5a: Exact time filter across modes", answer_y5a,
     expected_contains=["maths test"],
     must_not_contain=[FALLBACK, "team meeting", "doctor appointment"])

answer_y5b = chat("What do I have tomorrow between 2 PM and 4 PM?", "hospital",
                  use_test_memories=False)
test("Y5b: Time range filter across modes", answer_y5b,
     expected_contains=["team meeting", "maths test", "doctor appointment"],
     must_not_contain=[FALLBACK])

# Y6: Ordinary conversation stays inside the active mode
answer_y6a = chat("What did my doctor tell me?", "classroom",
                  use_test_memories=False)
test("Y6a: A conversation question does not reach the hospital mode",
     answer_y6a, must_not_contain=[APPOINTMENT_TEXT])

answer_y6b = chat("What did my teacher explain?", "hospital",
                  use_test_memories=False)
test("Y6b: A conversation question does not reach the classroom mode",
     answer_y6b, must_not_contain=[LESSON_TEXT])

answer_y6c = chat("What did we learn about neural networks?", "hospital",
                  use_test_memories=False)
test("Y6c: A conversation question does not reach the classroom mode",
     answer_y6c, must_not_contain=[LESSON_TEXT])

answer_y6d = chat("Do I have anything scheduled?", "classroom",
                  use_test_memories=False)
test("Y6d: Schedule enumeration lists events, not conversation records",
     answer_y6d,
     expected_contains=["maths test", "team meeting", "doctor appointment"],
     must_not_contain=[LESSON_TEXT, GROCERY_TEXT])

# Y7: A payment reminder is found across modes
answer_y7 = chat("Do I have any pending payment tomorrow?", "classroom",
                 use_test_memories=False)
test("Y7: Payment reminder lookup across modes", answer_y7,
     expected_contains=["payment"],
     must_not_contain=[FALLBACK, GROCERY_TEXT, "maths test", "team meeting"])

# Y8: Without a matching payment reminder the standard fallback is returned
clear_memories()
add_memories(schedule_memories)

answer_y8 = chat("Do I have any pending payment tomorrow?", "classroom",
                 use_test_memories=False)
test("Y8: Payment query without a matching reminder returns the fallback",
     answer_y8, expected_equals=FALLBACK)

# ==================================================
# SECTION Z: EVENT NAME MATCHING
# ==================================================
#
# A named schedule question must match the requested event as well as
# the memory type, so a physics exam question never answers with a
# maths test. A payment question only answers with a payment reminder.

print("Running Step 9 event name matching tests...\n")

PHYSICS_TEXT = "Physics exam is tomorrow at 1 PM."
MATHS_TEXT = "Maths test is tomorrow at 3 PM."
TEAM_TEXT = "Team meeting is tomorrow at 2 PM."
DOCTOR_TEXT = "Doctor appointment is tomorrow at 4 PM."
GROCERY_TEXT_Z = "Reminder to buy groceries."
PAYMENT_TEXT_Z = "Payment reminder: electricity bill is due tomorrow."

named_memories = [
    {
        "mode": "classroom",
        "text": PHYSICS_TEXT,
        "memory_type": "exam",
        "importance": "high",
        "date": TOMORROW_STR,
        "time": "01:00 PM"
    },
    {
        "mode": "classroom",
        "text": MATHS_TEXT,
        "memory_type": "exam",
        "importance": "high",
        "date": TOMORROW_STR,
        "time": "03:00 PM"
    },
    {
        "mode": "meeting",
        "text": TEAM_TEXT,
        "memory_type": "meeting",
        "importance": "high",
        "date": TOMORROW_STR,
        "time": "02:00 PM"
    },
    {
        "mode": "hospital",
        "text": DOCTOR_TEXT,
        "memory_type": "appointment",
        "importance": "high",
        "date": TOMORROW_STR,
        "time": "04:00 PM"
    },
    {
        "mode": "hospital",
        "text": "Dentist appointment is tomorrow at 5 PM.",
        "memory_type": "appointment",
        "importance": "high",
        "date": TOMORROW_STR,
        "time": "05:00 PM"
    },
    {
        "mode": "general",
        "text": GROCERY_TEXT_Z,
        "memory_type": "reminder",
        "importance": "low",
        "date": TOMORROW_STR,
        "time": None
    }
]

payment_memory_z = {
    "mode": "general",
    "text": PAYMENT_TEXT_Z,
    "memory_type": "reminder",
    "importance": "high",
    "date": TOMORROW_STR,
    "time": None
}

clear_memories()
add_memories(named_memories + [payment_memory_z])

# Z1: The event name is matched, not only the memory type
answer_z1 = chat("When is my physics exam?", "hospital",
                 use_test_memories=False)
test("Z1: A named exam matches its event, not just its type", answer_z1,
     expected_contains=["physics exam"],
     must_not_contain=[FALLBACK, "maths test"])

# Z2: Another exam of the same type is not returned
answer_z2 = chat("When is my maths test?", "meeting",
                 use_test_memories=False)
test("Z2: Another named exam matches only its own event", answer_z2,
     expected_contains=["maths test"],
     must_not_contain=[FALLBACK, "physics exam"])

# Z3: An event that is not stored is never invented
answer_z3 = chat("When is my chemistry exam?", "classroom",
                 use_test_memories=False)
test("Z3: An unknown named event returns the standard fallback", answer_z3,
     expected_equals=FALLBACK)

# Z4: The event name also applies to meetings
answer_z4 = chat("When is my team meeting?", "classroom",
                 use_test_memories=False)
test("Z4: A named meeting matches its event, not just its type", answer_z4,
     expected_contains=["team meeting"],
     must_not_contain=[FALLBACK, "doctor appointment", "physics exam"])

# Z5: A payment query returns only payment reminders
answer_z5 = chat("Do I have a pending payment tomorrow?", "classroom",
                 use_test_memories=False)
test("Z5: A payment query never returns a grocery reminder", answer_z5,
     expected_contains=["payment"],
     must_not_contain=[FALLBACK, GROCERY_TEXT_Z, "maths test", "physics exam"])

# Z6: Without a payment reminder the answer is exactly the fallback
clear_memories()
add_memories(named_memories)

answer_z6 = chat("Do I have a pending payment tomorrow?", "classroom",
                 use_test_memories=False)
test("Z6: Without a payment reminder the exact fallback is returned",
     answer_z6, expected_equals=FALLBACK)

# Restore the full store for the remaining filters
clear_memories()
add_memories(named_memories + [payment_memory_z])

# Z7: A named appointment is cross-mode and event specific
answer_z7 = chat("When is my doctor appointment?", "classroom",
                 use_test_memories=False)
test("Z7: A named appointment stays cross-mode and event specific",
     answer_z7,
     expected_contains=["doctor appointment"],
     must_not_contain=[FALLBACK, "team meeting", "physics exam",
                       "maths test"])

# Z8: A named date is an exact filter for a named event
answer_z8 = chat(f"When is my physics exam on {TODAY_STR}?", "hospital",
                 use_test_memories=False)
test("Z8: A named date is an exact filter for a named event", answer_z8,
     expected_equals=FALLBACK)

# Z9: The event name and its date are both applied
answer_z9 = chat("When is my physics exam tomorrow?", "hospital",
                 use_test_memories=False)
test("Z9: A named event and its date are both applied", answer_z9,
     expected_contains=["physics exam"],
     must_not_contain=[FALLBACK, "maths test"])

# Z10: The exact time filter is preserved
answer_z10 = chat("What do I have tomorrow at 1 PM?", "hospital",
                  use_test_memories=False)
test("Z10: An exact time filter is preserved", answer_z10,
     expected_contains=["physics exam"],
     must_not_contain=[FALLBACK, "maths test", "team meeting",
                       "doctor appointment"])

# Z11: The time range filter is preserved
answer_z11 = chat("What do I have tomorrow between 1 PM and 3 PM?",
                  "classroom", use_test_memories=False)
test("Z11: A time range filter is preserved", answer_z11,
     expected_contains=["physics exam", "team meeting", "maths test"],
     must_not_contain=[FALLBACK, "doctor appointment"])

# Z12: A conversation question stays inside the active mode
answer_z12 = chat("What did the teacher explain about the physics exam?",
                  "hospital", use_test_memories=False)
test("Z12: A conversation question mentioning an event stays in its mode",
     answer_z12, must_not_contain=[PHYSICS_TEXT, LESSON_TEXT])

# Z13: A payment word does not turn a conversation question into a lookup
answer_z13 = chat("What did my doctor say about the payment?", "classroom",
                  use_test_memories=False)
test("Z13: A payment word does not hijack a conversation question",
     answer_z13, must_not_contain=[PAYMENT_TEXT_Z])

# Z14: A topic question does not expose another mode's event
answer_z14 = chat("What is my physics exam?", "hospital",
                  use_test_memories=False)
test("Z14: A topic question does not expose another mode's event",
     answer_z14, must_not_contain=[PHYSICS_TEXT])

# Z15: The appointment qualifier is matched, not only the type
answer_z15 = chat("When is my dentist appointment?", "classroom",
                  use_test_memories=False)
test("Z15: A dentist appointment never answers a doctor appointment",
     answer_z15,
     expected_contains=["dentist appointment"],
     must_not_contain=[FALLBACK, "doctor appointment"])

# Z16: A known appointment synonym still finds the stored event
answer_z16 = chat("When is my checkup?", "classroom",
                  use_test_memories=False)
test("Z16: A checkup finds the doctor appointment", answer_z16,
     expected_contains=["doctor appointment"],
     must_not_contain=[FALLBACK, "dentist appointment"])

# Z17: A reminder typed as a payment needs no payment word in its text
clear_memories()
add_memories([
    {
        "mode": "general",
        "text": "Electricity charge is due tomorrow.",
        "memory_type": "payment",
        "importance": "high",
        "date": TOMORROW_STR,
        "time": None
    },
    {
        "mode": "general",
        "text": GROCERY_TEXT_Z,
        "memory_type": "reminder",
        "importance": "low",
        "date": TOMORROW_STR,
        "time": None
    }
])

answer_z17 = chat("Do I have a pending payment tomorrow?", "classroom",
                  use_test_memories=False)
test("Z17: A payment-typed reminder is returned, a grocery reminder is not",
     answer_z17,
     expected_contains=["electricity charge"],
     must_not_contain=[FALLBACK, GROCERY_TEXT_Z])

# ==================================================
# RESULTS SUMMARY
# ==================================================

print("\n" + "=" * 60)
print("STEP 9 TEST RESULTS")
print("=" * 60)
for line in results_log:
    print(line)
print(f"\n{'=' * 60}")
print(f"Total: {passed + failed} | Passed: {passed} | Failed: {failed}")
print("=" * 60)

if failed == 0:
    print("\nAll tests passed!")
else:
    print(f"\n{failed} test(s) failed.")
    sys.exit(1)