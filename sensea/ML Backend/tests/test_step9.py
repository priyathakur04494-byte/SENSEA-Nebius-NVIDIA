# test_step9.py
# Step 9 tests + full Step 7/8 regression suite.
# Run: python test_step9.py

import sys
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


def test(name, result, expected_contains=None, must_not_contain=None, expected_exact=None):
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
test("Q: Unknown info returns not found", answer_q, expected_contains=["information not found"])

# R: Classroom cannot access hospital memory
answer_r = chat("When is my doctor appointment?", "classroom", use_test_memories=False)
test("R: Classroom cannot access hospital appointment", answer_r,
     expected_contains=["information not found"])

# S: Hospital cannot access classroom memory
answer_s = chat("What did the professor explain?", "hospital", use_test_memories=False)
test("S: Hospital cannot access classroom memory", answer_s,
     expected_contains=["information not found"])

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
     must_not_contain=["information not found"])

answer_v3 = chat("What did the professor explain?", "classroom", use_test_memories=True)
test("V3: [Reg] Classroom — professor explained", answer_v3,
     must_not_contain=["information not found"])

answer_v4 = chat("What did the lecturer explain?", "classroom", use_test_memories=True)
test("V4: [Reg] Classroom — lecturer not found", answer_v4,
     expected_contains=["information not found"])

answer_v5 = chat("What is my physics exam?", "classroom", use_test_memories=True)
test("V5: [Reg] Classroom — physics exam not found", answer_v5,
     expected_contains=["information not found"])

answer_v6 = chat("When is my appointment?", "hospital", use_test_memories=True)
test("V6: [Reg] Hospital — when is my appointment", answer_v6,
     expected_contains=["doctor appointment"])

answer_v7 = chat("When is my exam?", "hospital", use_test_memories=True)
test("V7: [Reg] Hospital — exam not found", answer_v7,
     expected_contains=["information not found"])

answer_v8 = chat("What do I have tomorrow at 3 PM?", "meeting", use_test_memories=True)
test("V8: [Reg] Cross-mode exact time — 3 PM", answer_v8, expected_contains=["maths test"])

answer_v9 = chat("What do I have tomorrow between 2 PM and 4 PM?", "hospital",
                 use_test_memories=True)
test("V9: [Reg] Cross-mode time range", answer_v9,
     expected_contains=["team meeting", "maths test", "doctor appointment"])

answer_v10 = chat("What is my physics exam?", "meeting", use_test_memories=True)
test("V10: [Reg] Unknown — meeting mode", answer_v10,
     expected_contains=["information not found"])

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
test("W4: [Reg] NVIDIA gen — classroom neural networks", answer_w4,
     must_not_contain=["information not found"])

answer_w5 = chat("What is my physics exam?", "classroom", use_test_memories=True)
test("W5: [Reg] Not-found — physics exam", answer_w5,
     expected_contains=["information not found"])

answer_w6 = chat("What did we learn about neural networks?", "hospital",
                 use_test_memories=True)
test("W6: [Reg] Mode protection — hospital", answer_w6,
     expected_contains=["information not found"])

answer_w7 = chat("When is my doctor appointment?", "classroom", use_test_memories=True)
test("W7: [Reg] Mode protection — classroom/hospital", answer_w7,
     expected_contains=["information not found"])

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
     answer_x2, must_not_contain=["information not found"])

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