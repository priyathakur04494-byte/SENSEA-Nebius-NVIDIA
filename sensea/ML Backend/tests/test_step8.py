# test_step8.py
# Automated tests for Step 8 RAG.
# Also runs all Step 7 regression tests.
# Run with: python test_step8.py

import sys
from chatbot import chat
from rag import semantic_search, retrieve_context, MIN_SIMILARITY, TEST_MEMORIES
from memory_store import add_memories, clear_memories

# --------------------------------------------------
# TEST SETUP
#
# Step 8 was written against the hardcoded TEST_MEMORIES
# fixture. Since Step 9 the default semantic_search()/chat()
# path reads from memory_store, so seed the store with the
# same fixture before running. No assertions are changed.
# --------------------------------------------------

clear_memories()
add_memories(TEST_MEMORIES)

# --------------------------------------------------
# TEST RUNNER
# --------------------------------------------------

passed = 0
failed = 0
results_log = []


def test(name: str, result: str, expected_contains: list[str] = None,
         expected_exact: str = None, must_not_contain: list[str] = None):
    global passed, failed

    ok = True
    notes = []

    if expected_contains:
        for phrase in expected_contains:
            if phrase.lower() not in result.lower():
                ok = False
                notes.append(f"MISSING: '{phrase}'")

    if expected_exact:
        if expected_exact.strip().lower() not in result.strip().lower():
            ok = False
            notes.append(f"EXPECTED EXACT: '{expected_exact}', GOT: '{result}'")

    if must_not_contain:
        for phrase in must_not_contain:
            if phrase.lower() in result.lower():
                ok = False
                notes.append(f"MUST NOT CONTAIN: '{phrase}'")

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
# SECTION A: BASIC SEMANTIC RETRIEVAL (Step 8)
# ==================================================

print("Running Step 8 RAG tests...\n")

# A1: Semantic search returns classroom lesson memories
retrieved = semantic_search(
    "What did we learn about neural network training?",
    current_mode="classroom"
)
texts = [r["memory"]["text"] for r in retrieved]
test(
    "A1: Semantic search — classroom neural networks",
    " | ".join(texts),
    expected_contains=["neural networks", "deep learning"]
)

# A2: All retrieved memories are classroom mode
all_classroom = all(r["memory"]["mode"] == "classroom" for r in retrieved)
test(
    "A2: Semantic search — all results are classroom mode",
    "classroom" if all_classroom else "non-classroom mode found",
    expected_contains=["classroom"]
)

# ==================================================
# SECTION B: MODE FILTERING (Step 8)
# ==================================================

# B1: hospital mode — neural network question should return nothing
retrieved_hospital = semantic_search(
    "What did we learn about neural network training?",
    current_mode="hospital"
)
test(
    "B1: Mode filtering — hospital blocks classroom memories",
    str(len(retrieved_hospital)),
    expected_exact="0"
)

# B2: meeting mode — neural network question returns nothing
retrieved_meeting = semantic_search(
    "What did we learn about neural network training?",
    current_mode="meeting"
)
test(
    "B2: Mode filtering — meeting blocks classroom memories",
    str(len(retrieved_meeting)),
    expected_exact="0"
)

# ==================================================
# SECTION C: SIMILARITY THRESHOLD (Step 8)
# ==================================================

# C1: Completely unrelated query in hospital mode
retrieved_unrelated = semantic_search(
    "What is the capital of France?",
    current_mode="hospital"
)
test(
    "C1: Similarity threshold — unrelated query rejected",
    str(len(retrieved_unrelated)),
    expected_exact="0"
)

# C2: Scores above threshold are >= MIN_SIMILARITY
retrieved_classroom = semantic_search(
    "What did we learn about neural network training?",
    current_mode="classroom"
)
all_above_threshold = all(r["score"] >= MIN_SIMILARITY for r in retrieved_classroom)
test(
    "C2: Similarity threshold — all scores above threshold",
    "above" if all_above_threshold else "below threshold found",
    expected_contains=["above"]
)

# ==================================================
# SECTION D: NVIDIA GENERATION (via chat())
# ==================================================

# D1: Classroom semantic question → LLM-generated answer
answer_d1 = chat(
    "What did we learn about neural networks?",
    current_mode="classroom"
)
test(
    "D1: NVIDIA generation — classroom neural networks answer",
    answer_d1,
    must_not_contain=["I don't have enough information to answer that"],
    expected_contains=[]  # LLM response; just check it's non-empty
)
# Just check answer is non-empty and not "Information not found"
if answer_d1 and "i don't have enough information to answer that" not in answer_d1.lower():
    test("D1b: NVIDIA generation — answer is non-empty", answer_d1, expected_contains=[])
else:
    test("D1b: NVIDIA generation — answer is non-empty", "", expected_contains=["non-empty answer"])

# ==================================================
# SECTION E: NOT-FOUND PROTECTION (Step 8)
# ==================================================

# E1: Unknown topic in classroom
answer_e1 = chat(
    "What is my physics exam?",
    current_mode="classroom"
)
test(
    "E1: Not-found — unknown topic (physics exam)",
    answer_e1,
    expected_contains=["I don't have enough information to answer that"]
)

# E2: Quantum physics — no matching memory
answer_e2 = chat(
    "What did we learn about quantum physics?",
    current_mode="classroom"
)
test(
    "E2: Not-found — quantum physics not in memories",
    answer_e2,
    expected_contains=["I don't have enough information to answer that"]
)

# ==================================================
# SECTION F: CURRENT-MODE PROTECTION (Step 8)
# ==================================================

# F1: Hospital semantic query should not see classroom data
answer_f1 = chat(
    "What did we learn about neural networks?",
    current_mode="hospital"
)
test(
    "F1: Mode protection — hospital cannot access classroom memories",
    answer_f1,
    expected_contains=["I don't have enough information to answer that"]
)

# F2: Classroom cannot access hospital appointment
answer_f2 = chat(
    "When is my doctor appointment?",
    current_mode="classroom"
)
test(
    "F2: Mode protection — classroom cannot access hospital appointment",
    answer_f2,
    expected_contains=["I don't have enough information to answer that"]
)

# ==================================================
# SECTION G: CROSS-MODE SCHEDULE (Step 7 + Step 8)
# ==================================================

answer_g1 = chat(
    "What do I have tomorrow?",
    current_mode="classroom"
)
test(
    "G1: Cross-mode schedule — all tomorrow events",
    answer_g1,
    expected_contains=["meeting", "maths", "doctor"]
)

# ==================================================
# SECTION H: EXACT TIME (Step 7 preserved)
# ==================================================

answer_h1 = chat(
    "What do I have tomorrow at 3 PM?",
    current_mode="classroom"
)
test(
    "H1: Exact time — 3 PM returns classroom exam",
    answer_h1,
    expected_contains=["maths test"]
)

# ==================================================
# SECTION I: TIME RANGE (Step 7 preserved)
# ==================================================

answer_i1 = chat(
    "What do I have tomorrow between 2 PM and 4 PM?",
    current_mode="classroom"
)
test(
    "I1: Time range — 2 PM to 4 PM returns all three events",
    answer_i1,
    expected_contains=["team meeting", "maths test", "doctor appointment"]
)

# ==================================================
# SECTION J: STEP 7 REGRESSION TESTS
# ==================================================

# J1: classroom — when is my exam?
answer_j1 = chat("When is my exam?", current_mode="classroom")
test("J1: Classroom — when is my exam", answer_j1, expected_contains=["maths test", "3 pm"])

# J2: classroom — what did the teacher explain?
answer_j2 = chat("What did the teacher explain?", current_mode="classroom")
test("J2: Classroom — teacher explained", answer_j2, must_not_contain=["I don't have enough information to answer that"])

# J3: classroom — what did the professor explain?
answer_j3 = chat("What did the professor explain?", current_mode="classroom")
test("J3: Classroom — professor explained", answer_j3, must_not_contain=["I don't have enough information to answer that"])

# J4: classroom — what did the lecturer explain? (no lecturer in memories)
answer_j4 = chat("What did the lecturer explain?", current_mode="classroom")
test("J4: Classroom — lecturer not found", answer_j4, expected_contains=["I don't have enough information to answer that"])

# J5: classroom — what is my physics exam? (not in memories)
answer_j5 = chat("What is my physics exam?", current_mode="classroom")
test("J5: Classroom — physics exam not found", answer_j5, expected_contains=["I don't have enough information to answer that"])

# J6: hospital — when is my appointment?
answer_j6 = chat("When is my appointment?", current_mode="hospital")
test("J6: Hospital — when is my appointment", answer_j6, expected_contains=["doctor appointment"])

# J7: hospital — when is my exam? (no exam in hospital)
answer_j7 = chat("When is my exam?", current_mode="hospital")
test("J7: Hospital — exam not found", answer_j7, expected_contains=["I don't have enough information to answer that"])

# J8: cross-mode — what do I have tomorrow at 3 PM?
answer_j8 = chat("What do I have tomorrow at 3 PM?", current_mode="meeting")
test("J8: Cross-mode exact time — 3 PM", answer_j8, expected_contains=["maths test"])

# J9: cross-mode — what do I have tomorrow between 2 PM and 4 PM?
answer_j9 = chat("What do I have tomorrow between 2 PM and 4 PM?", current_mode="hospital")
test(
    "J9: Cross-mode time range",
    answer_j9,
    expected_contains=["team meeting", "maths test", "doctor appointment"]
)

# J10: unknown question — physics exam
answer_j10 = chat("What is my physics exam?", current_mode="meeting")
test("J10: Unknown — physics exam not found in meeting", answer_j10, expected_contains=["I don't have enough information to answer that"])

# ==================================================
# RESULTS SUMMARY
# ==================================================

print("\n" + "=" * 60)
print("STEP 8 TEST RESULTS")
print("=" * 60)

for line in results_log:
    print(line)

print(f"\n{'=' * 60}")
print(f"Total: {passed + failed} | Passed: {passed} | Failed: {failed}")
print("=" * 60)

if failed == 0:
    print("\nAll tests passed!")
else:
    print(f"\n{failed} test(s) failed. Review above.")
    sys.exit(1)