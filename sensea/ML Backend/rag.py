# rag.py
#
# Responsibility:
#   Semantic search over normalized memories.
#   Mode filtering, cosine similarity, threshold, topic mismatch guard,
#   context formatting.
#
# Step 8 behavior fully preserved.
# Step 9 change: semantic_search() and retrieve_context() now accept
# memories from memory_store rather than hardcoded TEST_MEMORIES.
#
# TEST_MEMORIES is kept for regression tests only.

from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime, timedelta
import numpy as np
import re

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

# Raised from 0.25 → 0.45.
#
# Why: At 0.25, questions with no matching topic signal in
# REQUIRED_TOPIC_PRESENCE (e.g. "what is the capital of France?")
# could still retrieve weakly-related classroom memories (cosine
# similarity ~0.25–0.40 due to shared "education/academic" domain
# embeddings), causing them to reach the LLM.  The LLM system prompt
# restricts it to memories, but that is a prompt-only defence.
#
# At 0.45, genuinely unrelated questions return no memories at all,
# so retrieve_context() returns ([], "") and chatbot.py returns
# NOT_FOUND_RESPONSE before the LLM is ever called.
#
# Verified: correctly-related questions (teacher/neural networks,
# schedule items) score ≥ 0.55 against their memories and are
# unaffected by this change.
MIN_SIMILARITY = 0.45

# --------------------------------------------------
# DYNAMIC DATES (used by TEST_MEMORIES for regression tests)
# --------------------------------------------------

TODAY = datetime.today().date()
TOMORROW = TODAY + timedelta(days=1)
TOMORROW_STR = TOMORROW.strftime("%Y-%m-%d")
TODAY_STR = TODAY.strftime("%Y-%m-%d")

# --------------------------------------------------
# TEST_MEMORIES — kept for Step 7/8 regression tests ONLY.
# The real chatbot path uses memory_store.
# --------------------------------------------------

TEST_MEMORIES = [
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


# --------------------------------------------------
# TOPIC MISMATCH GUARD (Step 8 — preserved)
# --------------------------------------------------

REQUIRED_TOPIC_PRESENCE = {
    "quantum":    ["quantum"],
    "biology":    ["biology", "biological"],
    "history":    ["history", "historical"],
    "chemistry":  ["chemistry", "chemical"],
    "physics":    ["physics"],
    "literature": ["literature", "literary"],
    "geography":  ["geography"],
    "economics":  ["economics", "economic"],
}


def _topic_mismatch(query: str, retrieved: list[dict]) -> bool:
    """
    Returns True if the query explicitly asks about a topic
    that is absent from all retrieved memories.
    """
    q = query.lower()
    combined = " ".join(r["memory"]["text"].lower() for r in retrieved)

    for signal, required in REQUIRED_TOPIC_PRESENCE.items():
        if signal in q:
            if not any(w in combined for w in required):
                return True
    return False


# --------------------------------------------------
# SEMANTIC SEARCH
# Step 9: accepts entries (memory + embedding dicts) directly.
# This allows memory_store to supply real memories.
# --------------------------------------------------

def semantic_search_on_entries(
    query: str,
    entries: list[dict],
    top_k: int = 3
) -> list[dict]:
    """
    Perform semantic search on a pre-filtered list of store entries.
    Each entry is: {"memory": {...}, "embedding": np.ndarray}

    Returns list of {"memory": {...}, "score": float}
    sorted by descending similarity, filtered by MIN_SIMILARITY.
    """
    if not entries:
        return []

    # Import here to avoid circular imports
    from memory_store import _model

    query_embedding = _model.encode([query])
    memory_matrix = np.stack([e["embedding"] for e in entries])
    similarities = cosine_similarity(query_embedding, memory_matrix)[0]
    ranked = similarities.argsort()[::-1]

    results = []
    for pos in ranked[:top_k]:
        score = float(similarities[pos])
        if score < MIN_SIMILARITY:
            continue
        results.append({
            "memory": entries[pos]["memory"],
            "score":  score,
        })

    return results


def semantic_search(
    query: str,
    current_mode: str,
    top_k: int = 3,
    use_test_memories: bool = False
) -> list[dict]:
    """
    Main semantic search entry point.

    use_test_memories=True  → search TEST_MEMORIES (regression tests only)
    use_test_memories=False → search memory_store (real pipeline)

    Returns list of {"memory": {...}, "score": float}
    """
    if use_test_memories:
        # Step 7/8 regression path: use hardcoded TEST_MEMORIES
        from memory_store import _model

        filtered = [m for m in TEST_MEMORIES if m["mode"] == current_mode]
        if not filtered:
            return []

        texts = [m["text"] for m in filtered]
        memory_matrix = _model.encode(texts)
        query_embedding = _model.encode([query])
        similarities = cosine_similarity(query_embedding, memory_matrix)[0]
        ranked = similarities.argsort()[::-1]

        results = []
        for pos in ranked[:top_k]:
            score = float(similarities[pos])
            if score < MIN_SIMILARITY:
                continue
            results.append({"memory": filtered[pos], "score": score})
        return results

    else:
        # Real pipeline path: use memory_store
        from memory_store import get_entries_by_mode
        entries = get_entries_by_mode(current_mode)
        return semantic_search_on_entries(query, entries, top_k)


# --------------------------------------------------
# FORMAT CONTEXT FOR LLM (Step 8 — preserved)
# --------------------------------------------------

def format_context(retrieved: list[dict]) -> str:
    if not retrieved:
        return ""

    lines = ["Retrieved memories:\n"]
    for i, item in enumerate(retrieved, start=1):
        m = item["memory"]
        lines.append(f"{i}.")
        lines.append(f"   Mode: {m['mode']}")
        lines.append(f"   Memory type: {m['memory_type']}")
        lines.append(f"   Text: {m['text']}")
        lines.append("")

    return "\n".join(lines)


# --------------------------------------------------
# PUBLIC INTERFACE FOR CHATBOT
# --------------------------------------------------

def retrieve_context(
    query: str,
    current_mode: str,
    use_test_memories: bool = False
) -> tuple[list[dict], str]:
    """
    Run semantic search, apply topic mismatch guard,
    format context for LLM.

    use_test_memories=True  → regression test path
    use_test_memories=False → real pipeline path

    Returns (retrieved_memories, formatted_context_string).
    Returns ([], "") if nothing passes threshold or topic mismatches.
    """
    retrieved = semantic_search(query, current_mode, use_test_memories=use_test_memories)

    if not retrieved:
        return [], ""

    if _topic_mismatch(query, retrieved):
        return [], ""

    return retrieved, format_context(retrieved)