# memory_store.py
#
# Responsibility:
#   Store and retrieve normalized memory objects.
#   Also manages per-memory embeddings.
#
# This is NOT RAG. It does not perform search.
# It is a simple in-memory store for the prototype.
#
# Designed so it can later be replaced with PostgreSQL + pgvector
# without changing RAG, chatbot, or normalizer.

from sentence_transformers import SentenceTransformer
import numpy as np

# --------------------------------------------------
# EMBEDDING MODEL — loaded once, shared across the system
# --------------------------------------------------

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# --------------------------------------------------
# IN-MEMORY STORE
# Each entry:
# {
#     "memory":    { ...standard memory object... },
#     "embedding": np.ndarray of shape (384,)
# }
# --------------------------------------------------

_store: list[dict] = []


def _embed(text: str) -> np.ndarray:
    """Embed a single text string. Returns shape (384,)."""
    return _model.encode([text])[0]


# --------------------------------------------------
# PUBLIC API
# --------------------------------------------------

def add_memory(memory: dict) -> bool:
    """
    Add one normalized memory object to the store.
    Computes and stores its embedding immediately.
    Returns True on success, False if memory is invalid.
    """
    if not isinstance(memory, dict):
        return False

    text = memory.get("text", "").strip()
    if not text:
        return False

    embedding = _embed(text)

    _store.append({
        "memory":    memory,
        "embedding": embedding,
    })
    return True


def add_memories(memories: list[dict]) -> int:
    """
    Add multiple normalized memory objects.
    Returns count of successfully added memories.
    """
    count = 0
    for memory in memories:
        if add_memory(memory):
            count += 1
    return count


def get_all_memories() -> list[dict]:
    """Return all stored memory objects (without embeddings)."""
    return [entry["memory"] for entry in _store]


def get_memories_by_mode(mode: str) -> list[dict]:
    """Return all memory objects for a specific mode."""
    return [
        entry["memory"]
        for entry in _store
        if entry["memory"].get("mode") == mode
    ]


def get_all_entries() -> list[dict]:
    """
    Return all store entries including embeddings.
    Used internally by RAG.
    Shape: [{"memory": {...}, "embedding": np.ndarray}, ...]
    """
    return list(_store)


def get_entries_by_mode(mode: str) -> list[dict]:
    """
    Return store entries (memory + embedding) for a specific mode.
    Used internally by RAG for mode-filtered semantic search.
    """
    return [
        entry
        for entry in _store
        if entry["memory"].get("mode") == mode
    ]


def clear_memories() -> None:
    """Clear all stored memories and embeddings."""
    _store.clear()


def memory_count() -> int:
    """Return total number of stored memories."""
    return len(_store)


def memory_count_by_mode(mode: str) -> int:
    """Return number of stored memories for a specific mode."""
    return len(get_memories_by_mode(mode))