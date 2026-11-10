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


def _normalize_user_id(user_id):
    """
    Validate an explicit user scope.

    None keeps the legacy, unscoped prototype behaviour. An explicit
    scope must be a non-empty string. It must come from a trusted
    authentication layer, never from the question or another arbitrary
    client field.
    """
    if user_id is None:
        return None

    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("user_id must be a non-empty string or None")

    return user_id.strip()


def _belongs_to(entry: dict, user_id) -> bool:
    """
    True when the entry belongs to the requested scope.

    user_id=None means the legacy unscoped view (all entries). This is
    retained for backward compatibility and is NOT multi-user isolation.
    """
    if user_id is None:
        return True

    return entry.get("user_id") == user_id


# --------------------------------------------------
# PUBLIC API
# --------------------------------------------------

def add_memory(memory: dict, user_id: str | None = None) -> bool:
    """
    Add one normalized memory object to the store.
    Computes and stores its embedding immediately.
    Returns True on success, False if memory is invalid.

    user_id is an optional trusted owner scope. It is stored beside the
    memory, not inside the memory object, so the existing memory schema
    is unchanged. None preserves the legacy unscoped behaviour.
    """
    if not isinstance(memory, dict):
        return False

    text = memory.get("text", "").strip()
    if not text:
        return False

    user_id = _normalize_user_id(user_id)
    embedding = _embed(text)

    _store.append({
        "memory":    memory,
        "embedding": embedding,
        "user_id":   user_id,
    })
    return True


def add_memories(
    memories: list[dict],
    user_id: str | None = None
) -> int:
    """
    Add multiple normalized memory objects.
    Returns count of successfully added memories.
    """
    count = 0
    for memory in memories:
        if add_memory(memory, user_id=user_id):
            count += 1
    return count


def get_all_memories(user_id: str | None = None) -> list[dict]:
    """
    Return stored memory objects (without embeddings).

    user_id=None returns the legacy unscoped view. An explicit user_id
    returns only that user's memories.
    """
    user_id = _normalize_user_id(user_id)

    return [
        entry["memory"]
        for entry in _store
        if _belongs_to(entry, user_id)
    ]


def get_memories_by_mode(
    mode: str,
    user_id: str | None = None
) -> list[dict]:
    """Return memory objects for a mode, optionally scoped to a user."""
    user_id = _normalize_user_id(user_id)

    return [
        entry["memory"]
        for entry in _store
        if entry["memory"].get("mode") == mode
        and _belongs_to(entry, user_id)
    ]


def get_all_entries(user_id: str | None = None) -> list[dict]:
    """
    Return store entries including embeddings.
    Used internally by RAG.
    Shape: [{"memory": {...}, "embedding": np.ndarray}, ...]

    user_id=None returns the legacy unscoped view.
    """
    user_id = _normalize_user_id(user_id)
    return [
        entry
        for entry in _store
        if _belongs_to(entry, user_id)
    ]


def get_entries_by_mode(
    mode: str,
    user_id: str | None = None
) -> list[dict]:
    """
    Return store entries (memory + embedding) for a mode, optionally
    scoped to a user. Used internally by RAG for semantic search.
    """
    user_id = _normalize_user_id(user_id)

    return [
        entry
        for entry in _store
        if entry["memory"].get("mode") == mode
        and _belongs_to(entry, user_id)
    ]


def clear_memories(user_id: str | None = None) -> None:
    """
    Clear stored memories and embeddings.

    user_id=None clears the whole legacy store. An explicit user_id
    clears only that user's records and leaves other users untouched.
    """
    user_id = _normalize_user_id(user_id)

    if user_id is None:
        _store.clear()
        return

    _store[:] = [
        entry
        for entry in _store
        if not _belongs_to(entry, user_id)
    ]


def memory_count(user_id: str | None = None) -> int:
    """Return number of stored memories in the requested scope."""
    return len(get_all_entries(user_id=user_id))


def memory_count_by_mode(
    mode: str,
    user_id: str | None = None
) -> int:
    """Return number of stored memories for a mode in the requested scope."""
    return len(get_memories_by_mode(mode, user_id=user_id))