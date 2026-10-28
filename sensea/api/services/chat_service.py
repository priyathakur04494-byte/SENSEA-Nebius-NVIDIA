# ============================================================
# SENSEA API — Chat / RAG / Memory Service
# ============================================================
#
# Wraps:
#   - chatbot.py   (query routing + schedule search)
#   - rag.py       (semantic retrieval)
#   - memory_store.py (in-memory vector store)
#   - normalizer.py   (extractor JSON → standard format)
#   - llm.py          (Nemotron answer generation)
#
# The sentence-transformer embedding model (all-MiniLM-L6-v2)
# is loaded once at startup via _init_embeddings().
# ============================================================

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

_embeddings_ready = False


def _init_embeddings() -> None:
    """Load the sentence-transformer model once."""
    global _embeddings_ready
    if _embeddings_ready:
        return
    try:
        import memory_store  # noqa: F401  # type: ignore
        _embeddings_ready = True
        logger.info("Memory store (MiniLM-L6-v2) loaded.")
    except Exception as exc:
        logger.warning("Memory store unavailable: %s", exc)


# ── Chat ─────────────────────────────────────────────────────

def answer_question(
    question: str,
    mode: str = "general",
    user_id: Optional[str] = None,
) -> str:
    """
    Answer a user question using RAG over stored memories.

    Routes through chatbot.answer_question() which handles:
      - schedule queries (structured search)
      - semantic queries (RAG)
      - fallback to NOT_FOUND_RESPONSE

    Returns the answer string.
    """
    _init_embeddings()

    try:
        from chatbot import chat as _chat  # type: ignore
        return _chat(
            question=question,
            current_mode=mode,
            user_id=user_id,
        )
    except Exception as exc:
        logger.error("Chat failed: %s", type(exc).__name__)
        raise RuntimeError(f"Chat query failed: {exc}") from exc


# ── Memories ─────────────────────────────────────────────────

VALID_MODES = {"general", "hospital", "meeting", "classroom"}


def add_memories(
    raw_memories: list[dict],
    mode: str,
    user_id: Optional[str] = None,
) -> int:
    """
    Normalize and store memories.

    Parameters
    ----------
    raw_memories : list[dict]
        Each dict has: text, memory_type, importance, date, time.
    mode : str
        Active mode for tagging memories.
    user_id : str | None
        Optional user scope.

    Returns
    -------
    int
        Number of successfully added memories.
    """
    _init_embeddings()

    if mode.lower() not in VALID_MODES:
        raise ValueError(
            f"Invalid mode '{mode}'. Must be one of: {sorted(VALID_MODES)}"
        )

    from memory_store import add_memory  # type: ignore
    from normalizer import normalize_memory_item  # type: ignore

    added = 0
    for item in raw_memories:
        normalized = normalize_memory_item(item, mode.lower())
        if normalized and add_memory(normalized, user_id=user_id):
            added += 1
    return added


def get_memories(
    mode: Optional[str] = None,
    user_id: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Return stored memory objects, optionally filtered by mode."""
    _init_embeddings()

    from memory_store import (  # type: ignore
        get_all_memories,
        get_memories_by_mode,
    )

    if mode:
        return get_memories_by_mode(mode.lower(), user_id=user_id)
    return get_all_memories(user_id=user_id)


def clear_memories(user_id: Optional[str] = None) -> None:
    """Clear all stored memories for the given user scope."""
    _init_embeddings()

    from memory_store import clear_memories as _clear  # type: ignore
    _clear(user_id=user_id)
