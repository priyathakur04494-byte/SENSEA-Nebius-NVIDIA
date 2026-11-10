# SENSEA API — Conversation Analysis Service (wraps ai_engine.py / Nemotron)
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

VALID_MODES = {"general", "hospital", "meeting", "classroom"}


def analyze_conversation(mode: str, segments: list[dict]) -> dict[str, Any]:
    mode = mode.lower().strip()
    if mode not in VALID_MODES:
        raise ValueError(f"Invalid mode '{mode}'. Must be one of: {sorted(VALID_MODES)}")
    if not segments:
        raise ValueError("At least one conversation segment is required.")

    internal_segments = [
        {
            "speaker_id": s.get("speaker_id", "Speaker"),
            "text": s.get("text", ""),
            "translation": s.get("translation", ""),
        }
        for s in segments
    ]

    # ai_engine raises ValueError AT IMPORT if NVIDIA_API_KEY is missing.
    # That is a server problem (503), not a bad request (400).
    try:
        from ai_engine import analyze_conversation as _analyze  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"Analysis engine unavailable: {exc}") from exc

    try:
        return _analyze(mode, internal_segments)
    except ValueError as exc:
        # ai_engine raises ValueError when Nemotron returns malformed JSON.
        raise RuntimeError(f"Nemotron returned an unusable response: {exc}") from exc
    except Exception as exc:
        logger.error("Conversation analysis failed: %s", type(exc).__name__)
        raise RuntimeError(f"Conversation analysis failed: {exc}") from exc
