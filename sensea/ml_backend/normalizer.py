# normalizer.py
#
# Responsibility:
#   Convert extractor JSON output + selected mode
#   into the standard internal memory format.
#
# This is the ONLY file that knows about the extractor's JSON shape.
# If the extractor changes, only this file needs updating.
# RAG, chatbot, and memory_store never import extractor code directly.

from datetime import datetime


# --------------------------------------------------
# STANDARD MEMORY FORMAT
# This is the contract between extraction and the rest of the system.
#
# {
#     "mode":         str   — application mode (attached here, not by extractor)
#     "text":         str   — the memory sentence
#     "memory_type":  str   — e.g. "lesson", "exam", "appointment", "meeting"
#     "importance":   str   — "high" / "medium" / "low"
#     "date":         str or None — "YYYY-MM-DD" or None
#     "time":         str or None — "HH:MM AM/PM" or None
# }
# --------------------------------------------------

VALID_MODES = {"general", "meeting", "hospital", "classroom"}

REQUIRED_FIELDS = {"text", "memory_type"}


def _safe_str(value, field_name: str, default=None):
    """Return stripped string or default. Never raises."""
    if value is None:
        return default
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else default
    return default


def _validate_date(date_str):
    """Return date string if valid YYYY-MM-DD, else None."""
    if not date_str or not isinstance(date_str, str):
        return None
    try:
        datetime.strptime(date_str.strip(), "%Y-%m-%d")
        return date_str.strip()
    except ValueError:
        return None


def _validate_time(time_str):
    """Return time string if valid HH:MM AM/PM, else None."""
    if not time_str or not isinstance(time_str, str):
        return None
    t = time_str.strip()
    for fmt in ("%I:%M %p", "%H:%M"):
        try:
            parsed = datetime.strptime(t, fmt)
            # Normalise to HH:MM AM/PM
            return parsed.strftime("%I:%M %p")
        except ValueError:
            continue
    return None


def normalize_memory_item(item: dict, mode: str) -> dict | None:
    """
    Convert one extractor memory item + selected mode
    into one standard internal memory object.

    Returns None if the item is malformed (missing required fields).
    Never raises.
    """
    if not isinstance(item, dict):
        return None

    if mode not in VALID_MODES:
        return None

    text = _safe_str(item.get("text"), "text")
    if not text:
        return None  # text is mandatory

    memory_type = _safe_str(item.get("memory_type"), "memory_type", default="general")
    importance = _safe_str(item.get("importance"), "importance", default="medium")
    date = _validate_date(item.get("date"))
    time = _validate_time(item.get("time"))

    return {
        "mode":        mode,
        "text":        text,
        "memory_type": memory_type,
        "importance":  importance,
        "date":        date,
        "time":        time,
    }


def normalize_extractor_output(extractor_json: dict, mode: str) -> list[dict]:
    """
    Convert the full extractor JSON output + selected mode
    into a list of standard memory objects.

    Accepts teammate's format:
    {
      "simplified_text": "...",
      "summary": "...",
      "important_information": [
        {
          "text": "...",
          "memory_type": "...",
          "importance": "...",
          "date": null,
          "time": null
        }
      ]
    }

    Returns [] on malformed input. Never raises.
    """
    if not isinstance(extractor_json, dict):
        return []

    items = extractor_json.get("important_information")

    if not isinstance(items, list):
        return []

    normalized = []
    for item in items:
        memory = normalize_memory_item(item, mode)
        if memory is not None:
            normalized.append(memory)

    return normalized