VALID_MODES = {"general", "meeting", "hospital", "classroom"}


def _validate_mode(mode):
    if mode not in VALID_MODES:
        raise ValueError(f"Invalid mode: {mode}")


def answer_question(question, mode):
    _validate_mode(mode)

    from chatbot import chat

    return chat(
        question=question,
        current_mode=mode
    )


def add_memories(raw_memories, mode):
    _validate_mode(mode)

    from normalizer import normalize_memory_item
    from memory_store import add_memory

    added = 0

    for item in raw_memories:
        normalized = normalize_memory_item(item, mode)

        if normalized is not None:
            add_memory(normalized)
            added += 1

    return added


def get_memories(mode=None):
    from memory_store import get_all_memories, get_memories_by_mode

    if mode is not None:
        _validate_mode(mode)
        return get_memories_by_mode(mode)

    return get_all_memories()


def clear_memories():
    from memory_store import clear_memories, memory_count

    cleared = memory_count()
    clear_memories()

    return cleared