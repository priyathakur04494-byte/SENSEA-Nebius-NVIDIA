def analyze_conversation(mode, segments):
    from mode_router import analyze_mode

    return analyze_mode(
        mode=mode,
        segments=segments
    )