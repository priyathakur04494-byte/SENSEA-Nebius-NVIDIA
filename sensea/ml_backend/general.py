from ai_engine import analyze_conversation


def analyze_general(segments):
    return analyze_conversation(
        mode="general",
        segments=segments
    )