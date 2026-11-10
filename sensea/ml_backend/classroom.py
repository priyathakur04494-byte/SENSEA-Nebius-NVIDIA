from ai_engine import analyze_conversation


def analyze_classroom(segments):
    return analyze_conversation(
        mode="classroom",
        segments=segments
    )