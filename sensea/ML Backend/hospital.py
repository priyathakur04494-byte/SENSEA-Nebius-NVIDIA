from ai_engine import analyze_conversation


def analyze_hospital(segments):
    return analyze_conversation(
        mode="hospital",
        segments=segments
    )