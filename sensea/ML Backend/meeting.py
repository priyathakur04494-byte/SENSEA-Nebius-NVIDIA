from ai_engine import analyze_conversation


def analyze_meeting(segments):
    return analyze_conversation(
        mode="meeting",
        segments=segments
    )