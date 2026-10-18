from general import analyze_general
from hospital import analyze_hospital
from meeting import analyze_meeting
from classroom import analyze_classroom


def analyze_mode(mode, segments):

    mode = mode.lower().strip()

    if mode == "general":
        return analyze_general(segments)

    elif mode == "hospital":
        return analyze_hospital(segments)

    elif mode == "meeting":
        return analyze_meeting(segments)

    elif mode == "classroom":
        return analyze_classroom(segments)

    else:
        raise ValueError(
            "Unknown mode. "
            "Choose general, hospital, meeting, or classroom."
        )