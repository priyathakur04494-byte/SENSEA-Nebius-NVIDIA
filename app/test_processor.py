
from app.schemas import InputData
from app.processor import process_information

test_cases = [
    ("classroom", "The DSA exam is on Monday at 10 AM."),
    ("meeting", "Rahul will prepare the presentation by Friday."),
    ("hospital", "Your follow-up appointment is on September 5 at 4 PM."),
    ("general", "The college orientation will happen next Wednesday."),
]

for mode, transcript in test_cases:
    print("\n" + "=" * 60)
    print("MODE:", mode)
    print("INPUT:", transcript)

    data = InputData(
        user_id="test_user",
        session_id="test_session",
        mode=mode,
        source_language="English",
        timestamp="2026-10-25T14:45:00",
        transcript=transcript,
    )

    try:
        result = process_information(data)
        print("OUTPUT:")
        print(result)
    except Exception as e:
        print("ERROR:", e)
