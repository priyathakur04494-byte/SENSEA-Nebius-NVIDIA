from ai_engine import analyze_conversation


segments = [

    {
        "speaker_id": "Doctor",
        "text": "You should take this tablet twice a day after meals.",
        "translation": ""
    },

    {
        "speaker_id": "Patient",
        "text": "For how many days?",
        "translation": ""
    },

    {
        "speaker_id": "Doctor",
        "text": "Take it for five days and come back for a follow-up next Monday.",
        "translation": ""
    }
]


mode = mode = mode = "hospital"

result = analyze_conversation(
    mode=mode,
    segments=segments
)

print("\n")
print("=" * 60)
print("AI RESULT")
print("=" * 60)
print(result)