from ai_engine import analyze_conversation


segments = [
    {
        "speaker_id": "Speaker 1",
        "text": "Hello everyone, today we are going to learn neural networks.",
        "translation": "ਸਭ ਨੂੰ ਹੈਲੋ, ਅੱਜ ਅਸੀਂ ਨਿਊਰਲ ਨੈੱਟਵਰਕ ਸਿੱਖਣ ਜਾ ਰਹੇ ਹਾਂ।"
    },
    {
        "speaker_id": "Speaker 2",
        "text": "Will we learn about artificial neurons?",
        "translation": "ਕੀ ਅਸੀਂ ਨਕਲੀ ਨਿਊਰੋਨ ਬਾਰੇ ਸਿੱਖਾਂਗੇ?"
    },
    {
        "speaker_id": "Speaker 1",
        "text": "Yes, and we will also learn backpropagation.",
        "translation": "ਹਾਂ, ਅਤੇ ਅਸੀਂ ਬੈਕਪ੍ਰੋਪਾਗੇਸ਼ਨ ਵੀ ਸਿੱਖਾਂਗੇ।"
    },
    {
        "speaker_id": "Speaker 1",
        "text": "The assignment is due on Friday.",
        "translation": "ਅਸਾਈਨਮੈਂਟ ਸ਼ੁੱਕਰਵਾਰ ਨੂੰ ਜਮ੍ਹਾਂ ਕਰਵਾਉਣੀ ਹੈ।"
    }
]


print("\n========================================")
print("       DEAF ASSIST AI ANALYSIS")
print("========================================")

result = analyze_conversation(
    mode="classroom",
    segments=segments
)

print("\n===== NVIDIA RESULT =====\n")
print(result)