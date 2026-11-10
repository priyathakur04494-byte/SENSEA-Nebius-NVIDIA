
import soundfile as sf
from kokoro import KPipeline

# American English
pipeline = KPipeline(lang_code="b")

text = "Hello! Welcome to Sensea. This is a test of the Kokoro text to speech system."

# Generate speech
generator = pipeline(
    text,
    voice="af_heart"
)

for i, (gs, ps, audio) in enumerate(generator):

    print("Generating speech...")
    print("Text:", text)
    print("Voice: af_heart")

    output_file = f"kokoro_output_{i}.wav"

    sf.write(
        output_file,
        audio,
        24000
    )

    print(f"✅ Saved: {output_file}")

