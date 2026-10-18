
import soundfile as sf
import subprocess
import platform
import numpy as np

from kokoro import KPipeline


# ============================================================
# SENSEA TEXT TO SPEECH
# ============================================================

VOICE = "bf_emma"
OUTPUT_FILE = "sensea_tts.wav"

# British English
pipeline = KPipeline(lang_code="b")


def get_multiline_input():
    """
    Allows the user to paste multiple lines of text.

    Type END on a new line when finished.
    """

    print("\nEnter or paste your text below.")
    print("When you are finished, type END on a new line and press Enter.")
    print("-" * 55)

    lines = []

    while True:
        try:
            line = input()

        except EOFError:
            break

        # END marks the end of the text
        if line.strip().upper() == "END":
            break

        lines.append(line)

    return "\n".join(lines).strip()


def text_to_speech(text):

    if not text:
        print("\n❌ No text entered.")
        return

    print("\n🔄 Generating speech...")

    try:

        # ----------------------------------------------------
        # Generate ALL audio chunks
        # ----------------------------------------------------

        generator = pipeline(
            text,
            voice=VOICE
        )

        audio_chunks = []

        for _, _, audio in generator:
            audio_chunks.append(audio)

        if not audio_chunks:
            print("❌ No audio was generated.")
            return

        # ----------------------------------------------------
        # Combine all Kokoro chunks
        # ----------------------------------------------------

        full_audio = np.concatenate(audio_chunks)

        # ----------------------------------------------------
        # Save complete audio
        # ----------------------------------------------------

        sf.write(
            OUTPUT_FILE,
            full_audio,
            24000
        )

        duration = len(full_audio) / 24000

        print("\n✅ Speech generated successfully!")
        print(f"📁 Saved as: {OUTPUT_FILE}")
        print(f"⏱️ Duration: {duration:.2f} seconds")

        # ----------------------------------------------------
        # Play audio automatically
        # ----------------------------------------------------

        print("🔊 Playing speech...")

        if platform.system() == "Darwin":

            # macOS
            subprocess.run(
                ["afplay", OUTPUT_FILE]
            )

        elif platform.system() == "Windows":

            import os

            os.startfile(
                OUTPUT_FILE
            )

        elif platform.system() == "Linux":

            subprocess.run(
                ["aplay", OUTPUT_FILE]
            )

        else:

            print(
                "⚠️ Automatic playback is not supported "
                "on this operating system."
            )

        print("✅ Done!")

    except Exception as e:

        print("\n❌ TTS Error:")
        print(e)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 55)
    print("       SENSEA — TEXT TO SPEECH")
    print("=" * 55)

    print("\n🇬🇧 Voice: British English (bf_emma)")

    text = get_multiline_input()

    text_to_speech(text)

