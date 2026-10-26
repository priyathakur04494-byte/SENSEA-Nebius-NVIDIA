
import soundfile as sf
import subprocess
import platform
import numpy as np

from kokoro import KPipeline


# ============================================================
# SENSEA TEXT TO SPEECH
# ============================================================

VOICE = "bf_emma"
HINDI_VOICE = "hf_alpha"
OUTPUT_FILE = "sensea_tts.wav"

# Kokoro language codes.
#   "b" -> British English
#   "h" -> Hindi (Devanagari)
LANG_CODE_ENGLISH = "b"
LANG_CODE_HINDI = "h"

# Voice used per language code.
VOICES = {
    LANG_CODE_ENGLISH: VOICE,
    LANG_CODE_HINDI: HINDI_VOICE,
}

# Pipelines are built on demand.
# Building one loads the Kokoro model, which is slow, so it must
# not happen at import time and must not happen twice.
_pipelines = {}


def detect_lang_code(text):
    """
    Pick the Kokoro language code from the script of the text.

    Devanagari text has to be phonemised with the Hindi rules
    (lang_code="h"). The English rules pronounce it incorrectly.
    """

    for char in text:
        if "\u0900" <= char <= "\u097F":
            return LANG_CODE_HINDI

    return LANG_CODE_ENGLISH


def get_pipeline(lang_code):
    """
    Return the Kokoro pipeline for a language code.

    The pipeline is created once and then reused.
    """

    if lang_code not in _pipelines:

        _pipelines[lang_code] = KPipeline(
            lang_code=lang_code
        )

    return _pipelines[lang_code]


def __getattr__(name):
    """
    Keep `tts.pipeline` working.

    It used to be a module level constant built at import time.
    It is now built the first time it is asked for.
    """

    if name == "pipeline":
        return get_pipeline(LANG_CODE_ENGLISH)

    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )


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

    lang_code = detect_lang_code(text)
    voice = VOICES.get(lang_code, VOICE)

    print("\n🔄 Generating speech...")

    try:

        # ----------------------------------------------------
        # Get the pipeline for the script of the text
        # ----------------------------------------------------

        pipeline = get_pipeline(lang_code)

        # ----------------------------------------------------
        # Generate ALL audio chunks
        # ----------------------------------------------------

        generator = pipeline(
            text,
            voice=voice
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
        print(f"🗣️  Voice: {voice} (lang_code={lang_code})")
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

    print("\n🗣️  English voice: British English (bf_emma)")
    print("   Devanagari text is spoken with the Hindi voice "
          f"({HINDI_VOICE})")

    text = get_multiline_input()

    text_to_speech(text)
