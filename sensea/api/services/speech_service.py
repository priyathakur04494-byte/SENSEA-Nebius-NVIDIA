
import subprocess
import tempfile
from pathlib import Path


# Whisper.cpp paths on your Windows computer
WHISPER_CLI = (
    Path.home()
    / "whisper.cpp"
    / "build"
    / "bin"
    / "Release"
    / "whisper-cli.exe"
)

WHISPER_MODEL = (
    Path.home()
    / "whisper.cpp"
    / "models"
    / "ggml-small.bin"
)


def transcribe_audio(audio_bytes: bytes, filename: str, language: str):
    # English-only for now; Hindi and Hinglish need their own model setup.
    if language != "eng_Latn":
        raise ValueError("Currently, only English transcription is supported.")

    if not WHISPER_CLI.is_file():
        raise RuntimeError(f"Whisper executable not found: {WHISPER_CLI}")

    if not WHISPER_MODEL.is_file():
        raise RuntimeError(f"Whisper model not found: {WHISPER_MODEL}")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        input_path = temp_path / Path(filename).name
        wav_path = temp_path / "audio.wav"
        output_prefix = temp_path / "transcript"

        input_path.write_bytes(audio_bytes)

        # Convert the uploaded audio into Whisper-compatible WAV.
        conversion = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i", str(input_path),
                "-ar", "16000",
                "-ac", "1",
                "-c:a", "pcm_s16le",
                str(wav_path),
            ],
            capture_output=True,
            text=True,
        )

        if conversion.returncode != 0:
            raise RuntimeError("Audio conversion failed: " + conversion.stderr[-1000:])

        # Run Whisper.cpp.
        result = subprocess.run(
            [
                str(WHISPER_CLI),
                "-m", str(WHISPER_MODEL),
                "-f", str(wav_path),
                "-l", "en",
                "-t", "4",
                "--beam-size", "3",
                "-nt",
                "-otxt",
                "-of", str(output_prefix),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode != 0:
            raise RuntimeError("Whisper transcription failed: " + result.stderr[-1000:])

        transcript_file = Path(str(output_prefix) + ".txt")

        if not transcript_file.is_file():
            raise RuntimeError("Whisper did not produce a transcript file.")

        transcript = transcript_file.read_text(encoding="utf-8").strip()

        return {
            "transcript": transcript,
            "language": "eng_Latn",
            "whisper_lang": "en",
        }
