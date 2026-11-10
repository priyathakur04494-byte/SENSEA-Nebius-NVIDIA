
import config  # Adds Simran's ML Backend folder to the Python import path

import numpy as np
import soundfile as sf
from io import BytesIO

from tts import pipeline, VOICE


SAMPLE_RATE = 24000


def generate_speech(text: str) -> bytes:
    if not text or not text.strip():
        raise ValueError("Text cannot be empty.")

    # Generate audio using Simran's Kokoro pipeline and voice.
    generator = pipeline(
        text.strip(),
        voice=VOICE
    )

    audio_chunks = []

    for _, _, audio in generator:
        audio_chunks.append(audio)

    if not audio_chunks:
        raise RuntimeError("No audio was generated.")

    # Combine all generated audio chunks.
    full_audio = np.concatenate(audio_chunks)

    # Create a WAV file in memory instead of saving it on the backend.
    audio_buffer = BytesIO()

    sf.write(
        audio_buffer,
        full_audio,
        SAMPLE_RATE,
        format="WAV"
    )

    audio_buffer.seek(0)

    return audio_buffer.getvalue()
