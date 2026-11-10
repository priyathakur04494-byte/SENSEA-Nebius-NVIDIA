# SENSEA API — Full pipeline: diarize -> transcribe each utterance -> translate
from __future__ import annotations

import io
import logging
import os
from typing import Optional

import soundfile as sf

logger = logging.getLogger(__name__)


def run_pipeline(
    audio_bytes: bytes,
    speech_language: str = "eng_Latn",
    translate_to: Optional[str] = None,
) -> dict:
    from services import speech_service as ss  # type: ignore
    from services.translation_service import translate_text  # type: ignore

    if speech_language not in ss.SPEECH_LANGUAGE_CONFIGS:
        raise ValueError(
            f"Unsupported speech language '{speech_language}'. "
            f"Supported: {list(ss.SPEECH_LANGUAGE_CONFIGS)}"
        )
    if translate_to and speech_language == "hinglish":
        # Hinglish transcripts are romanized, which IndicTrans2 cannot read.
        raise ValueError("Translation from 'hinglish' is not supported; use 'hin_Deva'.")

    ss._init_diarization()
    if not ss._diarization_ready:
        raise RuntimeError("Speaker diarization models are not available.")

    from speaker_diarization import SpeakerRegistry, diarize_clip  # type: ignore

    wav_path = ss._audio_bytes_to_wav(audio_bytes)
    try:
        audio, _ = sf.read(wav_path, dtype="float32", always_2d=False)
    finally:
        try:
            os.unlink(wav_path)
        except OSError:
            pass
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    segments: list[dict] = []
    for match, seg in diarize_clip(audio, SpeakerRegistry()):
        buf = io.BytesIO()
        sf.write(buf, seg, 16000, subtype="PCM_16", format="WAV")
        text = ss.transcribe_audio(buf.getvalue(), language_code=speech_language)["transcript"]
        if not text.strip():
            continue

        translation = ""
        if translate_to and translate_to != speech_language:
            translation = translate_text(text, speech_language, translate_to)

        segments.append({
            "speaker_id": match.label,
            "text": text,
            "translation": translation,
            "confidence": round(float(match.similarity), 3),
            "duration_seconds": round(len(seg) / 16000, 2),
        })

    return {
        "segments": segments,
        "total_speakers": len({s["speaker_id"] for s in segments}),
    }
