# ============================================================
# SENSEA API — Speech Service
# ============================================================
#
# Wraps:
#   - transcribe.py  (Whisper.cpp STT)
#   - speaker_diarization.py (Silero VAD + ECAPA-TDNN)
#
# Heavy models are loaded ONCE at startup via _init().
# The service is safe to import even when Whisper.cpp binary
# is absent — errors surface at call time, not import time.
# ============================================================

from __future__ import annotations

import io
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

# ── Lazy singletons ─────────────────────────────────────────
_registry: "Optional[SpeakerRegistry]" = None  # type: ignore[name-defined]
_diarization_ready = False


def _init_diarization() -> None:
    """Load Silero VAD + ECAPA-TDNN once."""
    global _registry, _diarization_ready
    if _diarization_ready:
        return
    try:
        from speaker_diarization import SpeakerRegistry, load_models  # type: ignore
        load_models()
        _registry = SpeakerRegistry()
        _diarization_ready = True
        logger.info("Speaker diarization models loaded.")
    except Exception as exc:
        logger.warning("Speaker diarization unavailable: %s", exc)


# ── Whisper helpers ──────────────────────────────────────────

def _get_whisper_bin(model_name: str) -> tuple[str, str]:
    """Return (binary_path, model_path) for whisper-cli."""
    from config import WHISPER_CPP_PATH  # type: ignore

    base = Path(WHISPER_CPP_PATH)

    # Binary name differs on Windows vs Unix
    if sys.platform == "win32":
        binary = base / "build" / "bin" / "Release" / "whisper-cli.exe"
        if not binary.exists():
            binary = base / "build" / "bin" / "whisper-cli.exe"
    else:
        binary = base / "build" / "bin" / "whisper-cli"

    model = base / "models" / f"ggml-{model_name}.bin"
    return str(binary), str(model)


def _run_whisper(wav_path: str, whisper_lang: str, model_name: str) -> str:
    """Invoke whisper-cli subprocess and return clean transcript."""
    import re

    binary, model = _get_whisper_bin(model_name)

    if not Path(binary).exists():
        raise FileNotFoundError(
            f"whisper-cli not found at {binary}. "
            "Build whisper.cpp or set WHISPER_CPP_PATH in .env."
        )
    if not Path(model).exists():
        raise FileNotFoundError(
            f"Whisper model ggml-{model_name}.bin not found at {model}."
        )

    cmd = [
        binary,
        "-m", model,
        "-f", wav_path,
        "-l", whisper_lang,
        "-t", "4",
        "--beam-size", "3",
        "-nt",
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        raise TimeoutError("Whisper timed out processing the audio.")

    if result.returncode != 0:
        raise RuntimeError(
            f"Whisper failed (exit {result.returncode}): {result.stderr[:300]}"
        )

    # ── strip ANSI + timestamps + log lines ─────────────────
    ANSI = re.compile(r"\x1b\[[0-9;]*[mGKHF]")
    TIMESTAMP = re.compile(r"\[\d+:\d+\.\d+\s*-->\s*\d+:\d+\.\d+\]")
    LOG_PREFIXES = (
        "whisper_", "main:", "system_info", "ggml_", "metal_",
        "log_", "operator()", "inf ", "sampling", "encode",
        "decode", "autodetect", "init ", "load ", "mel ",
        "read_audio", "trying to", "output_",
    )

    lines = []
    for line in result.stdout.splitlines():
        line = ANSI.sub("", line).strip()
        if not line:
            continue
        if any(line.lower().startswith(p) for p in LOG_PREFIXES):
            continue
        if re.match(r"^[\d\s.,\[\]\-:/|ms%]+$", line):
            continue
        lines.append(line)

    transcript = " ".join(lines)
    transcript = TIMESTAMP.sub("", transcript)
    transcript = re.sub(r"\s+", " ", transcript).strip()
    return transcript


def _audio_bytes_to_wav(audio_bytes: bytes, suffix: str = ".wav") -> str:
    """
    Save raw audio bytes (any format soundfile can read, or m4a/AAC via ffmpeg)
    to a 16 kHz mono WAV temp file.  Returns the temp file path (caller must
    delete it).
    """
    import shutil

    # ── Primary path: soundfile (handles WAV, FLAC, OGG…) ───
    try:
        buf = io.BytesIO(audio_bytes)
        audio, sr = sf.read(buf, dtype="float32", always_2d=False)
    except Exception as _sf_err:
        # ── Fallback: ffmpeg subprocess (handles m4a/AAC/MP4…) ─
        if not shutil.which("ffmpeg"):
            raise ValueError(
                "Cannot decode audio (soundfile failed and ffmpeg not found). "
                "Upload WAV/FLAC/OGG or install ffmpeg."
            ) from _sf_err

        # Write raw bytes to a temp file preserving the original suffix
        _in_tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        try:
            _in_tmp.write(audio_bytes)
            _in_tmp.close()

            _out_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            _out_tmp.close()

            try:
                subprocess.run(
                    [
                        "ffmpeg", "-y",
                        "-i", _in_tmp.name,
                        "-ar", "16000",
                        "-ac", "1",
                        "-f", "wav",
                        _out_tmp.name,
                    ],
                    capture_output=True,
                    check=True,
                    timeout=60,
                )
                audio, sr = sf.read(_out_tmp.name, dtype="float32", always_2d=False)
            except Exception as _ff_err:
                raise ValueError(
                    "Cannot decode audio. Upload WAV/FLAC/OGG, "
                    "or install ffmpeg for m4a/AAC support."
                ) from _ff_err
            finally:
                try:
                    os.unlink(_out_tmp.name)
                except OSError:
                    pass
        finally:
            try:
                os.unlink(_in_tmp.name)
            except OSError:
                pass

    # Downmix to mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Resample to 16 kHz if needed
    if sr != 16000:
        try:
            import librosa  # type: ignore
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        except ImportError:
            # Basic linear resample as fallback
            ratio = 16000 / sr
            new_len = int(len(audio) * ratio)
            audio = np.interp(
                np.linspace(0, len(audio) - 1, new_len),
                np.arange(len(audio)),
                audio,
            ).astype(np.float32)

    # Normalize
    peak = float(np.abs(audio).max())
    if peak >= 0.01:
        audio = audio * (0.90 / peak)
    else:
        audio = np.zeros_like(audio)
    audio = np.clip(audio, -1.0, 1.0).astype(np.float32)

    # Write temp WAV
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, audio, 16000, subtype="PCM_16")
    tmp.close()
    return tmp.name


# ── Public API ───────────────────────────────────────────────

SPEECH_LANGUAGE_CONFIGS = {
    "eng_Latn": {
        "name": "English",
        "whisper_lang": "en",
        "model": "base.en",
        "romanize": False,
    },
    "hin_Deva": {
        "name": "Hindi",
        "whisper_lang": "hi",
        "model": "medium",
        "romanize": False,
    },
    "hinglish": {
        "name": "Hinglish",
        "whisper_lang": "hi",
        "model": "medium",
        "romanize": True,
    },
}


def transcribe_audio(
    audio_bytes: bytes,
    language_code: str = "eng_Latn",
    suffix: str = ".wav",
) -> dict:
    """
    Transcribe uploaded audio bytes using Whisper.cpp.

    Parameters
    ----------
    audio_bytes : bytes
        Raw audio file content (WAV, FLAC, OGG, m4a/AAC with ffmpeg).
    language_code : str
        One of: 'eng_Latn', 'hin_Deva', 'hinglish'.
    suffix : str
        Original file extension (e.g. '.m4a') forwarded to ffmpeg when
        soundfile cannot parse the format directly.

    Returns
    -------
    dict with keys: transcript, language, whisper_lang
    """
    cfg = SPEECH_LANGUAGE_CONFIGS.get(language_code)
    if cfg is None:
        raise ValueError(
            f"Unsupported speech language '{language_code}'. "
            f"Supported: {list(SPEECH_LANGUAGE_CONFIGS)}"
        )

    wav_path = _audio_bytes_to_wav(audio_bytes, suffix=suffix)
    try:
        transcript = _run_whisper(
            wav_path,
            whisper_lang=cfg["whisper_lang"],
            model_name=cfg["model"],
        )

        # Hinglish display: convert Devanagari output to romanized Hinglish
        if cfg["romanize"] and transcript:
            try:
                from indic_transliteration import sanscript  # type: ignore
                from indic_transliteration.sanscript import transliterate  # type: ignore
                import re as _re
                DEVA_RE = _re.compile(r"[\u0900-\u097F]+")

                def _to_hinglish(text: str) -> str:
                    def _replace(m: object) -> str:
                        return transliterate(
                            m.group(0),
                            sanscript.DEVANAGARI,
                            sanscript.ITRANS,
                        ).lower()
                    return DEVA_RE.sub(_replace, text)

                transcript = _to_hinglish(transcript)
            except Exception as exc:
                logger.debug("Hinglish romanization failed: %s", exc)

    finally:
        try:
            os.unlink(wav_path)
        except OSError:
            pass

    return {
        "transcript": transcript,
        "language": cfg["name"],
        "whisper_lang": cfg["whisper_lang"],
    }


def diarize_audio(audio_bytes: bytes, suffix: str = ".wav") -> dict:
    """
    Run speaker diarization on uploaded audio bytes.

    Parameters
    ----------
    suffix : str
        Original file extension forwarded to ffmpeg when soundfile
        cannot parse the format (e.g. '.m4a').

    Returns a dict with keys:
        speakers: list of segment dicts
        total_speakers: int
    """
    _init_diarization()

    if not _diarization_ready or _registry is None:
        raise RuntimeError(
            "Speaker diarization models are not available. "
            "Check that speechbrain and silero-vad are installed."
        )

    from speaker_diarization import (  # type: ignore
        SpeakerRegistry,
        StreamingSegmenter,
        extract_embedding,
    )

    wav_path = _audio_bytes_to_wav(audio_bytes, suffix=suffix)
    try:
        audio, _ = sf.read(wav_path, dtype="float32", always_2d=False)
    finally:
        try:
            os.unlink(wav_path)
        except OSError:
            pass

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Use a fresh registry per request (stateless)
    registry = SpeakerRegistry()
    segments: list[dict] = []

    def _on_utterance(seg: np.ndarray) -> None:
        emb = extract_embedding(seg)
        if emb is None:
            return
        duration = len(seg) / 16000
        match = registry.identify(emb, duration)
        segments.append({
            "speaker_id": match.label,
            "is_new": match.is_new,
            "confidence": round(match.similarity, 3),
            "duration_seconds": round(duration, 2),
        })

    segmenter = StreamingSegmenter(_on_utterance)
    segmenter.feed(audio)
    segmenter.flush()

    unique_speakers = len({s["speaker_id"] for s in segments})
    return {
        "speakers": segments,
        "total_speakers": unique_speakers,
    }
