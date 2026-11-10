"""
deaf_assist_live.py — Deaf Assist with REAL-TIME speaker diarization

═══════════════════════════════════════════════════════════════════════

WHAT IS PRESERVED:

  ✓ Language selection    (English / Hindi / Hinglish)

  ✓ Translation targets   (22 IndicTrans2 languages + English)

  ✓ whisper.cpp + Metal transcription

  ✓ translate.py / IndicTrans2

  ✓ Hinglish romanization for display

  ✓ SPACEBAR push-to-talk behavior

  ✓ Offline → transcription only; online → translation too

  ✓ Hallucination + repetition filtering


WHAT IS NEW:

  ★ Utterances are detected DURING recording (Silero VAD) and each one is
    immediately identified (ECAPA-TDNN) → transcribed → translated → printed
    while you keep holding SPACEBAR.

  ★ Persistent Speaker 1..4 identities for the whole session.

  ★ Speaker N speaking again is recognized as the SAME Speaker N.

  ★ Hard cap of 4 speakers; short utterances never mint new IDs.

  ★ AI mode selection:

        General
        Hospital
        Meeting
        Classroom

  ★ After the conversation ends, the complete session buffer is sent
    to the selected AI mode → prompt → Nemotron.

  ★ Nemotron AI analysis is returned as STRUCTURED JSON.


PIPELINE

  Microphone
      ↓
  Audio Buffer
      ↓
  Silero VAD
      ↓
  Speech Segment
      ↓
  ECAPA Embedding
      ↓
  Speaker ID
      ↓
  whisper.cpp
      ↓
  IndicTrans2
      ↓
  Session Buffer
      ↓
  Selected AI Mode
      ↓
  Nemotron
      ↓
  Structured JSON
      ↓
  AI Analysis




REQUIREMENTS

  pip install speechbrain silero-vad sounddevice soundfile numpy pynput

  torch/torchaudio already installed for IndicTrans2
"""

from __future__ import annotations


# ══════════════════════════════════════════════════════════════
# AI MODE ROUTER
# ══════════════════════════════════════════════════════════════

from mode_router import analyze_mode


# ══════════════════════════════════════════════════════════════
# STDLIB
# ══════════════════════════════════════════════════════════════

import json
import logging
import os
import queue
import re
import subprocess
import tempfile
import threading
import urllib.request


# ══════════════════════════════════════════════════════════════
# THIRD-PARTY
# ══════════════════════════════════════════════════════════════

import numpy as np
import sounddevice as sd
import soundfile as sf


try:
    from pynput import keyboard as pynput_keyboard

except ImportError:
    print("❌  pip install pynput")
    raise SystemExit


try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate

    _ITRANS_OK = True

except ImportError:
    _ITRANS_OK = False


# ══════════════════════════════════════════════════════════════
# DIARIZATION
# ══════════════════════════════════════════════════════════════

from speaker_diarization import (
    SAMPLE_RATE,
    SpeakerRegistry,
    StreamingSegmenter,
    extract_embedding,
    load_models,
)


logging.basicConfig(
    level=logging.WARNING,
    format="%(name)s: %(message)s"
)


# ╔══════════════════════════════════════════════════════════════╗
# ║ CONFIG                                                      ║
# ╚══════════════════════════════════════════════════════════════╝

WHISPER_CPP_PATH = os.path.expanduser("~/whisper.cpp")

SPEAKERS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "speakers.json"
)

BLOCKSIZE = 1600


SPEECH_LANGUAGES = {

    "1": {
        "code": "eng_Latn",
        "name": "English",
        "whisper_lang": "en",
        "model": "small",
        "romanize": False
    },

    "2": {
        "code": "hin_Deva",
        "name": "Hindi",
        "whisper_lang": "hi",
        "model": "medium",
        "romanize": False
    },

    "3": {
        "code": "hinglish",
        "name": "Hinglish",
        "whisper_lang": "hi",
        "model": "medium",
        "romanize": True
    },

}


TARGET_LANGUAGES = {

    "eng_Latn": "English",
    "asm_Beng": "Assamese",
    "ben_Beng": "Bengali",
    "brx_Deva": "Bodo",
    "doi_Deva": "Dogri",
    "guj_Gujr": "Gujarati",
    "hin_Deva": "Hindi",
    "kan_Knda": "Kannada",
    "kas_Arab": "Kashmiri",
    "gom_Deva": "Konkani",
    "mai_Deva": "Maithili",
    "mal_Mlym": "Malayalam",
    "mar_Deva": "Marathi",
    "mni_Beng": "Manipuri",
    "npi_Deva": "Nepali",
    "ory_Orya": "Odia",
    "pan_Guru": "Punjabi",
    "san_Deva": "Sanskrit",
    "sat_Olck": "Santali",
    "snd_Arab": "Sindhi",
    "tam_Taml": "Tamil",
    "tel_Telu": "Telugu",
    "urd_Arab": "Urdu",

}


# ╔══════════════════════════════════════════════════════════════╗
# ║ UTILITIES                                                   ║
# ╚══════════════════════════════════════════════════════════════╝

def _banner(text: str) -> None:

    bar = "═" * (len(text) + 4)

    print(f"\n╔{bar}╗")
    print(f"║  {text}  ║")
    print(f"╚{bar}╝\n")


def is_online() -> bool:

    try:

        urllib.request.urlopen(
            "https://huggingface.co",
            timeout=3
        )

        return True

    except Exception:

        return False


def clean_audio(audio: np.ndarray) -> np.ndarray:

    audio = np.asarray(
        audio,
        dtype=np.float32
    )

    if not np.all(np.isfinite(audio)):

        audio = np.nan_to_num(
            audio,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

    audio -= audio.mean()

    peak = float(
        np.abs(audio).max()
    )

    if peak >= 0.01:

        audio = audio * (
            0.90 / peak
        )

    else:

        return np.zeros(
            len(audio),
            dtype=np.float32
        )

    return np.ascontiguousarray(
        np.clip(
            audio,
            -1.0,
            1.0
        )
    )


def rms(audio: np.ndarray) -> float:

    return (
        float(np.sqrt(np.mean(audio ** 2)))
        if len(audio)
        else 0.0
    )


def save_wav(audio: np.ndarray) -> str:

    tmp = tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False
    )

    sf.write(
        tmp.name,
        audio,
        SAMPLE_RATE,
        subtype="PCM_16"
    )

    tmp.close()

    return tmp.name


# ══════════════════════════════════════════════════════════════
# WHISPER
# ══════════════════════════════════════════════════════════════

ANSI_ESC = re.compile(
    r"\x1b\[[0-9;]*[mGKHF]"
)

TSTAMP_RE = re.compile(
    r"\[\d+:\d+\.\d+\s*-->\s*\d+:\d+\.\d+\]"
)

LOG_PFX = (

    "whisper_",
    "main:",
    "system_info",
    "ggml_",
    "metal_",
    "log_",
    "operator()",
    "inf ",
    "sampling",
    "encode",
    "decode",
    "autodetect",
    "init ",
    "load ",
    "mel ",
    "read_audio",
    "trying to",
    "output_",

)


def _parse_whisper(raw: str) -> str:

    lines = []

    for line in raw.splitlines():

        line = ANSI_ESC.sub(
            "",
            line
        ).strip()

        if not line:
            continue

        if any(
            line.lower().startswith(p)
            for p in LOG_PFX
        ):
            continue

        if re.match(
            r"^[\d\s.,\[\]\-:/|ms%]+$",
            line
        ):
            continue

        lines.append(line)

    text = TSTAMP_RE.sub(
        "",
        " ".join(lines)
    )

    return re.sub(
        r"\s+",
        " ",
        text
    ).strip()


def _is_hallucination(text: str) -> bool:

    if not text:
        return True

    if re.match(
        r"^\s*[\[\(]",
        text
    ):
        return True

    if re.match(
        r"^\s*[\W_]+\s*$",
        text
    ):
        return True

    lower = text.strip().lower()

    for token in (
        "thank you",
        "please subscribe",
        "subtitles by",
        "www.",
        "[music]",
        "(music)",
        "♪"
    ):

        if (
            lower.startswith(token)
            or lower.endswith(token)
        ):
            return True

    return False


def check_whisper(
    model_name: str
) -> tuple[str, str]:

    binary = os.path.join(
        WHISPER_CPP_PATH,
        "build",
        "bin",
        "whisper-cli"
    )

    model = os.path.join(
        WHISPER_CPP_PATH,
        "models",
        f"ggml-{model_name}.bin"
    )

    ok = True

    print(
        "\n── Checking Whisper ─────────────────────────"
    )

    if os.path.isfile(binary):

        print(
            "  ✅  whisper-cli found"
        )

    else:

        print(
            f"  ❌  whisper-cli not found: {binary}"
        )

        ok = False

    if os.path.isfile(model):

        print(
            f"  ✅  ggml-{model_name}.bin  "
            f"({os.path.getsize(model) // 1_048_576} MB)"
        )

    else:

        print(
            f"  ❌  ggml-{model_name}.bin not found"
        )

        ok = False

    print(
        "─────────────────────────────────────────────\n"
    )

    if not ok:
        raise SystemExit

    return binary, model


def run_whisper(
    wav_path: str,
    whisper_bin: str,
    model_path: str,
    lang: str
) -> str:

    cmd = [

        whisper_bin,

        "-m",
        model_path,

        "-f",
        wav_path,

        "-l",
        lang,

        "-t",
        "4",

        "--beam-size",
        "3",

        "-nt"

    ]

    try:

        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )

        return (
            _parse_whisper(r.stdout)
            or
            _parse_whisper(r.stderr)
        )

    except subprocess.TimeoutExpired:

        print(
            "  [Whisper timed out]"
        )

        return ""

    except Exception as e:

        print(
            f"  [Whisper error: {e}]"
        )

        return ""


# ══════════════════════════════════════════════════════════════
# HINGLISH ROMANIZATION
# ══════════════════════════════════════════════════════════════

_ITRANS_MAP = [

    ("ksh", "ksh"),
    ("jny", "gya"),

    ("Ch", "ch"),
    ("Th", "th"),
    ("Dh", "dh"),
    ("Sh", "sh"),

    ("ph", "f"),
    ("bh", "bh"),
    ("kh", "kh"),
    ("gh", "gh"),
    ("ch", "ch"),
    ("jh", "jh"),
    ("th", "th"),
    ("dh", "dh"),
    ("sh", "sh"),

    ("RI", "ri"),

    ("aa", "aa"),
    ("ii", "i"),
    ("uu", "u"),

    ("ai", "e"),
    ("au", "o"),

    ("T", "t"),
    ("D", "d"),
    ("N", "n"),
    ("M", "n"),
    ("H", "h"),

    (".n", "n"),

]


_DEVA_RE = re.compile(
    r"[\u0900-\u097F]+"
)

_HINDI_PUNC = re.compile(
    r"[।॥]+"
)


def _word_to_hinglish(w: str) -> str:

    for src, dst in _ITRANS_MAP:

        w = w.replace(
            src,
            dst
        )

    w = w.lower()

    if (
        len(w) > 2
        and w.endswith("a")
        and not w.endswith(
            ("aa", "ia", "ua")
        )
    ):

        w = w[:-1]

    if (
        len(w) > 3
        and w.endswith("aa")
    ):

        w = w[:-1]

    w = re.sub(
        r"(.)\1{2,}",
        r"\1\1",
        w
    )

    return w


def devanagari_to_hinglish(
    text: str
) -> str:

    if not text or not _ITRANS_OK:
        return text

    text = _HINDI_PUNC.sub(
        ".",
        text
    )

    def _replace(m):

        itrans = transliterate(
            m.group(0),
            sanscript.DEVANAGARI,
            sanscript.ITRANS
        )

        return " ".join(
            _word_to_hinglish(w)
            for w in itrans.split()
        )

    return _DEVA_RE.sub(
        _replace,
        text
    ).strip()


# ══════════════════════════════════════════════════════════════
# TRANSLATION BRIDGE
# ══════════════════════════════════════════════════════════════

_translate_fn = None


def get_translator():

    global _translate_fn

    if _translate_fn is not None:
        return _translate_fn

    try:

        from translate import translate_text

        _translate_fn = translate_text

        print(
            "✅  Translation module loaded."
        )

        return _translate_fn

    except Exception as e:

        print(
            f"❌  Could not load translate.py: {e}"
        )

        return None


def translate(
    text: str,
    target_lang: str,
    source_code: str
) -> str | None:

    fn = get_translator()

    if fn is None:
        return None

    if source_code in (
        "hin_Deva",
        "hinglish"
    ):

        source_code = "hin_Deva"

    try:

        return fn(
            text,
            source_code,
            target_lang
        )

    except Exception as e:

        print(
            f"  ❌  Translation error: {e}"
        )

        return None


# ╔══════════════════════════════════════════════════════════════╗
# ║ LIVE PIPELINE WORKER                                        ║
# ║ utterance → speaker ID → whisper → translate → print       ║
# ╚══════════════════════════════════════════════════════════════╝

class LivePipeline:

    """
    Consumes finished utterances from the StreamingSegmenter
    and runs:

        embedding
        speaker match
        transcription
        translation

    on a worker thread.

    The microphone callback and VAD are NEVER blocked.
    """

    def __init__(
        self,
        registry: SpeakerRegistry,
        whisper_bin: str,
        model_path: str,
        whisper_lang: str,
        romanize: bool,
        source_code: str,
        target_lang: str | None,
        target_lang_name: str,
        online: bool
    ):

        self.registry = registry

        self.whisper_bin = whisper_bin
        self.model_path = model_path
        self.whisper_lang = whisper_lang

        self.romanize = romanize

        self.source_code = source_code

        self.target_lang = target_lang
        self.target_lang_name = target_lang_name

        self.online = online

        # Complete conversation buffer
        self.segments = []

        self._q: queue.Queue = queue.Queue()

        self._print_lock = threading.Lock()

        self._worker = threading.Thread(
            target=self._run,
            daemon=True
        )

        self._worker.start()


    # ══════════════════════════════════════════════════════════
    # SEGMENTER CALLBACK
    # Must be fast
    # ══════════════════════════════════════════════════════════

    def submit(
        self,
        utterance: np.ndarray
    ) -> None:

        self._q.put(
            utterance
        )


    def get_segments(self):

        return self.segments


    def stop(self) -> None:

        self._q.put(None)

        self._worker.join(
            timeout=10
        )


    def wait_idle(self) -> None:

        self._q.join()


    def _run(self) -> None:

        while True:

            utt = self._q.get()

            try:

                if utt is None:
                    return

                self._process(
                    utt
                )

            except Exception as e:

                with self._print_lock:

                    print(
                        f"\n❌  Pipeline error: {e}"
                    )

            finally:

                self._q.task_done()


    def _process(
        self,
        utt: np.ndarray
    ) -> None:

        # ══════════════════════════════════════
        # 1. CLEAN AUDIO
        # ══════════════════════════════════════

        audio = clean_audio(
            utt
        )

        duration = (
            len(audio)
            / SAMPLE_RATE
        )

        if (
            duration < 0.5
            or rms(audio) < 0.004
        ):

            return


        # ══════════════════════════════════════
        # 2. SPEAKER DIARIZATION
        # ══════════════════════════════════════

        emb = extract_embedding(
            audio
        )

        if emb is None:
            return

        match = self.registry.identify(
            emb,
            duration
        )


        # ══════════════════════════════════════
        # 3. SPEECH RECOGNITION
        # ══════════════════════════════════════

        wav_path = None

        try:

            wav_path = save_wav(
                audio
            )

            transcript = run_whisper(
                wav_path,
                self.whisper_bin,
                self.model_path,
                self.whisper_lang
            )

        finally:

            if (
                wav_path
                and os.path.exists(wav_path)
            ):

                os.remove(
                    wav_path
                )


        if _is_hallucination(
            transcript
        ):

            return


        words = transcript.split()

        if (
            len(words) > 3
            and len(set(words)) / len(words) < 0.40
        ):

            return


        display_text = (

            devanagari_to_hinglish(
                transcript
            )

            if self.romanize

            else transcript

        )


        # ══════════════════════════════════════
        # 4. TRANSLATION
        # ══════════════════════════════════════

        translation: str | None = None

        translation_status = ""


        if self.target_lang is None:

            translation_status = (
                "(not selected)"
            )


        elif not self.online:

            translation_status = (
                "(offline — skipped)"
            )


        else:

            translation = translate(
                transcript,
                self.target_lang,
                self.source_code
            )

            if translation is None:

                translation_status = "❌ failed"


        # ══════════════════════════════════════
        # OUTPUT
        # ══════════════════════════════════════

        with self._print_lock:

            confidence = (

                ""

                if match.confident

                else "  (~low confidence)"

            )

            print()

            print(
                "━" * 24
            )

            print(
                f"{match.label}{confidence}"
            )

            print(
                "━" * 24
            )

            print()

            print(
                "Transcription:"
            )

            print(
                display_text
            )


            if translation is not None:

                print()

                print(
                    "Translation:"
                )

                print(
                    translation
                )


            elif translation_status:

                print()

                print(
                    f"Translation: {translation_status}"
                )

            print()


            # ══════════════════════════════════════
            # SAVE TO SESSION BUFFER
            # ══════════════════════════════════════

            self.segments.append(
                {
                    "speaker_id": match.label,
                    "text": transcript,
                    "translation": translation or ""
                }
            )


# ╔══════════════════════════════════════════════════════════════╗
# ║ MAIN                                                        ║
# ╚══════════════════════════════════════════════════════════════╝

mic_active = threading.Event()

stop_event = threading.Event()

AUDIO_QUEUE: queue.Queue = queue.Queue()


def audio_callback(
    indata,
    frames,
    time_info,
    status
) -> None:

    if status:

        print(
            f"\n[mic: {status}]"
        )

    if mic_active.is_set():

        AUDIO_QUEUE.put(
            np.array(
                indata[:, 0],
                dtype=np.float32,
                copy=True
            )
        )


# ══════════════════════════════════════════════════════════════
# AI MODE SELECTION
# ══════════════════════════════════════════════════════════════

def select_ai_mode():

    print(
        "\n========== SELECT AI MODE =========="
    )

    print(
        "1. General"
    )

    print(
        "2. Hospital"
    )

    print(
        "3. Meeting"
    )

    print(
        "4. Classroom"
    )

    print(
        "===================================="
    )


    while True:

        choice = input(
            "Select mode (1-4): "
        ).strip()


        if choice == "1":

            return "general"


        elif choice == "2":

            return "hospital"


        elif choice == "3":

            return "meeting"


        elif choice == "4":

            return "classroom"


        else:

            print(
                "❌ Invalid choice. "
                "Please enter 1, 2, 3, or 4."
            )


# ══════════════════════════════════════════════════════════════
# SPEAKER NAME PROMPT
# ══════════════════════════════════════════════════════════════

def _prompt_name_speakers(
    registry: SpeakerRegistry
) -> None:

    unnamed = registry.unnamed_labels()

    if not unnamed:
        return


    print(
        "─" * 48
    )

    print(
        "  UNNAMED SPEAKER(S) THIS SESSION"
    )

    print(
        "─" * 48
    )


    for label in unnamed:

        ans = input(
            f"  Save '{label}' for next time? (y/n): "
        ).strip().lower()


        if ans == "y":

            while True:

                name = input(
                    f"  Enter a name for {label}: "
                ).strip()


                if name:
                    break


                print(
                    "  Name cannot be empty."
                )


            registry.assign_name(
                label,
                name
            )


            print(
                f"  ✅  Saved as '{name}'."
            )


# ╔══════════════════════════════════════════════════════════════╗
# ║ MAIN FUNCTION                                                ║
# ╚══════════════════════════════════════════════════════════════╝

def main() -> None:

    _banner(
        "DEAF ASSIST — Live Speech · Speaker ID · Translation"
    )


    # ══════════════════════════════════════════════════════════
    # LOAD VAD + ECAPA
    # ══════════════════════════════════════════════════════════

    load_models()


    registry = SpeakerRegistry(
        SPEAKERS_FILE
    )


    # ══════════════════════════════════════════════════════════
    # STEP 1 — SPEAKING LANGUAGE
    # ══════════════════════════════════════════════════════════

    print(
        "STEP 1 — Select speaking language\n"
    )

    print(
        "  1. English"
    )

    print(
        "  2. Hindi"
    )

    print(
        "  3. Hinglish"
    )


    speech_choice = input(
        "\nEnter 1, 2, or 3: "
    ).strip()


    if speech_choice not in SPEECH_LANGUAGES:

        print(
            "\n❌  Invalid choice."
        )

        return


    cfg = SPEECH_LANGUAGES[
        speech_choice
    ]


    print(
        f"\n✅  Speaking language: {cfg['name']}"
    )


    # ══════════════════════════════════════════════════════════
    # STEP 2 — TRANSLATION TARGET
    # ══════════════════════════════════════════════════════════

    print(
        "\nSTEP 2 — Select translation language\n"
    )

    print(
        "  0. No translation\n"
    )


    exclude = {
        cfg["code"],
        "hinglish"
    }


    if cfg["code"] == "hinglish":

        exclude.add(
            "hin_Deva"
        )


    available = [

        (c, n)

        for c, n in TARGET_LANGUAGES.items()

        if c not in exclude

    ]


    for i, (code, name) in enumerate(
        available,
        1
    ):

        print(
            f"  {i:2}. {name}"
        )


    target_choice = input(
        "\nEnter your choice: "
    ).strip()


    target_lang: str | None = None

    target_lang_name = "No translation"


    if target_choice != "0":

        try:

            idx = int(
                target_choice
            )


            if not 1 <= idx <= len(available):

                raise ValueError


            target_lang, target_lang_name = (
                available[idx - 1]
            )


        except ValueError:

            print(
                "\n❌  Invalid choice — "
                "translation disabled."
            )


    print(
        f"\n✅  Translation: {target_lang_name}"
    )


    # ══════════════════════════════════════════════════════════
    # STEP 3 — SELECT AI MODE
    # ══════════════════════════════════════════════════════════

    ai_mode = select_ai_mode()


    print(
        f"\n🤖 AI Mode selected: "
        f"{ai_mode.upper()}"
    )


    # ══════════════════════════════════════════════════════════
    # STEP 4 — INTERNET + TRANSLATION
    # ══════════════════════════════════════════════════════════

    online = is_online()


    if online:

        print(
            "🌐  Internet: ONLINE"
        )


        if target_lang:

            print(
                "    Pre-loading translation model…"
            )

            get_translator()


    else:

        print(
            "📴  Internet: OFFLINE  "
            "(transcription only)"
        )


    # ══════════════════════════════════════════════════════════
    # STEP 5 — WHISPER
    # ══════════════════════════════════════════════════════════

    whisper_bin, model_path = check_whisper(
        cfg["model"]
    )


    # ══════════════════════════════════════════════════════════
    # CREATE LIVE PIPELINE
    # ══════════════════════════════════════════════════════════

    pipeline = LivePipeline(

        registry,

        whisper_bin,

        model_path,

        cfg["whisper_lang"],

        cfg["romanize"],

        cfg["code"],

        target_lang,

        target_lang_name,

        online,

    )


    # ══════════════════════════════════════════════════════════
    # CREATE SEGMENTER
    # ══════════════════════════════════════════════════════════

    segmenter = StreamingSegmenter(
        on_utterance=pipeline.submit
    )


    # ╔════════════════════════════════════════════════════════╗
    # ║ READY                                                  ║
    # ╚════════════════════════════════════════════════════════╝

    _banner(
        "READY — LIVE MODE"
    )


    print(
        f"  Speaking : {cfg['name']}"
    )

    print(
        f"  Translate: {target_lang_name}"
    )


    known = registry.summary()


    if known:

        print(
            f"  Known voices: {', '.join(known)}"
        )


    print()


    print(
        "  🎤  Hold SPACEBAR → speak "
        "(results appear LIVE per turn)"
    )

    print(
        "  🛑  Release SPACEBAR → finish "
        "any in-flight utterance"
    )

    print(
        "  ❌  Ctrl-C → end conversation "
        "and run AI analysis"
    )

    print()


    # ══════════════════════════════════════════════════════════
    # KEYBOARD HANDLERS
    # ══════════════════════════════════════════════════════════

    def on_press(key):

        if (
            key == pynput_keyboard.Key.space
            and not mic_active.is_set()
        ):

            mic_active.set()

            print(
                "🔴  Recording… "
                "(live results below — release SPACE when done)"
            )


    def on_release(key):

        if (
            key == pynput_keyboard.Key.space
            and mic_active.is_set()
        ):

            mic_active.clear()

            AUDIO_QUEUE.put(
                None
            )


    listener = pynput_keyboard.Listener(

        on_press=on_press,

        on_release=on_release

    )


    listener.start()


    # ══════════════════════════════════════════════════════════
    # RECORDING
    # ══════════════════════════════════════════════════════════

    try:

        with sd.InputStream(

            samplerate=SAMPLE_RATE,

            channels=1,

            dtype="float32",

            blocksize=BLOCKSIZE,

            callback=audio_callback

        ):

            while not stop_event.is_set():

                try:

                    chunk = AUDIO_QUEUE.get(
                        timeout=0.2
                    )

                except queue.Empty:

                    continue


                if chunk is None:

                    # Finish current VAD segment
                    segmenter.flush()

                    # Wait until Whisper /
                    # translation /
                    # speaker processing is finished
                    pipeline.wait_idle()


                else:

                    # LIVE:
                    # VAD + speaker identification +
                    # Whisper + translation

                    segmenter.feed(
                        chunk
                    )


    except KeyboardInterrupt:

        print(
            "\n\n🛑  Ending conversation..."
        )


    # ══════════════════════════════════════════════════════════
    # IMPORTANT:
    # FLUSH ANY REMAINING SPEECH
    # ══════════════════════════════════════════════════════════

    try:

        mic_active.clear()

        segmenter.flush()

        pipeline.wait_idle()

    except Exception as e:

        print(
            f"\n⚠️  Final processing warning: {e}"
        )


    # ══════════════════════════════════════════════════════════
    # SESSION BUFFER
    # ══════════════════════════════════════════════════════════

    segments = pipeline.get_segments()


    print(
        "\n========== SESSION BUFFER =========="
    )


    for segment in segments:

        print(
            segment
        )


    print(
        "====================================\n"
    )


    # ╔════════════════════════════════════════════════════════╗
    # ║ STEP 13 — NEMOTRON AI ANALYSIS                        ║
    # ╚════════════════════════════════════════════════════════╝

    if segments:

        print(
            "🤖 Analyzing conversation with Nemotron..."
        )

        print(
            f"   Mode: {ai_mode.upper()}"
        )

        print(
            f"   Segments: {len(segments)}"
        )

        print()


        try:

            # ════════════════════════════════════════════════
            # SEND COMPLETE CONVERSATION TO AI
            # ════════════════════════════════════════════════

            result = analyze_mode(
                ai_mode,
                segments
            )


            # ════════════════════════════════════════════════
            # STRUCTURED JSON OUTPUT
            # ════════════════════════════════════════════════

            print()

            print(
                "╔══════════════════════════════════════╗"
            )

            print(
                "║        🤖 AI ANALYSIS RESULT         ║"
            )

            print(
                "╚══════════════════════════════════════╝"
            )

            print()


            # Pretty-print structured JSON
            print(
                json.dumps(
                    result,
                    indent=2,
                    ensure_ascii=False
                )
            )


            print()

            print(
                "════════════════════════════════════════"
            )


        except Exception as e:

            print()

            print(
                "❌ AI analysis failed."
            )

            print(
                f"   Error: {e}"
            )

            print()


    else:

        print(
            "⚠️  No conversation segments to analyze."
        )


    # ══════════════════════════════════════════════════════════
    # SPEAKER NAME PROMPT
    # ══════════════════════════════════════════════════════════

    try:

        _prompt_name_speakers(
            registry
        )

        print(
            "🎤  Hold SPACEBAR to record again."
        )


    except KeyboardInterrupt:

        pass


    # ══════════════════════════════════════════════════════════
    # CLEANUP
    # ══════════════════════════════════════════════════════════

    finally:

        listener.stop()

        pipeline.stop()

        registry.save()

        print()

        _banner(
            "SESSION ENDED"
        )


        for line in registry.summary():

            print(
                f"    • {line}"
            )


        print()


# ══════════════════════════════════════════════════════════════
# PROGRAM ENTRY POINT
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":

    main()