
"""
Deaf Assist — Transcription + Translation + Speaker Identification

FLOW
----

1. Select speaking language:
       English
       Hindi
       Hinglish

2. Select translation language:
       No translation
       OR supported IndicTrans2 language

3. Hold SPACEBAR and speak.

4. Release SPACEBAR.

5. Whisper transcribes the speech.

6. Translation runs using translate.py when online.

7. Speaker diarization maintains up to 4 speakers
   across the WHOLE conversation.

8. Press ESC to finish the conversation.

9. Unknown speakers can be given names and saved.

IMPORTANT
---------
Speaker diarization is performed on the accumulated conversation
because running diarization independently on every utterance can
create inconsistent speaker IDs.

The current saved names are conversation-local IDs.
Persistent voice recognition across future conversations will be
added separately using speaker embeddings.
"""

import os
import re
import json
import time
import queue
import threading
import tempfile
import urllib.request
import subprocess

import numpy as np
import sounddevice as sd
import soundfile as sf

from pynput import keyboard

from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

import sherpa_onnx


# ============================================================
# CONFIG
# ============================================================

SAMPLE_RATE = 16000
BLOCKSIZE = 1600

# Maximum number of speakers for this prototype
MAX_SPEAKERS = 4

# Whisper
WHISPER_CPP_PATH = os.path.expanduser("~/whisper.cpp")

# Speaker diarization models
SPEAKER_MODEL_DIR = os.path.join(
    os.getcwd(),
    "speaker_models"
)

SEGMENTATION_MODEL = os.path.join(
    SPEAKER_MODEL_DIR,
    "sherpa-onnx-pyannote-segmentation-3-0",
    "model.onnx"
)

EMBEDDING_MODEL = os.path.join(
    SPEAKER_MODEL_DIR,
    "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
)

# Saved speaker names
SPEAKER_NAMES_FILE = os.path.join(
    os.getcwd(),
    "speaker_names.json"
)


# ============================================================
# GLOBAL STATE
# ============================================================

recording = False
stop_program = False

session_id = 0

lock = threading.Lock()

audio_queue = queue.Queue()
transcription_queue = queue.Queue()

# Every released Spacebar recording is stored here.
conversation_chunks = []

# Information about each spoken chunk.
#
# {
#   "audio": np.ndarray,
#   "start": float,
#   "end": float,
#   "transcript": str,
#   "translation": str
# }
utterances = []

conversation_start_time = None


# ============================================================
# SPEECH LANGUAGES
# ============================================================

SPEECH_LANGUAGES = {

    "1": {
        "code": "eng_Latn",
        "name": "English",
        "whisper_lang": "en",
        "model": "small",
        "romanize": False,
    },

    "2": {
        "code": "hin_Deva",
        "name": "Hindi",
        "whisper_lang": "hi",
        "model": "medium",
        "romanize": False,
    },

    "3": {
        "code": "hinglish",
        "name": "Hinglish",

        # We intentionally force Hindi recognition.
        #
        # Whisper produces proper Devanagari which can then be
        # translated correctly by IndicTrans2.
        #
        # The display is converted to casual Roman Hinglish.
        "whisper_lang": "hi",
        "model": "medium",
        "romanize": True,
    },
}


# ============================================================
# TARGET LANGUAGES
# ============================================================

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


# ============================================================
# INTERNET CHECK
# ============================================================

def is_online():

    try:

        urllib.request.urlopen(
            "https://huggingface.co",
            timeout=3
        )

        return True

    except Exception:

        return False


# ============================================================
# WHISPER
# ============================================================

def get_whisper_paths(model_name):

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

    return binary, model


def check_whisper_setup(model_name):

    binary, model = get_whisper_paths(model_name)

    print()
    print("── Checking Whisper setup ─────────────────────")

    if not os.path.isfile(binary):

        print("❌ whisper-cli NOT found:")
        print(binary)
        raise SystemExit

    print("✅ whisper-cli found")

    if not os.path.isfile(model):

        print("❌ Whisper model NOT found:")
        print(model)
        raise SystemExit

    size_mb = os.path.getsize(model) // (1024 * 1024)

    print(
        f"✅ model: ggml-{model_name}.bin "
        f"({size_mb} MB)"
    )

    print("───────────────────────────────────────────────")

    return binary, model


# ============================================================
# HINGLISH CONVERSION
# ============================================================

ITRANS_MAP = [

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


DEVANAGARI_RE = re.compile(
    r"[\u0900-\u097F]+"
)

HINDI_PUNCT = re.compile(
    r"[।॥]+"
)


def word_to_hinglish(word):

    result = word

    for src, dst in ITRANS_MAP:

        result = result.replace(src, dst)

    result = result.lower()

    if (
        len(result) > 2
        and result.endswith("a")
        and not result.endswith(
            ("aa", "ia", "ua")
        )
    ):
        result = result[:-1]

    if len(result) > 3 and result.endswith("aa"):

        result = result[:-1]

    result = re.sub(
        r"(.)\1{2,}",
        r"\1\1",
        result
    )

    return result


def devanagari_to_hinglish(text):

    if not text:
        return text

    text = HINDI_PUNCT.sub(
        ".",
        text
    )

    def replace(match):

        itrans = transliterate(
            match.group(0),
            sanscript.DEVANAGARI,
            sanscript.ITRANS
        )

        return " ".join(
            word_to_hinglish(word)
            for word in itrans.split()
        )

    return DEVANAGARI_RE.sub(
        replace,
        text
    ).strip()


# ============================================================
# WHISPER OUTPUT PARSER
# ============================================================

ANSI_ESCAPE = re.compile(
    r"\x1b\[[0-9;]*[mGKHF]"
)

TIMESTAMP_RE = re.compile(
    r"\[\d+:\d+\.\d+\s*-->\s*\d+:\d+\.\d+\]"
)

LOG_PREFIXES = (

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


def parse_output(raw):

    lines = []

    for line in raw.splitlines():

        line = ANSI_ESCAPE.sub(
            "",
            line
        ).strip()

        if not line:
            continue

        if any(
            line.lower().startswith(prefix)
            for prefix in LOG_PREFIXES
        ):
            continue

        if re.match(
            r"^[\d\s.,\[\]\-:/|ms%]+$",
            line
        ):
            continue

        lines.append(line)

    transcript = " ".join(lines)

    transcript = TIMESTAMP_RE.sub(
        "",
        transcript
    )

    transcript = re.sub(
        r"\s+",
        " ",
        transcript
    ).strip()

    return transcript


# ============================================================
# AUDIO CLEANING
# ============================================================

def clean_audio(audio):

    audio = np.asarray(
        audio,
        dtype=np.float32
    )

    if not np.all(
        np.isfinite(audio)
    ):

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


# ============================================================
# SAVE TEMP WAV
# ============================================================

def save_wav(audio):

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


# ============================================================
# RUN WHISPER
# ============================================================

WHISPER_BIN = None
MODEL_PATH = None
whisper_lang = None


def run_whisper(audio):

    wav_path = save_wav(audio)

    command = [

        WHISPER_BIN,

        "-m",
        MODEL_PATH,

        "-f",
        wav_path,

        "-l",
        whisper_lang,

        "-t",
        "4",

        "--beam-size",
        "3",

        "-nt",
    ]

    try:

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=30
        )

        transcript = parse_output(
            result.stdout
        )

        if not transcript:

            transcript = parse_output(
                result.stderr
            )

        return transcript

    except subprocess.TimeoutExpired:

        print(
            "⚠️ Whisper timed out."
        )

        return ""

    except Exception as e:

        print(
            f"❌ Whisper error: {e}"
        )

        return ""

    finally:

        if os.path.exists(wav_path):

            os.remove(wav_path)


# ============================================================
# TRANSLATION
# ============================================================

translation_function = None


def load_translation():

    global translation_function

    if translation_function is not None:

        return translation_function

    print()
    print("Loading translation module...")

    try:

        from translate import translate_text

        translation_function = translate_text

        print(
            "✅ Translation module ready."
        )

        return translation_function

    except Exception as e:

        print()
        print(
            "❌ Could not load translation.py"
        )

        print(
            f"Error: {e}"
        )

        return None


# ============================================================
# SPEAKER DIARIZATION
# ============================================================

def create_diarization():

    print()
    print(
        "Loading speaker diarization model..."
    )

    if not os.path.isfile(
        SEGMENTATION_MODEL
    ):

        raise FileNotFoundError(
            f"Segmentation model missing:\n"
            f"{SEGMENTATION_MODEL}"
        )

    if not os.path.isfile(
        EMBEDDING_MODEL
    ):

        raise FileNotFoundError(
            f"Speaker embedding model missing:\n"
            f"{EMBEDDING_MODEL}"
        )

    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(

        segmentation=
        sherpa_onnx.OfflineSpeakerSegmentationModelConfig(

            pyannote=
            sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(

                model=SEGMENTATION_MODEL
            )
        ),

        embedding=
        sherpa_onnx.SpeakerEmbeddingExtractorConfig(

            model=EMBEDDING_MODEL
        ),

        clustering=
        sherpa_onnx.FastClusteringConfig(

            num_clusters=MAX_SPEAKERS,

            threshold=0.5
        ),

        min_duration_on=0.3,

        min_duration_off=0.5
    )

    if not config.validate():

        raise RuntimeError(
            "Speaker diarization configuration is invalid."
        )

    diarizer = sherpa_onnx.OfflineSpeakerDiarization(
        config
    )

    print(
        "✅ Speaker diarization model ready."
    )

    print(
        f"Maximum speakers: {MAX_SPEAKERS}"
    )

    return diarizer


# ============================================================
# DIARIZE COMPLETE CONVERSATION
# ============================================================

def diarize_conversation(
    diarizer,
    audio
):

    if audio is None:
        return []

    if len(audio) < SAMPLE_RATE:

        return []

    audio = clean_audio(audio)

    # Sherpa expects its own sample rate.
    if diarizer.sample_rate != SAMPLE_RATE:

        raise RuntimeError(
            f"Diarizer expects "
            f"{diarizer.sample_rate} Hz, "
            f"but microphone is {SAMPLE_RATE} Hz."
        )

    print()
    print(
        "🔊 Updating speaker identification..."
    )

    start_time = time.time()

    raw_result = diarizer.process(audio)

    elapsed = time.time() - start_time

    # Newer sherpa_onnx versions return a plain list of segments.
    # Older versions return an object with .sort_by_start_time().
    if isinstance(raw_result, list):
        segments_raw = sorted(
            raw_result,
            key=lambda s: s.start
        )
    else:
        segments_raw = raw_result.sort_by_start_time()

    segments = []

    for segment in segments_raw:

        segments.append({
            "start": float(segment.start),
            "end": float(segment.end),
            "speaker": int(segment.speaker),
        })

    num_speakers = len(
        set(s["speaker"] for s in segments)
    )

    print(
        f"Speaker analysis: "
        f"{elapsed:.2f}s"
    )

    print(
        f"Detected speakers: "
        f"{num_speakers}"
    )
    return segments


# ============================================================
# SPEAKER ASSIGNMENT
# ============================================================

def speaker_for_utterance(
    start,
    end,
    diarization_segments
):

    if not diarization_segments:

        return None

    best_speaker = None
    best_overlap = 0.0

    for segment in diarization_segments:

        overlap_start = max(
            start,
            segment["start"]
        )

        overlap_end = min(
            end,
            segment["end"]
        )

        overlap = max(
            0.0,
            overlap_end - overlap_start
        )

        if overlap > best_overlap:

            best_overlap = overlap

            best_speaker = segment["speaker"]

    return best_speaker


# ============================================================
# SAVED SPEAKER NAMES
# ============================================================

def load_speaker_names():

    if not os.path.isfile(
        SPEAKER_NAMES_FILE
    ):

        return {}

    try:

        with open(
            SPEAKER_NAMES_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)

    except Exception:

        return {}


def save_speaker_names(names):

    with open(
        SPEAKER_NAMES_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            names,
            f,
            indent=2,
            ensure_ascii=False
        )


# ============================================================
# DISPLAY FINAL CONVERSATION
# ============================================================

def display_conversation(
    diarization_segments
):

    print()
    print()
    print(
        "╔══════════════════════════════════════════════╗"
    )
    print(
        "║             CONVERSATION RESULT             ║"
    )
    print(
        "╚══════════════════════════════════════════════╝"
    )

    if not utterances:

        print(
            "\nNo speech recorded."
        )

        return

    for index, utterance in enumerate(
        utterances,
        start=1
    ):

        speaker = speaker_for_utterance(
            utterance["start"],
            utterance["end"],
            diarization_segments
        )

        if speaker is None:

            speaker_label = "Unknown Speaker"

        else:

            speaker_label = (
                f"Speaker {speaker + 1}"
            )

        print()
        print(
            "----------------------------------------"
        )

        print(
            f"{speaker_label}"
        )

        print(
            f"Transcript: "
            f"{utterance['display_transcript']}"
        )

        if utterance["translation"]:

            print(
                f"Translation: "
                f"{utterance['translation']}"
            )

    print()
    print(
        "========================================"
    )


# ============================================================
# ASK TO NAME SPEAKERS
# ============================================================

def ask_speaker_names(
    diarization_segments
):

    if not diarization_segments:

        print(
            "\nNo speakers were detected."
        )

        return

    detected_ids = sorted(
        set(
            segment["speaker"]
            for segment in diarization_segments
        )
    )

    saved_names = load_speaker_names()

    print()
    print(
        "╔══════════════════════════════════════════════╗"
    )
    print(
        "║              SPEAKER NAMES                  ║"
    )
    print(
        "╚══════════════════════════════════════════════╝"
    )

    for speaker_id in detected_ids:

        key = str(speaker_id)

        existing = saved_names.get(
            key
        )

        if existing:

            print()
            print(
                f"Speaker {speaker_id + 1} "
                f"was previously named: "
                f"{existing}"
            )

            continue

        print()
        print(
            f"Unknown Speaker {speaker_id + 1}"
        )

        answer = input(
            "Would you like to name and save this speaker? (y/n): "
        ).strip().lower()

        if answer == "y":

            name = input(
                "Enter speaker name: "
            ).strip()

            if name:

                saved_names[key] = name

                print(
                    f"✅ Saved as {name}"
                )

    save_speaker_names(
        saved_names
    )


# ============================================================
# MENU
# ============================================================

def choose_languages():

    print()
    print(
        "╔══════════════════════════════════════════════╗"
    )
    print(
        "║        DEAF ASSIST — AI CONVERSATION        ║"
    )
    print(
        "║     Whisper + Translation + Speakers        ║"
    )
    print(
        "╚══════════════════════════════════════════════╝"
    )

    print()
    print(
        "STEP 1 — Speaking language"
    )

    print()
    print(
        "1. English"
    )
    print(
        "2. Hindi"
    )
    print(
        "3. Hinglish"
    )

    speech_choice = input(
        "\nEnter 1, 2, or 3: "
    ).strip()

    if speech_choice not in SPEECH_LANGUAGES:

        print(
            "❌ Invalid choice."
        )

        raise SystemExit

    speech_config = SPEECH_LANGUAGES[
        speech_choice
    ]

    print()
    print(
        f"✅ Speaking language: "
        f"{speech_config['name']}"
    )

    source_language = speech_config["code"]

    print()
    print(
        "STEP 2 — Translation language"
    )

    print()
    print(
        "0. No translation"
    )

    available_targets = {}

    for code, name in TARGET_LANGUAGES.items():

        if (
            source_language == "eng_Latn"
            and code == "eng_Latn"
        ):
            continue

        if (
            source_language == "hin_Deva"
            and code == "hin_Deva"
        ):
            continue

        available_targets[
            code
        ] = name

    target_items = list(
        available_targets.items()
    )

    print()

    for index, (_, name) in enumerate(
        target_items,
        start=1
    ):

        print(
            f"{index:2}. {name}"
        )

    choice = input(
        "\nEnter translation choice: "
    ).strip()

    if choice == "0":

        target_language = None

        target_language_name = (
            "No translation"
        )

    else:

        try:

            index = int(choice)

            if not (
                1 <= index <= len(target_items)
            ):

                raise ValueError

        except ValueError:

            print(
                "❌ Invalid translation choice."
            )

            raise SystemExit

        target_language, target_language_name = (
            target_items[index - 1]
        )

    print()
    print(
        f"✅ Translation: "
        f"{target_language_name}"
    )

    return (
        speech_config,
        source_language,
        target_language,
        target_language_name
    )


# ============================================================
# KEYBOARD
# ============================================================

def on_press(key):

    global recording
    global session_id

    if key == keyboard.Key.space:

        with lock:

            if not recording:

                session_id += 1

                clear_queue(
                    audio_queue
                )

                recording = True

                print()
                print(
                    "🎤 MICROPHONE ON"
                )

                print(
                    "Speak now..."
                )


def on_release(key):

    global recording
    global stop_program

    if key == keyboard.Key.space:

        with lock:

            if recording:

                recording = False

                print()
                print(
                    "🛑 MICROPHONE OFF"
                )

    elif key == keyboard.Key.esc:

        with lock:

            stop_program = True

        return False


# ============================================================
# QUEUE CLEAR
# ============================================================

def clear_queue(q):

    while True:

        try:

            q.get_nowait()

        except queue.Empty:

            break


# ============================================================
# MICROPHONE CALLBACK
# ============================================================

def audio_callback(
    indata,
    frames,
    time_info,
    status
):

    if status:

        print(
            f"\n[mic: {status}]"
        )

    with lock:

        active = recording
        current_session = session_id

    if active:

        audio = np.array(
            indata[:, 0],
            dtype=np.float32,
            copy=True
        )

        audio = clean_audio(
            audio
        )

        audio_queue.put(
            (
                current_session,
                audio
            )
        )


# ============================================================
# PROCESS ONE UTTERANCE
# ============================================================

def process_utterance(
    audio,
    start_time,
    end_time
):

    if audio is None:

        return

    audio = clean_audio(
        audio
    )

    duration = (
        len(audio) / SAMPLE_RATE
    )

    if duration < 0.6:

        print(
            "⚠️ Recording too short."
        )

        return

    rms = float(
        np.sqrt(
            np.mean(
                audio ** 2
            )
        )
    )

    if rms < 0.004:

        print(
            "⚠️ No clear speech detected."
        )

        return

    print()
    print(
        "⏳ Transcribing..."
    )

    transcript = run_whisper(
        audio
    )

    if not transcript:

        print(
            "⚠️ No speech detected."
        )

        return

    # Remove obvious hallucinated repetition
    words = transcript.split()

    if len(words) > 3:

        unique_ratio = (
            len(set(words))
            / len(words)
        )

        if unique_ratio < 0.40:

            print(
                "⚠️ Ignored repeated Whisper hallucination."
            )

            return

    display_transcript = transcript

    translation_source_text = transcript

    if romanize_output:

        display_transcript = (
            devanagari_to_hinglish(
                transcript
            )
        )

    translated = ""

    # ========================================================
    # TRANSLATION
    # ========================================================

    if target_language is not None:

        if online:

            translator = load_translation()

            if translator:

                try:

                    if source_language in (
                        "hin_Deva",
                        "hinglish"
                    ):

                        translation_source = (
                            "hin_Deva"
                        )

                    else:

                        translation_source = (
                            source_language
                        )

                    translated = translator(
                        translation_source_text,
                        translation_source,
                        target_language
                    )

                except Exception as e:

                    print()
                    print(
                        "❌ Translation error:"
                    )

                    print(e)

        else:

            print(
                "📴 Offline — translation skipped."
            )

    # ========================================================
    # STORE UTTERANCE
    # ========================================================

    utterance = {

        "start": start_time,

        "end": end_time,

        "transcript": transcript,

        "display_transcript":
            display_transcript,

        "translation": translated,
    }

    utterances.append(
        utterance
    )

    # ========================================================
    # SHOW IMMEDIATE RESULT
    # ========================================================

    print()
    print(
        "========================================"
    )

    print(
        "TRANSCRIPTION"
    )

    print(
        display_transcript
    )

    if translated:

        print()
        print(
            f"TRANSLATION → "
            f"{target_language_name}"
        )

        print(
            translated
        )

    print(
        "========================================"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    global WHISPER_BIN
    global MODEL_PATH
    global whisper_lang
    global romanize_output
    global online
    global source_language
    global target_language
    global target_language_name
    global conversation_start_time

    # --------------------------------------------------------
    # MENU
    # --------------------------------------------------------

    (
        speech_config,
        source_language,
        target_language,
        target_language_name
    ) = choose_languages()

    whisper_lang = speech_config[
        "whisper_lang"
    ]

    romanize_output = speech_config[
        "romanize"
    ]

    # --------------------------------------------------------
    # INTERNET
    # --------------------------------------------------------

    online = is_online()

    print()

    if online:

        print(
            "🌐 Internet: ONLINE"
        )

    else:

        print(
            "📴 Internet: OFFLINE"
        )

        if target_language:

            print(
                "⚠️ Translation will be skipped."
            )

    # --------------------------------------------------------
    # WHISPER
    # --------------------------------------------------------

    WHISPER_BIN, MODEL_PATH = (
        check_whisper_setup(
            speech_config["model"]
        )
    )

    # --------------------------------------------------------
    # DIARIZATION
    # --------------------------------------------------------

    diarizer = create_diarization()

    # --------------------------------------------------------
    # DISPLAY CONFIG
    # --------------------------------------------------------

    print()
    print(
        "========================================"
    )

    print(
        f"Speaking : "
        f"{speech_config['name']}"
    )

    print(
        f"Translate: "
        f"{target_language_name}"
    )

    print(
        f"Speakers : up to "
        f"{MAX_SPEAKERS}"
    )

    print(
        "========================================"
    )

    print()
    print(
        "🎤 HOLD SPACEBAR → microphone ON"
    )

    print(
        "🛑 RELEASE SPACEBAR → microphone OFF"
    )

    print(
        "❌ PRESS ESC → finish conversation"
    )

    print()

    # --------------------------------------------------------
    # KEYBOARD
    # --------------------------------------------------------

    listener = keyboard.Listener(
        on_press=on_press,
        on_release=on_release
    )

    listener.start()

    # --------------------------------------------------------
    # AUDIO STREAM
    # --------------------------------------------------------

    conversation_start_time = time.time()

    current_audio = []

    current_session = None

    recording_start = None

    last_session_state = False

    with sd.InputStream(

        samplerate=SAMPLE_RATE,

        channels=1,

        dtype="float32",

        blocksize=BLOCKSIZE,

        callback=audio_callback
    ):

        try:

            while True:

                with lock:

                    should_stop = (
                        stop_program
                    )

                    is_recording = (
                        recording
                    )

                    current_session = (
                        session_id
                    )

                if should_stop:

                    break

                # ------------------------------------------------
                # NEW SPACEBAR SESSION
                # ------------------------------------------------

                if (
                    is_recording
                    and not last_session_state
                ):

                    current_audio = []

                    current_session = (
                        session_id
                    )

                    recording_start = (
                        time.time()
                        - conversation_start_time
                    )

                # ------------------------------------------------
                # GET MIC AUDIO
                # ------------------------------------------------

                try:

                    (
                        chunk_session,
                        audio_chunk
                    ) = audio_queue.get(
                        timeout=0.1
                    )

                    if (
                        chunk_session
                        == current_session
                    ):

                        current_audio.append(
                            audio_chunk
                        )

                except queue.Empty:

                    pass

                # ------------------------------------------------
                # SPACEBAR RELEASE
                # ------------------------------------------------

                if (
                    last_session_state
                    and not is_recording
                    and current_audio
                ):

                    final_audio = np.concatenate(
                        current_audio
                    ).copy()

                    recording_end = (
                        time.time()
                        - conversation_start_time
                    )

                    print()
                    print(
                        "Processing spoken turn..."
                    )

                    # --------------------------------------------
                    # TRANSCRIBE + TRANSLATE
                    # --------------------------------------------

                    process_utterance(
                        final_audio,
                        recording_start,
                        recording_end
                    )

                    # --------------------------------------------
                    # STORE AUDIO FOR WHOLE CONVERSATION
                    # --------------------------------------------

                    conversation_chunks.append(
                        (
                            recording_start,
                            recording_end,
                            final_audio
                        )
                    )

                    current_audio = []

                    recording_start = None

                last_session_state = (
                    is_recording
                )

        except KeyboardInterrupt:

            print(
                "\nStopped."
            )

        finally:

            listener.stop()

    # ============================================================
    # FINAL SPEAKER DIARIZATION
    # ============================================================

    if conversation_chunks:

        print()
        print(
            "========================================"
        )

        print(
            "FINAL SPEAKER IDENTIFICATION"
        )

        print(
            "========================================"
        )

        # --------------------------------------------------------
        # Build conversation audio with silence preserved.
        #
        # This lets diarization keep approximately the same
        # timeline as the actual conversation.
        # --------------------------------------------------------

        total_duration = max(
            item[1]
            for item in conversation_chunks
        )

        total_samples = int(
            total_duration
            * SAMPLE_RATE
        ) + 1

        conversation_audio = np.zeros(
            total_samples,
            dtype=np.float32
        )

        for (
            start,
            end,
            audio
        ) in conversation_chunks:

            start_sample = int(
                start
                * SAMPLE_RATE
            )

            end_sample = (
                start_sample
                + len(audio)
            )

            if end_sample > len(
                conversation_audio
            ):

                end_sample = len(
                    conversation_audio
                )

                audio = audio[
                    :end_sample
                    - start_sample
                ]

            conversation_audio[
                start_sample:end_sample
            ] = audio

        # --------------------------------------------------------
        # RUN DIARIZATION
        # --------------------------------------------------------

        diarization_segments = (
            diarize_conversation(
                diarizer,
                conversation_audio
            )
        )

        # --------------------------------------------------------
        # FINAL DISPLAY
        # --------------------------------------------------------

        display_conversation(
            diarization_segments
        )

        # --------------------------------------------------------
        # ASK NAMES
        # --------------------------------------------------------

        ask_speaker_names(
            diarization_segments
        )

    else:

        print()
        print(
            "No conversation was recorded."
        )

    print()
    print(
        "========================================"
    )

    print(
        "DEAF ASSIST SESSION FINISHED"
    )

    print(
        "========================================"
    )


if __name__ == "__main__":

    main()

