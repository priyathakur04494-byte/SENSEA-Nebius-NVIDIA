"""
Deaf Assist — Speech-to-Text + Translation Prototype

FLOW
----

1. Select speaking language:
       English
       Hindi
       Hinglish

2. Select translation language:
       No translation
       OR any of the 22 supported IndicTrans2 languages (+ English)

3. Hold SPACEBAR and speak.

4. Release SPACEBAR.

5. Whisper transcribes the speech.

6. If translation was selected AND internet is available:
       Transcript is translated.

7. If "No translation" was selected:
       Only transcription is displayed.

8. Offline:
       Speech-to-text continues to work.
       Translation is skipped.

Uses:
    whisper.cpp + Metal
    IndicTrans2
"""

import sounddevice as sd
import soundfile as sf
import numpy as np
import queue
import threading
import subprocess
import tempfile
import os
import re
import urllib.request

from pynput import keyboard

from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate


# ============================================================
# CONFIG
# ============================================================

WHISPER_CPP_PATH = os.path.expanduser("~/whisper.cpp")

SAMPLE_RATE = 16000
CHUNK_SECONDS = 4
BLOCKSIZE = 1600

audio_queue = queue.Queue()
transcription_queue = queue.Queue()

recording = False
stop_program = False
session_id = 0

lock = threading.Lock()


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
        # CHANGED from "auto" to "hi".
        #
        # "auto" let Whisper's language-ID decide per-chunk whether it
        # heard English or Hindi. Two problems followed from that:
        #   1. It occasionally decided "Urdu" instead of Hindi, writing
        #      Perso-Arabic script that nothing downstream could handle.
        #   2. Even when it correctly guessed Hindi/English, the mixed
        #      Roman+Devanagari output got romanized for display, and
        #      then translation tried to reverse THAT casual romanized
        #      text back into Devanagari via hinglish_to_devanagari().
        #      Casual spelling isn't valid ITRANS, so the reverse
        #      conversion produced nonsense syllables — and that
        #      nonsense, not the actual sentence, is what got sent to
        #      IndicTrans2. This was corrupting every Hinglish
        #      translation, not just specific target languages.
        #
        # Forcing "hi" always gets clean, correctly-spelled Devanagari
        # directly from Whisper — no reconstruction needed. English
        # loanwords ("coffee") get phonetically rendered in Devanagari
        # ("कॉफी") rather than staying in Roman script mid-sentence, but
        # that Devanagari is genuine and translates correctly. The
        # casual Roman "Hinglish" caption for display is still produced
        # via devanagari_to_hinglish() below — only the translation
        # input path changes.
        "whisper_lang": "hi",
        "model": "medium",
        "romanize": True,
    },
}


# ============================================================
# 22 INDIC LANGUAGE + ENGLISH TRANSLATION TARGETS
#
# These are the REAL IndicTrans2 language codes.
#
# Do NOT replace these with:
# as, bn, brx, gu, etc.
#
# IndicTrans2 expects:
# asm_Beng, ben_Beng, brx_Deva, etc.
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
# CHECK INTERNET
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
# GET WHISPER BINARY + MODEL
# ============================================================

def get_bin_and_model(model_name):

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


# ============================================================
# CHECK WHISPER SETUP
# ============================================================

def check_setup(model_name):

    binary, model = get_bin_and_model(model_name)

    ok = True

    print()
    print("── Checking Whisper setup ─────────────────────")

    if os.path.isfile(binary):

        print("  ✅ whisper-cli found")

    else:

        print("  ❌ whisper-cli NOT found:")
        print(f"     {binary}")

        print()
        print("Build Whisper with:")

        print(
            "cd ~/whisper.cpp && "
            "cmake -B build -DWHISPER_METAL=ON && "
            "cmake --build build --config Release -j8"
        )

        ok = False

    if os.path.isfile(model):

        size_mb = os.path.getsize(model) // (1024 * 1024)

        print(
            f"  ✅ model: ggml-{model_name}.bin "
            f"({size_mb} MB)"
        )

    else:

        print(
            f"  ❌ model NOT found: "
            f"ggml-{model_name}.bin"
        )

        print()
        print("Download it with:")

        print(
            f"cd ~/whisper.cpp && "
            f"bash models/download-ggml-model.sh {model_name}"
        )

        ok = False

    print("───────────────────────────────────────────────")
    print()

    if not ok:

        raise SystemExit

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


# ============================================================
# WORD → HINGLISH
# ============================================================

def word_to_hinglish(itrans_word):

    word = itrans_word

    for src, dst in ITRANS_MAP:

        word = word.replace(src, dst)

    word = word.lower()

    if (
        len(word) > 2
        and word.endswith("a")
        and not word.endswith(
            ("aa", "ia", "ua")
        )
    ):

        word = word[:-1]

    if len(word) > 3 and word.endswith("aa"):

        word = word[:-1]

    word = re.sub(
        r"(.)\1{2,}",
        r"\1\1",
        word
    )

    return word


# ============================================================
# DEVANAGARI → HINGLISH
#
# Still used — this produces the casual romanized caption shown on
# screen for Hinglish speech. Nothing about display changes.
# ============================================================

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
# HINGLISH → DEVANAGARI
#
# NO LONGER USED in the translation path (see the worker below).
# Reversing casual-Hinglish spelling back into strict ITRANS/Devanagari
# was the source of the garbled translations, since casual spelling
# isn't valid ITRANS to begin with. Kept only for reference.
# ============================================================

def hinglish_to_devanagari(text):

    if not text:

        return text

    try:

        words = text.split()

        output = []

        for word in words:

            if re.fullmatch(
                r"[A-Za-z0-9@#$%&_\-']+",
                word
            ):

                output.append(word)

            else:

                try:

                    converted = transliterate(
                        word,
                        sanscript.ITRANS,
                        sanscript.DEVANAGARI
                    )

                    output.append(converted)

                except Exception:

                    output.append(word)

        return " ".join(output)

    except Exception:

        return text


# ============================================================
# PARSE WHISPER OUTPUT
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
# CLEAR QUEUE
# ============================================================

def clear_queue(q):

    while True:

        try:

            q.get_nowait()

        except queue.Empty:

            break


# ============================================================
# SAVE WAV
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

def run_whisper(wav_path):

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
            "  [Whisper timed out]"
        )

        return ""

    except Exception as e:

        print(
            f"  [Whisper error: {e}]"
        )

        return ""


# ============================================================
# LOAD TRANSLATION MODULE
# ============================================================

translation_function = None


def load_translation():

    global translation_function

    if translation_function is not None:

        return translation_function

    print()
    print(
        "Loading translation module..."
    )

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
            "❌ Could not load translation module."
        )

        print(
            f"Error: {e}"
        )

        return None


# ============================================================
# HEADER
# ============================================================

print()

print(
    "╔══════════════════════════════════════════════╗"
)

print(
    "║       DEAF ASSIST — SPEECH TRANSLATOR       ║"
)

print(
    "║       Whisper + IndicTrans2                 ║"
)

print(
    "╚══════════════════════════════════════════════╝"
)

print()


# ============================================================
# STEP 1 — SPEAKING LANGUAGE
# ============================================================

print(
    "STEP 1 — Select speaking language"
)

print()

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
        "\n❌ Invalid choice."
    )

    raise SystemExit


speech_config = SPEECH_LANGUAGES[
    speech_choice
]

source_language = speech_config[
    "code"
]

language_name = speech_config[
    "name"
]

model_name = speech_config[
    "model"
]

whisper_lang = speech_config[
    "whisper_lang"
]

romanize_output = speech_config[
    "romanize"
]


print()

print(
    f"✅ Speaking language: {language_name}"
)


# ============================================================
# STEP 2 — TRANSLATION LANGUAGE
# ============================================================

print()

print(
    "STEP 2 — Select translation language"
)

print()

print(
    "  0. No translation"
)

print()

print(
    "Available languages:"
)

print()

available_targets = {}

for code, name in TARGET_LANGUAGES.items():

    # --------------------------------------------------------
    # English speech
    # Don't show English as a translation target.
    # --------------------------------------------------------

    if source_language == "eng_Latn":

        if code == "eng_Latn":
            continue

    # --------------------------------------------------------
    # Hindi speech
    # Don't show Hindi as a translation target.
    # --------------------------------------------------------

    elif source_language == "hin_Deva":

        if code == "hin_Deva":
            continue

    # --------------------------------------------------------
    # Hinglish
    #
    # Hinglish is not a separate IndicTrans2 language.
    # Keep English + Hindi available.
    # --------------------------------------------------------

    available_targets[code] = name


target_items = list(
    available_targets.items()
)


for index, (code, name) in enumerate(
    target_items,
    start=1
):

    print(
        f"  {index:2}. {name}"
    )


target_choice = input(
    "\nEnter your choice: "
).strip()


# ============================================================
# NO TRANSLATION
# ============================================================

if target_choice == "0":

    target_language = None

    target_language_name = (
        "No translation"
    )


# ============================================================
# TRANSLATION SELECTED
# ============================================================

else:

    try:

        target_index = int(
            target_choice
        )

        if not (
            1 <= target_index <= len(target_items)
        ):

            raise ValueError

    except ValueError:

        print(
            "\n❌ Invalid translation choice."
        )

        raise SystemExit


    target_language, target_language_name = (
        target_items[
            target_index - 1
        ]
    )


print()

print(
    f"✅ Translation: {target_language_name}"
)

print(
    f"   IndicTrans2 code: {target_language}"
)


# ============================================================
# CHECK INTERNET
# ============================================================

online = is_online()

print()

if online:

    print(
        "🌐 Internet: ONLINE"
    )

    if target_language is not None:

        print(
            "✅ Translation is enabled."
        )

    else:

        print(
            "ℹ️ No translation selected."
        )

else:

    print(
        "📴 Internet: OFFLINE"
    )

    print(
        "✅ Speech-to-text will still work."
    )

    if target_language is not None:

        print(
            "⚠️ Translation will be skipped."
        )


# ============================================================
# LOAD WHISPER
# ============================================================

WHISPER_BIN, MODEL_PATH = check_setup(
    model_name
)


print(
    f"Language : {language_name}"
)

print(
    f"Model    : {model_name}"
)

print(
    "Backend  : whisper.cpp + Metal GPU"
)

print(
    f"Translate: {target_language_name}"
)

print()


# ============================================================
# KEYBOARD CALLBACKS
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
                    "🎤 Listening..."
                )

                print(
                    "Release SPACEBAR to "
                    "transcribe"
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
                    "🛑 Recording stopped."
                )

                print(
                    "⏳ Processing..."
                )

    elif key == keyboard.Key.esc:

        with lock:

            stop_program = True

        return False


# ============================================================
# KEYBOARD LISTENER
# ============================================================

listener = keyboard.Listener(
    on_press=on_press,
    on_release=on_release
)

listener.start()


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

        should_record = recording

        current_session = session_id

    if should_record:

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
# TRANSCRIPTION + TRANSLATION WORKER
# ============================================================

def transcription_worker():

    while True:

        item = transcription_queue.get()

        wav_path = None

        try:

            if item is None:

                break

            chunk_session, audio = item

            audio = clean_audio(
                audio
            )

            duration = (
                len(audio)
                / SAMPLE_RATE
            )

            if duration < 0.6:

                continue

            rms = float(
                np.sqrt(
                    np.mean(
                        audio ** 2
                    )
                )
            )

            if rms < 0.004:

                continue

            wav_path = save_wav(
                audio
            )


            # =================================================
            # STEP 1 — SPEECH TO TEXT
            # =================================================

            transcript = run_whisper(
                wav_path
            )


            # =================================================
            # REMOVE HALLUCINATED REPETITION
            #
            # Runs on the raw Whisper output, before romanization,
            # same as before.
            # =================================================

            if transcript:

                words = transcript.split()

                if len(words) > 3:

                    unique_ratio = (
                        len(set(words))
                        / len(words)
                    )

                    if unique_ratio < 0.40:

                        transcript = ""


            # =================================================
            # SPLIT INTO TWO COPIES: translation vs. display
            # =================================================
            #
            # CHANGED: previously this file romanized the transcript
            # for display, then reversed that romanization back into
            # Devanagari via hinglish_to_devanagari() for translation.
            # That reverse step assumed strict ITRANS spelling, but
            # devanagari_to_hinglish() deliberately produces casual,
            # non-ITRANS spelling to read naturally — so the reverse
            # conversion produced nonsense syllables, and THAT
            # nonsense was what got sent to IndicTrans2.
            #
            # Since whisper_lang is now "hi" for both Hindi and
            # Hinglish (see SPEECH_LANGUAGES above), `transcript` here
            # is already correct, properly-spelled Devanagari for
            # both modes — no reverse-engineering needed. We keep it
            # untouched for translation, and derive the romanized
            # caption from it separately, purely for display.

            translation_source_text = transcript

            display_transcript = transcript

            if (
                transcript
                and romanize_output
            ):

                display_transcript = (
                    devanagari_to_hinglish(
                        transcript
                    )
                )


            # =================================================
            # DISPLAY TRANSCRIPTION
            # =================================================

            if display_transcript:

                print()

                print(
                    "========================================"
                )

                print(
                    "TRANSCRIPTION"
                )

                print(
                    "========================================"
                )

                print(
                    display_transcript
                )


                # =================================================
                # NO TRANSLATION
                # =================================================

                if target_language is None:

                    print()

                    print(
                        "ℹ️ Translation: Not selected"
                    )


                # =================================================
                # TRANSLATION
                # =================================================

                elif online:

                    print()

                    print(
                        f"TRANSLATION → "
                        f"{target_language_name}"
                    )

                    print(
                        f"IndicTrans2 → "
                        f"{target_language}"
                    )

                    print(
                        "----------------------------------------"
                    )

                    translator = load_translation()


                    if translator:

                        try:

                            # ------------------------------------------------
                            # TRANSLATE
                            #
                            # Both Hindi and Hinglish speech now transcribe
                            # via whisper_lang="hi", so translation_source
                            # is always "hin_Deva" for either. English
                            # speech stays "eng_Latn". No reverse-conversion
                            # step needed anymore.
                            #
                            # translate.py must accept IndicTrans2 codes:
                            #   eng_Latn → asm_Beng
                            #   hin_Deva → pan_Guru
                            #   eng_Latn → tam_Taml
                            # ------------------------------------------------

                            if source_language in ("hin_Deva", "hinglish"):

                                translation_source = "hin_Deva"

                            else:

                                translation_source = source_language

                            translated = translator(
                                translation_source_text,
                                translation_source,
                                target_language
                            )


                            print()

                            print(
                                translated
                            )


                        except Exception as e:

                            print()

                            print(
                                "❌ Translation error:"
                            )

                            print(
                                e
                            )

                            print()

                            print(
                                "Source code:"
                            )

                            print(
                                translation_source
                            )

                            print(
                                "Target code:"
                            )

                            print(
                                target_language
                            )


                    else:

                        print(
                            "❌ Translation module "
                            "could not be loaded."
                        )


                # =================================================
                # OFFLINE
                # =================================================

                else:

                    print()

                    print(
                        "TRANSLATION"
                    )

                    print(
                        "----------------------------------------"
                    )

                    print(
                        "📴 Translation unavailable "
                        "because the app is offline."
                    )


                print(
                    "========================================"
                )


            else:

                print(
                    "\n[no speech detected]"
                )


        except Exception as e:

            print(
                f"\n❌ Processing error: {e}"
            )


        finally:

            if (
                wav_path
                and os.path.exists(wav_path)
            ):

                os.remove(
                    wav_path
                )

            transcription_queue.task_done()


# ============================================================
# START WORKER
# ============================================================

worker = threading.Thread(
    target=transcription_worker,
    daemon=True
)

worker.start()


# ============================================================
# READY
# ============================================================

print()

print(
    "╔══════════════════════════════════════════════╗"
)

print(
    "║                 READY                        ║"
)

print(
    "╚══════════════════════════════════════════════╝"
)

print()

print(
    f"Speaking : {language_name}"
)

print(
    f"Translate: {target_language_name}"
)

print()

print(
    "🎤 Hold SPACEBAR → speak"
)

print(
    "🛑 Release SPACEBAR → transcribe"
)

if target_language is not None:

    print(
        "🌐 Translation → enabled when online"
    )

else:

    print(
        "ℹ️ Translation → disabled"
    )

print()

print(
    "❌ Press ESC → quit"
)

print()


# ============================================================
# MAIN AUDIO LOOP
# ============================================================

with sd.InputStream(

    samplerate=SAMPLE_RATE,

    channels=1,

    dtype="float32",

    blocksize=BLOCKSIZE,

    callback=audio_callback

):

    current_audio = []

    current_samples = 0

    active_session = None


    try:

        while True:

            with lock:

                should_stop = stop_program

                is_recording = recording

                current_session = session_id


            if should_stop:

                break


            # ------------------------------------------------
            # NEW RECORDING SESSION
            # ------------------------------------------------

            if (
                is_recording
                and active_session != current_session
            ):

                active_session = current_session

                current_audio = []

                current_samples = 0


            # ------------------------------------------------
            # GET MICROPHONE AUDIO
            # ------------------------------------------------

            try:

                chunk_session, audio_chunk = (
                    audio_queue.get(
                        timeout=0.1
                    )
                )


            except queue.Empty:

                # ------------------------------------------------
                # SPACEBAR RELEASED
                # ------------------------------------------------

                if (
                    not is_recording
                    and current_samples > 0
                ):

                    final_audio = (
                        np.concatenate(
                            current_audio
                        ).copy()
                    )

                    transcription_queue.put(
                        (
                            active_session,
                            final_audio
                        )
                    )

                    current_audio = []

                    current_samples = 0

                    active_session = None

                continue


            # ------------------------------------------------
            # IGNORE OLD SESSION AUDIO
            # ------------------------------------------------

            if (
                chunk_session
                != active_session
            ):

                continue


            # ------------------------------------------------
            # ADD AUDIO
            # ------------------------------------------------

            current_audio.append(
                audio_chunk
            )

            current_samples += len(
                audio_chunk
            )


            # ------------------------------------------------
            # PROCESS EVERY 4 SECONDS
            # ------------------------------------------------

            if (
                current_samples
                >= SAMPLE_RATE * CHUNK_SECONDS
            ):

                audio_to_process = (
                    np.concatenate(
                        current_audio
                    ).copy()
                )

                transcription_queue.put(
                    (
                        active_session,
                        audio_to_process
                    )
                )

                current_audio = []

                current_samples = 0


    except KeyboardInterrupt:

        print(
            "\n\nStopped."
        )


    finally:

        transcription_queue.put(
            None
        )

        listener.stop()