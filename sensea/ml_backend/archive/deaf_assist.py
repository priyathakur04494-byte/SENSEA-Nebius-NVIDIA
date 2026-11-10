"""
Deaf Assist — Unified Pipeline
═══════════════════════════════════════════════════════════════
FLO
────
1. Select speaking language  (English / Hindi / Hinglish)
2. Select translation target (or "No translation")
3. Hold SPACEBAR → speak (one or multiple speakers)
4. Release SPACEBAR →
      a. Spectral VAD rejects non-speech segments (wind, car, etc.)
      b. Audio is split into per-speaker segments (diarization)
      c. Each segment is transcribed by Whisper
      d. Whisper hallucinations ([BLANK_AUDIO], (wind), etc.) filtered
      e. Each segment is translated (if requested & online)
      f. Results printed: [Speaker Name] transcript / translation
      g. Unknown speakers: prompted to save with a name
5. Saved speakers are remembered across sessions (speakers.json)

REQUIREMENTS
────────────
pip install resemblyzer sounddevice numpy pynput soundfile

FILES EXPECTED ALONGSIDE THIS SCRIPT
──────────────────────────────────────
  translate.py   — your existing translate_text() function
  speakers.json  — auto-created on first save (don't touch)

macOS note
──────────
pynput needs Accessibility permission for your terminal.
System Settings → Privacy & Security → Accessibility → enable Terminal.

WHISPER
───────
Set WHISPER_CPP_PATH below to your whisper.cpp checkout.
"""

# ── stdlib ────────────────────────────────────────────────────
import json
import os
import queue
import re
import subprocess
import tempfile
import threading
import urllib.request

# ── third-party ───────────────────────────────────────────────
import numpy as np
import sounddevice as sd
import soundfile as sf
from resemblyzer import VoiceEncoder, preprocess_wav

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


# ╔══════════════════════════════════════════════════════════════╗
# ║  CONFIG — edit these paths if needed                        ║
# ╚══════════════════════════════════════════════════════════════╝

WHISPER_CPP_PATH   = os.path.expanduser("~/whisper.cpp")
SPEAKERS_FILE      = os.path.join(os.path.dirname(__file__), "speakers.json")

SAMPLE_RATE        = 16000
BLOCKSIZE          = 1600          # ~100 ms per audio callback
MAX_SPEAKERS       = 4

# Used only for the FINAL per-segment match against the persistent
# speaker DB (named_speakers + session_unknowns).
SIMILARITY_THRESHOLD = 0.75

# Used only for grouping windows WITHIN a single recording into
# tentative local clusters ("is this window the same voice as the
# last cluster I saw in THIS clip"). Never touches the persistent DB.
# Slightly lower than SIMILARITY_THRESHOLD because single 3s windows
# are noisier than a full merged-segment embedding.
LOCAL_CLUSTER_THRESHOLD = 0.70

# ── Diarization ────────────────────────────────────────────────
# Longer window = more stable per-window embedding = fewer phantom speakers.
# Raised from 1.5 s → 3.0 s so each window contains a full sentence.
DIARIZE_WIN_SEC    = 3.0
DIARIZE_STEP_SEC   = 1.5

# A speaker boundary is only accepted when the new label appears in at
# least this many consecutive windows. Filters single-window noise blips.
DIARIZE_MIN_STREAK = 2

# ── Spectral VAD ───────────────────────────────────────────────
# Human voiced speech concentrates energy in 80–3000 Hz.
# Segments below this threshold are non-speech (wind, traffic, etc.)
# and are skipped entirely — no transcription, no speaker ID.
SPEECH_BAND_LO      = 80      # Hz
SPEECH_BAND_HI      = 3000    # Hz
SPEECH_ENERGY_RATIO = 0.55    # speech band must be ≥ 55 % of total energy

# ── Whisper hallucination filter ───────────────────────────────
# Whisper outputs these tokens when fed silence or non-speech audio.
# Any transcript matching this pattern is discarded.
_HALLUCINATION_RE = re.compile(
    r"^\s*[\[\(]"                   # starts with [ or (  e.g. [BLANK_AUDIO]
    r"|^\s*-?\s*$"                  # empty or just a dash
    r"|^\s*\.\s*$"                  # just a period
    r"|thank you\.?\s*$"            # Whisper's most common silence hallucination
    r"|please subscribe"
    r"|subtitles by"
    r"|www\.",
    re.IGNORECASE,
)


# ╔══════════════════════════════════════════════════════════════╗
# ║  SPEECH LANGUAGE OPTIONS                                     ║
# ╚══════════════════════════════════════════════════════════════╝

SPEECH_LANGUAGES = {
    "1": {
        "code":         "eng_Latn",
        "name":         "English",
        "whisper_lang": "en",
        "model":        "small",
        "romanize":     False,
    },
    "2": {
        "code":         "hin_Deva",
        "name":         "Hindi",
        "whisper_lang": "hi",
        "model":        "medium",
        "romanize":     False,
    },
    "3": {
        "code":         "hinglish",
        "name":         "Hinglish",
        "whisper_lang": "hi",
        "model":        "medium",
        "romanize":     True,
    },
}


# ╔══════════════════════════════════════════════════════════════╗
# ║  TRANSLATION TARGET LANGUAGES                                ║
# ╚══════════════════════════════════════════════════════════════╝

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
# ║  GLOBAL STATE                                                ║
# ╚══════════════════════════════════════════════════════════════╝

mic_active  = threading.Event()
stop_event  = threading.Event()
AUDIO_QUEUE: queue.Queue = queue.Queue()


# ╔══════════════════════════════════════════════════════════════╗
# ║  BANNER                                                      ║
# ╚══════════════════════════════════════════════════════════════╝

def _banner(text: str) -> None:
    bar = "═" * (len(text) + 4)
    print(f"\n╔{bar}╗")
    print(f"║  {text}  ║")
    print(f"╚{bar}╝\n")


# ╔══════════════════════════════════════════════════════════════╗
# ║  INTERNET CHECK                                              ║
# ╚══════════════════════════════════════════════════════════════╝

def is_online() -> bool:
    try:
        urllib.request.urlopen("https://huggingface.co", timeout=3)
        return True
    except Exception:
        return False


# ╔══════════════════════════════════════════════════════════════╗
# ║  SPECTRAL VAD — is this segment human speech?               ║
# ╚══════════════════════════════════════════════════════════════╝

def is_speech(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bool:
    """
    Return True only if the audio segment has enough energy in the
    human-speech band (80–3000 Hz) relative to its total energy.

    Wind, car horns, and general ambient noise spread energy across
    all frequencies, so their speech-band ratio falls below the threshold.
    """
    if len(audio) < 512:
        return False

    # FFT magnitude spectrum
    fft_mag  = np.abs(np.fft.rfft(audio))
    freqs    = np.fft.rfftfreq(len(audio), d=1.0 / sample_rate)

    total_energy = float(np.sum(fft_mag ** 2))
    if total_energy < 1e-10:
        return False   # silence

    speech_mask   = (freqs >= SPEECH_BAND_LO) & (freqs <= SPEECH_BAND_HI)
    speech_energy = float(np.sum(fft_mag[speech_mask] ** 2))

    ratio = speech_energy / total_energy
    return ratio >= SPEECH_ENERGY_RATIO


# ╔══════════════════════════════════════════════════════════════╗
# ║  SPEAKER DATABASE  (in-memory + JSON persistence)            ║
# ╚══════════════════════════════════════════════════════════════╝

named_speakers:   dict[str, np.ndarray] = {}
session_unknowns: dict[str, np.ndarray] = {}
_last_segment_label: str | None = None
_last_segment_embedding: np.ndarray | None = None
_unknown_counter = 0


def _next_unknown_id() -> str:
    global _unknown_counter
    try:
        match = re.search(r"Unknown-(\d+)$", _last_segment_label or "")
        if match:
            _unknown_counter = max(_unknown_counter, int(match.group(1)))
    except Exception:
        pass
    _unknown_counter += 1
    return f"Unknown-{_unknown_counter}"


def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def load_speakers() -> None:
    global named_speakers
    if not os.path.isfile(SPEAKERS_FILE):
        return
    try:
        with open(SPEAKERS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        named_speakers = {
            name: _normalize(np.array(vec, dtype=np.float32))
            for name, vec in data.items()
        }
        print(f"✅  Loaded {len(named_speakers)} saved speaker(s): "
              f"{', '.join(named_speakers.keys())}")
    except Exception as e:
        print(f"⚠️  Could not load speakers.json: {e}")


def save_speakers() -> None:
    try:
        data = {name: emb.tolist() for name, emb in named_speakers.items()}
        with open(SPEAKERS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"⚠️  Could not save speakers.json: {e}")


def identify_embedding(emb: np.ndarray) -> tuple[str, float]:
    """
    Compare embedding against named speakers then session unknowns.
    Returns (label, similarity).
    If no match ≥ threshold, registers a new Unknown-N and returns that.

    IMPORTANT: this permanently registers a new identity on a miss.
    Only call this once per FINAL merged speaker segment — never per
    diarization window — or the persistent speaker DB gets polluted
    with noisy short-window embeddings and the same real speaker will
    keep minting new Unknown-N slots. Use the local, non-mutating
    clustering inside diarize() for window-level grouping instead.

    Also guard against real voice changes within the same recording:
    if the current segment is clearly different from the previous
    speaker turn, create a new Unknown-N rather than reusing the same
    speaker label.
    """
    global _unknown_counter, _last_segment_label, _last_segment_embedding
    emb = _normalize(emb)
    best_label: str | None = None
    best_sim = -1.0

    for name, saved in named_speakers.items():
        sim = float(np.dot(emb, _normalize(saved)))
        if sim > best_sim:
            best_sim, best_label = sim, name

    for uid, saved in session_unknowns.items():
        sim = float(np.dot(emb, _normalize(saved)))
        if sim > best_sim:
            best_sim, best_label = sim, uid

    if _last_segment_embedding is not None and _last_segment_label is not None:
        prev_sim = float(np.dot(emb, _normalize(_last_segment_embedding)))
        if prev_sim < 0.45 and best_sim < SIMILARITY_THRESHOLD:
            uid = _next_unknown_id()
            session_unknowns[uid] = emb
            _last_segment_label = uid
            _last_segment_embedding = emb.copy()
            return uid, best_sim

    if best_sim >= SIMILARITY_THRESHOLD and best_label is not None:
        _last_segment_label = best_label
        _last_segment_embedding = emb.copy()
        return best_label, best_sim

    # Brand-new voice
    uid = _next_unknown_id()
    session_unknowns[uid] = emb
    _last_segment_label = uid
    _last_segment_embedding = emb.copy()
    return uid, best_sim


def _prompt_save_subset(subset: dict[str, np.ndarray]) -> None:
    """Ask user to name a specific subset of unknown speakers."""
    print()
    print("─" * 48)
    print("  UNKNOWN SPEAKER(S) DETECTED IN THIS CLIP")
    print("─" * 48)
    to_delete = []
    for uid, emb in subset.items():
        print(f"\n  {uid} — save this voice for next time?")
        ans = input("  Save? (y/n): ").strip().lower()
        if ans == "y":
            while True:
                name = input(f"  Enter a name for {uid}: ").strip()
                if name:
                    break
                print("  Name cannot be empty.")
            named_speakers[name] = emb
            to_delete.append(uid)
            print(f"  ✅  Saved as '{name}'.")
    for uid in to_delete:
        session_unknowns.pop(uid, None)
    if to_delete:
        save_speakers()


# ╔══════════════════════════════════════════════════════════════╗
# ║  RESEMBLYZER — SPEAKER DIARIZATION                          ║
# ╚══════════════════════════════════════════════════════════════╝

_voice_encoder: VoiceEncoder | None = None


def get_voice_encoder() -> VoiceEncoder:
    global _voice_encoder
    if _voice_encoder is None:
        print("Loading Resemblyzer speaker model…")
        _voice_encoder = VoiceEncoder()
        print("✅  Speaker model loaded.")
    return _voice_encoder


def diarize(audio: np.ndarray) -> list[tuple[str, float, np.ndarray]]:
    """
    Slide a window over `audio` and group windows into tentative
    LOCAL clusters (same voice within THIS clip only — no persistent
    DB writes, no Unknown-N minted here). Apply majority-vote
    smoothing + min-streak enforcement on those local labels, merge
    consecutive same-label windows into segments, then — ONCE per
    final merged segment — identify against the persistent speaker
    DB (named_speakers / session_unknowns) to get the real label.

    This two-stage approach (local grouping, then a single DB lookup
    per segment) is what prevents the same physical speaker from
    fragmenting into multiple Unknown-N identities within one clip:
    previously every window called identify_embedding() directly,
    which both compared against AND mutated the persistent DB,
    silently burning counter slots and poisoning it with noisy
    single-window embeddings.

    Returns list of (speaker_label, similarity, segment_audio).
    """
    encoder      = get_voice_encoder()
    win_samples  = int(DIARIZE_WIN_SEC  * SAMPLE_RATE)
    step_samples = int(DIARIZE_STEP_SEC * SAMPLE_RATE)

    # Not enough audio for even one window → single segment.
    # This is the only place a single window's embedding is allowed
    # to hit the persistent DB directly, since there's nothing to group.
    if len(audio) < win_samples:
        try:
            wav = preprocess_wav(audio, source_sr=SAMPLE_RATE)
            emb = encoder.embed_utterance(wav)
            label, sim = identify_embedding(emb)
        except Exception:
            label, sim = "Unknown-?", 0.0
        return [(label, sim, audio)]

    # ── Per-window LOCAL clustering (no DB writes here) ────────
    window_local_labels: list[str] = []
    window_starts: list[int] = []
    local_clusters: list[np.ndarray] = []   # centroids, scoped to this clip only

    i = 0
    while i + win_samples <= len(audio):
        chunk = audio[i: i + win_samples]

        # Skip windows that are not speech (VAD guard inside diarizer)
        if not is_speech(chunk):
            window_local_labels.append("__noise__")
            window_starts.append(i)
            i += step_samples
            continue

        try:
            wav = preprocess_wav(chunk, source_sr=SAMPLE_RATE)
            emb = _normalize(encoder.embed_utterance(wav))
        except Exception:
            emb = None

        if emb is None:
            window_local_labels.append("__noise__")
            window_starts.append(i)
            i += step_samples
            continue

        # Match against LOCAL clusters only — this never touches
        # named_speakers or session_unknowns, so it can't mint a
        # spurious Unknown-N or pollute the persistent DB.
        best_idx, best_sim = -1, -1.0
        for idx, centroid in enumerate(local_clusters):
            sim = float(np.dot(emb, centroid))
            if sim > best_sim:
                best_sim, best_idx = sim, idx

        if best_sim >= LOCAL_CLUSTER_THRESHOLD:
            label = f"__local_{best_idx}__"
            # running-average the centroid so it drifts gracefully
            # with natural variation in the same speaker's voice
            local_clusters[best_idx] = _normalize(
                0.7 * local_clusters[best_idx] + 0.3 * emb
            )
        else:
            local_clusters.append(emb)
            label = f"__local_{len(local_clusters) - 1}__"

        window_local_labels.append(label)
        window_starts.append(i)
        i += step_samples

    if not window_local_labels:
        return []

    # ── Majority-vote smoothing ────────────────────────────────
    # Replace isolated label flips with the surrounding label so
    # a single noisy window cannot create a phantom speaker.
    smoothed = list(window_local_labels)
    for idx in range(1, len(smoothed) - 1):
        prev_label = smoothed[idx - 1]
        next_label = smoothed[idx + 1]
        if smoothed[idx] != prev_label and prev_label == next_label:
            smoothed[idx] = prev_label   # lone flip → revert

    # ── Minimum-streak enforcement ─────────────────────────────
    # A new label must appear consecutively DIARIZE_MIN_STREAK times
    # before we commit to a boundary.
    final_labels = list(smoothed)
    current      = smoothed[0]
    streak       = 1
    for idx in range(1, len(smoothed)):
        if smoothed[idx] == current:
            streak += 1
        else:
            if streak < DIARIZE_MIN_STREAK:
                # Revert the previous short run back to whatever came before
                for j in range(idx - streak, idx):
                    final_labels[j] = final_labels[idx - streak - 1] if (idx - streak - 1) >= 0 else smoothed[idx]
            current = smoothed[idx]
            streak  = 1

    # ── Merge consecutive same-label windows into segments ─────
    # This is the ONLY place identify_embedding() is called for a
    # multi-window clip — once per merged segment, using a clean
    # embedding of the full segment audio (much more stable than
    # any single window), and it's what actually reads/writes the
    # persistent speaker DB.
    segments: list[tuple[str, float, np.ndarray]] = []
    seg_label = final_labels[0]
    seg_start = 0

    def _commit_segment(label: str, start: int, end: int) -> None:
        if label == "__noise__":
            return                      # drop noise segments entirely
        seg_audio = audio[start:end]
        if len(seg_audio) < int(SAMPLE_RATE * 0.5):
            return                      # too short to transcribe
        try:
            wav   = preprocess_wav(seg_audio, source_sr=SAMPLE_RATE)
            emb   = encoder.embed_utterance(wav)
            final_label, sim = identify_embedding(emb)
        except Exception:
            final_label, sim = "Unknown-?", 0.0
        segments.append((final_label, sim, seg_audio))

    for idx in range(1, len(final_labels)):
        if final_labels[idx] != seg_label:
            _commit_segment(seg_label, seg_start, window_starts[idx])
            seg_label = final_labels[idx]
            seg_start = window_starts[idx]

    _commit_segment(seg_label, seg_start, len(audio))

    return segments


# ╔══════════════════════════════════════════════════════════════╗
# ║  AUDIO UTILITIES                                             ║
# ╚══════════════════════════════════════════════════════════════╝

def clean_audio(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if not np.all(np.isfinite(audio)):
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    audio -= audio.mean()
    peak = float(np.abs(audio).max())
    if peak >= 0.01:
        audio = audio * (0.90 / peak)
    else:
        return np.zeros(len(audio), dtype=np.float32)
    return np.ascontiguousarray(np.clip(audio, -1.0, 1.0))


def save_wav(audio: np.ndarray) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, audio, SAMPLE_RATE, subtype="PCM_16")
    tmp.close()
    return tmp.name


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(audio ** 2)))


# ╔══════════════════════════════════════════════════════════════╗
# ║  WHISPER                                                     ║
# ╚══════════════════════════════════════════════════════════════╝

ANSI_ESC  = re.compile(r"\x1b\[[0-9;]*[mGKHF]")
TSTAMP_RE = re.compile(r"\[\d+:\d+\.\d+\s*-->\s*\d+:\d+\.\d+\]")
LOG_PFX   = (
    "whisper_", "main:", "system_info", "ggml_", "metal_", "log_",
    "operator()", "inf ", "sampling", "encode", "decode", "autodetect",
    "init ", "load ", "mel ", "read_audio", "trying to", "output_",
)


def _parse_whisper(raw: str) -> str:
    lines = []
    for line in raw.splitlines():
        line = ANSI_ESC.sub("", line).strip()
        if not line:
            continue
        if any(line.lower().startswith(p) for p in LOG_PFX):
            continue
        if re.match(r"^[\d\s.,\[\]\-:/|ms%]+$", line):
            continue
        lines.append(line)
    text = TSTAMP_RE.sub("", " ".join(lines))
    return re.sub(r"\s+", " ", text).strip()


def _is_hallucination(text: str) -> bool:
    """
    Return True for known Whisper hallucinations:
      [BLANK_AUDIO], (wind blowing), (sounds of a car), Thank you., etc.
    These appear when Whisper is fed silence or non-speech audio.
    """
    if not text:
        return True
    # Bracket/paren descriptions — e.g. [BLANK_AUDIO], (wind blowing)
    if re.match(r"^\s*[\[\(]", text):
        return True
    # Only punctuation / whitespace
    if re.match(r"^\s*[\W_]+\s*$", text):
        return True
    # Common Whisper silence tokens
    lower = text.strip().lower()
    for token in ("thank you", "please subscribe", "subtitles by",
                  "www.", "[music]", "(music)", "♪"):
        if lower.startswith(token) or lower.endswith(token):
            return True
    return False


def check_whisper(model_name: str) -> tuple[str, str]:
    binary = os.path.join(WHISPER_CPP_PATH, "build", "bin", "whisper-cli")
    model  = os.path.join(WHISPER_CPP_PATH, "models", f"ggml-{model_name}.bin")
    ok = True
    print("\n── Checking Whisper ─────────────────────────")
    if os.path.isfile(binary):
        print("  ✅  whisper-cli found")
    else:
        print(f"  ❌  whisper-cli not found: {binary}")
        ok = False
    if os.path.isfile(model):
        print(f"  ✅  ggml-{model_name}.bin  "
              f"({os.path.getsize(model) // 1_048_576} MB)")
    else:
        print(f"  ❌  ggml-{model_name}.bin not found")
        ok = False
    print("─────────────────────────────────────────────\n")
    if not ok:
        raise SystemExit
    return binary, model


def run_whisper(wav_path: str, whisper_bin: str, model_path: str,
                lang: str) -> str:
    cmd = [
        whisper_bin, "-m", model_path, "-f", wav_path,
        "-l", lang, "-t", "4", "--beam-size", "3", "-nt",
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        text = _parse_whisper(r.stdout) or _parse_whisper(r.stderr)
        return text
    except subprocess.TimeoutExpired:
        print("  [Whisper timed out]")
        return ""
    except Exception as e:
        print(f"  [Whisper error: {e}]")
        return ""


# ╔══════════════════════════════════════════════════════════════╗
# ║  HINGLISH ROMANIZATION  (display only)                      ║
# ╚══════════════════════════════════════════════════════════════╝

_ITRANS_MAP = [
    ("ksh","ksh"),("jny","gya"),("Ch","ch"),("Th","th"),("Dh","dh"),
    ("Sh","sh"),("ph","f"),("bh","bh"),("kh","kh"),("gh","gh"),
    ("ch","ch"),("jh","jh"),("th","th"),("dh","dh"),("sh","sh"),
    ("RI","ri"),("aa","aa"),("ii","i"),("uu","u"),("ai","e"),
    ("au","o"),("T","t"),("D","d"),("N","n"),("M","n"),("H","h"),
    (".n","n"),
]
_DEVA_RE    = re.compile(r"[\u0900-\u097F]+")
_HINDI_PUNC = re.compile(r"[।॥]+")


def _word_to_hinglish(w: str) -> str:
    for src, dst in _ITRANS_MAP:
        w = w.replace(src, dst)
    w = w.lower()
    if len(w) > 2 and w.endswith("a") and not w.endswith(("aa","ia","ua")):
        w = w[:-1]
    if len(w) > 3 and w.endswith("aa"):
        w = w[:-1]
    w = re.sub(r"(.)\1{2,}", r"\1\1", w)
    return w


def devanagari_to_hinglish(text: str) -> str:
    if not text or not _ITRANS_OK:
        return text
    text = _HINDI_PUNC.sub(".", text)
    def _replace(m):
        itrans = transliterate(m.group(0), sanscript.DEVANAGARI, sanscript.ITRANS)
        return " ".join(_word_to_hinglish(w) for w in itrans.split())
    return _DEVA_RE.sub(_replace, text).strip()


# ╔══════════════════════════════════════════════════════════════╗
# ║  TRANSLATION                                                 ║
# ╚══════════════════════════════════════════════════════════════╝

_translate_fn = None


def get_translator():
    global _translate_fn
    if _translate_fn is not None:
        return _translate_fn
    try:
        from translate import translate_text
        _translate_fn = translate_text
        print("✅  Translation module loaded.")
        return _translate_fn
    except Exception as e:
        print(f"❌  Could not load translate.py: {e}")
        return None


def translate(text: str, source_lang: str, target_lang: str,
              source_code: str) -> str | None:
    fn = get_translator()
    if fn is None:
        return None
    if source_code == "hinglish":
        source_code = "hin_Deva"
    try:
        return fn(text, source_code, target_lang)
    except Exception as e:
        print(f"  ❌  Translation error: {e}")
        return None


# ╔══════════════════════════════════════════════════════════════╗
# ║  CORE PIPELINE  (runs after each spacebar release)           ║
# ╚══════════════════════════════════════════════════════════════╝

def process_recording(
    audio:            np.ndarray,
    whisper_bin:      str,
    model_path:       str,
    whisper_lang:     str,
    romanize:         bool,
    source_code:      str,
    target_lang:      str | None,
    target_lang_name: str,
    online:           bool,
) -> None:
    """Full pipeline for one push-to-talk recording."""

    audio = clean_audio(audio)

    if rms(audio) < 0.004:
        print("\n  [Too quiet — nothing to process]")
        return

    # ── Whole-clip VAD guard ───────────────────────────────────
    if not is_speech(audio):
        print("\n  [No human speech detected in recording — skipped]")
        return

    print("\n⏳  Diarizing speakers…")
    segments = diarize(audio)

    if not segments:
        print("  [No speech segments after VAD filtering]")
        return

    print(f"  → {len(segments)} speaker segment(s) found.\n")

    new_unknowns_this_call: list[str] = []
    printed_any = False

    for seg_idx, (speaker_label, sim, seg_audio) in enumerate(segments, 1):

        # Short or silent segments are already filtered in diarize(),
        # but double-check here for safety.
        if len(seg_audio) < int(SAMPLE_RATE * 0.5):
            continue
        if rms(seg_audio) < 0.004:
            continue

        # ── Per-segment VAD ────────────────────────────────────
        if not is_speech(seg_audio):
            continue

        # ── Track new unknowns ─────────────────────────────────
        if speaker_label.startswith("Unknown-"):
            if speaker_label not in new_unknowns_this_call:
                new_unknowns_this_call.append(speaker_label)

        # ── Transcribe ─────────────────────────────────────────
        wav_path = None
        try:
            wav_path  = save_wav(seg_audio)
            transcript = run_whisper(wav_path, whisper_bin, model_path, whisper_lang)
        finally:
            if wav_path and os.path.exists(wav_path):
                os.remove(wav_path)

        # ── Filter hallucinations ──────────────────────────────
        if _is_hallucination(transcript):
            continue

        # Filter repetitive hallucinations (Whisper looping)
        words = transcript.split()
        if len(words) > 3 and len(set(words)) / len(words) < 0.40:
            continue

        # ── Romanize for display if Hinglish ──────────────────
        display_text = devanagari_to_hinglish(transcript) if romanize else transcript

        # ── Print result ───────────────────────────────────────
        printed_any = True
        print("═" * 48)
        is_known = not speaker_label.startswith("Unknown-")
        tag = f"🎤  [{speaker_label}]" if is_known else f"❓  [{speaker_label}]"
        print(f"{tag}  (match: {sim:.2f})")
        print()
        print(f"  TRANSCRIPT : {display_text}")

        # ── Translate ──────────────────────────────────────────
        if target_lang is None:
            print("  TRANSLATION: (not selected)")
        elif not online:
            print("  TRANSLATION: (offline — skipped)")
        else:
            result = translate(transcript, whisper_lang, target_lang, source_code)
            if result:
                print(f"  → {target_lang_name:10}: {result}")
            else:
                print("  TRANSLATION: ❌  failed")

        print("═" * 48)

    if not printed_any:
        print("  [No speech content found after filtering]")

    # ── Prompt to save unknown speakers ───────────────────────
    relevant = {
        uid: session_unknowns[uid]
        for uid in new_unknowns_this_call
        if uid in session_unknowns
    }
    if relevant:
        _prompt_save_subset(relevant)

    print()
    print("🎤  Hold SPACEBAR to record again.")


# ╔══════════════════════════════════════════════════════════════╗
# ║  MICROPHONE CALLBACK                                         ║
# ╚══════════════════════════════════════════════════════════════╝

def audio_callback(indata, frames, time_info, status) -> None:
    if status:
        print(f"\n[mic: {status}]")
    if mic_active.is_set():
        AUDIO_QUEUE.put(np.array(indata[:, 0], dtype=np.float32, copy=True))


# ╔══════════════════════════════════════════════════════════════╗
# ║  MAIN                                                        ║
# ╚══════════════════════════════════════════════════════════════╝

def main() -> None:

    _banner("DEAF ASSIST — Speech · ID · Translation")

    load_speakers()
    get_voice_encoder()

    # ════════════════════════════════════════════════════════════
    # STEP 1 — Speaking language
    # ════════════════════════════════════════════════════════════
    print("STEP 1 — Select speaking language\n")
    print("  1. English")
    print("  2. Hindi")
    print("  3. Hinglish")
    speech_choice = input("\nEnter 1, 2, or 3: ").strip()
    if speech_choice not in SPEECH_LANGUAGES:
        print("\n❌  Invalid choice.")
        return

    cfg           = SPEECH_LANGUAGES[speech_choice]
    source_code   = cfg["code"]
    language_name = cfg["name"]
    model_name    = cfg["model"]
    whisper_lang  = cfg["whisper_lang"]
    romanize      = cfg["romanize"]
    print(f"\n✅  Speaking language: {language_name}")

    # ════════════════════════════════════════════════════════════
    # STEP 2 — Translation target
    # ════════════════════════════════════════════════════════════
    print("\nSTEP 2 — Select translation language\n")
    print("  0. No translation\n")

    exclude = {source_code, "hinglish"}
    if source_code == "hinglish":
        exclude.add("hin_Deva")
    available = [(c, n) for c, n in TARGET_LANGUAGES.items() if c not in exclude]

    for i, (code, name) in enumerate(available, 1):
        print(f"  {i:2}. {name}")

    target_choice    = input("\nEnter your choice: ").strip()
    target_lang:      str | None = None
    target_lang_name: str        = "No translation"

    if target_choice != "0":
        try:
            idx = int(target_choice)
            if not 1 <= idx <= len(available):
                raise ValueError
            target_lang, target_lang_name = available[idx - 1]
        except ValueError:
            print("\n❌  Invalid choice — translation disabled.")

    print(f"\n✅  Translation: {target_lang_name}")

    # ════════════════════════════════════════════════════════════
    # STEP 3 — Internet + pre-load translation
    # ════════════════════════════════════════════════════════════
    online = is_online()
    if online:
        print("🌐  Internet: ONLINE")
        if target_lang:
            print("    Pre-loading translation model…")
            get_translator()
    else:
        print("📴  Internet: OFFLINE  (transcription only)")

    # ════════════════════════════════════════════════════════════
    # STEP 4 — Verify Whisper
    # ════════════════════════════════════════════════════════════
    whisper_bin, model_path = check_whisper(model_name)

    # ════════════════════════════════════════════════════════════
    # STEP 5 — Keyboard + mic
    # ════════════════════════════════════════════════════════════
    _banner("READY")
    print(f"  Speaking : {language_name}")
    print(f"  Translate: {target_lang_name}")
    if named_speakers:
        print(f"  Known voices: {', '.join(named_speakers.keys())}")
    print()
    print("  🎤  Hold SPACEBAR → speak")
    print("  🛑  Release SPACEBAR → transcribe + identify")
    print("  ❌  Ctrl-C → quit")
    print()

    buffer: list[np.ndarray] = []

    def on_press(key):
        if key == pynput_keyboard.Key.space:
            if not mic_active.is_set():
                mic_active.set()
                print("🔴  Recording… (release SPACE when done)", end="\r", flush=True)

    def on_release(key):
        if key == pynput_keyboard.Key.space:
            if mic_active.is_set():
                mic_active.clear()
                AUDIO_QUEUE.put(None)
                print(" " * 50, end="\r")

    listener = pynput_keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=BLOCKSIZE,
            callback=audio_callback,
        ):
            while not stop_event.is_set():
                try:
                    chunk = AUDIO_QUEUE.get(timeout=0.2)
                except queue.Empty:
                    continue

                if chunk is None:
                    if buffer:
                        full_audio = np.concatenate(buffer)
                        buffer = []
                        process_recording(
                            full_audio,
                            whisper_bin,
                            model_path,
                            whisper_lang,
                            romanize,
                            source_code,
                            target_lang,
                            target_lang_name,
                            online,
                        )
                else:
                    buffer.append(chunk)

    except KeyboardInterrupt:
        pass

    finally:
        listener.stop()
        print()
        _banner("SESSION ENDED")
        print(f"  Named speakers in database: {len(named_speakers)}")
        for name in named_speakers:
            print(f"    • {name}")
        print()


if __name__ == "__main__":
    # The old pipeline in this file waits until the spacebar is released before
    # processing audio. To get simultaneous live transcription + translation +
    # speaker labels while the key is held, run the streaming implementation.
    from deaf_assist2 import main as live_main
    live_main()