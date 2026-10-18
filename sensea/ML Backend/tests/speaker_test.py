"""
Deaf Assist — Speaker Identification Test

Purpose:
- Detect different speakers from microphone audio
- Support up to 8 speakers
- Use Resemblyzer voice embeddings
- Work locally/offline after the model is available
- PUSH-TO-TALK: Hold SPACEBAR to record, release to process
- This is a TEST ONLY — not yet integrated with Whisper.cpp

Requirements:
    pip install resemblyzer sounddevice numpy pynput

macOS note:
    pynput uses the Accessibility API — no sudo needed.
    If you see a permissions prompt, go to:
    System Settings → Privacy & Security → Accessibility
    and enable your terminal app (e.g. Terminal or iTerm2).
"""

import queue
import threading

import numpy as np
import sounddevice as sd
from resemblyzer import VoiceEncoder, preprocess_wav

try:
    from pynput import keyboard as pynput_keyboard
except ImportError:
    print("❌ Missing 'pynput' package. Run: pip install pynput")
    raise SystemExit


# ============================================================
# CONFIG
# ============================================================

SAMPLE_RATE = 16000

# Maximum number of different speakers
MAX_SPEAKERS = 8

# Minimum audio duration (seconds) to bother processing
MIN_SECONDS = 0.5

# Similarity threshold.
# Higher = stricter about deciding whether a voice is new.
SIMILARITY_THRESHOLD = 0.72

# ============================================================
# STATE
# ============================================================

AUDIO_QUEUE = queue.Queue()

# Event that is SET when the spacebar is held down
mic_active = threading.Event()

# Event that is SET when the program should exit
stop_event = threading.Event()

# ============================================================
# LOAD SPEAKER MODEL
# ============================================================

print()
print("========================================")
print("   DEAF ASSIST — SPEAKER TEST")
print("========================================")
print()

print("Loading speaker identification model...")
print("Using Resemblyzer...")

try:
    encoder = VoiceEncoder()
    print("✅ Speaker model loaded.")
except Exception as e:
    print()
    print("❌ Could not load speaker model.")
    print(e)
    raise SystemExit


# ============================================================
# SPEAKER DATABASE
# ============================================================

speaker_embeddings: list[tuple[int, np.ndarray]] = []


# ============================================================
# NORMALIZE EMBEDDING
# ============================================================

def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    embedding = np.asarray(embedding, dtype=np.float32)
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return embedding
    return embedding / norm


# ============================================================
# COMPARE VOICE
# ============================================================

def find_speaker(embedding: np.ndarray) -> tuple[int | None, float]:
    """
    Compare embedding against all saved speakers.
    Returns (speaker_id, best_similarity).
    speaker_id is None if no match exceeds threshold.
    """
    if not speaker_embeddings:
        return None, 0.0

    embedding = normalize_embedding(embedding)
    best_speaker = None
    best_similarity = -1.0

    for speaker_id, saved_embedding in speaker_embeddings:
        saved_embedding = normalize_embedding(saved_embedding)
        similarity = float(np.dot(embedding, saved_embedding))
        if similarity > best_similarity:
            best_similarity = similarity
            best_speaker = speaker_id

    if best_similarity >= SIMILARITY_THRESHOLD:
        return best_speaker, best_similarity

    return None, best_similarity


# ============================================================
# CREATE NEW SPEAKER
# ============================================================

def create_speaker(embedding: np.ndarray) -> int | None:
    """
    Register a new speaker and return their assigned ID.
    Returns None if the speaker limit has been reached.
    """
    if len(speaker_embeddings) >= MAX_SPEAKERS:
        return None

    speaker_number = len(speaker_embeddings) + 1
    speaker_embeddings.append(
        (speaker_number, normalize_embedding(embedding))
    )
    return speaker_number


# ============================================================
# PROCESS AUDIO
# ============================================================

def process_audio(audio: np.ndarray) -> None:
    """Embed the audio clip and identify or register the speaker."""

    if audio is None or len(audio) == 0:
        return

    audio = np.asarray(audio, dtype=np.float32)

    # Enforce minimum duration
    if len(audio) < SAMPLE_RATE * MIN_SECONDS:
        print("  [Too short — ignored]")
        return

    # Remove DC offset
    audio = audio - np.mean(audio)

    # Check volume (skip silence)
    rms = float(np.sqrt(np.mean(audio ** 2)))
    if rms < 0.005:
        print("  [Too quiet — ignored]")
        return

    try:
        # Resemblyzer preprocessing
        wav = preprocess_wav(audio, source_sr=SAMPLE_RATE)

        if len(wav) < SAMPLE_RATE * MIN_SECONDS:
            print("  [Too short after preprocessing — ignored]")
            return

        # Generate voice embedding
        embedding = encoder.embed_utterance(wav)

        speaker_id, similarity = find_speaker(embedding)

        # --------------------------------------------------
        # EXISTING SPEAKER
        # --------------------------------------------------
        if speaker_id is not None:
            print(
                f"🎤 Speaker {speaker_id}  "
                f"(similarity: {similarity:.2f})"
            )

        # --------------------------------------------------
        # NEW SPEAKER
        # --------------------------------------------------
        else:
            if len(speaker_embeddings) < MAX_SPEAKERS:
                new_id = create_speaker(embedding)
                print()
                print("🆕 NEW SPEAKER DETECTED")
                print(f"   → Assigned: Speaker {new_id}")
                print(
                    f"   → Closest existing speaker "
                    f"similarity: {similarity:.2f}"
                )
                print()
            else:
                print(
                    f"⚠️  Maximum of {MAX_SPEAKERS} speakers reached. "
                    "Could not register new speaker."
                )

    except Exception as e:
        print(f"❌ Speaker processing error: {e}")


# ============================================================
# MICROPHONE CALLBACK
# ============================================================

def audio_callback(
    indata: np.ndarray,
    frames: int,
    time_info,
    status,
) -> None:
    """
    Called by sounddevice for every audio block.
    Only enqueues audio while the spacebar is held.
    """
    if status:
        print(f"\n[microphone: {status}]")

    if mic_active.is_set():
        audio = np.array(indata[:, 0], dtype=np.float32, copy=True)
        AUDIO_QUEUE.put(audio)


# ============================================================
# KEYBOARD LISTENER  (pynput — works on macOS without sudo)
# ============================================================

def on_press(key: pynput_keyboard.Key) -> None:
    """Activate the microphone when spacebar is pressed."""
    if key == pynput_keyboard.Key.space:
        if not mic_active.is_set():
            mic_active.set()
            print("🔴 Recording... (release SPACE to process)", end="\r", flush=True)


def on_release(key: pynput_keyboard.Key) -> None:
    """
    Deactivate the microphone when spacebar is released.
    Signals the main loop to drain the queue and process audio.
    """
    if key == pynput_keyboard.Key.space:
        if mic_active.is_set():
            mic_active.clear()
            # Sentinel tells the main loop to process whatever is buffered
            AUDIO_QUEUE.put(None)
            print(" " * 60, end="\r")  # Clear the recording line


# ============================================================
# MAIN LOOP
# ============================================================

def main() -> None:
    print()
    print("========================================")
    print("READY")
    print("========================================")
    print()
    print(f"Maximum speakers  : {MAX_SPEAKERS}")
    print(f"Min clip length   : {MIN_SECONDS} s")
    print(f"Similarity thresh : {SIMILARITY_THRESHOLD}")
    print()
    print("Hold SPACEBAR to record.")
    print("Release SPACEBAR to identify the speaker.")
    print("Press Ctrl+C to stop.")
    print()

    buffer: list[np.ndarray] = []

    # pynput listener runs in its own daemon thread automatically
    listener = pynput_keyboard.Listener(
        on_press=on_press,
        on_release=on_release,
    )
    listener.start()

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=1600,
            callback=audio_callback,
        ):
            while not stop_event.is_set():
                try:
                    chunk = AUDIO_QUEUE.get(timeout=0.2)
                except queue.Empty:
                    continue

                # None is the sentinel that means "process now"
                if chunk is None:
                    if buffer:
                        audio_chunk = np.concatenate(buffer)
                        buffer = []
                        process_audio(audio_chunk)
                else:
                    buffer.append(chunk)

    except KeyboardInterrupt:
        pass

    finally:
        listener.stop()
        print()
        print()
        print("========================================")
        print("TEST STOPPED")
        print("========================================")
        print()
        print(f"Different speakers detected: {len(speaker_embeddings)}")
        for speaker_id, _ in speaker_embeddings:
            print(f"  Speaker {speaker_id}")
        print()


if __name__ == "__main__":
    main()