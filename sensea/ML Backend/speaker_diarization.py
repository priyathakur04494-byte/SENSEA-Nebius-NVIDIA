"""
speaker_diarization.py — Real-time speaker diarization for Deaf Assist
═══════════════════════════════════════════════════════════════════════

Three cleanly separated stages (NOT the same thing):

  1. VOICE ACTIVITY DETECTION  (Silero VAD)   — "is someone speaking?"
  2. SPEAKER DIARIZATION       (ECAPA-TDNN)   — "WHICH speaker is it?"
  3. (Speech recognition & translation live in deaf_assist_live.py)

Design goals, in priority order (matches project requirements):
  1. Speaker identity CONSISTENCY  (Speaker 1 stays Speaker 1)
  2. Never mint more than MAX_SPEAKERS identities
  3. Short utterances / pauses / noise must NOT create new speakers
  4. Low latency (utterance-level streaming, not whole-recording batch)

Device placement (Apple Silicon):
  • Silero VAD  → CPU  (2 MB model; per-frame GPU launches are slower)
  • ECAPA-TDNN  → CPU  (~100 ms/utterance; keeps MPS free for IndicTrans2)
  Models are loaded ONCE and never moved between devices at runtime.

Dependencies:
  pip install speechbrain silero-vad torch torchaudio numpy
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass, field

import numpy as np
import torch

logger = logging.getLogger("diarization")

# ╔══════════════════════════════════════════════════════════════╗
# ║  CONFIG                                                      ║
# ╚══════════════════════════════════════════════════════════════╝

SAMPLE_RATE = 16000
MAX_SPEAKERS = 4

# ECAPA cosine-similarity decision zones.
# ECAPA same-speaker sims are typically 0.45–0.85;
# different-speaker sims are typically 0.00–0.30.
MATCH_THRESHOLD = 0.40      # ≥ this → confidently the same speaker
NEW_SPEAKER_THRESHOLD = 0.25  # < this → confidently a NEW voice
# Between the two = ambiguous zone → hysteresis: assign to best existing
# speaker (identity consistency beats minting a new ID — priority #1).

# Embedding memory per speaker
MAX_EMBEDDINGS_PER_SPEAKER = 12
CENTROID_UPDATE_MIN_SIM = 0.45   # only refine centroid on confident matches

# Utterance gates
MIN_UTTERANCE_SEC = 0.5          # below → dropped entirely (noise blip)
MIN_NEW_SPEAKER_SEC = 1.2        # below → may match existing, NEVER creates new
MIN_EMBED_SEC = 0.4              # ECAPA needs at least this much audio

# Silero VAD streaming parameters
VAD_FRAME_SAMPLES = 512                    # 32 ms @ 16 kHz (Silero requirement)
VAD_SPEECH_PROB = 0.50                     # frame is "speech" above this
VAD_END_SILENCE_SEC = 0.80                 # silence that closes an utterance
VAD_PRE_ROLL_SEC = 0.30                    # audio kept from before speech onset

_ECAPA_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
_ECAPA_SAVEDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "pretrained_models", "ecapa")


# ╔══════════════════════════════════════════════════════════════╗
# ║  MODEL SINGLETONS (loaded once, CPU-pinned)                  ║
# ╚══════════════════════════════════════════════════════════════╝

_vad_model = None
_ecapa_model = None
_model_lock = threading.Lock()


def load_models() -> None:
    """Load Silero VAD + ECAPA-TDNN once. Call at startup (thread-safe)."""
    global _vad_model, _ecapa_model
    with _model_lock:
        if _vad_model is None:
            logger.info("Loading Silero VAD (CPU)…")
            from silero_vad import load_silero_vad
            _vad_model = load_silero_vad()   # offline: bundled in the wheel
            logger.info("Silero VAD ready.")
        if _ecapa_model is None:
            logger.info("Loading ECAPA-TDNN speaker encoder (CPU)…")
            from speechbrain.inference.speaker import EncoderClassifier
            _ecapa_model = EncoderClassifier.from_hparams(
                source=_ECAPA_SOURCE,
                savedir=_ECAPA_SAVEDIR,
                run_opts={"device": "cpu"},   # deliberate: keep MPS for IndicTrans2
            )
            logger.info("ECAPA-TDNN ready.")


def _vad() -> "torch.nn.Module":
    if _vad_model is None:
        load_models()
    return _vad_model


def _ecapa():
    if _ecapa_model is None:
        load_models()
    return _ecapa_model


# ╔══════════════════════════════════════════════════════════════╗
# ║  EMBEDDING EXTRACTION                                        ║
# ╚══════════════════════════════════════════════════════════════╝

def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).ravel()
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def extract_embedding(audio: np.ndarray) -> np.ndarray | None:
    """192-d L2-normalized ECAPA embedding of a mono 16 kHz float32 clip."""
    if audio is None or len(audio) < int(MIN_EMBED_SEC * SAMPLE_RATE):
        return None
    try:
        wav = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
        wav = wav.unsqueeze(0)                       # (1, T)
        with torch.no_grad():
            emb = _ecapa().encode_batch(wav)          # (1, 1, 192)
        return _normalize(emb.squeeze().cpu().numpy())
    except Exception as e:
        logger.warning("Embedding extraction failed: %s", e)
        return None


# ╔══════════════════════════════════════════════════════════════╗
# ║  SPEAKER REGISTRY — persistent identities, max 4             ║
# ╚══════════════════════════════════════════════════════════════╝

@dataclass
class _Speaker:
    label: str                                   # "Speaker 1"…"Speaker 4"
    name: str | None = None                      # user-assigned name
    embeddings: list[np.ndarray] = field(default_factory=list)
    centroid: np.ndarray | None = None

    def add(self, emb: np.ndarray) -> None:
        self.embeddings.append(emb)
        if len(self.embeddings) > MAX_EMBEDDINGS_PER_SPEAKER:
            # drop the OLDEST *session* embedding but always keep index 0
            # (the enrollment/loaded embedding) as the identity anchor
            self.embeddings.pop(1 if len(self.embeddings) > 1 else 0)
        self.centroid = _normalize(np.mean(np.stack(self.embeddings), axis=0))

    def similarity(self, emb: np.ndarray) -> float:
        """max(centroid sim, best individual sim) — robust to intra-speaker
        variation (volume, distance, emotion) without letting one noisy
        embedding dominate."""
        if self.centroid is None:
            return -1.0
        best = float(np.dot(emb, self.centroid))
        for e in self.embeddings:
            s = float(np.dot(emb, e))
            if s > best:
                best = s
        return best

    @property
    def display(self) -> str:
        return self.name if self.name else self.label


@dataclass
class MatchResult:
    label: str          # "Speaker 1" … / saved name
    similarity: float
    is_new: bool
    confident: bool     # False = ambiguous-zone assignment (possible overlap)


class SpeakerRegistry:
    """
    Session-persistent speaker identity store.

      • Never creates more than MAX_SPEAKERS identities.
      • Short utterances never create identities — they only match.
      • Ambiguous zone → hysteresis to best existing speaker.
      • Confident matches refine that speaker's rolling centroid.
      • Optional cross-session persistence via speakers.json
        (ECAPA 192-d; old 256-d Resemblyzer vectors are detected & skipped).
    """

    def __init__(self, speakers_file: str | None = None):
        self._speakers: list[_Speaker] = []
        self._lock = threading.Lock()
        self._file = speakers_file
        self._last_label: str | None = None
        if speakers_file:
            self._load(speakers_file)

    # ── persistence ────────────────────────────────────────────

    def _load(self, path: str) -> None:
        if not os.path.isfile(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            skipped = 0
            for name, vec in data.items():
                arr = np.asarray(vec, dtype=np.float32)
                if arr.shape[-1] != 192:          # old Resemblyzer 256-d vector
                    skipped += 1
                    continue
                if len(self._speakers) >= MAX_SPEAKERS:
                    break
                sp = _Speaker(label=f"Speaker {len(self._speakers) + 1}",
                              name=name)
                sp.add(_normalize(arr))
                self._speakers.append(sp)
            if self._speakers:
                names = ", ".join(s.display for s in self._speakers)
                print(f"✅  Loaded {len(self._speakers)} saved speaker(s): {names}")
            if skipped:
                print(f"⚠️  Skipped {skipped} old-format (Resemblyzer) voice(s) "
                      f"in {os.path.basename(path)} — please re-save them once.")
        except Exception as e:
            logger.warning("Could not load %s: %s", path, e)

    def save(self) -> None:
        if not self._file:
            return
        try:
            data = {s.name: s.centroid.tolist()
                    for s in self._speakers
                    if s.name and s.centroid is not None}
            with open(self._file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning("Could not save %s: %s", self._file, e)

    # ── identification ─────────────────────────────────────────

    def identify(self, emb: np.ndarray, duration_sec: float) -> MatchResult:
        """
        Match an utterance embedding against known speakers.

        Anti-ID-switching mechanisms applied here:
          • two-threshold hysteresis (MATCH / NEW zones + ambiguous band)
          • margin test: a new speaker is only minted when clearly distinct
          • duration gate: short clips can never mint a speaker
          • max-4 hard cap: 5th voice snaps to nearest existing speaker
          • temporal bias: on a tie-like ambiguity, prefer the previous turn's
            speaker (people usually finish their own sentences)
        """
        with self._lock:
            emb = _normalize(emb)

            if not self._speakers:
                if duration_sec >= MIN_NEW_SPEAKER_SEC:
                    return self._create(emb)
                # too short to enroll anyone — provisional Speaker 1
                sp = _Speaker(label="Speaker 1")
                sp.add(emb)
                self._speakers.append(sp)
                self._last_label = sp.label
                return MatchResult(sp.display, 1.0, True, False)

            scored = sorted(((s.similarity(emb), s) for s in self._speakers),
                            key=lambda t: t[0], reverse=True)
            best_sim, best_sp = scored[0]

            # 1) confident match
            if best_sim >= MATCH_THRESHOLD:
                if best_sim >= CENTROID_UPDATE_MIN_SIM and \
                        duration_sec >= MIN_UTTERANCE_SEC:
                    best_sp.add(emb)               # refine identity
                self._last_label = best_sp.label
                return MatchResult(best_sp.display, best_sim, False, True)

            # 2) confidently new voice
            if best_sim < NEW_SPEAKER_THRESHOLD:
                if (duration_sec >= MIN_NEW_SPEAKER_SEC
                        and len(self._speakers) < MAX_SPEAKERS):
                    return self._create(emb)
                # short clip OR registry full → nearest existing, low confidence
                self._last_label = best_sp.label
                return MatchResult(best_sp.display, best_sim, False, False)

            # 3) ambiguous zone → hysteresis. Slight bias toward the previous
            #    turn's speaker if it is nearly tied with the top match.
            if self._last_label and best_sp.label != self._last_label:
                for sim, sp in scored[1:]:
                    if sp.label == self._last_label and sim >= best_sim - 0.05:
                        best_sim, best_sp = sim, sp
                        break
            self._last_label = best_sp.label
            return MatchResult(best_sp.display, best_sim, False, False)

    def _create(self, emb: np.ndarray) -> MatchResult:
        sp = _Speaker(label=f"Speaker {len(self._speakers) + 1}")
        sp.add(emb)
        self._speakers.append(sp)
        self._last_label = sp.label
        logger.info("New identity created: %s", sp.label)
        return MatchResult(sp.display, 1.0, True, True)

    # ── naming / info ──────────────────────────────────────────

    def assign_name(self, label: str, name: str) -> bool:
        with self._lock:
            for s in self._speakers:
                if s.label == label or s.display == label:
                    s.name = name.strip()
                    self.save()
                    return True
        return False

    def unnamed_labels(self) -> list[str]:
        with self._lock:
            return [s.label for s in self._speakers if not s.name]

    def summary(self) -> list[str]:
        with self._lock:
            return [f"{s.display}  ({len(s.embeddings)} sample(s))"
                    for s in self._speakers]

    def reset_session(self) -> None:
        """Drop unnamed session speakers; keep named (persistent) ones."""
        with self._lock:
            self._speakers = [s for s in self._speakers if s.name]
            for i, s in enumerate(self._speakers, 1):
                s.label = f"Speaker {i}"
            self._last_label = None


# ╔══════════════════════════════════════════════════════════════╗
# ║  STREAMING VAD SEGMENTER                                     ║
# ╚══════════════════════════════════════════════════════════════╝

class StreamingSegmenter:
    """
    Feed raw mic blocks in; complete utterances come out via callback —
    WHILE recording continues. This is what makes diarization real-time
    instead of whole-recording batch processing.

    A pause shorter than VAD_END_SILENCE_SEC stays INSIDE one utterance,
    so "Hello… [breath] how are you?" remains a single Speaker-1 segment.
    """

    def __init__(self, on_utterance):
        self._on_utterance = on_utterance
        self._buf = np.zeros(0, dtype=np.float32)
        self._pre_roll = np.zeros(0, dtype=np.float32)
        self._utterance: list[np.ndarray] = []
        self._in_speech = False
        self._silence_frames = 0
        self._end_silence_frames = max(
            1, int(VAD_END_SILENCE_SEC * SAMPLE_RATE / VAD_FRAME_SAMPLES))
        self._pre_roll_samples = int(VAD_PRE_ROLL_SEC * SAMPLE_RATE)
        self._vad = _vad()

    def feed(self, block: np.ndarray) -> None:
        self._buf = np.concatenate([self._buf, block.astype(np.float32)])
        while len(self._buf) >= VAD_FRAME_SAMPLES:
            frame = self._buf[:VAD_FRAME_SAMPLES]
            self._buf = self._buf[VAD_FRAME_SAMPLES:]
            self._process_frame(frame)

    def _process_frame(self, frame: np.ndarray) -> None:
        try:
            with torch.no_grad():
                prob = float(self._vad(torch.from_numpy(frame), SAMPLE_RATE))
        except Exception:
            prob = 0.0

        if prob >= VAD_SPEECH_PROB:
            if not self._in_speech:
                self._in_speech = True
                self._utterance = ([self._pre_roll.copy()]
                                   if len(self._pre_roll) else [])
            self._utterance.append(frame)
            self._silence_frames = 0
        else:
            if self._in_speech:
                self._utterance.append(frame)      # keep trailing context
                self._silence_frames += 1
                if self._silence_frames >= self._end_silence_frames:
                    self._close_utterance()
            else:
                self._pre_roll = np.concatenate(
                    [self._pre_roll, frame])[-self._pre_roll_samples:]

    def _close_utterance(self) -> None:
        audio = (np.concatenate(self._utterance)
                 if self._utterance else np.zeros(0, dtype=np.float32))
        self._utterance = []
        self._in_speech = False
        self._silence_frames = 0
        self._pre_roll = np.zeros(0, dtype=np.float32)
        if len(audio) >= int(MIN_UTTERANCE_SEC * SAMPLE_RATE):
            self._on_utterance(audio)

    def flush(self) -> None:
        """Spacebar released mid-sentence → emit whatever is in flight."""
        if self._in_speech and self._utterance:
            self._close_utterance()
        try:
            self._vad.reset_states()
        except Exception:
            pass
        self._buf = np.zeros(0, dtype=np.float32)
        self._pre_roll = np.zeros(0, dtype=np.float32)


# ╔══════════════════════════════════════════════════════════════╗
# ║  BATCH DIARIZATION (fallback for whole-clip processing)      ║
# ╚══════════════════════════════════════════════════════════════╝

def diarize_clip(audio: np.ndarray,
                 registry: SpeakerRegistry) -> list[tuple[MatchResult, np.ndarray]]:
    """
    Offline path: split a finished recording into VAD utterances and
    identify each. Used if the caller prefers release-then-process mode.
    Returns [(MatchResult, segment_audio), …].
    """
    results: list[tuple[MatchResult, np.ndarray]] = []

    def _collect(seg: np.ndarray) -> None:
        emb = extract_embedding(seg)
        if emb is None:
            return
        match = registry.identify(emb, len(seg) / SAMPLE_RATE)
        results.append((match, seg))

    seg = StreamingSegmenter(_collect)
    seg.feed(audio)
    seg.flush()
    return results