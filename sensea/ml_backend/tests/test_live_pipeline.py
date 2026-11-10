# test_live_pipeline.py
#
# Regression tests for the live speaker pipeline.
#
# Run with: python test_live_pipeline.py
#
# The previous version of this file imported `LiveSpeakerTracker` and
# `VADSplitter` from `deaf_assist2`. Both were replaced by
# `SpeakerRegistry` and `StreamingSegmenter` in
# `speaker_diarization.py`, so the file could not even be imported.
# It now tests the current implementation.

import os
import sys

import numpy as np

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

import torch

import speaker_diarization as sd
from speaker_diarization import (
    MAX_SPEAKERS,
    MIN_NEW_SPEAKER_SEC,
    MIN_UTTERANCE_SEC,
    SpeakerRegistry,
    StreamingSegmenter,
    VAD_END_SILENCE_SEC,
    VAD_FRAME_SAMPLES,
    diarize_clip,
)


# --------------------------------------------------
# TEST RUNNER
# --------------------------------------------------

passed = 0
failed = 0
results_log = []


def test(name: str, ok: bool, notes: str = ""):
    global passed, failed

    status = "PASS" if ok else "FAIL"

    if ok:
        passed += 1
    else:
        failed += 1

    results_log.append(f"[{status}] {name}")

    if not ok:
        results_log.append(f"        → {notes}")


def one_hot(index: int, size: int = 192) -> np.ndarray:
    """An embedding that is orthogonal to every other one_hot vector."""
    vec = np.zeros(size, dtype=np.float32)
    vec[index] = 1.0
    return vec


class StubVad:
    """
    Stand-in for the Silero VAD.

    The segmenter's own logic (pre-roll, end silence, minimum
    utterance length) is what is under test here, so the speech
    probability is derived from the signal level instead of
    running the model.
    """

    def __init__(self, threshold: float = 0.01):
        self.threshold = threshold

    def __call__(self, frame, sample_rate):
        if hasattr(frame, "detach"):
            samples = frame.detach().cpu().numpy()
        else:
            samples = np.asarray(frame, dtype=np.float32)

        if samples.size == 0:
            return torch.tensor(0.0)

        rms = float(
            np.sqrt(np.mean(np.square(samples)))
        )

        return torch.tensor(
            0.9 if rms >= self.threshold else 0.0
        )

    def reset_states(self):
        pass


def feed(segmenter, seconds: float, value: float):
    """Feed `seconds` of a constant signal into the segmenter."""
    samples = int(seconds * sd.SAMPLE_RATE)
    block = np.full(samples, value, dtype=np.float32)
    segmenter.feed(block)


# ==================================================
# SECTION A: CONSTANTS
# ==================================================

test(
    "A1: MAX_SPEAKERS is four",
    MAX_SPEAKERS == 4,
    f"MAX_SPEAKERS={MAX_SPEAKERS}"
)

# ==================================================
# SECTION B: VAD / SEGMENTER
# ==================================================

original_vad = sd._vad
sd._vad = lambda: StubVad()

try:

    utterances = []
    segmenter = StreamingSegmenter(
        lambda audio: utterances.append(audio)
    )

    feed(segmenter, 1.5, 0.1)
    feed(segmenter, VAD_END_SILENCE_SEC + 0.5, 0.0)
    segmenter.flush()

    test(
        "B1: speech followed by silence emits one utterance",
        len(utterances) == 1,
        f"emitted={len(utterances)}"
    )

    test(
        "B2: the emitted utterance keeps the speech",
        len(utterances) == 1
        and len(utterances[0]) >= int(1.0 * sd.SAMPLE_RATE),
        f"len={len(utterances[0]) if utterances else 0}"
    )

    # Silence only -> nothing to emit
    utterances = []
    segmenter = StreamingSegmenter(
        lambda audio: utterances.append(audio)
    )

    feed(segmenter, 2.0, 0.0)
    segmenter.flush()

    test(
        "B3: silence only emits no utterance",
        len(utterances) == 0,
        f"emitted={len(utterances)}"
    )

    # A blip flushed before the minimum length is dropped
    utterances = []
    segmenter = StreamingSegmenter(
        lambda audio: utterances.append(audio)
    )

    feed(segmenter, MIN_UTTERANCE_SEC / 2, 0.1)
    segmenter.flush()

    test(
        "B4: a blip below MIN_UTTERANCE_SEC is dropped",
        len(utterances) == 0,
        f"emitted={len(utterances)}"
    )

finally:
    sd._vad = original_vad

# ==================================================
# SECTION C: SPEAKER REGISTRY
# ==================================================

registry = SpeakerRegistry()

first = registry.identify(one_hot(0), MIN_NEW_SPEAKER_SEC)

test(
    "C1: the first voice becomes Speaker 1",
    first.label == "Speaker 1" and first.is_new,
    f"label={first.label} is_new={first.is_new}"
)

second = registry.identify(one_hot(1), MIN_NEW_SPEAKER_SEC)

test(
    "C2: a clearly different voice becomes a new speaker",
    second.label == "Speaker 2" and second.is_new,
    f"label={second.label} is_new={second.is_new}"
)

third = registry.identify(one_hot(2), MIN_NEW_SPEAKER_SEC)
fourth = registry.identify(one_hot(3), MIN_NEW_SPEAKER_SEC)

test(
    "C3: four distinct voices fill the registry",
    third.label == "Speaker 3"
    and fourth.label == "Speaker 4"
    and len(registry._speakers) == MAX_SPEAKERS,
    f"labels={third.label}/{fourth.label} "
    f"count={len(registry._speakers)}"
)

fifth = registry.identify(one_hot(4), MIN_NEW_SPEAKER_SEC)

test(
    "C4: a fifth voice never creates a fifth speaker",
    len(registry._speakers) == MAX_SPEAKERS and not fifth.is_new,
    f"count={len(registry._speakers)} is_new={fifth.is_new}"
)

same = registry.identify(one_hot(0), MIN_NEW_SPEAKER_SEC)

test(
    "C5: the same voice keeps its identity",
    same.label == "Speaker 1" and not same.is_new,
    f"label={same.label} is_new={same.is_new}"
)

# A short clip can never mint a new speaker
short = registry.identify(one_hot(5), MIN_NEW_SPEAKER_SEC - 0.5)

test(
    "C6: a short clip does not create a new speaker",
    not short.is_new and len(registry._speakers) == MAX_SPEAKERS,
    f"is_new={short.is_new} count={len(registry._speakers)}"
)

# ==================================================
# SECTION D: EMPTY INPUT
# ==================================================

test(
    "D1: diarize_clip(None) returns an empty list",
    diarize_clip(None, SpeakerRegistry()) == [],
    "diarize_clip(None) raised or returned something else"
)

test(
    "D2: diarize_clip(empty) returns an empty list",
    diarize_clip(
        np.zeros(0, dtype=np.float32),
        SpeakerRegistry()
    ) == [],
    "diarize_clip(empty) raised or returned something else"
)

# ==================================================
# RESULTS SUMMARY
# ==================================================

print("\n" + "=" * 60)
print("LIVE PIPELINE TEST RESULTS")
print("=" * 60)

for line in results_log:
    print(line)

print("\n" + "=" * 60)
print(f"Total: {passed + failed} | Passed: {passed} | Failed: {failed}")
print("=" * 60)

if failed:
    print("\nSome tests failed.")
    sys.exit(1)

print("\nAll tests passed!")
