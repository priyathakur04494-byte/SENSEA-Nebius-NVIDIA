import os
import queue
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from deaf_assist2 import MAX_SPEAKERS, SAMPLE_RATE, LiveSpeakerTracker, VADSplitter


class DummyStream:
    def __init__(self, emb):
        self.embedding = emb

    def accept_waveforms(self, sample_rate, audio):
        pass


class DummyExtractor:
    def __init__(self):
        self._count = 0

    def create_stream(self):
        emb = np.zeros(16, dtype=np.float32)
        emb[self._count % 16] = 1.0
        self._count += 1
        return DummyStream(emb)

    def compute(self, stream):
        pass


def test_max_speakers_is_four():
    assert MAX_SPEAKERS == 4


def test_vad_emits_during_continuous_speech():
    q = queue.Queue()
    vad = VADSplitter(q)

    for i in range(12):
        block = np.ones(1600, dtype=np.float32) * 0.1
        vad.feed(block, i * 0.1)

    assert not q.empty(), "speech should emit before silence when speaking continues"


def test_tracker_never_creates_fifth_speaker():
    tracker = LiveSpeakerTracker(DummyExtractor())
    tracker._centroids = [
        np.array([1.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0], dtype=np.float32),
        np.array([-1.0, 0.0], dtype=np.float32),
        np.array([0.0, -1.0], dtype=np.float32),
    ]
    tracker._counts = [1, 1, 1, 1]

    new_emb = np.array([0.8, 0.6], dtype=np.float32)
    sid = tracker._match_or_create(new_emb, update_allowed=True)

    assert sid in range(MAX_SPEAKERS)
    assert len(tracker._centroids) == MAX_SPEAKERS


def test_short_second_voice_creates_new_speaker():
    tracker = LiveSpeakerTracker(DummyExtractor())
    tracker._centroids = [np.array([1.0, 0.0], dtype=np.float32)]
    tracker._counts = [1]

    new_emb = np.array([-1.0, 0.0], dtype=np.float32)
    sid = tracker._match_or_create(new_emb, update_allowed=False)

    assert sid == 1
    assert len(tracker._centroids) == 2
