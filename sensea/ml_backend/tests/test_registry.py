"""
test_registry.py — Simulates the 5 required test scenarios against
SpeakerRegistry using synthetic ECAPA-like embeddings (no audio needed).

Synthetic model: each "person" is a random unit base vector; each
utterance = base + Gaussian noise, renormalized. In 192 dims a
per-dim scale s gives a noise vector of norm s*sqrt(192), so
s=0.040 → noise norm ≈ 0.55 → same-speaker cosine ≈ 0.87, while
random different-speaker bases give cosine ≈ 0.00 ± 0.07 — matching
real ECAPA-TDNN separation.

Run:  python test_registry.py
"""

import sys
import types

# Stub torch just enough for speaker_diarization import (registry logic
# itself is pure numpy). On your Mac, real torch is installed and used.
if "torch" not in sys.modules:
    try:
        import torch  # noqa: F401
    except ImportError:
        stub = types.ModuleType("torch")
        stub.no_grad = lambda: types.SimpleNamespace(
            __enter__=lambda s: None, __exit__=lambda s, *a: False)
        stub.from_numpy = lambda x: x
        stub.nn = types.SimpleNamespace(Module=object)
        sys.modules["torch"] = stub

import numpy as np
from speaker_diarization import SpeakerRegistry

rng = np.random.default_rng(42)


def make_person() -> np.ndarray:
    v = rng.normal(size=192).astype(np.float32)
    return v / np.linalg.norm(v)


def utterance_of(person: np.ndarray, noise: float = 0.040) -> np.ndarray:
    v = person + rng.normal(scale=noise, size=192).astype(np.float32)
    return v / np.linalg.norm(v)


def run_test(name: str, sequence: list, expected: list,
             durations: list | None = None) -> bool:
    registry = SpeakerRegistry()          # fresh session, no persistence
    people = {}
    got = []
    for i, pid in enumerate(sequence):
        if pid not in people:
            people[pid] = make_person()
        dur = durations[i] if durations else 3.0
        emb = utterance_of(people[pid])
        match = registry.identify(emb, dur)
        got.append(match.label)
    ok = got == expected
    status = "✅ PASS" if ok else "❌ FAIL"
    print(f"{status}  {name}")
    if not ok:
        print(f"        expected: {expected}")
        print(f"        got     : {got}")
    return ok


results = []

# Test 1 — Two speakers alternating
results.append(run_test(
    "Test 1: A→B→A→B",
    ["A", "B", "A", "B"],
    ["Speaker 1", "Speaker 2", "Speaker 1", "Speaker 2"]))

# Test 2 — Three speakers
results.append(run_test(
    "Test 2: A→B→C→A→C→B",
    ["A", "B", "C", "A", "C", "B"],
    ["Speaker 1", "Speaker 2", "Speaker 3",
     "Speaker 1", "Speaker 3", "Speaker 2"]))

# Test 3 — Four speakers, two rounds
results.append(run_test(
    "Test 3: A→B→C→D→A→B→C→D",
    ["A", "B", "C", "D", "A", "B", "C", "D"],
    ["Speaker 1", "Speaker 2", "Speaker 3", "Speaker 4",
     "Speaker 1", "Speaker 2", "Speaker 3", "Speaker 4"]))

# Test 4 — Short utterances must not mint new speakers.
# A and B enroll with long utterances; then short (0.6 s) turns follow.
results.append(run_test(
    "Test 4: short utterances (A:Hello B:Hi A:Yes B:Okay A:Let's go)",
    ["A", "B", "A", "B", "A"],
    ["Speaker 1", "Speaker 2", "Speaker 1", "Speaker 2", "Speaker 1"],
    durations=[3.0, 3.0, 0.6, 0.6, 0.6]))

# Test 5 — Same speaker many times, interleaved
results.append(run_test(
    "Test 5: A→B→A→C→A→B→A",
    ["A", "B", "A", "C", "A", "B", "A"],
    ["Speaker 1", "Speaker 2", "Speaker 1", "Speaker 3",
     "Speaker 1", "Speaker 2", "Speaker 1"]))

# Test 6 (extra) — 5th distinct voice must NOT create Speaker 5
registry = SpeakerRegistry()
people = [make_person() for _ in range(5)]
labels = [registry.identify(utterance_of(p), 3.0).label for p in people]
n_ids = len(set(labels))
cap_ok = n_ids <= 4
print(f"{'✅ PASS' if cap_ok else '❌ FAIL'}  Test 6: 5 voices → max 4 IDs "
      f"(got {n_ids} distinct: {labels})")
results.append(cap_ok)

# Test 7 (extra) — 20 noisy re-utterances of one person stay Speaker 1
registry = SpeakerRegistry()
p = make_person()
stable = all(registry.identify(utterance_of(p, noise=0.060), 2.0).label
             == "Speaker 1" for _ in range(20))
print(f"{'✅ PASS' if stable else '❌ FAIL'}  Test 7: 20 noisy turns of one "
      f"person never switch ID")
results.append(stable)

print()
print(f"{'ALL TESTS PASSED' if all(results) else 'SOME TESTS FAILED'} "
      f"({sum(results)}/{len(results)})")
sys.exit(0 if all(results) else 1)
