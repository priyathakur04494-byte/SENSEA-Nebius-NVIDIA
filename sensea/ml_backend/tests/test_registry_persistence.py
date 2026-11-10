import json

import numpy as np

from speaker_diarization import SpeakerRegistry


def make_embedding(seed=42, dimensions=192):
    rng = np.random.default_rng(seed)
    embedding = rng.normal(size=dimensions).astype(np.float32)
    return embedding / np.linalg.norm(embedding)


def test_missing_speakers_file_loads_safely(tmp_path):
    path = tmp_path / "missing_speakers.json"
    registry = SpeakerRegistry(str(path))
    assert registry.summary() == []


def test_named_speaker_is_saved_and_reloaded(tmp_path):
    path = tmp_path / "speakers.json"
    registry = SpeakerRegistry(str(path))

    result = registry.identify(make_embedding(), 2.0)
    assert result.is_new
    assert registry.assign_name(result.label, "Test Speaker")

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert "Test Speaker" in saved
    assert len(saved["Test Speaker"]) == 192

    reloaded = SpeakerRegistry(str(path))
    assert any(s.name == "Test Speaker" for s in reloaded._speakers)

    # A slightly changed embedding should still match the saved speaker.
    similar = make_embedding()
    similar = similar + np.random.default_rng(43).normal(
        0, 0.005, size=192
    ).astype(np.float32)
    similar /= np.linalg.norm(similar)

    match = reloaded.identify(similar, 2.0)
    assert match.label == "Test Speaker"
    assert not match.is_new


def test_unnamed_speaker_is_not_persisted(tmp_path):
    path = tmp_path / "speakers.json"
    registry = SpeakerRegistry(str(path))

    registry.identify(make_embedding(), 2.0)
    registry.save()

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved == {}


def test_legacy_256_dimension_embedding_is_skipped(tmp_path):
    path = tmp_path / "speakers.json"
    path.write_text(
        json.dumps({"Legacy Speaker": [0.1] * 256}),
        encoding="utf-8",
    )

    registry = SpeakerRegistry(str(path))
    assert not any(
        s.name == "Legacy Speaker" for s in registry._speakers
    )


def test_malformed_json_loads_without_crashing(tmp_path):
    path = tmp_path / "speakers.json"
    path.write_text("{invalid json", encoding="utf-8")

    registry = SpeakerRegistry(str(path))
    assert registry.summary() == []
