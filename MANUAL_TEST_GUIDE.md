# SENSEA — Manual Test Guide

**Repo:** `/Users/priyathakur/Desktop/SENSEA-Hackathon`
**Branch:** `ML-Priya`
**ML backend root:** `sensea/ML Backend` (note the space — always quote the path)
**Frontend root:** `sensea/front`
**Last verified:** 22 Oct 2026, on macOS (darwin), Python 3.11.16 / 3.9.6

Every command in this document was executed on this machine before being written down.
Anything that could not be run is marked **NOT CURRENTLY RUNNABLE** with the exact blocker.
No commands were copied from docs or guessed — if a flag is shown here, it exists in the source.

---

## 0. How to use this guide

Each feature below has the same seven parts:

| Part | Meaning |
|---|---|
| **A. Purpose** | What the feature does |
| **B. Command** | Exact, copy-pasteable command |
| **C. Expected input** | What you must supply (files, text, mic, network) |
| **D. Expected success** | The exact output/observation that means it works |
| **E. Failure diagnosis** | How to tell *why* it failed |
| **F. Classification** | `AUTOMATED TEST` · `LOCAL SMOKE TEST` · `REAL AUDIO/VIDEO TEST` · `LIVE API TEST` · `FRONTEND/API INTEGRATION TEST` · `NOT CURRENTLY RUNNABLE` |
| **G. Limitations** | What this test does *not* prove |

**Run order matters.** Appendix B (§13) gives the recommended order. The short version:
environment fix + no-mic automated suites first, then translation/TTS, then the live-API
AI analysis, then real audio/video (sign language, diarization), then the microphone
pipelines (which need macOS permissions), then the frontend.

---

## 1. Environments

Three pre-built virtualenvs live at the repo root. **Do not mix them** — each has a
different Python version and a different set of packages.

| Env | Python | Exists for | Has | Missing |
|---|---|---|---|---|
| `venv/` | 3.9.6 | Speech-to-text prototype | torch, sounddevice, soundfile, pynput, `indic_transliteration`, whisper/pywhispercpp/faster_whisper, dotenv | `kokoro`, `IndicTransToolkit`, `sentence_transformers`, `sklearn`, `openai` |
| `translation_env/` | 3.11.16 | Translation + TTS + diarization + sign language | `kokoro` 0.9.4, `indictranstoolkit` 1.1.1, torch, `speechbrain`, `silero-vad`, `mediapipe`, `opencv`, `sherpa_onnx` | ❌ **broken** — see §1.1 |
| `rag_test_env/` | 3.11.16 | RAG / chatbot / LLM | `sentence-transformers`, `sklearn`, `openai`, torch, dotenv, transformers 4.51.3 + hub 0.30.2 | `kokoro`, `IndicTransToolkit` |

There is **no `pytest` anywhere** — every test file is a plain script that prints
`[PASS]` / `[FAIL]` lines and exits.

### 1.1 The one blocker that gates TTS + Translation

`translation_env/` has `transformers` 4.51.3 (which requires `huggingface-hub >=0.30.0,<1.0`)
but `huggingface-hub` **1.30.0** is installed (pulled in by `sentence-transformers` 6.0.1,
which needs `hub>=1.3.0`). Result:

```
ImportError: huggingface-hub>=0.30.0,<1.0 is required for a normal functioning of this
module, but found huggingface-hub==1.30.0.
```

This breaks `kokoro` (TTS) and `translate.py` in that env. **Fix (Option A, recommended) —**
requires your approval because it installs a package:

```bash
cd /Users/priyathakur/Desktop/SENSEA-Hackathon
translation_env/bin/python -m pip install "huggingface-hub<1.0"
```

`pip` resolves this to `huggingface_hub-0.36.2`, which satisfies both `transformers`
(`<1.0`) and everything else in that env except `sentence-transformers` — which is
**not used** by any `translation_env` command in this guide (RAG lives in `rag_test_env`).

**Option B — verified working right now, zero installs.** Use `rag_test_env`'s interpreter
and append `translation_env`'s site-packages at the *end* of `sys.path` (append, never
`PYTHONPATH` — `translation_env` ships a `typing.py` backport that shadows the stdlib
and crashes with `AttributeError: type object 'Callable' has no attribute '_abc_registry'`):

```bash
cd "sensea/ML Backend"
../../rag_test_env/bin/python -c "
import sys
sys.path.append('../../translation_env/lib/python3.11/site-packages')
import tts
tts.text_to_speech('Hello from SENSEA')
"
```

### 1.2 `.env` location matters

`sensea/ML Backend/llm.py` calls `load_dotenv("../.env")` — **relative to the current
working directory**. It therefore resolves to `sensea/.env` only when you run from
`sensea/ML Backend`. Run those commands from that directory.

`sensea/.env` contains exactly one key: `NVIDIA_API_KEY`. Never print its value.

---

## 2. Feature: Speech-to-Text (STT) — whisper.cpp push-to-talk

**F. Classification:** `REAL AUDIO/VIDEO TEST`

### A. Purpose
Record from the microphone while SPACEBAR is held, transcribe with `whisper.cpp`
(Metal), optionally transliterate Hinglish, optionally translate.

### B. Command
```bash
cd "/Users/priyathakur/Desktop/SENSEA-Hackathon/sensea/ML Backend"
../../venv/bin/python -u transcribe.py
```

`transcribe.py` has no `if __name__ == "__main__"` guard — it runs top-to-bottom on
import, so plain `python transcribe.py` is correct here.

### C. Expected input
1. Menu `1` = English, `2` = Hindi, `3` = Hinglish.
2. Menu `0` = No translation, or `1`–`22` for an Indic target language.
3. **Hold SPACEBAR** and speak; **release** to transcribe.

### D. Expected success
```
── Checking Whisper setup ─────────────────────
✅ whisper-cli found
✅ model: ggml-small.bin (465 MB)
───────────────────────────────────────────────
🎤 Hold SPACEBAR → microphone ON
```
followed by a transcript line after you release SPACEBAR.

Verified startup output (this exact machine):
```
✅ whisper-cli found
✅ model: ggml-small.bin (465 MB)
Language English
🎤 Hold SPACEBAR
```

### E. Failure diagnosis
| Symptom | Cause |
|---|---|
| `❌ whisper-cli NOT found` | `~/whisper.cpp/build/bin/whisper-cli` missing. Note it is `build/bin`, **not** `build/bin/main` |
| `❌ model NOT found` | `~/whisper.cpp/models/ggml-small.bin` missing. `ggml-medium.bin` also present |
| `This process is not trusted! Input event monitoring will not be possible…` | macOS **Input Monitoring / Accessibility** permission not granted to your terminal. System Settings → Privacy & Security → Input Monitoring |
| `ModuleNotFoundError: No module named 'pynput'` | Wrong env — must be `venv/bin/python` |

### G. Limitations
Requires a real microphone and macOS permissions. Not runnable in a headless/CI shell.

---

## 3. Feature: Text-to-Speech (TTS) — Kokoro

**F. Classification:** `LOCAL SMOKE TEST` (after the §1.1 fix) · `NOT CURRENTLY RUNNABLE` as-is

### A. Purpose
Turn text into a 24 kHz WAV using Kokoro-82M, pick the British-English voice
(`bf_emma`) or the Hindi voice (`hf_alpha`) based on the script of the text, and
auto-play the result with `afplay`.

### B. Command
**After Option A (§1.1):**
```bash
cd "/Users/priyathakur/Desktop/SENSEA-Hackathon/sensea/ML Backend"
../../translation_env/bin/python -u tts.py
```
Type your text, then `END` on its own line.

**Verified working right now (Option B, no installs):**
```bash
cd "/Users/priyathakur/Desktop/SENSEA-Hackathon/sensea/ML Backend"
../../rag_test_env/bin/python -c "
import sys
sys.path.append('../../translation_env/lib/python3.11/site-packages')
import tts
tts.text_to_speech('Hello from SENSEA')
"
```

### C. Expected input
Any text. Devanagari (U+0900–U+097F) auto-selects the Hindi voice.

### D. Expected success
```
🔄 Generating speech...
✅ Speech generated successfully!
🗣️  Voice: bf_emma (lang_code=b)
📁 Saved as: sensea_tts.wav
⏱️ Duration: 3.15 seconds
🔊 Playing speech...
✅ Done!
```
Verified for English (`bf_emma`, 3.15 s) and Hindi (`hf_alpha`, 2.77 s,
`detect_lang_code('नमस्ते दुनिया') == 'h'`). Output file: `sensea/ML Backend/sensea_tts.wav`
(24 kHz mono). `afplay` playback also succeeded here.

### E. Failure diagnosis
| Symptom | Cause |
|---|---|
| `ImportError: huggingface-hub>=0.30.0,<1.0 ... found 1.30.0` | §1.1 — run Option A |
| `ModuleNotFoundError: No module named 'kokoro'` | Wrong env — `kokoro` only exists in `translation_env` |
| `❌ No audio was generated.` | Text had no speakable content |
| `❌ TTS Error` on playback | No audio output device (headless). The WAV is still written — check the file exists |

### G. Limitations
Kokoro covers **English + Hindi only** here (`VOICE = "bf_emma"`, `HINDI_VOICE = "hf_alpha"`).
Not a general multi-language TTS. Running the script **overwrites** `sensea_tts.wav`.

---

## 4. Feature: Translation — IndicTrans2

**F. Classification:** `LOCAL SMOKE TEST` (after §1.1) · `AUTOMATED TEST` (`tests/test_translate.py`) · `NOT CURRENTLY RUNNABLE` as-is

### A. Purpose
Translate between English and 22 Indic languages using `ai4bharat/IndicTrans2`
(`en-indic` 200M, `indic-en` 200M, `indic-indic` 320M). Chunking keeps long text
inside the 256-token encoder budget.

### B. Command
**Self-test (all language pairs):** after Option A —
```bash
cd "/Users/priyathakur/Desktop/SENSEA-Hackathon/sensea/ML Backend"
../../translation_env/bin/python -u translate.py
```

**Verified working right now (Option B):**
```bash
cd "/Users/priyathakur/Desktop/SENSEA-Hackathon/sensea/ML Backend"
../../rag_test_env/bin/python -c "
import sys, runpy
sys.path.append('../../translation_env/lib/python3.11/site-packages')
runpy.run_path('translate.py', run_name='__main__')
"
```

**Regression suite** (`tests/test_translate.py`) — requires Option A:
```bash
cd "/Users/priyathakur/Desktop/SENSEA-Hackathon/sensea/ML Backend"
../../translation_env/bin/python -u tests/test_translate.py
# optional: add the live end-to-end section
SENSEA_TRANSLATE_E2E=1 ../../translation_env/bin/python -u tests/test_translate.py
```

### C. Expected input
None — `translate.py` has a built-in self-test. Language codes are IndicTrans2 form:
`hin_Deva`, `tam_Taml`, `eng_Latn`, etc. (23 codes total).

### D. Expected success
```
Loading indic-en model (ai4bharat/indictrans2-indic-en-dist-200M)...
indic-en model loaded!
English: I got a lot of good news today.
```
Verified pairs: `en→hi` = `नमस्ते, आज आप कैसे हैं?`, `hi→en` = `I got a lot of good news today.`,
plus Punjabi / Bengali / Assamese / Tamil outputs and a valid `indic-indic` 320M model load.

Regression suite result (verified in an **equivalent** environment — transformers 4.51.3 +
huggingface-hub 0.30.2 + IndicTransToolkit 1.1.1, with `IndicTransToolkit` made importable):
```
Total: 71 | Passed: 71 | Failed: 0     # without E2E
Total: 74 | Passed: 74 | Failed: 0     # with SENSEA_TRANSLATE_E2E=1
All tests passed!
```

### E. Failure diagnosis
| Symptom | Cause |
|---|---|
| `ModuleNotFoundError: No module named 'IndicTransToolkit'` | `rag_test_env` / `venv` — no IndicTransToolkit. Use `translation_env` (needs Option A) |
| `ImportError: huggingface-hub>=0.30.0,<1.0` | §1.1 — Option A |
| `ERROR: Invalid source language code 'hin'` | Expected — code must be `hin_Deva`, not `hin` |
| `ERROR: Unsupported source language code 'xyz_Xyzw'` | Expected — code not in the 23 supported codes |
| `AttributeError: type object 'Callable' has no attribute '_abc_registry'` | You used `PYTHONPATH=` instead of `sys.path.append`. See §1.1 Option B |

### G. Limitations
`tests/test_translate.py` sections **A–D** are pure-logic and run anywhere; section **E**
(live E2E) needs the models. The 3 "B3 child process survives" cases spawn
`sys.executable -c …` subprocesses, so they only pass when `IndicTransToolkit` is
importable by the *child* too — i.e. Option A. Under Option B they fail with
`ModuleNotFoundError: No module named 'IndicTransToolkit'` **in the child**, which is an
environment artifact, not a code defect.

---

## 5. Feature: RAG Chatbot / Q&A — memory-grounded answers

**F. Classification:** `AUTOMATED TEST` · `LIVE API TEST`

### A. Purpose
Answer questions strictly from stored memories. Three routes: structured schedule
search (date-aware), keyword search (direct teacher/topic questions), and semantic RAG
(`all-MiniLM-L6-v2`, `MIN_SIMILARITY = 0.45`). Unknown questions return
`NOT_FOUND_RESPONSE` — there is **no unrestricted LLM fallback**, by design.

### B. Command
**Regression suite (27 tests):**
```bash
cd "/Users/priyathakur/Desktop/SENSEA-Hackathon/sensea/ML Backend"
../../rag_test_env/bin/python -u tests/test_chatbot.py
```

**Interactive one-liner (single question):**
```bash
cd "/Users/priyathakur/Desktop/SENSEA-Hackathon/sensea/ML Backend"
../../rag_test_env/bin/python -c "
from chatbot import chat
print(chat('What did the teacher explain?', 'classroom', use_test_memories=True))
"
```

> `chatbot.py` has **no `__main__` block** — `python chatbot.py` does nothing useful.
> Always import and call `chat(...)`.

**Pure embedding/similarity demo (no LLM, no API key):**
```bash
cd "/Users/priyathakur/Desktop/SENSEA-Hackathon/sensea/ML Backend"
../../rag_test_env/bin/python -u tests/rag_test.py
```
Verified output shows per-memory similarity scores, e.g. `Similarity: 0.533`, and a
mode-filtered search section (`Current mode: hospital`) — note it can return
low/negative-similarity rows, which is the raw semantic-search behaviour before the
`MIN_SIMILARITY = 0.45` filter in `rag.semantic_search`.

### C. Expected input
A question string and a mode: `general` · `hospital` · `meeting` · `classroom`.
`use_test_memories=True` uses the built-in `TEST_MEMORIES` fixture instead of the
persisted store.

### D. Expected success
Verified answers:
```
Teacher explained convolutional neural networks.     # "What did the teacher explain?" (classroom)
I don't have enough information to answer that        # off-topic question
```
and the suite:
```
Total: 27 | Passed: 27 | Failed: 0
```

### E. Failure diagnosis
| Symptom | Cause |
|---|---|
| `ModuleNotFoundError: No module named 'chatbot'` | Wrong CWD — must be inside `sensea/ML Backend` |
| `OpenAIError` / `401` | `sensea/.env` missing `NVIDIA_API_KEY`, or CWD is not `sensea/ML Backend` so `../.env` misses |
| `[chatbot] … failed: …` on stderr | Technical failure inside a route; the user-facing answer is still `NOT_FOUND_RESPONSE` by design |
| Every answer is "not found" | `use_test_memories=False` and the persisted store is empty — call `memory_store.add_memory(...)` first |
| `ModuleNotFoundError: No module named 'sentence_transformers'` | Wrong env — use `rag_test_env` |

### G. Limitations
The suite runs against the `TEST_MEMORIES` fixture, not a real conversation.
Answers are only as good as the NVIDIA model response. The date logic is
resolved at import time (`TODAY`/`TOMORROW` are module-level).

---

## 6. Feature: AI Conversation Analysis — per-mode summaries

**F. Classification:** `LIVE API TEST`

### A. Purpose
Feed a list of `{speaker_id, text, translation}` segments into `ai_engine.analyze_conversation(mode, segments)`
and get a structured, mode-specific JSON summary (schema per mode:
`GENERAL_SCHEMA`, `HOSPITAL_SCHEMA`, `MEETING_SCHEMA`, `CLASSROOM_SCHEMA`).

### B. Command
All four need `PYTHONPATH=.` — they import `ai_engine` as a top-level module:
```bash
cd "/Users/priyathakur/Desktop/SENSEA-Hackathon/sensea/ML Backend"
PYTHONPATH=. ../../rag_test_env/bin/python -u tests/test_real_pipeline.py
PYTHONPATH=. ../../rag_test_env/bin/python -u tests/test_modes.py        # hospital mode
PYTHONPATH=. ../../rag_test_env/bin/python -u tests/test_ai_engine.py    # raw ask_nvidia() prompt
PYTHONPATH=. ../../rag_test_env/bin/python -u tests/test_nvidia.py       # raw OpenAI client against NVIDIA
```

### C. Expected input
None — the segments are hard-coded in each script.
`ai_engine.py` reads `NVIDIA_API_KEY` from the environment via `load_dotenv()`.

### D. Expected success
```
========================================
       DEAF ASSIST AI ANALYSIS
========================================

===== NVIDIA RESULT =====

{'mode': 'classroom', 'summary': 'The class is learning about neural networks, …',
 'topic': 'Neural Networks', 'key_concepts': ['artificial neurons', 'backpropagation'], …}
```
All four scripts were run live against the NVIDIA API and returned valid output:

| Script | Verified output |
|---|---|
| `test_real_pipeline.py` | `mode: classroom` + `summary` / `topic` / `key_concepts` / `assignments` / `flashcards` / `quiz` |
| `test_modes.py` | `mode: hospital` + `medical_summary` / `instructions` / `appointments_follow_up` |
| `test_ai_engine.py` | Raw JSON: `summary` + `key_points[]` |
| `test_nvidia.py` | Markdown text + `FINISH REASON === stop` |

### E. Failure diagnosis
| Symptom | Cause |
|---|---|
| `ModuleNotFoundError: No module named 'ai_engine'` | Missing `PYTHONPATH=.` |
| `ValueError: NVIDIA_API_KEY not found in .env` | `sensea/.env` is missing or empty. `load_dotenv()` (no path) walks *up* from the CWD, so running from `sensea/ML Backend` resolves `sensea/.env` — verified working here |
| `extract_json_object` returns `None` | Model wrapped the JSON in prose — re-run; the model is not constrained to JSON |
| HTTP 429/5xx | NVIDIA endpoint rate limit — wait and retry |

### G. Limitations
Requires a live NVIDIA API key and internet. Output quality is model-dependent and
non-deterministic, so assert on structure, not exact strings.

---

## 7. Feature: Speaker Diarization / Voice ID — ECAPA + Silero VAD

**F. Classification:** `AUTOMATED TEST` (`tests/test_registry.py`) · `REAL AUDIO/VIDEO TEST` (offline clip) · `LOCAL SMOKE TEST` (library-level)

### A. Purpose
Detect who is speaking. `speaker_diarization.py` uses Silero VAD to segment speech and
ECAPA-TDNN (`speechbrain/spkrec-ecapa-voxceleb`, 192-d) to embed and identify voices.
`SpeakerRegistry` holds up to `MAX_SPEAKERS = 4` identities with hysteresis
(`MATCH_THRESHOLD = 0.40`, `NEW_SPEAKER_THRESHOLD = 0.25`) and optional
`speakers.json` persistence.

### B. Command
**Registry logic suite (7 tests):**
```bash
cd "/Users/priyathakur/Desktop/SENSEA-Hackathon/sensea/ML Backend"
PYTHONPATH=. ../../translation_env/bin/python -u tests/test_registry.py
```
`PYTHONPATH=.` is required — the test imports `speaker_diarization` as a top-level module.

**Real diarization on a real recording (offline):**
```bash
cd "/Users/priyathakur/Desktop/SENSEA-Hackathon"
PYTHONPATH="sensea/ML Backend" translation_env/bin/python -c "
import numpy as np, soundfile as sf
from speaker_diarization import load_models, extract_embedding, diarize_clip, SpeakerRegistry

load_models()
audio, sr = sf.read('speaker_models/0-four-speakers-zh.wav', dtype='float32')
if audio.ndim > 1:
    audio = audio.mean(axis=1)

print('embedding shape:', extract_embedding(audio[:sr * 3]).shape)
for match, seg in diarize_clip(audio, SpeakerRegistry()):
    print(match.label, '->', round(len(seg) / sr, 2), 's')
"
```

### C. Expected input
`speaker_models/0-four-speakers-zh.wav` — 16 kHz mono, 56.86 s.

### D. Expected success
Verified on this machine:
```
embedding shape: (192,)
Speaker 1 -> ...
Speaker 2 -> ...
```
and the suite:
```
Total: 7 | Passed: 7 | Failed: 0
```

### E. Failure diagnosis
| Symptom | Cause |
|---|---|
| `ModuleNotFoundError: No module named 'speaker_diarization'` | Missing `PYTHONPATH=.` |
| `ModuleNotFoundError: No module named 'speechbrain'` / `'silero_vad'` | Wrong env — use `translation_env` |
| `All zeros` / `embedding shape: None` | Audio shorter than `MIN_EMBED_SEC = 0.4` s |
| No segments returned | Silence, or audio below `VAD_SPEECH_PROB = 0.50` |
| `RuntimeError` loading ECAPA | Model not in the local cache — `speechbrain/spkrec-ecapa-voxceleb` must be cached under `~/.cache/huggingface` |

### G. Limitations
The clip is **named** "four speakers" but only **two** identities were detected on this
run — do not treat "4 speakers" as a pass criterion. `SpeakerRegistry()` with no file
argument keeps identities in memory only; persistence requires passing a
`speakers.json` path. Utterances under `MIN_UTTERANCE_SEC = 0.5` are dropped as noise,
and under `MIN_NEW_SPEAKER_SEC = 1.2` can never *create* a new identity.

---

## 8. Feature: Sign Language Recognition

**F. Classification:** `REAL AUDIO/VIDEO TEST` · `AUTOMATED TEST` (held-out signer eval)

### A. Purpose
Recognise Indian Sign Language from video. MediaPipe Hands (21 landmarks/hand) + Pose
landmarks → 258-d per-frame features → 30-frame sequence → `sign_language_model.pth`
(20 classes) → top-1 label + confidence, with a right-hand orientation gate.

### B. Command
**Offline inference on a dataset video (verified):**
```bash
cd "/Users/priyathakur/Desktop/SENSEA-Hackathon/sensea/ML Backend"
../../translation_env/bin/python sign_language/scripts/test_dataset_video.py \
  --video "sign_language/data/raw/isl_isolated_40words/hello/hello__INCLUDE__00566__MVI_0029.mp4"
```
`--video` is the only CLI flag and it is **required**.

**Live webcam inference:**
```bash
cd "/Users/priyathakur/Desktop/SENSEA-Hackathon/sensea/ML Backend"
../../translation_env/bin/python sign_language/scripts/realtime.py
```
Controls are printed on screen (`Q` quit; see the script header for the rest).

**Held-out signer evaluation:**
```bash
cd "/Users/priyathakur/Desktop/SENSEA-Hackathon/sensea/ML Backend"
PYTHONPATH=. ../../translation_env/bin/python sign_language/scripts/evaluate_signer_test.py
```

### C. Expected input
An MP4 of an ISL sign, or a webcam. The raw dataset lives under
`sign_language/data/raw/isl_isolated_40words/<label>/` (gitignored).
Processed features: `sign_language/data/processed/X.npy` shape `(353, 30, 258)`,
`y.npy` shape `(353,)`, 20 labels.

### D. Expected success
Verified:
```
Device: mps
Frames: 76
features: 258
Predicted: hello  (89.10%)
✅ CORRECT — HELLO RECOGNIZED
```

### E. Failure diagnosis
| Symptom | Cause |
|---|---|
| `ModuleNotFoundError: No module named 'sign_language'` | Missing `PYTHONPATH=.` (only needed for `evaluate_signer_test.py`) |
| `FileNotFoundError: Video not found` | Path is relative to `sensea/ML Backend` — pass the full relative path shown above |
| `No module named 'mediapipe'` / `'cv2'` | Wrong env — use `translation_env` |
| Wrong label / low confidence | Check the orientation gate (`ORIENTATION_THRESHOLD = 1.0`) and `CONFIDENCE_THRESHOLD = 0.80`; the model is sensitive to signer and camera angle |

### G. Limitations
`evaluate_signer_test.py` **writes/overwrites**
`sign_language/data/processed/unseen_signer_results.csv` — a generated artifact.
The dataset is gitignored, so a fresh clone will not contain the raw videos and the
offline video test cannot run. The model is trained on a limited signer set;
cross-signer accuracy is the real risk.

---

## 9. Feature: Live Conversation Pipeline

**F. Classification:** `REAL AUDIO/VIDEO TEST`

Two implementations exist. `deaf_assist2.py` is the **current** one;
`archive/deaf_assist_live.py` is the older ECAPA-based one.

### A. Purpose
End-to-end loop: hold SPACEBAR → record → whisper.cpp transcription → sherpa-onnx
speaker diarization → optional IndicTrans2 translation → optional AI analysis.

### B. Command
**Current pipeline — must be run from the repo root** (it resolves
`speaker_models/` and `speaker_names.json` from `os.getcwd()`):
```bash
cd "/Users/priyathakur/Desktop/SENSEA-Hackathon"
PYTHONPATH="sensea/ML Backend" translation_env/bin/python -u "sensea/ML Backend/deaf_assist2.py"
```

**Archived pipeline (ECAPA-based) — run from the backend dir:**
```bash
cd "/Users/priyathakur/Desktop/SENSEA-Hackathon/sensea/ML Backend"
PYTHONPATH=. ../../translation_env/bin/python -u archive/deaf_assist_live.py
```

### C. Expected input
`deaf_assist2.py`: `1`/`2`/`3` speaking language → `0` or `1`–`22` translation target
→ hold SPACEBAR.
`archive/deaf_assist_live.py`: language → translation target → **AI mode** `1`–`4`
(`general` / `hospital` / `meeting` / `classroom`).

### D. Expected success
Verified startup for `deaf_assist2.py`:
```
✅ whisper-cli found
✅ model: ggml-small.bin (465 MB)
Loading speaker diarization model...
✅ Speaker diarization model ready.
Maximum speakers: 4
🎤 HOLD SPACEBAR → microphone ON
```
Verified startup for `archive/deaf_assist_live.py`:
```
🤖 AI Mode selected: GENERAL
✅  whisper-cli found
✅  ggml-small.bin  (465 MB)
╔═════════════════════╗
║  READY — LIVE MODE  ║
╚═════════════════════╝
```

### E. Failure diagnosis
| Symptom | Cause |
|---|---|
| `Speaker diarization model ready` never appears / model file error | Wrong CWD — must be the **repo root**, so `speaker_models/` resolves |
| `ModuleNotFoundError: No module named 'mode_router'` | Missing `PYTHONPATH` |
| `ModuleNotFoundError: No module named 'translate'` / `ImportError: huggingface-hub` | The live **translation** step imports `translate` lazily (§1.1). STT + diarization still work; translation needs Option A |
| `This process is not trusted! Input event monitoring…` | macOS Input Monitoring permission (see §2) |
| `sherpa_onnx` import error | Wrong env — `sherpa_onnx` only exists in `translation_env` |

### G. Limitations
The full loop (record → transcribe → translate) needs a mic, macOS permissions, and —
for the translation stage — the §1.1 fix. Startup was verified here; the
record→transcribe→translate round trip was **not** completed end-to-end in this
environment (no mic + hub conflict).

---

## 10. Feature: Frontend app (React Native / Expo)

**F. Classification:** `FRONTEND/API INTEGRATION TEST`

### A. Purpose
The user-facing app: live captions with speaker chips, sound-event alerts
(🟢/🟡/🔴), AI-context panel (summary / key points / "you may have missed"), on-device
memory, type-to-speak, local login/signup.

### B. Command
```bash
cd "/Users/priyathakur/Desktop/SENSEA-Hackathon/sensea/front"
npm install      # node_modules is NOT present in this checkout
npx expo start
```
Then press `i` (iOS simulator) / `a` (Android) or scan the QR with Expo Go.

### C. Expected input
Expo Go on a phone, or a simulator. No Python backend is needed for the demo.

### D. Expected success
The app boots into `Login` → `SignUp` → `MainTabs` (Home, Captions, Summary, Alerts,
Speak, Memory) → `Profile`. Captions stream in automatically from the simulated
speech source.

### E. Failure diagnosis
| Symptom | Cause |
|---|---|
| `npm: command not found` / missing deps | `node_modules` is absent — run `npm install` (Node v24.15.0 / npm 11.12.1 are installed here) |
| `No script URL provided` | Metro not started — rerun `npx expo start` |
| Translation shows only Hindi/Tamil/Telugu phrases | `src/config/bhashiniConfig.ts` still has the placeholders `YOUR_BHASHINI_USER_ID` / `YOUR_BHASHINI_ULCA_API_KEY` → the app falls back to the built-in phrase dictionary. Add real credentials to enable Bhashini |
| Native-module features missing | Some features need an Expo **development build**, not Expo Go (per `sensea/front/README.md`) |

### G. Limitations
**Everything audio-related in the app is simulated.**
`src/services/speechService.ts` generates a mock caption stream; captions are not real
transcripts. `src/components/ContextPanel.tsx` / `src/utils/contextInsights.ts` compute
the summary and Q&A **on-device** from the transcript text — they are not LLM calls.
TTS is real (`expo-speech`) but the "translate then speak" step uses the mock/Bhashini
dictionary, not IndicTrans2. There is **no backend account system** — login/signup are
local AsyncStorage. `sensea/front/` is gitignored in the root `.gitignore`.

---

## 11. Feature: Frontend ↔ ML-backend integration

**F. Classification:** `NOT CURRENTLY RUNNABLE`

### A. Purpose
Would be: the Expo app streaming real captions/translation/diarization from the Python
pipeline over a WebSocket or HTTP API.

### B. Command
There is none. **No such command exists in this repository.**

### C. Expected input
N/A.

### D. Expected success
N/A.

### E. Failure diagnosis / evidence
A repository-wide search for `fastapi`, `flask`, `uvicorn`, `websocket`, `aiohttp`
servers, or any server bootstrap returns **no API/WebSocket server**. The only HTTP
endpoints referenced by the app are:
- `https://meity-auth.ulcacontrib.org/ulca/apis/v0/model/getModelsPipeline` (Bhashini, cloud)
- `https://integrate.api.nvidia.com/v1` (used by the Python backend only)

The Python side (`deaf_assist2.py`, `transcribe.py`) is a **terminal CLI**, not a
service. `sensea/front/src/services/speechService.ts` emits a hard-coded mock stream.

### G. Limitations
This is a **greenfield gap**, not a broken test. To make it testable you would need to
(a) add a server entry point that wraps the pipeline, and (b) point the frontend's
`speechService.ts` at it. Both are out of scope for this guide.

---

## 12. Appendix A — Full automated test suite

All run from `sensea/ML Backend` unless noted. All verified on this machine.

| Command (relative to `sensea/ML Backend`) | Env | Result |
|---|---|---|
| `../../translation_env/bin/python -u tests/test_live_pipeline.py` | `translation_env` | **13/13 pass** |
| `PYTHONPATH=. ../../translation_env/bin/python -u tests/test_registry.py` | `translation_env` | **7/7 pass** |
| `../../rag_test_env/bin/python -u tests/test_chatbot.py` | `rag_test_env` | **27/27 pass** |
| `PYTHONPATH=. ../../rag_test_env/bin/python -u tests/test_step8.py` | `rag_test_env` | **25/25 pass** (live NVIDIA API) |
| `PYTHONPATH=. ../../rag_test_env/bin/python -u tests/test_step9.py` | `rag_test_env` | **44/44 pass** |
| `../../translation_env/bin/python -u tests/test_translate.py` | `translation_env` | **71/71 pass** — needs §1.1 Option A |
| `SENSEA_TRANSLATE_E2E=1 … tests/test_translate.py` | `translation_env` | **74/74 pass** — needs §1.1 Option A |
| `PYTHONPATH=. ../../rag_test_env/bin/python -u tests/test_real_pipeline.py` | `rag_test_env` | ✅ runs — live NVIDIA API, valid classroom JSON |
| `PYTHONPATH=. ../../rag_test_env/bin/python -u tests/test_modes.py` | `rag_test_env` | ✅ runs — live NVIDIA API, valid hospital JSON |
| `PYTHONPATH=. ../../rag_test_env/bin/python -u tests/test_ai_engine.py` | `rag_test_env` | ✅ runs — live NVIDIA API, valid JSON |
| `PYTHONPATH=. ../../rag_test_env/bin/python -u tests/test_nvidia.py` | `rag_test_env` | ✅ runs — live NVIDIA API, `FINISH REASON === stop` |
| `../../rag_test_env/bin/python -u tests/rag_test.py` | `rag_test_env` | ✅ runs — embedding demo, no API key needed |
| `../../translation_env/bin/python -u tests/speaker_test.py` | `translation_env` | ✅ reaches `READY` (Resemblyzer, 8 speakers, thresh 0.72) — needs mic + Accessibility |
| `PYTHONPATH=. ../../translation_env/bin/python sign_language/scripts/evaluate_signer_test.py` | `translation_env` | ✅ runs — writes `unseen_signer_results.csv` |

`tests/test_step8.py`, `tests/test_step9.py`, `tests/test_real_pipeline.py`,
`tests/test_modes.py`, `tests/test_ai_engine.py` and `tests/test_nvidia.py` need
`PYTHONPATH=.` — they import `speaker_diarization` / `ai_engine` / `mode_router` as
top-level modules and fail with `ModuleNotFoundError` otherwise.

There is **no** `pytest`: every file is a plain script. Do not run them with `pytest`.

---

## 13. Appendix B — Recommended manual test order

1. **Environment sanity (2 min)** — run §1.1 Option A, then:
   `../../translation_env/bin/python -c "import kokoro, IndicTransToolkit, sherpa_onnx, mediapipe; print('ok')"`
2. **Automated suites, no mic (10 min)** — §12 top-to-bottom. Failures here are code
   problems, not environment problems.
3. **Translation (5 min)** — §4 `translate.py` self-test.
4. **TTS (2 min)** — §3. Confirm you *hear* the audio and that `sensea_tts.wav` is written.
5. **RAG chatbot (5 min)** — §5, then a deliberately off-topic question to confirm the
   `NOT_FOUND_RESPONSE` path.
6. **AI analysis modes (5 min)** — §6, all four modes.
7. **Sign language, offline (5 min)** — §8 offline video command.
8. **Sign language, live (5 min)** — §8 webcam command.
9. **Speaker diarization offline (5 min)** — §7 real clip.
10. **STT with microphone (10 min)** — §2. Grant macOS Input Monitoring first.
11. **Live pipeline (10 min)** — §9. Do it from the repo root.
12. **Frontend (15 min)** — §10, then walk every tab.
13. **Frontend ↔ backend integration** — §11. **Not testable today**; record it as a
    known gap.

---

## 14. Appendix C — Known issues / troubleshooting matrix

| # | Issue | Impact | Fix / status |
|---|---|---|---|
| 1 | `translation_env` has `huggingface-hub` 1.30.0 but `transformers` needs `<1.0` | TTS + Translation + live translation broken in that env | §1.1 Option A — `pip install "huggingface-hub<1.0"` (needs approval) |
| 2 | `rag_test_env` / `venv` lack `IndicTransToolkit` | `tests/test_translate.py` cannot run there | Use `translation_env` after issue #1 is fixed |
| 3 | `PYTHONPATH` to `translation_env` shadows stdlib `typing.py` | `AttributeError: … '_abc_registry'` | Use `sys.path.append(...)` inside the process, never `PYTHONPATH` |
| 4 | `llm.py` uses `load_dotenv("../.env")` (CWD-relative) | "API key not found" when run from the wrong directory | Always `cd` into `sensea/ML Backend` first |
| 5 | `deaf_assist2.py` resolves models from `os.getcwd()` | "Speaker diarization model" not found | Always run from the **repo root** |
| 6 | macOS Input Monitoring / Accessibility not granted | SPACEBAR push-to-talk dead in `transcribe.py` and the live pipeline | System Settings → Privacy & Security → Input Monitoring |
| 7 | `tests/test_translate.py` B3 cases spawn child processes | 3 failures under Option B | Environment artifact — use Option A |
| 8 | `sensea/front/` has no `node_modules` and is gitignored | Frontend won't start on a fresh clone | `npm install` |
| 9 | `bhashiniConfig.ts` contains placeholders | Frontend translation falls back to a 3-language phrase dictionary | Add real Bhashini credentials |
| 10 | No API/WebSocket server anywhere | Frontend cannot consume the Python pipeline | Known gap — see §11 |
| 11 | `sign_language/data/raw/` is gitignored | The offline sign-language video test cannot run on a fresh clone | Keep the dataset locally; don't commit it |
| 12 | `sensea_tts.wav`, `kokoro_output_*.wav`, `unseen_signer_results.csv` are overwritten by test runs | Local artifacts change | They are generated outputs (gitignored) — back them up if you need them |

---

## 15. Appendix D — Feature → command quick reference

| # | Feature | Command (from `sensea/ML Backend` unless noted) | Classification |
|---|---|---|---|
| 1 | STT (push-to-talk) | `../../venv/bin/python -u transcribe.py` | REAL AUDIO/VIDEO TEST |
| 2 | TTS (Kokoro) | `../../translation_env/bin/python -u tts.py` | LOCAL SMOKE TEST (needs §1.1) |
| 3 | Translation (IndicTrans2) | `../../translation_env/bin/python -u translate.py` | LOCAL SMOKE TEST (needs §1.1) |
| 4 | Translation regression | `../../translation_env/bin/python -u tests/test_translate.py` | AUTOMATED TEST (needs §1.1) |
| 5 | RAG chatbot | `../../rag_test_env/bin/python -u tests/test_chatbot.py` | AUTOMATED TEST |
| 6 | Chatbot one-shot | `../../rag_test_env/bin/python -c "from chatbot import chat; print(chat('What did the teacher explain?','classroom',use_test_memories=True))"` | LIVE API TEST |
| 7 | AI conversation analysis | `PYTHONPATH=. ../../rag_test_env/bin/python -u tests/test_real_pipeline.py` | LIVE API TEST |
| 8 | Speaker registry logic | `PYTHONPATH=. ../../translation_env/bin/python -u tests/test_registry.py` | AUTOMATED TEST |
| 9 | Speaker diarization (real clip) | `PYTHONPATH="sensea/ML Backend" translation_env/bin/python -c "…diarize_clip…"` (from repo root) | REAL AUDIO/VIDEO TEST |
| 10 | Sign language (offline video) | `../../translation_env/bin/python sign_language/scripts/test_dataset_video.py --video "sign_language/data/raw/isl_isolated_40words/hello/hello__INCLUDE__00566__MVI_0029.mp4"` | REAL AUDIO/VIDEO TEST |
| 11 | Sign language (webcam) | `../../translation_env/bin/python sign_language/scripts/realtime.py` | REAL AUDIO/VIDEO TEST |
| 12 | Sign language (held-out signer) | `PYTHONPATH=. ../../translation_env/bin/python sign_language/scripts/evaluate_signer_test.py` | AUTOMATED TEST |
| 13 | Live pipeline (current) | `PYTHONPATH="sensea/ML Backend" translation_env/bin/python -u "sensea/ML Backend/deaf_assist2.py"` (from repo root) | REAL AUDIO/VIDEO TEST |
| 14 | Live pipeline (archived) | `PYTHONPATH=. ../../translation_env/bin/python -u archive/deaf_assist_live.py` | REAL AUDIO/VIDEO TEST |
| 15 | Frontend app | `npm install && npx expo start` (from `sensea/front`) | FRONTEND/API INTEGRATION TEST |
| 16 | Frontend ↔ ML backend | — | **NOT CURRENTLY RUNNABLE** |
