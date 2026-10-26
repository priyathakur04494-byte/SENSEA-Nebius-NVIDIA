# test_translate.py
# Regression tests for the translation layer.
#
#   Fix 1  — malformed language codes used to SIGSEGV the native
#            IndicTransToolkit processor (processor.pyx:380 reads
#            `src_lang.split("_")[1]` with an unchecked list index).
#            Codes are now validated in Python first.
#   Fix 8  — long inputs used to be silently truncated by the
#            tokenizer and capped at 256 output tokens. They are now
#            chunked.
#
# Run: ../../translation_env/bin/python tests/test_translate.py
#
# The end-to-end section (real model, needs the HF cache) only runs
# when SENSEA_TRANSLATE_E2E=1 is set:
#   SENSEA_TRANSLATE_E2E=1 ../../translation_env/bin/python tests/test_translate.py

import ast
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)

if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

import language_codes
from language_codes import (
    SUPPORTED_LANGUAGE_CODES,
    TARGET_LANGUAGES,
    UnsupportedLanguageError,
    validate_language_code,
)

passed = 0
failed = 0
results_log = []


def test(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        results_log.append(f"[PASS] {name}")
    else:
        failed += 1
        results_log.append(f"[FAIL] {name}")
        if detail:
            results_log.append(f"        → {detail}")


def test_raises(name, fn, exc_type=UnsupportedLanguageError):
    try:
        fn()
    except exc_type as e:
        test(name, True)
        return
    except Exception as e:
        test(name, False, f"raised {type(e).__name__}: {e}")
        return
    test(name, False, "no exception raised")


print("Running translation regression tests...\n")

# ==================================================
# SECTION A: LANGUAGE CODE VALIDATION
# ==================================================

# A1: every supported code passes validation
for code in sorted(SUPPORTED_LANGUAGE_CODES):
    try:
        validate_language_code(code)
        ok = True
        detail = ""
    except Exception as e:
        ok = False
        detail = str(e)
    test(f"A1: supported code accepted ({code})", ok, detail)

# A2: malformed codes are rejected with a clear error.
# "hin" / "mar" / "klingon" are the codes that used to crash the
# interpreter with SIGSEGV.
MALFORMED = [
    "hin",
    "mar",
    "klingon",
    "",
    "   ",
    "HIN_DEVA",
    "hinDeva",
    "hin_Deva_Extra",
    "hin_",
    "_Deva",
    "hin_deva_extra",
    "en_Latn",
    "zzzz_Zzzz",
]

for code in MALFORMED:
    test_raises(
        f"A2: malformed code rejected ({code!r})",
        lambda code=code: validate_language_code(code),
    )

# A3: non-string codes are rejected instead of crashing the toolkit
for value in [None, 123, 1.5, ["hin_Deva"], {"lang": "hin_Deva"}]:
    test_raises(
        f"A3: non-string code rejected ({value!r})",
        lambda value=value: validate_language_code(value),
    )

# A4: the error message names the offending code
try:
    validate_language_code("klingon")
except UnsupportedLanguageError as e:
    test("A4: error message names the bad code", "klingon" in str(e), str(e))
else:
    test("A4: error message names the bad code", False, "no exception raised")

# A5: UnsupportedLanguageError is a ValueError subclass, so existing
# `except ValueError` handlers around the translation call still work.
test(
    "A5: UnsupportedLanguageError is a ValueError",
    issubclass(UnsupportedLanguageError, ValueError),
)

# ==================================================
# SECTION B: translate_text VALIDATES BEFORE THE NATIVE CALL
# ==================================================

import translate

for code in ["hin", "mar", "klingon", "xyz_Xyzw"]:
    test_raises(
        f"B1: translate_text rejects {code!r} (no SIGSEGV)",
        lambda code=code: translate.translate_text("hello", code, "eng_Latn"),
    )
    test_raises(
        f"B2: translate_text rejects {code!r} as target",
        lambda code=code: translate.translate_text("hello", "eng_Latn", code),
    )

# B3: the same codes raise in a CHILD process — the interpreter must
# survive (before the fix the child died with signal 11).
CHILD = """
import sys
sys.path.insert(0, %r)
from translate import translate_text
try:
    translate_text("hello", "hin", "eng_Latn")
    print("NO_ERROR")
except Exception as e:
    print(type(e).__name__)
""" % BACKEND

for code in ["hin", "mar", "klingon"]:
    proc = subprocess.run(
        [sys.executable, "-c", CHILD],
        capture_output=True,
        text=True,
        timeout=600,
    )
    test(
        f"B3: child process survives {code!r} (exit {proc.returncode})",
        proc.returncode == 0 and "UnsupportedLanguageError" in proc.stdout,
        f"stdout={proc.stdout.strip()!r} stderr={proc.stderr.strip()[-300:]!r}",
    )

# B4: unsupported but well-formed codes are rejected too
test_raises(
    "B4: unknown well-formed code rejected",
    lambda: translate.translate_text("hello", "xyz_Xyzw", "eng_Latn"),
)

# B5: non-string input is rejected with a clear TypeError
test_raises(
    "B5: non-string text rejected",
    lambda: translate.translate_text(123, "eng_Latn", "hin_Deva"),
    TypeError,
)

# B6: empty text returns an empty string (unchanged behaviour)
test(
    "B6: empty text returns empty string",
    translate.translate_text("", "eng_Latn", "hin_Deva") == "",
)

test(
    "B7: whitespace-only text returns empty string",
    translate.translate_text("   \n ", "eng_Latn", "hin_Deva") == "",
)

# B8: same source and target returns the text unchanged
test(
    "B8: same language returns the input unchanged",
    translate.translate_text("Hello there", "eng_Latn", "eng_Latn") == "Hello there",
)

# ==================================================
# SECTION C: THE COPIES OF TARGET_LANGUAGES STAY IN SYNC
# ==================================================

COPIES = [
    os.path.join(BACKEND, "transcribe.py"),
    os.path.join(BACKEND, "deaf_assist2.py"),
    os.path.join(BACKEND, "archive", "deaf_assist.py"),
    os.path.join(BACKEND, "archive", "deaf_assist_live.py"),
]


def read_target_languages(path):
    """Read the TARGET_LANGUAGES literal without importing the module."""
    with open(path, "r", encoding="utf-8") as handle:
        tree = ast.parse(handle.read())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "TARGET_LANGUAGES":
                    return ast.literal_eval(node.value)
    return None


for path in COPIES:
    name = os.path.relpath(path, BACKEND)
    if not os.path.exists(path):
        test(f"C1: {name} exists", False, "file not found")
        continue
    found = read_target_languages(path)
    test(
        f"C1: {name} TARGET_LANGUAGES matches language_codes.py",
        found == TARGET_LANGUAGES,
        f"found={found!r}",
    )

test(
    "C2: 22 Indic languages + English are supported",
    len(SUPPORTED_LANGUAGE_CODES) == 23
    and "eng_Latn" in SUPPORTED_LANGUAGE_CODES
    and "hin_Deva" in SUPPORTED_LANGUAGE_CODES
    and "urd_Arab" in SUPPORTED_LANGUAGE_CODES,
    f"count={len(SUPPORTED_LANGUAGE_CODES)}",
)

# ==================================================
# SECTION D: LONG INPUT IS CHUNKED, NOT TRUNCATED
# ==================================================

LONG_TEXT = (
    "The professor explained convolutional neural networks in detail. "
    "Students practised gradient descent with handwritten examples. "
    "The assignment on backpropagation is due next week. "
    "Everyone should review the lecture notes before the seminar. "
    "The library closes at six in the evening on weekdays. "
    "Guest lectures are scheduled every second Friday afternoon. "
    "The research group meets to discuss the new dataset. "
)


class FakeTokenizer:
    """Minimal stand-in: one token per whitespace separated word.

    The encoder limit is deliberately small (64) so the chunking logic
    can be exercised without loading a real model.
    """

    model_max_length = 64

    def encode(self, text, add_special_tokens=False):
        return text.split()


fake = FakeTokenizer()

# D1: short text is passed through untouched (one chunk, identical text)
chunks = translate._split_into_chunks("Hello world. This is short.", fake)
test(
    "D1: short text stays a single untouched chunk",
    chunks == ["Hello world. This is short."],
    f"chunks={chunks!r}",
)

# D2: long text is split into several chunks
long_chunks = translate._split_into_chunks(LONG_TEXT, fake)
test(
    "D2: long text is split into multiple chunks",
    len(long_chunks) > 1,
    f"chunks={long_chunks!r}",
)

# D3: no content is lost by the chunking
rejoined = " ".join(long_chunks)
test(
    "D3: chunking loses no words",
    rejoined.split() == LONG_TEXT.split(),
    f"rejoined={rejoined!r}",
)

# D4: every chunk fits the encoder budget
budget = min(translate.CHUNK_TOKEN_BUDGET, translate._encoder_limit(fake) - 16)
test(
    "D4: every chunk fits the encoder budget",
    all(len(c.split()) <= budget for c in long_chunks),
    f"budget={budget} chunks={[len(c.split()) for c in long_chunks]}",
)

# D5: a single sentence longer than the budget is split on words
huge = " ".join(f"word{i}" for i in range(1200))
huge_chunks = translate._split_into_chunks(huge, fake)
test(
    "D5: one oversized sentence is split on words",
    len(huge_chunks) > 1
    and all(len(c.split()) <= budget for c in huge_chunks)
    and " ".join(huge_chunks).split() == huge.split(),
    f"n={len(huge_chunks)}",
)

# D6: the output budget scales with the input instead of a flat 256
test(
    "D6: output budget scales with input length",
    translate.MIN_OUTPUT_TOKENS == 256
    and translate.OUTPUT_TOKEN_MULTIPLIER >= 2
    and translate.MAX_OUTPUT_TOKENS > translate.MIN_OUTPUT_TOKENS,
)

# D7: the encoder budget matches IndicTrans2's 256 source positions
test(
    "D7: encoder limit matches the model's 256 source positions",
    translate.ENCODER_TOKEN_LIMIT == 256
    and translate.CHUNK_TOKEN_BUDGET < translate.ENCODER_TOKEN_LIMIT,
    f"limit={translate.ENCODER_TOKEN_LIMIT} budget={translate.CHUNK_TOKEN_BUDGET}",
)

# ==================================================
# SECTION E: END-TO-END TRANSLATION (opt-in)
# ==================================================

if os.environ.get("SENSEA_TRANSLATE_E2E") == "1":
    try:
        out = translate.translate_text("Hello, how are you today?", "eng_Latn", "hin_Deva")
        test(
            "E1: en -> hi translation returns Devanagari",
            out and any("\u0900" <= ch <= "\u097f" for ch in out),
            f"out={out!r}",
        )

        # A long Hindi paragraph whose LAST sentence carries a marker
        # word. Before the fix the output was capped at 256 tokens, so
        # the marker sentence never made it into the translation.
        long_hi = (
            "आज प्रोफेसर ने कॉन्वोल्यूशनल न्यूरल नेटवर्क के बारे में विस्तार से समझाया। "
            "छात्रों ने हाथ से लिखे उदाहरणों के साथ ग्रेडिएंट डिसेंट का अभ्यास किया। "
            "बैकप्रोपेगेशन पर असाइनमेंट अगले सप्ताह जमा करनी है। "
            "हर किसी को सेमिनार से पहले लेक्चर नोट्स दोबारा पढ़ने चाहिए। "
            "पुस्तकालय सप्ताह के दिनों में शाम छह बजे बंद हो जाता है। "
            "अतिथि व्याख्यान हर दूसरे शुक्रवार को दोपहर में होते हैं। "
            "अनुसंधान समूह नए डेटा सेट पर चर्चा करने के लिए मिलता है। "
            "आखिरी वाक्य में एक बिल्ली का जिक्र है।"
        )
        long_out = translate.translate_text(long_hi, "hin_Deva", "eng_Latn")
        test(
            "E2: long hi -> en translation is complete (last sentence kept)",
            "cat" in (long_out or "").lower(),
            f"out={long_out!r}",
        )

        long_en = translate.translate_text(
            " ".join(["The seminar about neural networks starts at four."] * 12),
            "eng_Latn",
            "hin_Deva",
        )
        test(
            "E3: long en -> hi translation keeps every sentence",
            (long_en or "").count("।") >= 11,
            f"danda_count={(long_en or '').count('।')} out={(long_en or '')[-120:]!r}",
        )
    except Exception as e:
        test("E1/E2/E3: end-to-end translation", False, f"{type(e).__name__}: {e}")
else:
    results_log.append("[SKIP] Section E (end-to-end) — set SENSEA_TRANSLATE_E2E=1 to run")

# ==================================================
# RESULTS SUMMARY
# ==================================================

print("\n" + "=" * 60)
print("TRANSLATION TEST RESULTS")
print("=" * 60)
for line in results_log:
    print(line)
print(f"\n{'=' * 60}")
print(f"Total: {passed + failed} | Passed: {passed} | Failed: {failed}")
print("=" * 60)

if failed == 0:
    print("\nAll tests passed!")
else:
    print(f"\n{failed} test(s) failed.")
    sys.exit(1)
