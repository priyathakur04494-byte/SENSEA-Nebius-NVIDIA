# ============================================================
# language_codes.py
# ============================================================
#
# Single source of truth for the IndicTrans2 language codes SENSEA
# supports, plus the validation that keeps malformed codes out of the
# native IndicTransToolkit processor.
#
#
# WHY THE VALIDATION EXISTS
# ============================================================
#
# IndicTransToolkit/processor.pyx:380 reads the script part of a
# language code like this:
#
#     cdef str script_part = src_lang.split("_")[1]
#
# Cython compiles that into an UNCHECKED list index:
#
#     __pyx_t_2 = __Pyx_PyList_GET_ITEM(__pyx_t_1, 1);
#     __Pyx_INCREF(__pyx_t_2);
#
# `split("_")` returns ONE element for a code without an underscore
# ("hin", "mar", "klingon"), so index 1 reads past the end of the
# list's item array. The garbage pointer is then INCREF'd and the
# process dies with SIGSEGV (exit status -11) inside
# processor.cpython-*.so.
#
# There is no Python exception to catch: `except Exception` never
# runs, the interpreter is gone. The crash needs neither MPS/Metal,
# nor torch inference, nor the tokenizer - `preprocess_batch()` alone
# reproduces it, before a single model is loaded.
#
# So every language code must be validated in Python BEFORE it reaches
# the toolkit. That is what `validate_language_code()` does.
#
#
# NOTE
# ============================================================
#
# transcribe.py / deaf_assist2.py / archive/*.py keep their own
# TARGET_LANGUAGES literals (they are scripts, not importable
# modules). tests/test_translate.py asserts all of them stay in sync
# with the map below, so the copies cannot silently drift.

import re


# ============================================================
# SUPPORTED LANGUAGES (22 Indic + English)
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


SUPPORTED_LANGUAGE_CODES = frozenset(TARGET_LANGUAGES)


# IndicTrans2 codes look like "<3 letter language>_<Script>", e.g.
# "hin_Deva", "eng_Latn", "sat_Olck".
LANGUAGE_CODE_RE = re.compile(r"^[a-z]{3}_[A-Za-z]{4}$")


class UnsupportedLanguageError(ValueError):
    """Raised when a language code is malformed or not supported.

    Subclasses ValueError so existing `except ValueError` /
    `except Exception` handlers around the translation call keep
    working unchanged.
    """


def validate_language_code(code, role="language"):
    """Return the validated language code, or raise.

    Parameters
    ----------
    code : str
        The language code to check, e.g. "hin_Deva".
    role : str
        Human readable name of the slot being validated ("source
        language", "target language") - used in the error message.

    Raises
    ------
    UnsupportedLanguageError
        If the code is not a string, is empty, is not in IndicTrans2
        "<lang>_<Script>" form, or is a well formed code that this app
        does not support.
    """
    if not isinstance(code, str):
        raise UnsupportedLanguageError(
            f"Invalid {role} code {code!r}: expected a string, "
            f"got {type(code).__name__}"
        )

    normalised = code.strip()

    if not normalised:
        raise UnsupportedLanguageError(
            f"Invalid {role} code: the code is empty"
        )

    if not LANGUAGE_CODE_RE.match(normalised):
        raise UnsupportedLanguageError(
            f"Invalid {role} code {normalised!r}: expected the "
            f"IndicTrans2 form '<language>_<Script>' "
            f"(for example 'hin_Deva' or 'eng_Latn')"
        )

    if normalised not in SUPPORTED_LANGUAGE_CODES:
        raise UnsupportedLanguageError(
            f"Unsupported {role} code {normalised!r}. Supported codes: "
            + ", ".join(sorted(SUPPORTED_LANGUAGE_CODES))
        )

    return normalised


def is_supported_language_code(code) -> bool:
    """True if `code` is a valid, supported language code."""
    try:
        validate_language_code(code)
    except UnsupportedLanguageError:
        return False
    return True
