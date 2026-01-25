"""Reusable text normalization pipeline for medical narratives.

If unavailable, tts.py falls back to its internal cleaning.
"""

import re
from typing import List

UNIT_PATTERNS = [
    (r"(\d+)\s*(mL|ml)", r"\1 milliliters"),
    (r"(\d+)\s*(mmHg)", r"\1 millimeters of mercury"),
    (r"(\d+)\s*μg/L", r"\1 micrograms per liter"),
]


def _replace_blood_pressure(match: re.Match) -> str:
    systolic = match.group(1)
    diastolic = match.group(2)
    return (
        f"a systolic blood pressure of {systolic} millimeters of mercury "
        f"and a diastolic blood pressure of {diastolic} millimeters of mercury"
    )


def _normalize_units(text: str) -> str:
    text = re.sub(r"(\d+)/(\d+)\s*mmHg", _replace_blood_pressure, text)
    for pattern, repl in UNIT_PATTERNS:
        text = re.sub(pattern, repl, text)
    return text


def _remove_parentheses(text: str) -> str:
    return re.sub(r"\([^)]*\)", "", text)


def _strip_noise_chars(text: str) -> str:
    return text.replace("*", "").replace("-", "").replace("#", "")


def _expand_acronyms(text: str, expand_acronyms: bool) -> str:
    if not expand_acronyms:
        return text
    # Correct typo and expand common acronym
    text = text.replace("ICH", "intracerebral hemorrhage")
    return text


def _normalize_whitespace(text: str) -> str:
    text = re.sub(r" +", " ", text).strip()
    text = re.sub(r"\n+", "\n", text)
    return text


def normalize_text(text: str, expand_acronyms: bool = True) -> str:
    """Normalization pipeline combining multiple steps.

    - Strips common noise characters
    - Removes parentheses content
    - Normalizes medical units
    - Optionally expands acronyms (e.g., ICH)
    - Normalizes whitespace
    """
    t = _strip_noise_chars(text)
    t = t.replace("Dear [Patient’s Name],", "").replace("Dear [Patient's Name],", "")
    t = t.replace("AHA/ASA", "").replace("AHA", "")
    t = _remove_parentheses(t)
    t = _normalize_units(t)
    t = _expand_acronyms(t, expand_acronyms)
    t = _normalize_whitespace(t)
    return t
