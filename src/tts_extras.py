"""Optional utilities for batch TTS workflows.
"""

import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, Any


def _sanitize_filename(name: str) -> str:
    # Normalize unicode and strip problematic characters
    name = unicodedata.normalize("NFKD", name)
    name = re.sub(r"[\\/:*?\"<>|]", "_", name)
    name = re.sub(r"\s+", "_", name).strip(" _")
    # Keep it reasonably short
    if len(name) > 64:
        name = name[:64]
    return name or "unnamed"


def derive_filename(index: int, item: Dict[str, Any], field: str) -> str:
    """Derive an output filename from a selected item field.

    Falls back to indexed naming if field is missing or invalid.
    """
    value = item.get(field)
    if isinstance(value, str) and value.strip():
        return _sanitize_filename(value.strip())
    return f"{index:03d}"


def should_skip(out_wav: Path, resume: bool) -> bool:
    return resume and out_wav.exists()


def write_metadata_sidecar(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def strip_newlines(text: str) -> str:
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
