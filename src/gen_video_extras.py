"""Extras for gen_vedio video synthesis.

"""

import json
from pathlib import Path
from typing import Optional, Dict, Any


def derive_audio_url(filename: str, base_url: Optional[str]) -> str:
    if not base_url:
        return filename
    return f"{base_url.rstrip('/')}/{filename}"


def write_metadata_sidecar(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def should_skip_task(meta_path: Path, resume: bool) -> bool:
    return resume and meta_path.exists()
