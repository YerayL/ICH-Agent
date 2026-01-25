"""Text-to-Speech batch synthesizer.

This module provides a CLI and utilities to synthesize speech for a batch of
patient result items loaded from a JSON file. It focuses on professionalized
structure, safer error handling, and optional features while preserving the
original logic and defaults.

Key features:
- Deterministic medical unit conversions and blood pressure normalization
- Text cleaning pipeline with optional acronym expansion and newline handling
- Configurable CPU/GPU selection (unchanged default behavior)
- Optional extras: resume, dry-run, metadata sidecar, filename derivation

"""

import argparse
import json
import re
import time
import logging
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Dict, Any

from TTS.api import TTS

# Optional, plug-in utilities. These modules are designed to be self-contained
# and can be removed without breaking core logic if not used.
try:
    from .text_normalizer import normalize_text
except Exception:
    # Fallback: use internal cleaning if optional module isn't available
    normalize_text = None  # type: ignore

try:
    from .tts_extras import (
        derive_filename,
        should_skip,
        write_metadata_sidecar,
        strip_newlines as extras_strip_newlines,
    )
except Exception:
    # Graceful fallback if extras are not present
    derive_filename = None  # type: ignore
    should_skip = None  # type: ignore
    write_metadata_sidecar = None  # type: ignore
    extras_strip_newlines = None  # type: ignore


try:
    from .tts_optional_features import create_hooks
except Exception:
    create_hooks = None  # type: ignore


# Keep conversions explicit to ensure deterministic replacements.
UNIT_PATTERNS = [
    (r"(\d+)\s*(mL|ml)", r"\1 milliliters"),
    (r"(\d+)\s*(mmHg)", r"\1 millimeters of mercury"),
    (r"(\d+)\s*μg/L", r"\1 micrograms per liter"),
]


@dataclass
class TTSConfig:
    model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    input_json: Path = Path("/home/pc/lyy/ICH-agent/vedio_data/patient_results.json")
    output_dir: Path = Path("/home/pc/lyy/ICH-agent/agent/gen_audio_data_zxj")
    speaker_wav: Path = Path("/home/pc/lyy/ICH-agent/zxj.wav")
    language: str = "en"
    gpu: bool = True
    limit: Optional[int] = None  # Optional cap for quick runs.
    # Optional extras (all default to disabled or sane values)
    dry_run: bool = False
    resume: bool = False
    filename_field: Optional[str] = None
    write_metadata: bool = False
    strip_newlines: bool = False
    expand_acronyms: bool = True
    retries: int = 0
    sleep_seconds: float = 0.0
    hooks_log: Optional[Path] = None
    run_summary: Optional[Path] = None
    max_failures: Optional[int] = None


def convert_medical_units(text: str) -> str:
    text = re.sub(r"(\d+)/(\d+)\s*mmHg", _replace_blood_pressure, text)
    for pattern, repl in UNIT_PATTERNS:
        text = re.sub(pattern, repl, text)
    return text


def _replace_blood_pressure(match: re.Match) -> str:
    systolic = match.group(1)
    diastolic = match.group(2)
    return (
        f"a systolic blood pressure of {systolic} millimeters of mercury "
        f"and a diastolic blood pressure of {diastolic} millimeters of mercury"
    )


def clean_text(text: str) -> str:
    """Original cleaning pipeline preserved for backward compatibility.

    Applies basic punctuation removal, parenthetical stripping, medical unit
    conversions, acronym expansion for ICH, and whitespace normalization.
    """
    cleaned = text.replace("*", "").replace("-", "").replace("#", "")
    cleaned = cleaned.replace("Dear [Patient’s Name],", "")
    cleaned = cleaned.replace("Dear [Patient's Name],", "")
    cleaned = cleaned.replace("AHA/ASA", "")
    cleaned = cleaned.replace("AHA", "")
    cleaned = re.sub(r"\([^)]*\)", "", cleaned)
    cleaned = convert_medical_units(cleaned)
    # Fix typo: "hemorrahge" -> "hemorrhage"
    cleaned = cleaned.replace("ICH", "intracerebral hemorrhage")
    cleaned = re.sub(r" +", " ", cleaned).strip()
    cleaned = re.sub(r"\n+", "\n", cleaned)
    return cleaned


def clean_text_with_options(
    text: str,
    expand_acronyms: bool = True,
    strip_newlines: bool = False,
) -> str:
    """Enhanced cleaning with optional behaviors.

    - Uses external normalizer if available for extensibility.
    - Provides switches for acronym expansion and newline stripping.
    - Falls back to the original cleaning pipeline to preserve behavior.
    """
    base = clean_text(text) if normalize_text is None else normalize_text(
        text,
        expand_acronyms=expand_acronyms,
    )
    if strip_newlines:
        if extras_strip_newlines is not None:
            base = extras_strip_newlines(base)
        else:
            base = re.sub(r"\s*\n\s*", " ", base)
            base = re.sub(r"\s+", " ", base).strip()
    return base


def load_patient_results(path: Path, limit: Optional[int]) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
        if limit is not None:
            data = data[:limit]
        return data


def synthesize_batch(config: TTSConfig) -> None:
    """Run batch synthesis with optional extras while preserving defaults."""
    _configure_logging()
    logger = logging.getLogger("tts")

    config.output_dir.mkdir(parents=True, exist_ok=True)

    hooks = create_hooks(config.hooks_log, config.run_summary, config.max_failures) if create_hooks is not None else None

    # Validate speaker wav existence early for clearer error reporting
    if not config.speaker_wav.exists():
        logger.error("Speaker WAV not found: %s", config.speaker_wav)
        raise FileNotFoundError(f"Speaker WAV not found: {config.speaker_wav}")

    try:
        tts = TTS(config.model_name, gpu=config.gpu)
    except Exception as e:
        logger.exception("Failed to initialize TTS model '%s' (gpu=%s)", config.model_name, config.gpu)
        raise e

    data = load_patient_results(config.input_json, config.limit)
    total_items = len(data) if hasattr(data, "__len__") else None
    successes = 0
    failures = 0

    if hooks is not None:
        hooks.on_run_start(total_items, asdict(config))
    start_time = time.time()

    for i, item in enumerate(data, start=1):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info("Processing item %s at %s", i, timestamp)

        item_start = time.time()

        # Validate item content
        content = item.get("content")
        if not isinstance(content, str) or not content.strip():
            logger.warning("Item %s missing or empty 'content'; skipping.", i)
            failures += 1
            if hooks is not None:
                hooks.on_item_failure(i, "missing_or_empty_content", 0)
                if hooks.should_abort():
                    break
            continue

        cleaned = clean_text_with_options(
            content,
            expand_acronyms=config.expand_acronyms,
            strip_newlines=config.strip_newlines,
        )
        item["content"] = cleaned

        # Filename derivation (optional)
        if derive_filename is not None and config.filename_field:
            filename = derive_filename(i, item, config.filename_field)
        else:
            filename = f"{i:03d}"

        out_wav = config.output_dir / f"{filename}.wav"

        # Resume behavior: skip existing files
        if should_skip is not None:
            if should_skip(out_wav, resume=config.resume):
                logger.info("Skipping existing file due to resume: %s", out_wav)
                continue
        else:
            if config.resume and out_wav.exists():
                logger.info("Skipping existing file due to resume: %s", out_wav)
                continue

        if config.dry_run:
            logger.info("Dry-run: would synthesize to %s", out_wav)
            if hooks is not None:
                hooks.on_item_success(i, out_wav.name, 0.0, len(item["content"]))
            successes += 1
        else:
            # Retry loop for occasional synthesis failures
            attempts = 0
            last_error: Optional[Exception] = None
            while attempts <= config.retries:
                try:
                    tts.tts_to_file(
                        text=item["content"],
                        file_path=str(out_wav),
                        speaker_wav=str(config.speaker_wav),
                        language=config.language,
                    )
                    last_error = None
                    break
                except Exception as e:
                    last_error = e
                    attempts += 1
                    logger.warning("Synthesis failed for %s (attempt %s/%s): %s", out_wav, attempts, config.retries, e)
                    if attempts <= config.retries:
                        time.sleep(min(2.0 * attempts, 5.0))
            if last_error is not None:
                failures += 1
                if hooks is not None:
                    hooks.on_item_failure(i, str(last_error), attempts)
                    if hooks.should_abort():
                        break
                logger.error("Giving up on %s after %s attempts: %s", out_wav, attempts, last_error)
                continue
            if hooks is not None:
                hooks.on_item_success(i, out_wav.name, time.time() - item_start, len(item["content"]))
            successes += 1

        if config.dry_run and hooks is None:
            # Keep count parity with non-hook path
            successes += 1

        # Optional metadata sidecar
        if config.write_metadata:
            meta: Dict[str, Any] = {
                "index": i,
                "filename": os.path.basename(out_wav),
                "timestamp": timestamp,
                "model_name": config.model_name,
                "language": config.language,
                "speaker_wav": str(config.speaker_wav),
                "original_content": content,
                "cleaned_content": cleaned,
            }
            if write_metadata_sidecar is not None:
                write_metadata_sidecar(config.output_dir / f"{filename}.json", meta)
            else:
                _write_json(config.output_dir / f"{filename}.json", meta)

        # Optional pacing between items
        if config.sleep_seconds > 0:
            time.sleep(config.sleep_seconds)

        if hooks is not None and hooks.should_abort():
            logger.warning("Aborting run after reaching max_failures=%s", config.max_failures)
            break

    run_time = time.time() - start_time
    if hooks is not None:
        hooks.finalize(successes, failures, run_time)
    # Keep a concise terminal output of total runtime for continuity
    print(f"{run_time:.6f} 秒")


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _configure_logging() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s - %(message)s",
        )


def parse_args() -> TTSConfig:
    parser = argparse.ArgumentParser(description="Batch TTS synthesis for patient results")
    parser.add_argument("--model-name", default="tts_models/multilingual/multi-dataset/xtts_v2", help="TTS model name")
    parser.add_argument("--input-json", type=Path, default=Path("/home/pc/lyy/ICH-agent/vedio_data/patient_results.json"), help="Input JSON file path")
    parser.add_argument("--output-dir", type=Path, default=Path("/home/pc/lyy/ICH-agent/agent/gen_audio_data_zxj"), help="Directory to save wav files")
    parser.add_argument("--speaker-wav", type=Path, default=Path("/home/pc/lyy/ICH-agent/zxj.wav"), help="Reference speaker wav path")
    parser.add_argument("--language", default="en", help="Language code")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of items to synthesize")
    # Optional extras
    parser.add_argument("--dry-run", action="store_true", help="Do not synthesize audio; only process and log")
    parser.add_argument("--resume", action="store_true", help="Skip items whose output files already exist")
    parser.add_argument("--filename-field", default=None, help="Field name in JSON item used to derive output filename")
    parser.add_argument("--write-metadata", action="store_true", help="Write a JSON sidecar with metadata for each item")
    parser.add_argument("--strip-newlines", action="store_true", help="Normalize newlines to spaces in cleaned text")
    parser.add_argument("--no-acronym", action="store_true", help="Disable acronym expansion (e.g., ICH -> intracerebral hemorrhage)")
    parser.add_argument("--retries", type=int, default=0, help="Number of retries per item on synthesis failure")
    parser.add_argument("--sleep-seconds", type=float, default=0.0, help="Seconds to sleep between items")
    parser.add_argument("--hooks-log", type=Path, default=None, help="Optional JSONL log for per-item telemetry")
    parser.add_argument("--run-summary", type=Path, default=None, help="Optional summary JSON with run aggregates")
    parser.add_argument("--max-failures", type=int, default=None, help="Abort run after this many failures (optional)")
    args = parser.parse_args()

    return TTSConfig(
        model_name=args.model_name,
        input_json=args.input_json,
        output_dir=args.output_dir,
        speaker_wav=args.speaker_wav,
        language=args.language,
        gpu=not args.cpu,
        limit=args.limit,
        dry_run=args.dry_run,
        resume=args.resume,
        filename_field=args.filename_field,
        write_metadata=args.write_metadata,
        strip_newlines=args.strip_newlines,
        expand_acronyms=not args.no_acronym,
        retries=max(0, int(args.retries)),
        sleep_seconds=max(0.0, float(args.sleep_seconds)),
        hooks_log=args.hooks_log,
        run_summary=args.run_summary,
        max_failures=args.max_failures,
    )


def main() -> None:
    config = parse_args()
    synthesize_batch(config)


if __name__ == "__main__":
    main()


