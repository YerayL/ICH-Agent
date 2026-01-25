"""Video synthesis orchestrator.

This script submits audio + reference video to a local service to generate
face-sync videos. It preserves the original behavior while adding structured
logging, CLI configuration, optional retries, metadata, and progress polling.

Defaults:
- Text file: /home/pc/lyy/ICH-agent/vedio_data/patient_results.json
- Reference video: ref_face.mp4
- Audio filename pattern: 001.wav, 002.wav, ... (produced by tts.py)
"""

import requests
import uuid
import time
import os
import json
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any
import logging

from gen_video_extras import (
    derive_audio_url,
    write_metadata_sidecar,
    should_skip_task,
)


def _configure_logging() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")


@dataclass
class GenVideoConfig:
    text_file: Path = Path("/home/pc/lyy/ICH-agent/vedio_data/patient_results.json")
    ref_video_path: str = "ref_face.mp4"
    submit_url: str = "http://127.0.0.1:8383/easy/submit"
    query_url_base: str = "http://127.0.0.1:8383/easy/query"
    audio_base_url: Optional[str] = None  # If set, prefix to audio filenames
    dry_run: bool = False
    resume: bool = False
    write_metadata: bool = False
    retries: int = 0
    timeout: float = 15.0
    sleep_seconds: float = 0.0
    poll_progress: bool = False
    poll_interval: float = 2.0
    poll_timeout: float = 300.0

def preprocess_audio(reference_audio_path: str, base_url: str = "http://127.0.0.1:18180", timeout: float = 15.0) -> Optional[Dict[str, Any]]:
    url = f"{base_url.rstrip('/')}/v1/preprocess_and_tran"
    payload = {
        "format": "wav",
        "reference_audio": reference_audio_path,
        "lang": "en",
    }
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logging.error("Preprocess failed: %s", e)
        return None

def synthesize_audio(text: str, preprocess_result: Dict[str, Any], speaker_id: str, base_url: str = "http://127.0.0.1:18180", timeout: float = 15.0) -> Optional[str]:
    url = f"{base_url.rstrip('/')}/v1/invoke"
    payload = {
        "speaker": speaker_id,
        "text": text,
        "format": "wav",
        "topP": 0.7,
        "max_new_tokens": 1024,
        "chunk_length": 100,
        "repetition_penalty": 1.2,
        "temperature": 0.7,
        "need_asr": False,
        "streaming": False,
        "is_fixed_seed": 0,
        "is_norm": 0,
        "reference_audio": preprocess_result.get("asr_format_audio_url"),
        "reference_text": preprocess_result.get("reference_audio_text"),
    }
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        filename = save_synthesized_audio(resp, speaker_id)
        return filename
    except Exception as e:
        logging.error("Synthesize audio failed: %s", e)
        return None

def save_synthesized_audio(response: requests.Response, speaker_id: str, save_dir: str = "/home/pc/heygem_data/face2face/temp/") -> str:
    os.makedirs(save_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d%H%M%S")
    filename = f"{speaker_id}_{timestamp}.wav"
    save_path = os.path.join(save_dir, filename)
    with open(save_path, "wb") as f:
        f.write(response.content)
    return filename


def synthesize_video(audio_url: str, video_url: str, speaker_id: str, submit_url: str, timeout: float = 15.0, retries: int = 0) -> Optional[str]:
    payload = {
        "audio_url": audio_url,
        "video_url": video_url,
        "code": speaker_id,
        "chaofen": 0,
        "watermark_switch": 0,
        "pn": 1,
    }
    attempt = 0
    while attempt <= max(0, retries):
        try:
            resp = requests.post(submit_url, json=payload, timeout=timeout)
            resp.raise_for_status()
            return speaker_id
        except Exception as e:
            logging.warning("Submit failed (attempt %s/%s): %s", attempt + 1, retries, e)
            attempt += 1
            if attempt <= retries:
                time.sleep(min(2.0 * attempt, 5.0))
    return None

def check_video_progress(task_code: str, query_url_base: str, timeout: float = 15.0) -> Optional[Dict[str, Any]]:
    url = f"{query_url_base.rstrip('/')}/easy/query?code={task_code}"
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logging.error("Query failed: %s", e)
        return None

def parse_args() -> GenVideoConfig:
    import argparse
    parser = argparse.ArgumentParser(description="Submit audio + ref video to synthesize lip-sync videos")
    parser.add_argument("--text-file", type=Path, default=Path("/home/pc/lyy/ICH-agent/vedio_data/patient_results.json"), help="Input JSON file path used for iteration/length")
    parser.add_argument("--ref-video-path", default="ref_face.mp4", help="Reference video path")
    parser.add_argument("--submit-url", default="http://127.0.0.1:8383/easy/submit", help="Submit endpoint URL")
    parser.add_argument("--query-url-base", default="http://127.0.0.1:8383", help="Query endpoint base URL")
    parser.add_argument("--audio-base-url", default=None, help="If provided, prefix to audio filenames like 001.wav")
    parser.add_argument("--dry-run", action="store_true", help="Log actions without submitting")
    parser.add_argument("--resume", action="store_true", help="Skip tasks that appear already processed (metadata exists)")
    parser.add_argument("--write-metadata", action="store_true", help="Write sidecar metadata JSON per task")
    parser.add_argument("--retries", type=int, default=0, help="Submit retry count on failure")
    parser.add_argument("--timeout", type=float, default=15.0, help="HTTP timeout seconds")
    parser.add_argument("--sleep-seconds", type=float, default=0.0, help="Sleep seconds between tasks")
    parser.add_argument("--poll-progress", action="store_true", help="Poll task status after submission")
    parser.add_argument("--poll-interval", type=float, default=2.0, help="Polling interval seconds")
    parser.add_argument("--poll-timeout", type=float, default=300.0, help="Polling max duration seconds")
    args = parser.parse_args()
    return GenVideoConfig(
        text_file=args.text_file,
        ref_video_path=args.ref_video_path,
        submit_url=args.submit_url,
        query_url_base=args.query_url_base,
        audio_base_url=args.audio_base_url,
        dry_run=args.dry_run,
        resume=args.resume,
        write_metadata=args.write_metadata,
        retries=max(0, int(args.retries)),
        timeout=max(1.0, float(args.timeout)),
        sleep_seconds=max(0.0, float(args.sleep_seconds)),
        poll_progress=args.poll_progress,
        poll_interval=max(0.5, float(args.poll_interval)),
        poll_timeout=max(1.0, float(args.poll_timeout)),
    )


def main(config: Optional[GenVideoConfig] = None) -> None:
    _configure_logging()
    logger = logging.getLogger("video")

    cfg = config or parse_args()
    with cfg.text_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    start = time.time()
    for i, d in enumerate(data):
        speaker_id = f"{i+1:03d}"
        audio_filename = f"{speaker_id}.wav"
        audio_url = derive_audio_url(audio_filename, cfg.audio_base_url)

        meta_path = Path(f"{speaker_id}.json")
        if should_skip_task(meta_path, resume=cfg.resume):
            logger.info("Resume skip: %s", speaker_id)
            continue

        if cfg.dry_run:
            logger.info("Dry-run submit: audio=%s video=%s code=%s", audio_url, cfg.ref_video_path, speaker_id)
            task_code = speaker_id
        else:
            task_code = synthesize_video(audio_url, cfg.ref_video_path, speaker_id, cfg.submit_url, timeout=cfg.timeout, retries=cfg.retries)

        if not task_code:
            logger.error("Submission failed for %s", speaker_id)
            continue

        if cfg.write_metadata:
            meta = {
                "speaker_id": speaker_id,
                "audio": audio_url,
                "video": cfg.ref_video_path,
                "submit_url": cfg.submit_url,
                "timestamp": datetime.now().isoformat(),
            }
            write_metadata_sidecar(meta_path, meta)

        if cfg.poll_progress and not cfg.dry_run:
            logger.info("Polling task %s", task_code)
            deadline = time.time() + cfg.poll_timeout
            while time.time() < deadline:
                status = check_video_progress(task_code, cfg.query_url_base, timeout=cfg.timeout)
                if status is not None:
                    # Heuristic: stop if status shows completion flag
                    if status.get("status") in ("completed", "done", 2):
                        logger.info("Task %s completed", task_code)
                        break
                time.sleep(cfg.poll_interval)

        if cfg.sleep_seconds > 0:
            time.sleep(cfg.sleep_seconds)

    run_time = time.time() - start
    print(f"{run_time:.6f} 秒")

if __name__ == "__main__":
    main()

