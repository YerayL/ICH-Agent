"""Optional observability and guardrails for batch TTS runs.

Use ``create_hooks`` to obtain a lightweight hook manager that emits JSONL
telemetry, writes an aggregate summary, and optionally aborts after a
configurable number of failures.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class BatchRunState:
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    started_at: float = field(default_factory=time.time)
    failures: int = 0
    successes: int = 0


class HookManager:
    def __init__(
        self,
        log_path: Optional[Path] = None,
        summary_path: Optional[Path] = None,
        max_failures: Optional[int] = None,
    ) -> None:
        self.log_path = log_path
        self.summary_path = summary_path
        self.max_failures = max_failures
        self.state = BatchRunState()

        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if self.summary_path is not None:
            self.summary_path.parent.mkdir(parents=True, exist_ok=True)

    def on_run_start(self, total_items: Optional[int], config: Dict[str, Any]) -> None:
        payload = {
            "event": "run_start",
            "run_id": self.state.run_id,
            "total_items": total_items,
            "config": _safe_config_snapshot(config),
            "timestamp": time.time(),
        }
        self._append_log(payload)

    def on_item_success(self, index: int, filename: str, duration_sec: float, content_length: int) -> None:
        self.state.successes += 1
        payload = {
            "event": "item_success",
            "run_id": self.state.run_id,
            "index": index,
            "filename": filename,
            "duration_sec": round(duration_sec, 4),
            "content_length": content_length,
            "timestamp": time.time(),
        }
        self._append_log(payload)

    def on_item_failure(self, index: int, error: str, attempt: int) -> None:
        self.state.failures += 1
        payload = {
            "event": "item_failure",
            "run_id": self.state.run_id,
            "index": index,
            "attempt": attempt,
            "error": error,
            "timestamp": time.time(),
        }
        self._append_log(payload)

    def should_abort(self) -> bool:
        return self.max_failures is not None and self.state.failures >= self.max_failures

    def finalize(self, successes: int, failures: int, wall_time: float) -> None:
        summary = {
            "event": "run_complete",
            "run_id": self.state.run_id,
            "successes": successes,
            "failures": failures,
            "wall_time_sec": round(wall_time, 4),
            "started_at": self.state.started_at,
            "ended_at": time.time(),
        }
        self._append_log(summary)
        if self.summary_path is not None:
            _write_json(self.summary_path, summary)

    def _append_log(self, payload: Dict[str, Any]) -> None:
        if self.log_path is None:
            return
        try:
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:
            # Observability must never break the main flow.
            pass


def create_hooks(
    log_path: Optional[Path],
    summary_path: Optional[Path],
    max_failures: Optional[int],
) -> Optional[HookManager]:
    """Factory returning a HookManager when any feature is requested."""
    if log_path is None and summary_path is None and max_failures is None:
        return None
    return HookManager(log_path=log_path, summary_path=summary_path, max_failures=max_failures)


def _safe_config_snapshot(config: Dict[str, Any]) -> Dict[str, Any]:
    # Protect against non-serializable values by stringifying as needed.
    snapshot: Dict[str, Any] = {}
    for key, value in config.items():
        try:
            json.dumps(value)
            snapshot[key] = value
        except Exception:
            snapshot[key] = str(value)
    return snapshot


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
