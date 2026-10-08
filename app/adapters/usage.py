"""Local meter for Gnani spend, because the programme credits are a hard cap.

Published rates (dashboard, Oct 8 2026):
  Speech to text  Rs 27.00 per **hour of audio**   (REST, WebSocket and batch alike)
  Text to speech  Rs 27.00 per **10,000 characters**

So cost follows audio duration and character count, not the number of calls.
Reconstructing this session's runs gave Rs 4.03 against 5 credits reported
spent, which is how we concluded 1 credit = Rs 1 and that billing is
proportional rather than rounded up per request (DECISIONS.md D-020).

Every real adapter call records here, appending one JSON line per call to
`scripts/out/usage.jsonl`. `uv run python -m scripts.usage` summarises it.
Failures to write are swallowed: metering must never break a voice turn.
"""

import json
import os
import threading
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

STT_INR_PER_HOUR = 27.0
TTS_INR_PER_10K_CHARS = 27.0

# Published alongside the rates; the eval runner paces itself against this.
MAX_REQUESTS_PER_MINUTE = 60
MAX_CONCURRENT_STT_SESSIONS = 20

DEFAULT_LOG = Path(__file__).resolve().parents[2] / "scripts" / "out" / "usage.jsonl"
_lock = threading.Lock()


def stt_cost_inr(audio_seconds: float) -> float:
    return audio_seconds / 3600.0 * STT_INR_PER_HOUR


def tts_cost_inr(characters: int) -> float:
    return characters / 10_000.0 * TTS_INR_PER_10K_CHARS


@dataclass(frozen=True)
class UsageRecord:
    at: str
    service: str  # "stt" | "tts"
    model: str
    units: float  # audio seconds, or characters
    unit: str  # "s" | "chars"
    cost_inr: float
    note: str = ""


def _log_path() -> Path | None:
    """`KHATACALL_USAGE_LOG=off` disables metering, e.g. in tests."""
    raw = os.environ.get("KHATACALL_USAGE_LOG")
    if raw and raw.lower() in ("off", "none", "0", "false"):
        return None
    return Path(raw) if raw else DEFAULT_LOG


def _append(record: UsageRecord) -> UsageRecord:
    path = _log_path()
    if path is None:
        return record
    try:
        with _lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
    except OSError:
        pass  # never let bookkeeping break a call
    return record


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def record_stt(audio_seconds: float, *, model: str, note: str = "") -> UsageRecord:
    return _append(
        UsageRecord(_now(), "stt", model, round(audio_seconds, 3), "s", stt_cost_inr(audio_seconds), note)
    )


def record_tts(characters: int, *, model: str, note: str = "") -> UsageRecord:
    return _append(
        UsageRecord(_now(), "tts", model, characters, "chars", tts_cost_inr(characters), note)
    )


def read_log(path: Path | None = None) -> list[UsageRecord]:
    path = path or _log_path()
    if path is None or not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                records.append(UsageRecord(**json.loads(line)))
            except (json.JSONDecodeError, TypeError):
                continue
    return records
