"""Pieces shared by the Gnani Prisma and Timbre adapters."""

import asyncio
import io
import logging
import time
import wave
from collections.abc import Callable
from contextlib import AsyncExitStack
from typing import Any

from app.adapters.base import AudioClip

log = logging.getLogger(__name__)

# A WebSocket to Gnani that has not opened after this long gets a second attempt
# alongside it; whichever opens first is used. Measured Nov 10 from the demo laptop:
# TCP 68 ms and TLS 0.15-0.5 s every time, but the upgrade answered in 0.5-2.3 s
# usually and 5.7 s or 17.6 s now and then, past websockets' 10 s open timeout
# ("timed out during opening handshake"). The wait is per attempt, so a fresh one
# usually opens in the normal time (D-052).
HEDGE_AFTER_S = 2.5
HEDGE_ATTEMPTS = 3
_discarded: set[asyncio.Future] = set()


async def open_hedged(
    stack: AsyncExitStack,
    connect: Callable[[], Any],
    *,
    hedge_after_s: float = HEDGE_AFTER_S,
    attempts: int = HEDGE_ATTEMPTS,
    what: str = "Gnani",
) -> Any:
    """Enter `connect()` (an async context manager, such as `websockets.connect(...)`).

    While none has opened, another attempt starts every `hedge_after_s`, up to
    `attempts`; one that fails starts the next at once. The first to open is
    returned, with its exit registered on `stack`. Any other that opens later is
    closed straight away, having sent nothing. If every attempt fails, the first
    error is raised.
    """
    tries: list[tuple[Any, asyncio.Future]] = []
    errors: list[BaseException] = []
    started = time.perf_counter()

    def start() -> None:
        manager = connect()
        tries.append((manager, asyncio.ensure_future(manager.__aenter__())))

    start()
    while True:
        running = [task for _, task in tries if not task.done()]
        can_hedge = len(tries) < attempts
        if not running:
            if not can_hedge:
                raise errors[0]
            start()
            continue
        done, _ = await asyncio.wait(running, timeout=hedge_after_s if can_hedge else None,
                                     return_when=asyncio.FIRST_COMPLETED)
        if not done:
            log.info("%s socket not open after %.1f s; starting attempt %d alongside",
                     what, time.perf_counter() - started, len(tries) + 1)
            start()
            continue
        for manager, task in tries:
            if task not in done:
                continue
            if task.exception() is not None:
                errors.append(task.exception())
                continue
            for other_manager, other in tries:
                if other is not task:
                    _close_when_open(other_manager, other)
            stack.push_async_exit(manager)
            if len(tries) > 1:
                log.info("%s socket opened on attempt %d after %.1f s", what,
                         [t for _, t in tries].index(task) + 1, time.perf_counter() - started)
            return task.result()


def _close_when_open(manager: Any, task: asyncio.Future) -> None:
    """An attempt that lost: if it opens after all, close it at once."""

    def done(t: asyncio.Future) -> None:
        if t.cancelled() or t.exception() is not None:
            return
        closer = asyncio.ensure_future(manager.__aexit__(None, None, None))
        _discarded.add(closer)
        closer.add_done_callback(_discarded.discard)

    task.add_done_callback(done)

# Header name exactly as shown in https://docs.gnani.ai (both STT and TTS pages).
API_KEY_HEADER = "X-API-Key-ID"


class GnaniAPIError(RuntimeError):
    def __init__(self, message: str, *, status: int | None = None, error_type: str | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.error_type = error_type

    def __str__(self) -> str:
        parts = [p for p in (f"HTTP {self.status}" if self.status else None, self.error_type) if p]
        prefix = f"[{' '.join(parts)}] " if parts else ""
        return prefix + super().__str__()


def require_key(api_key: str) -> str:
    if not api_key:
        raise GnaniAPIError("GNANI_API_KEY is empty; set it in .env (see .env.example)")
    return api_key


def to_wav_bytes(audio: AudioClip) -> bytes:
    """Wrap mono PCM16 in a WAV container (Prisma REST takes a file upload)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(audio.sample_rate)
        w.writeframes(audio.pcm)
    return buf.getvalue()
