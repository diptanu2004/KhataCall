"""Gnani Timbre v2.5 text-to-speech, realtime WebSocket endpoint.

Docs: https://docs.gnani.ai/api/TTS/tts-websocket
Audio arrives as base64 inside JSON messages; we decode and yield raw PCM16.
"""

import asyncio
import base64
import json
import logging
import time
from collections.abc import AsyncIterator, Callable
from contextlib import AsyncExitStack
from typing import Any

import websockets

from app.adapters.gnani_common import API_KEY_HEADER, GnaniAPIError, open_hedged, require_key
from app.adapters.usage import record_tts

log = logging.getLogger(__name__)

# A pre-opened socket older than this is replaced rather than used. Measured Nov 8:
# one idle for 20 s still synthesised normally, so this is the limit of what has been
# tried, not a known server timeout. A normal turn uses its socket within ~8 s of the
# press; a spare left over from a tap can wait much longer (D-047).
SPARE_MAX_AGE_S = 20.0

# Only Poorvi is listed for hi-en (Hinglish) in the voice catalog: https://docs.gnani.ai/api/TTS/available-voices
DEFAULT_VOICE = "Poorvi"


class TimbreTTS:
    model_id = "timbre-v2.5"

    def __init__(
        self,
        api_key: str,
        *,
        ws_url: str = "wss://api.vachana.ai/api/v1/tts",
        sample_rate: int = 16_000,
        speed: float = 1.0,
        connect: Callable[..., Any] = websockets.connect,
    ) -> None:
        if not 8_000 <= sample_rate <= 48_000:
            raise ValueError("Timbre sample_rate must be 8000-48000 Hz")
        if not 0.85 <= speed <= 1.15:
            raise ValueError("Timbre speed must be 0.85-1.15")
        self._api_key = api_key
        self._ws_url = ws_url
        self.sample_rate = sample_rate
        self._speed = speed
        self._connect = connect
        self._spare: asyncio.Task[tuple[AsyncExitStack, Any]] | None = None
        self._spare_opened_at = 0.0
        self._closing: set[asyncio.Task] = set()

    async def prepare(self) -> None:
        """Open the next reply's socket now, so its handshake overlaps STT and Evon.

        The handshake is ~450 ms of a ~700 ms first chunk (D-046). Each socket
        still carries one request: a second request on an open socket returns a
        bare `complete` with no audio (D-018), so this changes *when* the socket
        opens, not how many requests it carries (HLD §12.2).
        """
        if self._spare is not None and time.monotonic() - self._spare_opened_at > SPARE_MAX_AGE_S:
            self._discard(self._spare)
            self._spare = None
        if self._spare is None:
            self._spare = asyncio.create_task(self._open())
            self._spare_opened_at = time.monotonic()

    async def _open(self) -> tuple[AsyncExitStack, Any]:
        stack = AsyncExitStack()
        headers = {API_KEY_HEADER: require_key(self._api_key)}
        ws = await open_hedged(stack, lambda: self._connect(self._ws_url, additional_headers=headers), what="Timbre")
        return stack, ws

    def _discard(self, spare: asyncio.Task) -> None:
        """Close a pre-opened socket in the background: a close is a handshake too."""

        async def close() -> None:
            try:
                stack, _ = await spare
            except Exception:
                return
            await stack.aclose()

        task = asyncio.create_task(close())
        self._closing.add(task)
        task.add_done_callback(self._closing.discard)

    async def _take_spare(self) -> tuple[AsyncExitStack, Any] | None:
        """The pre-opened socket, if it connected and is still fresh."""
        spare, self._spare = self._spare, None
        if spare is None:
            return None
        if time.monotonic() - self._spare_opened_at > SPARE_MAX_AGE_S:
            self._discard(spare)
            return None
        try:
            return await spare
        except Exception:
            log.info("pre-opened Timbre socket failed to connect; opening a fresh one")
            return None

    async def aclose(self) -> None:
        """Close a pre-opened socket nobody used."""
        spare, self._spare = self._spare, None
        if spare is not None:
            self._discard(spare)
        if self._closing:
            await asyncio.gather(*self._closing, return_exceptions=True)

    def _request(self, text: str, language: str, voice: str) -> dict[str, Any]:
        return {
            "text": text,
            "voice": voice,
            "model": self.model_id,
            "language": language,
            "speed": self._speed,
            "audio_config": {
                "sample_rate": self.sample_rate,
                "num_channels": 1,
                "sample_width": 2,
                # Docs output table: container "raw" + encoding "linear_pcm" -> raw 16-bit PCM, no header.
                # VERIFY: byte order is little-endian (assumed; pcm_s16le is also listed as an encoding).
                "encoding": "linear_pcm",
                "container": "raw",
            },
        }

    async def synthesize(self, text: str, *, language: str, voice: str) -> AsyncIterator[bytes]:
        require_key(self._api_key)
        # Billed per 10,000 characters of input text, so the cost is known up front.
        record_tts(len(text), model=self.model_id)
        request = json.dumps(self._request(text, language, voice), ensure_ascii=False)

        if (spare := await self._take_spare()) is not None:
            stack, ws = spare
            started = False
            try:
                async with stack:
                    await ws.send(request)
                    async for pcm in self._audio(ws):
                        started = True
                        yield pcm
                return
            except (websockets.ConnectionClosed, _ClosedEarly) as e:
                if started:
                    raise
                # The socket died while it waited (seen once, during a network fault, D-047),
                # before any audio: one retry on a fresh socket, metered in case the first
                # request was billed.
                log.info("pre-opened Timbre socket failed before any audio (%s); retrying on a fresh one", e)
                record_tts(len(text), model=self.model_id, note="retry")

        stack, ws = await self._open()
        async with stack:
            await ws.send(request)
            async for pcm in self._audio(ws):
                yield pcm

    async def _audio(self, ws: Any) -> AsyncIterator[bytes]:
        leftover = b""  # chunk boundaries are not guaranteed to fall on 2-byte samples
        async for message in ws:
            if isinstance(message, bytes):
                raise GnaniAPIError("unexpected binary frame from Timbre")
            event = json.loads(message)
            kind = event.get("type")
            if kind == "error":
                raise GnaniAPIError(event.get("message", "synthesis failed"))
            if kind not in ("audio", "complete"):
                continue
            pcm = leftover + base64.b64decode(event.get("data", {}).get("audio") or "")
            cut = len(pcm) - len(pcm) % 2
            leftover = pcm[cut:]
            if cut:
                yield pcm[:cut]
            if kind == "complete" or event.get("data", {}).get("is_final"):
                return
        raise _ClosedEarly("Timbre closed the connection before sending 'complete'")


class _ClosedEarly(GnaniAPIError):
    """The socket ended without `complete`."""
