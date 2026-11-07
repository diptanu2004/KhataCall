"""Gnani Prisma v2.5 speech-to-text, realtime WebSocket endpoint.

Docs: https://docs.gnani.ai/api/STT/stt-websocket

Protocol constraints the docs are explicit about:
  - PCM16 little-endian mono, **exactly 1024 bytes per frame** (512 samples)
  - frames sent at real-time cadence (32 ms at 16 kHz, 64 ms at 8 kHz);
    "Do not buffer and burst."
  - server-side VAD closes a segment after `x-min-silence-ms` of silence
    (default 500 ms), then emits `transcript`

Observed live, and *not* in the docs (see DECISIONS.md D-017):
  - a `speech_start` event, and `utterance_id` / `is_final` on transcripts
  - `connected.config` advertises `partials_enabled`, `partial_interval_ms`
    and `partial_max_window_s`, so interim transcripts exist but are off

**Two entry points, because the live path and the eval path differ.**

`live()` is for the browser: audio arrives from a microphone already at
real-time cadence, so frames are forwarded the moment 1024 bytes have
accumulated. Pacing them again would add a second real-time delay.

`transcribe()` is for the eval harness and tests: it has a whole clip in hand,
so it paces the frames itself to respect the protocol.

**Why latency is measured from the end of real audio.** In push-to-talk the
owner has already streamed their words while speaking; what they wait for is
whatever happens after their last word. `Transcript.latency_ms` is measured
from that moment, excluding the silence padding we add to trip the VAD.
"""

import asyncio
import json
import time
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import websockets

from app.adapters.base import AudioClip, Transcript
from app.adapters.gnani_common import GnaniAPIError, require_key
from app.adapters.usage import record_stt

FRAME_BYTES = 1024
FRAME_SAMPLES = FRAME_BYTES // 2  # 512, per the docs
SUPPORTED_SAMPLE_RATES = (8000, 16000, 44100, 48000)
VAD_MIN_SILENCE_RANGE_MS = (100, 2000)

# Whether this server has been seen to send the undocumented `speech_start` event
# (D-017). Measured Nov 8: every utterance got one before its transcript, and audio
# with no speech got none and no transcript either, so the old path waited out the
# whole timeout (D-047). The shortcut that relies on it stays off until the event
# has actually been seen in this process, so if Prisma stops sending it, behaviour
# falls back to waiting for the transcript.
_speech_start_seen = False


def frames(pcm: bytes) -> Iterator[bytes]:
    """Split PCM into exactly-1024-byte frames, zero-padding the last one."""
    for start in range(0, len(pcm), FRAME_BYTES):
        frame = pcm[start : start + FRAME_BYTES]
        yield frame if len(frame) == FRAME_BYTES else frame + bytes(FRAME_BYTES - len(frame))


@dataclass
class _Segment:
    text: str
    arrived_at: float
    raw: dict[str, Any]


@dataclass
class _Timings:
    opened_at: float
    speech_end_at: float | None = None  # last frame of real audio
    vad_end_at: float | None = None
    segments: list[_Segment] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    connected_config: dict[str, Any] = field(default_factory=dict)


class LiveTranscription:
    """One utterance on an open Prisma socket. Push audio, then `finish()`."""

    def __init__(
        self,
        ws: Any,
        *,
        sample_rate: int,
        padding_ms: int,
        min_silence_ms: int,
        burst_padding: bool,
        transcript_timeout_s: float,
        quiet_period_s: float,
        no_speech_after_s: float = 0.0,
    ) -> None:
        self._ws = ws
        self._sample_rate = sample_rate
        self._padding_ms = padding_ms
        self._min_silence_ms = min_silence_ms
        self._burst_padding = burst_padding
        self._transcript_timeout_s = transcript_timeout_s
        self._quiet_period_s = quiet_period_s
        self._no_speech_after_s = no_speech_after_s
        self._buffer = bytearray()
        self._t = _Timings(opened_at=time.perf_counter())
        self._error: str | None = None
        self._final_seen = asyncio.Event()
        self._segment_arrived = asyncio.Event()
        self._speech_heard = asyncio.Event()  # speech_start, a transcript, or the socket ending
        self._receiver: asyncio.Task | None = None
        self._metered = False
        self.frames_sent = 0

    # --- lifecycle, driven by PrismaStreamingSTT.live() ---

    def _start_receiver(self) -> None:
        self._receiver = asyncio.create_task(self._receive())

    def _stop_receiver(self) -> None:
        if self._receiver is not None:
            self._receiver.cancel()

    async def _receive(self) -> None:
        global _speech_start_seen
        try:
            async for message in self._ws:
                if isinstance(message, bytes):
                    continue
                event = json.loads(message)
                self._t.events.append(event)
                match event.get("type"):
                    case "error":
                        self._error = event.get("message", "transcription failed")
                        return
                    case "connected":
                        self._t.connected_config.update(event.get("config") or {})
                    case "speech_start":
                        _speech_start_seen = True
                        self._speech_heard.set()
                    # The docs name this "processing"; the live server sends "speech_end".
                    case "processing" | "speech_end" if self._t.vad_end_at is None:
                        self._t.vad_end_at = time.perf_counter()
                    case "transcript":
                        self._t.segments.append(_Segment(event.get("text", ""), time.perf_counter(), event))
                        self._segment_arrived.set()
                        self._speech_heard.set()
                        # is_final is undocumented but present; absent means treat as final.
                        if event.get("is_final", True):
                            self._final_seen.set()
        finally:
            # An error or a closed socket: unblock any waiter instead of letting it time out.
            self._final_seen.set()
            self._speech_heard.set()

    def _meter(self) -> None:
        """Billed per hour of audio. Counts everything streamed, padding included: the
        server received it. Once per utterance, finished or not."""
        if self._metered or not self.frames_sent:
            return
        self._metered = True
        record_stt(self.frames_sent * FRAME_SAMPLES / self._sample_rate, model="prisma-v2.5-stream", note="stream")

    # --- sending ---

    @property
    def _done(self) -> bool:
        return self._final_seen.is_set() or (self._receiver is not None and self._receiver.done())

    async def _send_frame(self, frame: bytes) -> None:
        await self._ws.send(frame)
        self.frames_sent += 1

    async def push(self, pcm: bytes) -> None:
        """Forward microphone audio. Whole frames are sent as soon as they fill."""
        if self._error:
            raise GnaniAPIError(self._error)
        self._buffer += pcm
        while len(self._buffer) >= FRAME_BYTES and not self._done:
            frame = bytes(self._buffer[:FRAME_BYTES])
            del self._buffer[:FRAME_BYTES]
            await self._send_frame(frame)

    async def finish(self) -> Transcript:
        """Mark end of speech, trip the VAD, and wait for the transcript."""
        if self._buffer and not self._done:
            await self._send_frame(bytes(self._buffer).ljust(FRAME_BYTES, b"\0"))
            self._buffer.clear()
        self._t.speech_end_at = time.perf_counter()

        padding = bytes(2 * (self._sample_rate * self._padding_ms // 1000))
        interval = FRAME_SAMPLES / self._sample_rate
        next_send = time.perf_counter()
        for frame in frames(padding):
            if self._done:
                break
            await self._send_frame(frame)
            if not self._burst_padding:
                next_send += interval
                if (delay := next_send - time.perf_counter()) > 0:
                    await asyncio.sleep(delay)
        # Metered here, before any wait, so a later timeout still records the spend.
        self._meter()

        if self._no_speech_after_s and _speech_start_seen:
            try:
                await asyncio.wait_for(self._speech_heard.wait(), timeout=self._no_speech_after_s)
            except TimeoutError:
                # Prisma's VAD never heard speech, so no transcript is coming (D-047).
                return Transcript(text="", latency_ms=None, raw={"no_speech": True, "frames_sent": self.frames_sent})

        try:
            await asyncio.wait_for(self._final_seen.wait(), timeout=self._transcript_timeout_s)
        except TimeoutError:
            raise GnaniAPIError(
                f"no transcript within {self._transcript_timeout_s:.0f} s of the last frame "
                f"(min_silence_ms={self._min_silence_ms}, padding={self._padding_ms} ms)"
            ) from None
        await self._wait_for_quiet()

        if self._error:
            raise GnaniAPIError(self._error)
        if not self._t.segments:
            raise GnaniAPIError("stream closed without a transcript")
        return self._build()

    async def _wait_for_quiet(self) -> None:
        """Wait out a short silence before declaring the utterance complete.

        The VAD can split one spoken sentence into several segments - "Sharma ji ka"
        then "dedh sau udhaar" if the owner pauses mid-phrase. Returning on the first
        segment would silently drop the rest, and in a ledger that means a wrong
        amount. So we wait for a quiet period, restarting it whenever another segment
        lands, rather than sleeping a fixed time once.

        All the audio has already been sent by this point, so a sibling segment
        arrives within roughly one inference (~150-300 ms) or not at all.
        """
        if not self._quiet_period_s:
            return
        while True:
            self._segment_arrived.clear()
            try:
                await asyncio.wait_for(self._segment_arrived.wait(), timeout=self._quiet_period_s)
            except TimeoutError:
                return

    def _build(self) -> Transcript:
        t = self._t
        first = t.segments[0].arrived_at
        tail = (first - t.speech_end_at) * 1000 if t.speech_end_at else None
        return Transcript(
            text=" ".join(s.text for s in t.segments if s.text).strip(),
            latency_ms=tail,
            raw={
                "tail_ms": tail,
                "total_ms": (t.segments[-1].arrived_at - t.opened_at) * 1000,
                "server_latency_ms": t.segments[0].raw.get("latency"),
                "vad_end_to_transcript_ms": (first - t.vad_end_at) * 1000 if t.vad_end_at else None,
                "speech_end_to_vad_end_ms": (
                    (t.vad_end_at - t.speech_end_at) * 1000 if t.vad_end_at and t.speech_end_at else None
                ),
                "num_segments": len(t.segments),
                "frames_sent": self.frames_sent,
                "connected_config": dict(t.connected_config),
                "segments": [s.raw for s in t.segments],
                "event_types": [e.get("type") for e in t.events],
            },
        )


class PrismaStreamingSTT:
    model_id = "prisma-v2.5-stream"

    def __init__(
        self,
        api_key: str,
        *,
        ws_url: str = "wss://api.vachana.ai/stt/v3/stream",
        fmt: str = "verbatim",
        # The VAD needs this much silence to close a segment. It is a direct latency cost,
        # so we ask for less than the 500 ms default. Range per docs: 100-2000 ms.
        min_silence_ms: int = 200,
        # Silence appended after the utterance so the VAD closes. Must exceed min_silence_ms.
        padding_margin_ms: int = 150,
        # Push-to-talk release is an explicit end of speech, so the padding carries no
        # information and need not be paced. Bursting it trades politeness for latency (D-016).
        burst_padding: bool = True,
        transcript_timeout_s: float = 15.0,
        # Silence required after the last segment before the utterance is declared done.
        # Guards against the VAD splitting a sentence; see LiveTranscription._wait_for_quiet.
        quiet_period_s: float = 0.2,
        # With no `speech_start` this long after the last frame, nobody spoke (D-047).
        no_speech_after_s: float = 1.5,
        pace_realtime: bool = True,
        connect: Callable[..., Any] = websockets.connect,
    ) -> None:
        if fmt not in ("verbatim", "transcribe"):
            raise ValueError(f"fmt must be 'verbatim' or 'transcribe', got {fmt!r}")
        low, high = VAD_MIN_SILENCE_RANGE_MS
        if not low <= min_silence_ms <= high:
            raise ValueError(f"min_silence_ms must be {low}-{high}, got {min_silence_ms}")
        self._api_key = api_key
        self._ws_url = ws_url
        self._fmt = fmt
        self._min_silence_ms = min_silence_ms
        self._padding_ms = min_silence_ms + padding_margin_ms
        self._burst_padding = burst_padding
        self._transcript_timeout_s = transcript_timeout_s
        self._quiet_period_s = quiet_period_s
        self._no_speech_after_s = no_speech_after_s
        self._pace_realtime = pace_realtime
        self._connect = connect

    def _headers(self, language: str, sample_rate: int) -> dict[str, str]:
        # Lowercase names as this endpoint's docs spell them, unlike the REST/TTS pages'
        # X-API-Key-ID. HTTP header names are case-insensitive either way.
        return {
            "x-api-key-id": require_key(self._api_key),
            "lang_code": language,
            "x-sample-rate": str(sample_rate),
            "x-format": self._fmt,
            "x-min-silence-ms": str(self._min_silence_ms),
        }

    @asynccontextmanager
    async def live(self, *, language: str, sample_rate: int) -> AsyncIterator[LiveTranscription]:
        """Open a socket for one utterance of live microphone audio."""
        if sample_rate not in SUPPORTED_SAMPLE_RATES:
            raise ValueError(f"Prisma realtime accepts {SUPPORTED_SAMPLE_RATES}, got {sample_rate}")
        headers = self._headers(language, sample_rate)
        async with self._connect(self._ws_url, additional_headers=headers) as ws:
            session = LiveTranscription(
                ws,
                sample_rate=sample_rate,
                padding_ms=self._padding_ms,
                min_silence_ms=self._min_silence_ms,
                burst_padding=self._burst_padding,
                transcript_timeout_s=self._transcript_timeout_s,
                quiet_period_s=self._quiet_period_s,
                no_speech_after_s=self._no_speech_after_s,
            )
            session._start_receiver()
            try:
                yield session
            finally:
                session._stop_receiver()
                session._meter()  # a turn abandoned before finish() still sent audio

    async def transcribe(self, audio: AudioClip, *, language: str) -> Transcript:
        """Transcribe a complete clip, pacing frames as the protocol requires."""
        async with self.live(language=language, sample_rate=audio.sample_rate) as session:
            interval = FRAME_SAMPLES / audio.sample_rate
            next_send = time.perf_counter()
            for frame in frames(audio.pcm):
                await session.push(frame)
                if self._pace_realtime:
                    next_send += interval
                    if (delay := next_send - time.perf_counter()) > 0:
                        await asyncio.sleep(delay)
            return await session.finish()
