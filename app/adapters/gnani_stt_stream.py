"""Gnani Prisma v2.5 speech-to-text, realtime WebSocket endpoint.

Docs: https://docs.gnani.ai/api/STT/stt-websocket

Protocol constraints the docs are explicit about:
  - PCM16 little-endian mono, **exactly 1024 bytes per frame** (512 samples)
  - frames sent at real-time cadence (32 ms at 16 kHz, 64 ms at 8 kHz);
    "Do not buffer and burst."
  - server-side VAD closes a segment after `x-min-silence-ms` of silence
    (default 500 ms), then emits `transcript`

Observed live, and *not* in the docs (see DECISIONS.md D-016):
  - a `speech_start` event, and `utterance_id` / `is_final` on transcripts
  - `connected.config` advertises `partials_enabled`, `partial_interval_ms`
    and `partial_max_window_s`, so interim transcripts exist but are off

**Why latency is measured from the end of real audio.** Frames are paced, so
wall-clock time for a clip is at least its duration. In push-to-talk the owner
has already streamed that audio while speaking; what they wait for is whatever
happens after their last word. `Transcript.latency_ms` is therefore measured
from the moment the last frame of *real* audio was sent, excluding the silence
padding we add to trip the VAD.
"""

import asyncio
import json
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import websockets

from app.adapters.base import AudioClip, Transcript
from app.adapters.gnani_common import GnaniAPIError, require_key

FRAME_BYTES = 1024
FRAME_SAMPLES = FRAME_BYTES // 2  # 512, per the docs
SUPPORTED_SAMPLE_RATES = (8000, 16000, 44100, 48000)
VAD_MIN_SILENCE_RANGE_MS = (100, 2000)


@dataclass
class _Segment:
    text: str
    arrived_at: float
    raw: dict[str, Any]


def frames(pcm: bytes) -> Iterator[bytes]:
    """Split PCM into exactly-1024-byte frames, zero-padding the last one."""
    for start in range(0, len(pcm), FRAME_BYTES):
        frame = pcm[start : start + FRAME_BYTES]
        yield frame if len(frame) == FRAME_BYTES else frame + bytes(FRAME_BYTES - len(frame))


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
        # information and need not be paced. Bursting it trades protocol politeness for
        # latency; measured in D-016.
        burst_padding: bool = True,
        transcript_timeout_s: float = 15.0,
        # Window after the first final transcript, in case the VAD split the utterance.
        extra_segment_grace_s: float = 0.3,
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
        self._extra_segment_grace_s = extra_segment_grace_s
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

    async def transcribe(self, audio: AudioClip, *, language: str) -> Transcript:
        if audio.sample_rate not in SUPPORTED_SAMPLE_RATES:
            raise ValueError(f"Prisma realtime accepts {SUPPORTED_SAMPLE_RATES}, got {audio.sample_rate}")

        padding = bytes(2 * (audio.sample_rate * self._padding_ms // 1000))
        frame_interval = FRAME_SAMPLES / audio.sample_rate

        segments: list[_Segment] = []
        events: list[dict[str, Any]] = []
        connected_config: dict[str, Any] = {}
        started = time.perf_counter()
        speech_end_at: float | None = None  # last frame of real audio
        vad_end_at: float | None = None
        error: str | None = None
        final_seen = asyncio.Event()

        async with self._connect(self._ws_url, additional_headers=self._headers(language, audio.sample_rate)) as ws:

            async def receive() -> None:
                nonlocal vad_end_at, error
                async for message in ws:
                    if isinstance(message, bytes):
                        continue
                    event = json.loads(message)
                    events.append(event)
                    match event.get("type"):
                        case "error":
                            error = event.get("message", "transcription failed")
                            final_seen.set()
                            return
                        case "connected":
                            connected_config.update(event.get("config") or {})
                        # The docs name this "processing"; the live server sends "speech_end".
                        case "processing" | "speech_end" if vad_end_at is None:
                            vad_end_at = time.perf_counter()
                        case "transcript":
                            segments.append(_Segment(event.get("text", ""), time.perf_counter(), event))
                            # is_final is undocumented but present; absent means treat as final.
                            if event.get("is_final", True):
                                final_seen.set()
                # Socket closed: unblock the waiter instead of letting it time out.
                final_seen.set()

            receiver = asyncio.create_task(receive())

            async def send_paced(pcm: bytes, *, paced: bool) -> None:
                nonlocal next_send
                for frame in frames(pcm):
                    if final_seen.is_set() or receiver.done():
                        return
                    await ws.send(frame)
                    if paced and self._pace_realtime:
                        next_send += frame_interval
                        if (delay := next_send - time.perf_counter()) > 0:
                            await asyncio.sleep(delay)

            try:
                next_send = time.perf_counter()
                await send_paced(audio.pcm, paced=True)
                speech_end_at = time.perf_counter()
                await send_paced(padding, paced=not self._burst_padding)

                try:
                    await asyncio.wait_for(final_seen.wait(), timeout=self._transcript_timeout_s)
                except TimeoutError:
                    raise GnaniAPIError(
                        f"no transcript within {self._transcript_timeout_s:.0f} s of the last frame "
                        f"(min_silence_ms={self._min_silence_ms}, padding={self._padding_ms} ms)"
                    ) from None
                if self._extra_segment_grace_s:
                    await asyncio.sleep(self._extra_segment_grace_s)
            finally:
                receiver.cancel()

        if error:
            raise GnaniAPIError(error)
        if not segments:
            raise GnaniAPIError("stream closed without a transcript")

        first = segments[0].arrived_at
        return Transcript(
            text=" ".join(s.text for s in segments if s.text).strip(),
            latency_ms=(first - speech_end_at) * 1000 if speech_end_at else None,
            raw={
                "tail_ms": (first - speech_end_at) * 1000 if speech_end_at else None,
                "total_ms": (segments[-1].arrived_at - started) * 1000,
                "server_latency_ms": segments[0].raw.get("latency"),
                "vad_end_to_transcript_ms": (first - vad_end_at) * 1000 if vad_end_at else None,
                "speech_end_to_vad_end_ms": (
                    (vad_end_at - speech_end_at) * 1000 if vad_end_at and speech_end_at else None
                ),
                "num_segments": len(segments),
                "connected_config": connected_config,
                "segments": [s.raw for s in segments],
                "event_types": [e.get("type") for e in events],
            },
        )
