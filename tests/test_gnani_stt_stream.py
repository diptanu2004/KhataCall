"""Prisma realtime WebSocket adapter against a fake socket. No network."""

import asyncio
import json

import pytest

from app.adapters.base import STT, AudioClip
from app.adapters.gnani_common import GnaniAPIError
from app.adapters.gnani_stt_stream import FRAME_BYTES, PrismaStreamingSTT, frames

CLIP = AudioClip(pcm=bytes(2 * 16_000), sample_rate=16_000)  # 1 s at 16 kHz


class FakeStream:
    """Collects sent frames and replays scripted server events after the send finishes."""

    def __init__(self, events: list[dict], *, emit_after_frames: int | None = None) -> None:
        self._events = events
        self._emit_after_frames = emit_after_frames
        self.sent: list[bytes] = []
        self._release = asyncio.Event()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def send(self, frame: bytes) -> None:
        self.sent.append(frame)
        if self._emit_after_frames is not None and len(self.sent) >= self._emit_after_frames:
            self._release.set()

    def __aiter__(self):
        return self._iter()

    async def _iter(self):
        if self._emit_after_frames is not None:
            await self._release.wait()
        else:
            # Default: let the sender drain first, then deliver events.
            await asyncio.sleep(0)
        for event in self._events:
            yield json.dumps(event)
            await asyncio.sleep(0)


CONNECTED = {"type": "connected", "message": "ok", "config": {"sample_rate": 16_000, "chunk_size": 512}}


def transcript_event(text: str, index: int = 0) -> dict:
    return {
        "type": "transcript",
        "text": text,
        "audio_duration_ms": 1000,
        "segment_id": f"seg_{index}",
        "segment_index": index,
        "latency": 120,
    }


def build(events: list[dict], **kwargs) -> tuple[PrismaStreamingSTT, FakeStream, dict]:
    stream = FakeStream(events, emit_after_frames=kwargs.pop("emit_after_frames", None))
    captured: dict = {}

    def connect(url, **ckwargs):
        captured.update(url=url, **ckwargs)
        return stream

    kwargs.setdefault("pace_realtime", False)
    kwargs.setdefault("extra_segment_grace_s", 0)
    kwargs.setdefault("padding_margin_ms", 0)
    kwargs.setdefault("min_silence_ms", 100)
    return PrismaStreamingSTT("test-key", connect=connect, **kwargs), stream, captured


def run(stt: PrismaStreamingSTT, clip: AudioClip = CLIP, language: str = "en-IN"):
    return asyncio.run(stt.transcribe(clip, language=language))


# --- framing ---------------------------------------------------------------


def test_frames_are_exactly_1024_bytes_with_the_last_padded():
    pcm = bytes(FRAME_BYTES * 2 + 10)

    got = list(frames(pcm))
    assert len(got) == 3
    assert all(len(f) == FRAME_BYTES for f in got)
    assert got[-1][10:] == bytes(FRAME_BYTES - 10)  # padding, not audio


def test_one_second_at_16k_sends_the_expected_frame_count():
    stt, stream, _ = build([CONNECTED, transcript_event("haan")])
    run(stt)
    # Audio and silence padding are framed separately, so each pads its own last frame:
    # audio 32000 B -> 32 frames; padding (min_silence 100 ms) 3200 B -> 4 frames.
    assert len(stream.sent) == 32 + 4
    assert all(len(f) == FRAME_BYTES for f in stream.sent)


def test_silence_padding_adds_frames_sized_from_min_silence():
    # padding = min_silence_ms + margin = 300 ms = 9600 B -> 10 frames, on top of 32 audio frames.
    stt, stream, _ = build(
        [CONNECTED, transcript_event("haan")], min_silence_ms=200, padding_margin_ms=100
    )
    run(stt)
    assert len(stream.sent) == 32 + 10


# --- headers and config ----------------------------------------------------


def test_sends_documented_upgrade_headers():
    stt, _, captured = build([CONNECTED, transcript_event("ok")], min_silence_ms=250)
    run(stt, language="en-IN")

    assert captured["url"] == "wss://api.vachana.ai/stt/v3/stream"
    assert captured["additional_headers"] == {
        "x-api-key-id": "test-key",
        "lang_code": "en-IN",
        "x-sample-rate": "16000",
        "x-format": "verbatim",
        "x-min-silence-ms": "250",
    }


def test_rejects_unsupported_sample_rate_before_connecting():
    stt, stream, captured = build([CONNECTED])
    with pytest.raises(ValueError, match="8000"):
        run(stt, AudioClip(bytes(200), 22_050))
    assert captured == {}


def test_missing_key_fails_before_connecting():
    stt, _, captured = build([CONNECTED])
    stt._api_key = ""
    with pytest.raises(GnaniAPIError, match="GNANI_API_KEY"):
        run(stt)
    assert captured == {}


# --- transcripts -----------------------------------------------------------


def test_returns_transcript_and_satisfies_protocol():
    stt, _, _ = build([CONNECTED, {"type": "speech_end"}, transcript_event("sharma ji ka dedh sau udhaar")])
    result = run(stt)

    assert result.text == "sharma ji ka dedh sau udhaar"
    assert result.raw["num_segments"] == 1
    assert result.raw["segments"][0]["segment_id"] == "seg_0"
    assert isinstance(stt, STT)


def test_joins_multiple_vad_segments():
    stt, _, _ = build(
        [CONNECTED, transcript_event("sharma ji ka", 0), transcript_event("dedh sau udhaar", 1)],
        extra_segment_grace_s=0.05,
    )
    result = run(stt)
    assert result.text == "sharma ji ka dedh sau udhaar"
    assert result.raw["num_segments"] == 2


def test_tail_latency_is_measured_from_the_last_frame():
    stt, _, _ = build([CONNECTED, transcript_event("ok")])
    result = run(stt)
    assert result.latency_ms is not None
    assert result.latency_ms >= 0
    assert result.raw["tail_ms"] == result.latency_ms


def test_error_event_raises():
    stt, _, _ = build([CONNECTED, {"type": "error", "message": "unsupported sample rate"}])
    with pytest.raises(GnaniAPIError, match="unsupported sample rate"):
        run(stt)


def test_stream_closing_without_transcript_raises():
    stt, _, _ = build([CONNECTED])
    with pytest.raises(GnaniAPIError, match="without a transcript"):
        run(stt)


def test_timeout_waiting_for_transcript_raises():
    # Server sends nothing after the frames; the wait must give up, not hang.
    class SilentStream(FakeStream):
        async def _iter(self):
            await asyncio.sleep(10)
            yield json.dumps(CONNECTED)

    stream = SilentStream([])
    stt = PrismaStreamingSTT(
        "k",
        connect=lambda url, **kw: stream,
        pace_realtime=False,
        transcript_timeout_s=0.05,
        min_silence_ms=100,
        padding_margin_ms=0,
    )
    with pytest.raises(GnaniAPIError, match="no transcript within"):
        run(stt)


def test_rejects_min_silence_outside_documented_range():
    for bad in (50, 3000):
        with pytest.raises(ValueError, match="min_silence_ms"):
            PrismaStreamingSTT("k", min_silence_ms=bad)


def test_non_final_transcript_does_not_end_the_wait():
    """Undocumented partials arrive with is_final=false; only a final ends the turn."""
    partial = transcript_event("sharma ji", 0) | {"is_final": False}
    stt, _, _ = build([CONNECTED, partial, transcript_event("sharma ji ka dedh sau", 1)])
    result = run(stt)
    assert result.text == "sharma ji sharma ji ka dedh sau"
    assert result.raw["num_segments"] == 2


def test_connected_config_is_captured():
    stt, _, _ = build([CONNECTED, transcript_event("ok")])
    assert run(stt).raw["connected_config"]["chunk_size"] == 512
