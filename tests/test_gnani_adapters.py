"""Gnani adapters against mocked transports. No network."""

import asyncio
import base64
import io
import json
import wave

import httpx
import pytest

from app.adapters.base import STT, TTS, AudioClip
from app.adapters.gnani_common import GnaniAPIError, to_wav_bytes
from app.adapters.gnani_stt import PrismaSTT
from app.adapters.gnani_tts import TimbreTTS

ONE_SECOND_8K = AudioClip(pcm=bytes(2 * 8000), sample_rate=8000)


# --- Prisma (REST) ---------------------------------------------------------


def prisma_with(handler) -> PrismaSTT:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return PrismaSTT("test-key", client=client)


def test_wav_wrapper_preserves_rate_and_samples():
    with wave.open(io.BytesIO(to_wav_bytes(ONE_SECOND_8K))) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()) == (1, 2, 8000, 8000)


def test_prisma_sends_documented_request_and_returns_transcript():
    seen: dict[str, httpx.Request] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["request"] = request
        return httpx.Response(200, json={"success": True, "request_id": "req_1", "transcript": "sharma ji ka dedh sau"})

    stt = prisma_with(handler)
    result = asyncio.run(stt.transcribe(ONE_SECOND_8K, language="en-IN"))

    request = seen["request"]
    body = request.content
    assert str(request.url) == "https://api.vachana.ai/stt/v3"
    assert request.headers["X-API-Key-ID"] == "test-key"
    assert b'name="language_code"\r\n\r\nen-IN' in body
    assert b'name="format"\r\n\r\nverbatim' in body
    assert b'name="audio_file"; filename="utterance.wav"' in body
    assert result.text == "sharma ji ka dedh sau"
    assert result.raw["request_id"] == "req_1"
    assert isinstance(stt, STT)


def test_prisma_sends_bias_list_as_json():
    seen: dict[str, bytes] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = request.content
        return httpx.Response(200, json={"success": True, "transcript": ""})

    stt = PrismaSTT("k", bias_list=["Sharma", "Gupta"], client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    asyncio.run(stt.transcribe(ONE_SECOND_8K, language="en-IN"))
    assert b'name="bias_list"\r\n\r\n["Sharma", "Gupta"]' in seen["body"]


def test_prisma_error_body_raises_with_type_and_status():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"success": False, "error": {"type": "INVALID_REQUEST_ERROR", "message": "bad audio"}})

    with pytest.raises(GnaniAPIError) as err:
        asyncio.run(prisma_with(handler).transcribe(ONE_SECOND_8K, language="en-IN"))
    assert (err.value.status, err.value.error_type) == (400, "INVALID_REQUEST_ERROR")
    assert "bad audio" in str(err.value)


def test_prisma_auth_error_detail_shape_raises():
    """Observed shape for auth failures; the docs only show the success/error shape."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401, json={"detail": {"error_code": "MISSING_API_KEY", "message": "Missing API key", "status_code": 401}}
        )

    with pytest.raises(GnaniAPIError) as err:
        asyncio.run(prisma_with(handler).transcribe(ONE_SECOND_8K, language="en-IN"))
    assert (err.value.status, err.value.error_type) == (401, "MISSING_API_KEY")
    assert "Missing API key" in str(err.value)


def test_prisma_string_detail_error_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(422, json={"detail": "Unprocessable entity"})

    with pytest.raises(GnaniAPIError, match="Unprocessable entity"):
        asyncio.run(prisma_with(handler).transcribe(ONE_SECOND_8K, language="en-IN"))


def test_prisma_non_json_error_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(502, text="Bad Gateway")

    with pytest.raises(GnaniAPIError) as err:
        asyncio.run(prisma_with(handler).transcribe(ONE_SECOND_8K, language="en-IN"))
    assert err.value.status == 502


def test_prisma_rejects_audio_over_60s_without_calling_api():
    calls = []
    stt = prisma_with(lambda r: calls.append(r) or httpx.Response(200, json={}))
    too_long = AudioClip(pcm=bytes(2 * 8000 * 61), sample_rate=8000)

    with pytest.raises(ValueError):
        asyncio.run(stt.transcribe(too_long, language="en-IN"))
    assert calls == []


def test_prisma_missing_key_fails_before_request():
    calls = []
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: calls.append(r) or httpx.Response(200)))
    with pytest.raises(GnaniAPIError, match="GNANI_API_KEY"):
        asyncio.run(PrismaSTT("", client=client).transcribe(ONE_SECOND_8K, language="en-IN"))
    assert calls == []


# --- Timbre (realtime WebSocket) ---------------------------------------------


class FakeTimbreSocket:
    def __init__(self, messages: list[dict]) -> None:
        self._messages = [json.dumps(m) for m in messages]
        self.sent: list[dict] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def send(self, data: str) -> None:
        self.sent.append(json.loads(data))

    def __aiter__(self):
        return self._iter()

    async def _iter(self):
        for m in self._messages:
            yield m


def audio_event(i: int, pcm: bytes) -> dict:
    return {"type": "audio", "data": {"chunk_index": i, "audio": base64.b64encode(pcm).decode(), "is_final": False}}


COMPLETE = {"type": "complete", "data": {"chunk_index": 99, "audio": "", "is_final": True}}


def timbre_with(messages: list[dict]) -> tuple[TimbreTTS, FakeTimbreSocket, dict]:
    socket = FakeTimbreSocket(messages)
    connect_args: dict = {}

    def connect(url, **kwargs):
        connect_args.update(url=url, **kwargs)
        return socket

    return TimbreTTS("test-key", connect=connect), socket, connect_args


def collect(tts: TimbreTTS, text: str = "haan ji") -> list[bytes]:
    async def run() -> list[bytes]:
        return [c async for c in tts.synthesize(text, language="hi-en", voice="Poorvi")]

    return asyncio.run(run())


def test_timbre_sends_documented_request():
    tts, socket, connect_args = timbre_with([COMPLETE])
    collect(tts, "Likh diya.")

    assert connect_args["url"] == "wss://api.vachana.ai/api/v1/tts"
    assert connect_args["additional_headers"] == {"X-API-Key-ID": "test-key"}
    request = socket.sent[0]
    assert (request["text"], request["voice"], request["model"], request["language"]) == (
        "Likh diya.", "Poorvi", "timbre-v2.5", "hi-en",
    )
    assert request["audio_config"]["container"] == "raw"
    assert request["audio_config"]["sample_rate"] == 16_000
    assert isinstance(tts, TTS)


def test_timbre_decodes_chunks_and_keeps_samples_aligned():
    pcm = bytes(range(10))
    # Split at odd byte offsets; output must still be whole 2-byte samples, in order.
    tts, _, _ = timbre_with([audio_event(0, pcm[:3]), audio_event(1, pcm[3:7]), audio_event(2, pcm[7:]), COMPLETE])
    chunks = collect(tts)

    assert b"".join(chunks) == pcm
    assert all(len(c) % 2 == 0 for c in chunks)


def test_timbre_ignores_unknown_event_types():
    tts, _, _ = timbre_with([{"type": "start"}, audio_event(0, b"\x01\x02"), COMPLETE])
    assert collect(tts) == [b"\x01\x02"]


def test_timbre_error_event_raises():
    tts, _, _ = timbre_with([{"type": "error", "message": "Invalid text or audio configuration."}])
    with pytest.raises(GnaniAPIError, match="Invalid text"):
        collect(tts)


def test_timbre_connection_closed_early_raises():
    tts, _, _ = timbre_with([audio_event(0, b"\x00\x00")])
    with pytest.raises(GnaniAPIError, match="complete"):
        collect(tts)


def test_timbre_rejects_out_of_range_config():
    with pytest.raises(ValueError):
        TimbreTTS("k", sample_rate=4000)
    with pytest.raises(ValueError):
        TimbreTTS("k", speed=2.0)
