"""Gnani Timbre v2.5 text-to-speech, realtime WebSocket endpoint.

Docs: https://docs.gnani.ai/api/TTS/tts-websocket
Audio arrives as base64 inside JSON messages; we decode and yield raw PCM16.
"""

import base64
import json
from collections.abc import AsyncIterator, Callable
from typing import Any

import websockets

from app.adapters.gnani_common import API_KEY_HEADER, GnaniAPIError, require_key
from app.adapters.usage import record_tts

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
        # One connection per utterance keeps this simple. Reusing a connection across turns
        # would save the handshake; that is an M4 latency experiment.
        headers = {API_KEY_HEADER: require_key(self._api_key)}
        # Billed per 10,000 characters of input text, so the cost is known up front.
        record_tts(len(text), model=self.model_id)
        async with self._connect(self._ws_url, additional_headers=headers) as ws:
            await ws.send(json.dumps(self._request(text, language, voice), ensure_ascii=False))
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
            raise GnaniAPIError("Timbre closed the connection before sending 'complete'")
