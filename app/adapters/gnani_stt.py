"""Gnani Prisma v2.5 speech-to-text, REST endpoint.

Docs: https://docs.gnani.ai/api/STT/speech-to-text
REST is used for both the live loop and the eval (DECISIONS.md D-003).
"""

import json
import time
from collections.abc import Sequence

import httpx

from app.adapters.base import AudioClip, Transcript
from app.adapters.gnani_common import API_KEY_HEADER, GnaniAPIError, require_key, to_wav_bytes

MAX_DURATION_S = 60.0  # documented hard limit for REST; ideal is <= 30 s


class PrismaSTT:
    # The REST request has no model field; the docs present this endpoint as Prisma v2.5.
    model_id = "prisma-v2.5"

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://api.vachana.ai",
        fmt: str = "verbatim",
        bias_list: Sequence[str] = (),
        client: httpx.AsyncClient | None = None,
        timeout_s: float = 30.0,
    ) -> None:
        # "verbatim" keeps numbers in spoken form so nlu/numbers.py parses them (rule 7);
        # "transcribe" would apply Prisma's ITN ("dedh sau" -> "150"?). DECISIONS.md D-010.
        if fmt not in ("verbatim", "transcribe"):
            raise ValueError(f"fmt must be 'verbatim' or 'transcribe', got {fmt!r}")
        if len(bias_list) > 100:
            raise ValueError("Prisma accepts at most 100 bias_list words")
        self._api_key = api_key
        self._url = base_url.rstrip("/") + "/stt/v3"
        self._fmt = fmt
        self._bias_list = list(bias_list)
        self._client = client or httpx.AsyncClient(timeout=timeout_s)

    async def transcribe(self, audio: AudioClip, *, language: str) -> Transcript:
        if audio.duration_s > MAX_DURATION_S:
            raise ValueError(f"audio is {audio.duration_s:.1f} s; Prisma REST accepts at most {MAX_DURATION_S:.0f} s")

        data = {"language_code": language, "format": self._fmt}
        if self._bias_list:
            data["bias_list"] = json.dumps(self._bias_list, ensure_ascii=False)
        files = {"audio_file": ("utterance.wav", to_wav_bytes(audio), "audio/wav")}

        start = time.perf_counter()
        response = await self._client.post(
            self._url, headers={API_KEY_HEADER: require_key(self._api_key)}, data=data, files=files
        )
        latency_ms = (time.perf_counter() - start) * 1000

        try:
            body = response.json()
        except ValueError:
            raise GnaniAPIError(response.text[:200] or "non-JSON response", status=response.status_code) from None

        if response.status_code != 200 or not body.get("success"):
            error = body.get("error") or {}
            raise GnaniAPIError(
                error.get("message", "transcription failed"),
                status=response.status_code,
                error_type=error.get("type"),
            )
        return Transcript(text=body.get("transcript", ""), latency_ms=latency_ms, raw=body)

    async def aclose(self) -> None:
        await self._client.aclose()
