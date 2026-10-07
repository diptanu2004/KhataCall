"""Pieces shared by the Gnani Prisma and Timbre adapters."""

import io
import wave

from app.adapters.base import AudioClip

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
