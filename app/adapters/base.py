"""Model-agnostic interfaces for speech-to-text, the LLM, and text-to-speech.

Everything outside app/adapters/ depends only on these types, never on a
Gnani client directly. That keeps the pipeline testable with fakes and keeps
every Gnani-specific detail in one place.
"""

import hashlib
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable


@dataclass(frozen=True)
class AudioClip:
    """Mono PCM16 little-endian audio."""

    pcm: bytes
    sample_rate: int

    def __post_init__(self) -> None:
        if len(self.pcm) % 2:
            raise ValueError("PCM16 audio must have an even number of bytes")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")

    @property
    def num_samples(self) -> int:
        return len(self.pcm) // 2

    @property
    def duration_s(self) -> float:
        return self.num_samples / self.sample_rate

    def sha256(self) -> str:
        """Content hash, used in the eval cache key (audio sha256, condition id, model id)."""
        h = hashlib.sha256()
        h.update(self.sample_rate.to_bytes(4, "little"))
        h.update(self.pcm)
        return h.hexdigest()


@dataclass(frozen=True)
class Transcript:
    text: str
    # Prisma returns no confidence score on either endpoint, so there is no confidence field.
    latency_ms: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Message:
    role: Literal["system", "user", "assistant", "tool"]
    content: str


@dataclass(frozen=True)
class ToolCall:
    """A tool call *proposed* by the LLM. The executor decides whether it runs."""

    name: str
    arguments: dict[str, Any]
    id: str | None = None


@dataclass(frozen=True)
class LLMReply:
    text: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()


@runtime_checkable
class STT(Protocol):
    model_id: str

    async def transcribe(self, audio: AudioClip, *, language: str) -> Transcript: ...


@runtime_checkable
class LLM(Protocol):
    model_id: str

    async def complete(
        self, messages: Sequence[Message], *, tools: Sequence[dict[str, Any]] = ()
    ) -> LLMReply:
        """`tools` are OpenAI-style function schemas built in agent/tools.py."""
        ...


@runtime_checkable
class TTS(Protocol):
    model_id: str
    sample_rate: int

    def synthesize(self, text: str, *, language: str, voice: str) -> AsyncIterator[bytes]:
        """Yield raw mono PCM16 chunks at `sample_rate` as soon as they are available."""
        ...
