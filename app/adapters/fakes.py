"""Deterministic, offline stand-ins for the Gnani adapters. Used by tests and USE_FAKES=true."""

from collections import deque
from collections.abc import AsyncIterator, Iterable, Sequence
from typing import Any

from app.adapters.base import AudioClip, LLMReply, Message, Transcript


class FakeSTT:
    """Returns scripted transcripts in order, and records every call."""

    model_id = "fake-stt"

    def __init__(self, transcripts: Iterable[str] = ()) -> None:
        self._queue = deque(transcripts)
        self.calls: list[tuple[AudioClip, str]] = []

    async def transcribe(self, audio: AudioClip, *, language: str) -> Transcript:
        self.calls.append((audio, language))
        if not self._queue:
            raise AssertionError("FakeSTT ran out of scripted transcripts")
        return Transcript(text=self._queue.popleft(), latency_ms=0.0)


class FakeLLM:
    """Returns scripted replies in order, and records the messages and tools it was given."""

    model_id = "fake-llm"

    def __init__(self, replies: Iterable[LLMReply] = ()) -> None:
        self._queue = deque(replies)
        self.calls: list[tuple[list[Message], list[dict[str, Any]]]] = []

    async def complete(
        self, messages: Sequence[Message], *, tools: Sequence[dict[str, Any]] = ()
    ) -> LLMReply:
        self.calls.append((list(messages), list(tools)))
        if not self._queue:
            raise AssertionError("FakeLLM ran out of scripted replies")
        return self._queue.popleft()


class FakeTTS:
    """Yields silence whose length grows with the text, in fixed-size chunks."""

    model_id = "fake-tts"

    def __init__(self, sample_rate: int = 16_000, ms_per_char: int = 60, chunk_ms: int = 100) -> None:
        self.sample_rate = sample_rate
        self.ms_per_char = ms_per_char
        self.chunk_ms = chunk_ms
        self.calls: list[tuple[str, str, str]] = []

    async def synthesize(self, text: str, *, language: str, voice: str) -> AsyncIterator[bytes]:
        self.calls.append((text, language, voice))
        total_samples = self.sample_rate * self.ms_per_char * len(text) // 1000
        chunk_samples = self.sample_rate * self.chunk_ms // 1000
        while total_samples > 0:
            n = min(chunk_samples, total_samples)
            yield bytes(2 * n)
            total_samples -= n
