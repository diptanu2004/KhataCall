"""Deterministic, offline stand-ins for the Gnani adapters. Used by tests and USE_FAKES=true."""

from collections import deque
from collections.abc import AsyncIterator, Iterable, Sequence
from contextlib import asynccontextmanager
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


class FakeLiveTranscription:
    """One utterance on a FakeLiveSTT. Records pushed audio."""

    def __init__(self, transcript: str, *, latency_ms: float = 0.0, fail_with: Exception | None = None) -> None:
        self._transcript = transcript
        self._latency_ms = latency_ms
        self._fail_with = fail_with
        self.pushed = bytearray()
        self.finished = False

    async def push(self, pcm: bytes) -> None:
        self.pushed += pcm

    async def finish(self) -> Transcript:
        self.finished = True
        if self._fail_with is not None:
            raise self._fail_with
        return Transcript(text=self._transcript, latency_ms=self._latency_ms)


class FakeLiveSTT:
    """Streaming stand-in for PrismaStreamingSTT: scripted transcripts, one per `live()`."""

    model_id = "fake-live-stt"

    def __init__(
        self, transcripts: Iterable[str] = (), *, fail_with: Exception | None = None
    ) -> None:
        self._queue = deque(transcripts)
        self._fail_with = fail_with
        self.sessions: list[FakeLiveTranscription] = []
        self.opened_with: list[tuple[str, int]] = []
        self.closed = 0

    @asynccontextmanager
    async def live(self, *, language: str, sample_rate: int) -> AsyncIterator[FakeLiveTranscription]:
        self.opened_with.append((language, sample_rate))
        if not self._queue and self._fail_with is None:
            raise AssertionError("FakeLiveSTT ran out of scripted transcripts")
        session = FakeLiveTranscription(
            self._queue.popleft() if self._queue else "", fail_with=self._fail_with
        )
        self.sessions.append(session)
        try:
            yield session
        finally:
            self.closed += 1


class FailingTTS:
    """Raises partway through synthesis, to exercise the error path."""

    model_id = "failing-tts"

    def __init__(self, sample_rate: int = 16_000, chunks_before_failure: int = 0) -> None:
        self.sample_rate = sample_rate
        self._chunks = chunks_before_failure

    async def synthesize(self, text: str, *, language: str, voice: str) -> AsyncIterator[bytes]:
        for _ in range(self._chunks):
            yield bytes(320)
        raise RuntimeError("synthesis failed")
