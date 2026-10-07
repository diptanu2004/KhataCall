import asyncio

import pytest

from app.adapters.base import LLM, STT, TTS, AudioClip, LLMReply, Message, ToolCall
from app.adapters.fakes import FakeLLM, FakeSTT, FakeTTS


def test_audio_clip_duration_from_sample_count():
    clip = AudioClip(pcm=bytes(2 * 8000), sample_rate=8000)
    assert clip.num_samples == 8000
    assert clip.duration_s == 1.0


def test_audio_clip_rejects_odd_byte_count():
    with pytest.raises(ValueError):
        AudioClip(pcm=b"\x00\x00\x00", sample_rate=16_000)


def test_audio_clip_hash_depends_on_sample_rate():
    pcm = bytes(320)
    assert AudioClip(pcm, 8000).sha256() != AudioClip(pcm, 16_000).sha256()
    assert AudioClip(pcm, 8000).sha256() == AudioClip(pcm, 8000).sha256()


def test_fakes_satisfy_protocols():
    assert isinstance(FakeSTT(), STT)
    assert isinstance(FakeLLM(), LLM)
    assert isinstance(FakeTTS(), TTS)


def test_fake_stt_returns_scripted_transcripts_in_order():
    stt = FakeSTT(["sharma ji ka dedh sau udhaar likho", "haan"])
    clip = AudioClip(bytes(320), 16_000)

    first = asyncio.run(stt.transcribe(clip, language="en-IN"))
    second = asyncio.run(stt.transcribe(clip, language="en-IN"))

    assert (first.text, second.text) == ("sharma ji ka dedh sau udhaar likho", "haan")
    assert [lang for _, lang in stt.calls] == ["en-IN", "en-IN"]


def test_fake_stt_fails_loudly_when_script_runs_out():
    with pytest.raises(AssertionError):
        asyncio.run(FakeSTT().transcribe(AudioClip(b"", 16_000), language="en-IN"))


def test_fake_llm_records_messages_and_tools():
    reply = LLMReply(tool_calls=(ToolCall("add_credit", {"customer_name": "Sharma ji", "amount_spoken": "dedh sau"}),))
    llm = FakeLLM([reply])
    tools = [{"type": "function", "function": {"name": "add_credit"}}]

    got = asyncio.run(llm.complete([Message("user", "sharma ji ka dedh sau udhaar likho")], tools=tools))

    assert got is reply
    messages, seen_tools = llm.calls[0]
    assert messages[0].content == "sharma ji ka dedh sau udhaar likho"
    assert seen_tools == tools


def test_fake_tts_audio_length_scales_with_text():
    tts = FakeTTS(sample_rate=16_000, ms_per_char=50, chunk_ms=100)

    async def collect() -> list[bytes]:
        return [c async for c in tts.synthesize("haan ji", language="hi-en", voice="Nalini")]

    chunks = asyncio.run(collect())

    total_samples = sum(len(c) for c in chunks) // 2
    assert total_samples == 16_000 * 50 * len("haan ji") // 1000
    assert all(len(c) <= 2 * 1600 for c in chunks)
