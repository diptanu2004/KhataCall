"""One turn through the session state machine, with fakes. No network."""

import asyncio
from contextlib import asynccontextmanager

import pytest

from app.adapters.base import LLMReply
from app.adapters.fakes import FailingTTS, FakeLiveSTT, FakeLLM, FakeTTS
from app.pipeline.session import COULD_NOT_HEAR, MAX_HISTORY_MESSAGES, SOMETHING_BROKE, Session
from app.protocol import CLIENT_SAMPLE_RATE, State

UTTERANCE = "sharma ji ka dedh sau udhaar likho"
REPLY = "Abhi main sirf baat kar sakta hoon."


class Recorder:
    """Captures everything the session would send to the browser."""

    def __init__(self) -> None:
        self.messages: list[dict] = []
        self.audio: list[bytes] = []

    async def send_json(self, message: dict) -> None:
        self.messages.append(message)

    async def send_bytes(self, data: bytes) -> None:
        self.audio.append(data)

    def types(self) -> list[str]:
        return [m["type"] for m in self.messages]

    def states(self) -> list[str]:
        return [m["state"] for m in self.messages if m["type"] == "state"]

    def first(self, type_: str) -> dict:
        return next(m for m in self.messages if m["type"] == type_)

    def all(self, type_: str) -> list[dict]:
        return [m for m in self.messages if m["type"] == type_]


def build(
    *, transcripts=(UTTERANCE,), replies=(REPLY,), stt=None, tts=None, llm=None
) -> tuple[Session, Recorder, FakeLiveSTT, FakeLLM, FakeTTS]:
    rec = Recorder()
    stt = stt or FakeLiveSTT(transcripts)
    llm = llm or FakeLLM([LLMReply(text=r) for r in replies])
    tts = tts or FakeTTS(sample_rate=16_000, ms_per_char=10, chunk_ms=100)
    session = Session(
        stt=stt,
        llm=llm,
        tts=tts,
        send_json=rec.send_json,
        send_bytes=rec.send_bytes,
        stt_language="en-IN",
        tts_language="hi-en",
        tts_voice="Poorvi",
        client_sample_rate=CLIENT_SAMPLE_RATE,
    )
    return session, rec, stt, llm, tts


async def one_turn(session: Session, *, audio_chunks: int = 3) -> None:
    await session.handle_json({"type": "start"})
    for _ in range(audio_chunks):
        await session.handle_audio(bytes(640))
    await session.handle_json({"type": "stop"})


# --- the happy path --------------------------------------------------------


def test_full_turn_produces_transcript_reply_and_audio():
    session, rec, stt, llm, tts = build()

    asyncio.run(one_turn(session))

    assert rec.states() == ["LISTENING", "TRANSCRIBING", "THINKING", "SPEAKING", "LISTENING"]
    assert rec.first("transcript")["text"] == UTTERANCE
    assert rec.first("agent_text")["text"] == REPLY
    assert rec.audio, "no TTS audio was forwarded"
    assert tts.calls == [(REPLY, "hi-en", "Poorvi")]
    assert session.state is State.LISTENING


def test_microphone_audio_reaches_the_stt_session():
    session, _, stt, _, _ = build()
    asyncio.run(one_turn(session, audio_chunks=5))
    assert len(stt.sessions[0].pushed) == 5 * 640
    assert stt.sessions[0].finished


def test_stt_socket_opens_on_button_press_not_on_release():
    """The handshake costs ~400 ms, so it must overlap with the owner speaking (D-016)."""
    session, _, stt, _, _ = build()

    asyncio.run(session.handle_json({"type": "start"}))

    assert stt.opened_with == [("en-IN", CLIENT_SAMPLE_RATE)]
    asyncio.run(session.aclose())


def test_stt_socket_is_closed_after_each_turn():
    """Teardown is backgrounded (it costs ~1.3 s), so wait for it explicitly."""
    session, _, stt, _, _ = build(transcripts=(UTTERANCE, "haan"), replies=(REPLY, "Theek hai."))

    async def two_turns():
        await one_turn(session)
        await one_turn(session)
        await session.aclose()

    asyncio.run(two_turns())
    assert stt.closed == 2


def test_turn_does_not_block_on_closing_the_stt_socket():
    """Closing a Prisma socket was measured at ~1.3 s; it must not delay the LLM call."""

    class SlowClosingSTT(FakeLiveSTT):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.close_started = False

        @asynccontextmanager
        async def live(self, *, language: str, sample_rate: int):
            async with super().live(language=language, sample_rate=sample_rate) as s:
                try:
                    yield s
                finally:
                    self.close_started = True
                    await asyncio.sleep(0.5)

    stt = SlowClosingSTT([UTTERANCE])
    session, rec, _, _, _ = build(stt=stt)

    async def run():
        start = asyncio.get_running_loop().time()
        await one_turn(session)
        elapsed = asyncio.get_running_loop().time() - start
        await session.aclose()
        return elapsed

    elapsed = asyncio.run(run())
    assert stt.close_started, "the socket should have been released"
    assert elapsed < 0.4, f"turn waited on teardown ({elapsed:.2f}s)"
    assert rec.first("latency")["stt_ms"] < 400


def test_audio_frames_bracket_the_binary_stream():
    session, rec, _, _, tts = build()
    asyncio.run(one_turn(session))

    types = rec.types()
    assert types.index("audio_start") < types.index("audio_end")
    assert rec.first("audio_start")["sample_rate"] == tts.sample_rate


def test_transcript_is_sent_before_the_llm_is_called():
    """The owner should see what was heard while the agent is still thinking."""
    session, rec, _, _, _ = build()
    asyncio.run(one_turn(session))

    types = rec.types()
    assert types.index("transcript") < types.index("agent_text")


# --- latency ---------------------------------------------------------------


def test_latency_breakdown_is_reported_and_adds_up():
    session, rec, _, _, _ = build()
    asyncio.run(one_turn(session))

    latency = rec.first("latency")
    for key in ("stt_ms", "llm_ms", "tts_first_byte_ms", "total_ms"):
        assert latency[key] is not None, key
        assert latency[key] >= 0
    stages = latency["stt_ms"] + latency["llm_ms"] + latency["tts_first_byte_ms"]
    # Total is measured end-to-end, so it is the sum plus a little bookkeeping.
    assert latency["total_ms"] >= stages
    assert latency["total_ms"] == pytest.approx(stages, abs=50)


def test_latency_is_sent_once_per_turn():
    session, rec, _, _, _ = build()
    asyncio.run(one_turn(session))
    assert len(rec.all("latency")) == 1


# --- conversation memory ---------------------------------------------------


def test_history_accumulates_and_is_passed_to_the_llm():
    session, _, _, llm, _ = build(
        transcripts=(UTTERANCE, "aur Gupta aunty ka?"), replies=(REPLY, "Woh bhi abhi nahi."),
    )
    asyncio.run(one_turn(session))
    asyncio.run(one_turn(session))

    second_call_messages, _ = llm.calls[1]
    contents = [m.content for m in second_call_messages]
    assert contents[0].startswith("Tum \"KhataCall\" ho")  # system prompt stays first
    assert UTTERANCE in contents
    assert REPLY in contents
    assert "aur Gupta aunty ka?" in contents


def test_history_is_capped_but_keeps_the_system_prompt():
    turns = 10
    session, _, _, llm, _ = build(
        transcripts=tuple(f"sawaal {i}" for i in range(turns)),
        replies=tuple(f"jawaab {i}" for i in range(turns)),
    )
    for _ in range(turns):
        asyncio.run(one_turn(session))

    assert len(session.history) <= MAX_HISTORY_MESSAGES + 1
    assert session.history[0].role == "system"
    # Oldest turns are dropped, newest kept.
    assert "sawaal 0" not in [m.content for m in session.history]
    assert "jawaab 9" in [m.content for m in session.history]


# --- failure handling (HLD 13) --------------------------------------------


def test_empty_transcript_re_asks_without_calling_the_llm():
    session, rec, _, llm, _ = build(transcripts=("   ",))

    asyncio.run(one_turn(session))

    assert llm.calls == [], "an empty transcript must not spend an LLM call"
    assert rec.first("agent_text")["text"] == COULD_NOT_HEAR
    assert session.state is State.LISTENING


def test_stt_failure_is_reported_and_spoken():
    session, rec, _, llm, _ = build(stt=FakeLiveSTT(fail_with=RuntimeError("no transcript")))

    asyncio.run(one_turn(session))

    error = rec.first("error")
    assert "no transcript" in error["message"]
    assert error["spoken"] is True
    assert rec.first("agent_text")["text"] == COULD_NOT_HEAR
    assert llm.calls == []
    assert session.state is State.LISTENING


def test_llm_failure_is_reported_and_spoken():
    session, rec, _, _, _ = build(llm=FakeLLM([]))  # empty script raises

    asyncio.run(one_turn(session))

    assert "Evon failed" in rec.first("error")["message"]
    assert rec.first("agent_text")["text"] == SOMETHING_BROKE
    assert session.state is State.LISTENING


def test_blank_llm_reply_falls_back_to_a_spoken_apology():
    session, rec, _, _, _ = build(replies=("",))
    asyncio.run(one_turn(session))
    assert rec.first("agent_text")["text"] == SOMETHING_BROKE


def test_tts_failure_still_shows_the_text_and_recovers():
    session, rec, _, _, _ = build(tts=FailingTTS(chunks_before_failure=1))

    asyncio.run(one_turn(session))

    # The reply was already sent as text, so the owner can read what we could not say.
    assert rec.first("agent_text")["text"] == REPLY
    assert "Timbre failed" in rec.first("error")["message"]
    assert rec.all("audio_end"), "the audio stream must still be closed"
    assert session.state is State.LISTENING


# --- out-of-order and config ----------------------------------------------


def test_audio_outside_a_turn_is_ignored():
    session, _, stt, _, _ = build()
    asyncio.run(session.handle_audio(bytes(640)))  # before any start
    assert stt.sessions == []


def test_stop_without_start_is_ignored():
    session, rec, _, llm, _ = build()
    asyncio.run(session.handle_json({"type": "stop"}))
    assert rec.messages == []
    assert llm.calls == []


def test_start_during_a_turn_is_ignored():
    session, _, stt, _, _ = build()

    async def run():
        await session.handle_json({"type": "start"})
        session.state = State.THINKING  # pretend a turn is in flight
        await session.handle_json({"type": "start"})

    asyncio.run(run())
    assert len(stt.opened_with) == 1


def test_phone_mode_request_is_acknowledged_as_not_built():
    session, rec, _, _, _ = build()
    asyncio.run(session.handle_json({"type": "config", "phone_mode": True, "snr_db": 10}))

    assert "M3" in rec.first("notice")["message"]
    assert session.config.phone_mode is True
    assert session.config.snr_db == 10


def test_config_without_phone_mode_is_silent():
    session, rec, _, _, _ = build()
    asyncio.run(session.handle_json({"type": "config", "phone_mode": False}))
    assert rec.messages == []


def test_unknown_message_type_raises():
    session, _, _, _, _ = build()
    with pytest.raises(Exception):
        asyncio.run(session.handle_json({"type": "nonsense"}))
