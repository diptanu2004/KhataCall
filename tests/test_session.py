"""One turn through the session state machine, with fakes. No network."""

import asyncio
import base64
from contextlib import asynccontextmanager

import numpy as np
import pytest

from app.adapters.base import LLMReply
from app.adapters.fakes import FailingTTS, FakeLiveSTT, FakeLLM, FakeSTT, FakeTTS
from app.channel.phone import PhoneLine
from app.pipeline.session import (
    COULD_NOT_HEAR,
    MAX_HISTORY_MESSAGES,
    NOT_UNDERSTOOD,
    SOMETHING_BROKE,
    Session,
    unsafe_to_say,
)
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
    *, transcripts=(UTTERANCE,), replies=(REPLY,), stt=None, tts=None, llm=None, batch_stt=None, phone_line=None
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
        batch_stt=batch_stt,
        phone_line=phone_line,
    )
    return session, rec, stt, llm, tts


# 20 browser chunks of 20 ms: 0.4 s, comfortably more than a tap (MIN_UTTERANCE_S).
async def one_turn(session: Session, *, audio_chunks: int = 20) -> None:
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
    asyncio.run(one_turn(session, audio_chunks=15))
    assert len(stt.sessions[0].pushed) == 15 * 640
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


def test_evon_sees_only_this_utterance_but_the_session_remembers_the_conversation():
    """D-051: with the history, real Evon answered balances from memory instead of calling
    get_balance (15 of 15), once with another customer's figure. Without it: 15 of 15 calls."""
    session, _, _, llm, _ = build(
        transcripts=(UTTERANCE, "aur Gupta aunty ka?"), replies=(REPLY, "Woh bhi abhi nahi."),
    )
    asyncio.run(one_turn(session))
    asyncio.run(one_turn(session))

    second_call_messages, _ = llm.calls[1]
    assert [m.role for m in second_call_messages] == ["system", "user"]
    assert second_call_messages[0].content.startswith("Tum \"KhataCall\" ho")
    assert second_call_messages[1].content == "aur Gupta aunty ka?"
    # The conversation is still kept: the name check reads the last few user turns.
    contents = [m.content for m in session.history]
    assert UTTERANCE in contents and REPLY in contents and "aur Gupta aunty ka?" in contents


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


def test_stop_without_start_is_ignored_but_the_page_is_told_where_things_stand():
    """A start that failed (Prisma unreachable) leaves nothing to stop. The page still
    gets a state, so it does not wait for a turn that is never coming (D-047)."""
    session, rec, _, llm, _ = build()
    asyncio.run(session.handle_json({"type": "stop"}))
    assert rec.types() == ["state"]
    assert llm.calls == []


def test_a_tap_is_not_transcribed_and_costs_nothing():
    """Prisma waits 15 s for speech that never comes; a tap gets a hint instead (D-047)."""
    session, rec, stt, llm, tts = build()
    asyncio.run(one_turn(session, audio_chunks=3))  # 60 ms

    assert stt.sessions[0].finished is False
    assert llm.calls == [] and tts.calls == []
    assert "hold the button" in rec.first("notice")["message"]
    assert rec.states() == ["LISTENING", "LISTENING"]


def test_start_during_a_turn_is_ignored():
    session, _, stt, _, _ = build()

    async def run():
        await session.handle_json({"type": "start"})
        session.state = State.THINKING  # pretend a turn is in flight
        await session.handle_json({"type": "start"})

    asyncio.run(run())
    assert len(stt.opened_with) == 1


# --- phone-line mode -------------------------------------------------------

NOISE_BANK = {"freesound_1": (0.05 * np.random.default_rng(0).standard_normal(16_000 * 5)).astype(np.float32)}


def speech_chunks(seconds: float = 0.6, chunk: int = 640) -> list[bytes]:
    """A tone standing in for speech, as the browser would send it: PCM16 at 16 kHz."""
    t = np.arange(int(seconds * CLIENT_SAMPLE_RATE)) / CLIENT_SAMPLE_RATE
    pcm = (0.3 * np.sin(2 * np.pi * 300 * t) * 32767).astype("<i2").tobytes()
    return [pcm[i : i + chunk] for i in range(0, len(pcm), chunk)]


def phone_build(bank=NOISE_BANK, transcripts=(UTTERANCE,)):
    batch = FakeSTT(transcripts)
    session, rec, stt, llm, tts = build(batch_stt=batch, phone_line=PhoneLine(bank))
    return session, rec, stt, batch


async def phone_turn(session: Session, chunks: list[bytes], **config) -> None:
    await session.handle_json({"type": "config", "phone_mode": True, **config})
    await session.handle_json({"type": "start"})
    for c in chunks:
        await session.handle_audio(c)
    await session.handle_json({"type": "stop"})


def test_phone_mode_is_refused_where_the_server_cannot_do_it():
    session, rec, _, _, _ = build()
    asyncio.run(session.handle_json({"type": "config", "phone_mode": True, "snr_db": 10}))

    assert "not available" in rec.first("notice")["message"]
    assert session.config.phone_mode is False


def test_a_phone_turn_degrades_the_whole_utterance_and_uses_rest():
    """The eval's path exactly: no live socket, one 8 kHz clip to Prisma REST."""
    session, rec, live, batch = phone_build()
    chunks = speech_chunks(0.6)

    asyncio.run(phone_turn(session, chunks, snr_db=10, loss_pct=5))

    assert live.opened_with == [], "phone mode must not open the streaming socket"
    clip, language = batch.calls[0]
    assert (clip.sample_rate, language) == (8_000, "en-IN")
    assert clip.num_samples == len(b"".join(chunks)) // 2 // 2  # 16 kHz in, 8 kHz out
    assert rec.first("transcript")["text"] == UTTERANCE
    assert rec.states() == ["LISTENING", "TRANSCRIBING", "THINKING", "SPEAKING", "LISTENING"]


def test_the_degraded_audio_is_sent_back_before_the_transcript():
    session, rec, _, batch = phone_build()
    asyncio.run(phone_turn(session, speech_chunks(0.6), snr_db=5, loss_pct=1))

    types = rec.types()
    assert types.index("heard_audio") < types.index("transcript")
    heard = rec.first("heard_audio")
    assert base64.b64decode(heard["pcm_b64"]) == batch.calls[0][0].pcm
    assert (heard["sample_rate"], heard["snr_db"], heard["loss_pct"], heard["noise"]) == (8_000, 5, 1, "freesound_1")


def test_a_tap_in_phone_mode_is_not_sent_to_prisma():
    session, rec, _, batch = phone_build(transcripts=())
    asyncio.run(phone_turn(session, speech_chunks(0.05)))

    assert batch.calls == []
    assert any("hold the button" in m["message"] for m in rec.all("notice"))
    assert not rec.all("agent_text")


def test_without_imported_noise_the_line_runs_noise_free():
    session, rec, _, batch = phone_build(bank={})
    asyncio.run(phone_turn(session, speech_chunks(0.6), snr_db=5))

    assert any("No shop noise" in m["message"] for m in rec.all("notice"))
    assert session.config.snr_db is None
    assert rec.first("heard_audio")["noise"] is None
    assert len(batch.calls) == 1


def test_turning_phone_mode_off_goes_back_to_streaming():
    session, rec, live, batch = phone_build()
    live._queue.append(UTTERANCE)  # one scripted transcript for the streaming turn

    async def run():
        await phone_turn(session, speech_chunks(0.6))
        await session.handle_json({"type": "config", "phone_mode": False})
        await one_turn(session)

    asyncio.run(run())
    assert len(batch.calls) == 1
    assert len(live.opened_with) == 1


def test_config_without_phone_mode_is_silent():
    session, rec, _, _, _ = build()
    asyncio.run(session.handle_json({"type": "config", "phone_mode": False}))
    assert rec.messages == []


def test_unknown_message_type_raises():
    session, _, _, _, _ = build()
    with pytest.raises(Exception):
        asyncio.run(session.handle_json({"type": "nonsense"}))


# --- the ledger panel -------------------------------------------------------


def test_the_opening_ledger_is_pushed_on_connect():
    """The shop's book should be on screen before anyone speaks."""
    from app.ledger.db import connect
    from app.ledger.repo import Ledger

    ledger = Ledger(connect(":memory:"))
    sharma = ledger.add_customer("Sharma ji")
    ledger.add_credit(sharma.id, 15_000)

    rec = Recorder()
    session = Session(
        stt=FakeLiveSTT([UTTERANCE]),
        llm=FakeLLM([]),
        tts=FakeTTS(),
        send_json=rec.send_json,
        send_bytes=rec.send_bytes,
        stt_language="en-IN",
        tts_language="hi-en",
        tts_voice="Poorvi",
        client_sample_rate=CLIENT_SAMPLE_RATE,
        ledger=ledger,
    )

    asyncio.run(session.send_opening_ledger())

    update = rec.first("ledger_update")
    assert update["customers"] == [{"name": "Sharma ji", "balance_paise": 15_000}]


# --- warming Evon's connection (D-046) ---------------------------------------


class WarmableLLM(FakeLLM):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.warmed = 0

    async def warm(self) -> None:
        self.warmed += 1


def test_push_to_talk_warms_the_llm_connection_while_the_owner_speaks():
    llm = WarmableLLM([LLMReply(text=REPLY), LLMReply(text=REPLY)])
    session, rec, *_ = build(transcripts=(UTTERANCE, UTTERANCE), llm=llm)

    async def two_turns():
        await one_turn(session)
        await asyncio.sleep(0)  # the fakes never yield; a real turn gives the warm-up time to finish
        await one_turn(session)
        await session.aclose()

    asyncio.run(two_turns())
    assert llm.warmed == 2
    assert rec.first("agent_text")["text"] == REPLY


def test_an_llm_without_warm_is_left_alone():
    session, rec, *_ = build()
    asyncio.run(one_turn(session))
    assert rec.first("agent_text")["text"] == REPLY


# --- Evon's own words, with the ledger on (D-051) --------------------------------


@pytest.mark.parametrize(
    "said, utterance",
    [
        ("Sharma ji ke khate me kitna h?", "Sharma ji ke khate me kitna h?"),  # Nov 10, on the phone
        ("Sharma ji ka ab 350 rupaye baaki hai.", "Sharma ji ke khate me kitna h?"),  # Mehra ji's figure
        ("Sharma ji ka ab paanch sau rupaye baaki hai.", "Sharma ji ka kitna baaki hai?"),
    ],
)
def test_evon_never_speaks_a_figure_or_the_owners_own_words(said, utterance):
    assert unsafe_to_say(said, utterance) is not None


@pytest.mark.parametrize(
    "said", ["Namaste! Boliye, kya likhna hai?", "Kiska khata? Naam boliye.", "Kitne rupaye?",
             "Main sirf dukaan ka hisaab sambhalta hoon."],
)
def test_greetings_and_questions_back_are_spoken(said):
    assert unsafe_to_say(said, "Sharma ji ka kitna baaki hai?") is None


def test_with_the_ledger_on_a_balance_from_evons_memory_is_not_spoken():
    from app.ledger.db import connect
    from app.ledger.repo import Ledger

    ledger = Ledger(connect(":memory:"))
    ledger.add_customer("Sharma ji")
    rec = Recorder()
    session = Session(
        stt=FakeLiveSTT(["Sharma ji ke khate me kitna h?"]),
        llm=FakeLLM([LLMReply(text="Sharma ji ka ab 350 rupaye baaki hai.")]),
        tts=FakeTTS(), send_json=rec.send_json, send_bytes=rec.send_bytes,
        stt_language="en-IN", tts_language="hi-en", tts_voice="Poorvi",
        client_sample_rate=CLIENT_SAMPLE_RATE, ledger=ledger,
    )
    asyncio.run(one_turn(session))
    assert rec.first("agent_text")["text"] == NOT_UNDERSTOOD
