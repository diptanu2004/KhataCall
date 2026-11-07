"""Per-connection turn state machine. See docs/HLD.md §5 and §6.

Half-duplex and push-to-talk: one turn at a time, driven by `start` / audio /
`stop` from the browser.

    IDLE -> LISTENING -> TRANSCRIBING -> THINKING -> [CONFIRMING] -> SPEAKING -> LISTENING

A write that Evon proposes is read back (CONFIRMING), and the owner's "haan"
on the next turn commits it without asking Evon again (D-006). Turns are
handled one at a time: audio pressed during a turn waits in the socket and is
handled when the turn ends.

The session is deliberately ignorant of FastAPI: it takes two send callables
and the three adapters, so the whole turn can be tested with fakes.
"""

import asyncio
import base64
import logging
import time
import zlib
from collections.abc import Awaitable, Callable, Sequence
from contextlib import AsyncExitStack
from typing import Any

import numpy as np

from app.adapters.base import LLM, STT, TTS, LLMReply, Message, Transcript
from app.agent.executor import Executor, Outcome
from app.agent.prompts import SYSTEM_PROMPT
from app.agent.tools import TOOL_SCHEMAS
from app.channel.phone import MIN_UTTERANCE_S, PhoneLine
from app.ledger.repo import Ledger
from app.protocol import (
    AgentTextMessage,
    AudioEndMessage,
    AudioStartMessage,
    ConfigMessage,
    ErrorMessage,
    HeardAudioMessage,
    LatencyMessage,
    LedgerUpdateMessage,
    NoticeMessage,
    ServerMessage,
    State,
    StateMessage,
    TranscriptMessage,
    client_message,
)

log = logging.getLogger(__name__)

# Spoken when something fails. Short, because it is synthesised on the critical path.
COULD_NOT_HEAR = "Maaf kijiye, phir se boliye?"
SOMETHING_BROKE = "Maaf kijiye, kuch gadbad ho gayi. Phir se boliye?"

# Turns of dialogue kept as context. Evon has a 128K window, but a long history costs
# latency on every turn and a kirana exchange rarely needs more.
MAX_HISTORY_MESSAGES = 12
# User turns the name check may find a customer's name in (fix 5).
RECENT_USER_TURNS = 3


class LiveSTT:
    """What `session.py` needs from a streaming STT adapter.

    Structural, not inherited: `PrismaStreamingSTT` satisfies it, and so does
    the fake in tests.
    """

    def live(self, *, language: str, sample_rate: int) -> Any: ...


class Session:
    def __init__(
        self,
        *,
        stt: LiveSTT,
        llm: LLM,
        tts: TTS,
        send_json: Callable[[dict], Awaitable[None]],
        send_bytes: Callable[[bytes], Awaitable[None]],
        stt_language: str,
        tts_language: str,
        tts_voice: str,
        client_sample_rate: int,
        system_prompt: str = SYSTEM_PROMPT,
        ledger: Ledger | None = None,
        session_id: str | None = None,
        batch_stt: STT | None = None,
        phone_line: PhoneLine | None = None,
    ) -> None:
        self._stt = stt
        self._llm = llm
        self._tts = tts
        self._send_json = send_json
        self._send_bytes = send_bytes
        self._stt_language = stt_language
        self._tts_language = tts_language
        self._tts_voice = tts_voice
        self._client_sample_rate = client_sample_rate
        self._history: list[Message] = [Message("system", system_prompt)]
        self._ledger = ledger
        # Without a ledger the agent can still talk, which is what USE_FAKES does.
        self._executor = Executor(ledger, session_id=session_id) if ledger is not None else None
        self._tools = TOOL_SCHEMAS if ledger is not None else ()
        # Phone-line mode: the utterance is buffered, degraded and sent to Prisma
        # REST in one piece, exactly as the eval did (app/channel/phone.py).
        self._batch_stt = batch_stt
        self._phone_line = phone_line
        self._session_id = session_id or "local"
        self._buffer: bytearray | None = None  # set for a phone-mode turn
        self._phone_turns = 0

        self.state = State.IDLE
        self.config = ConfigMessage()
        self._stack: AsyncExitStack | None = None
        self._live: Any = None
        self._closing: set[asyncio.Task] = set()
        self._warming: asyncio.Task | None = None
        # Diagnostics for one turn: how the browser's audio actually arrived.
        self._pushes = 0
        self._pushed_bytes = 0
        self._first_push_at: float | None = None
        self._last_push_at: float | None = None

    # --- plumbing ---

    async def _send(self, message: ServerMessage) -> None:
        await self._send_json(message.model_dump(mode="json"))

    async def _set_state(self, state: State) -> None:
        self.state = state
        await self._send(StateMessage(state=state))

    def _remember(self, role: str, content: str) -> None:
        self._history.append(Message(role, content))  # type: ignore[arg-type]
        if len(self._history) > MAX_HISTORY_MESSAGES + 1:  # +1 for the system prompt
            self._history = [self._history[0], *self._history[-MAX_HISTORY_MESSAGES:]]

    # --- inbound ---

    async def handle_json(self, raw: dict) -> None:
        message = client_message.validate_python(raw)
        match message:
            case ConfigMessage():
                await self._handle_config(message)
            case _ if message.type == "start":
                await self._handle_start()
            case _ if message.type == "stop":
                await self._handle_stop()

    async def handle_audio(self, pcm: bytes) -> None:
        """Forward microphone audio to Prisma as it arrives (or buffer it, in phone mode)."""
        if self.state is not State.LISTENING or (self._live is None and self._buffer is None):
            return  # frames outside a turn, e.g. a late flush after release
        now = time.perf_counter()
        if self._first_push_at is None:
            self._first_push_at = now
        self._last_push_at = now
        self._pushes += 1
        self._pushed_bytes += len(pcm)
        if self._buffer is not None:
            self._buffer.extend(pcm)
        else:
            await self._live.push(pcm)

    @property
    def phone_mode_available(self) -> bool:
        return self._batch_stt is not None and self._phone_line is not None

    async def _handle_config(self, message: ConfigMessage) -> None:
        if message.phone_mode and not self.phone_mode_available:
            message = message.model_copy(update={"phone_mode": False})
            await self._send(NoticeMessage(message="Phone-line mode is not available on this server."))
        if message.phone_mode and message.snr_db is not None and not self._phone_line.has_noise:
            message = message.model_copy(update={"snr_db": None})
            await self._send(
                NoticeMessage(message="No shop noise imported (docs/NOISE.md); the line is noise-free.")
            )
        self.config = message
        if self._executor is not None:
            # No STT confidence exists to key off, so phone mode always spells the
            # amount out digit by digit (D-004).
            self._executor.spell_amounts = message.phone_mode
        if message.phone_mode:
            noise = f"shop noise at {message.snr_db:g} dB SNR" if message.snr_db is not None else "no shop noise"
            loss = message.loss_pct or 0
            await self._send(
                NoticeMessage(message=f"Phone line on: mobile call (GSM + G.711), {noise}, {loss:g}% packet loss.")
            )

    async def _handle_start(self) -> None:
        if self.state in (State.TRANSCRIBING, State.THINKING, State.SPEAKING):
            log.warning("start ignored while %s", self.state)
            return
        if self.config.phone_mode:
            # No socket is opened: the whole utterance is needed before it can be degraded.
            self._release_stt()
            self._buffer = bytearray()
        else:
            self._buffer = None
            await self._open_stt()
        self._warm_llm()
        # Every turn ends in speech, so Timbre's socket opens now, behind the owner's own
        # words. Opened at release instead, a "haan" turn (~0.2 s of STT) could not hide a
        # slow handshake (D-046). Timbre bills characters, so an open socket costs nothing.
        if (prepare := getattr(self._tts, "prepare", None)) is not None:
            await prepare()
        self._pushes = self._pushed_bytes = 0
        self._first_push_at = self._last_push_at = None
        await self._set_state(State.LISTENING)

    def _warm_llm(self) -> None:
        """Open Evon's connection while the owner speaks, as the Prisma socket is (D-046)."""
        warm = getattr(self._llm, "warm", None)
        if warm is None or (self._warming is not None and not self._warming.done()):
            return
        self._warming = asyncio.create_task(warm())

    async def _open_stt(self) -> None:
        """Open the Prisma socket on button-press, so its handshake overlaps with speech."""
        self._release_stt()
        self._stack = AsyncExitStack()
        self._live = await self._stack.enter_async_context(
            self._stt.live(language=self._stt_language, sample_rate=self._client_sample_rate)
        )

    def _release_stt(self) -> None:
        """Hand the socket off to be closed in the background.

        Closing a WebSocket is a handshake with the server, and it was measured at
        ~1.3 s. Awaiting it here would add that to every turn for no benefit: the
        transcript is already in hand, so the LLM call should start immediately.
        """
        stack, self._stack, self._live = self._stack, None, None
        if stack is None:
            return
        task = asyncio.create_task(stack.aclose())
        self._closing.add(task)
        task.add_done_callback(self._closing.discard)

    # --- the turn ---

    async def _transcribe_phone_line(self) -> Transcript:
        """Degrade the buffered utterance like the eval's mob_* conditions, then Prisma REST."""
        pcm, self._buffer = bytes(self._buffer or b""), None
        if len(pcm) / 2 / self._client_sample_rate < MIN_UTTERANCE_S:
            return Transcript(text="")  # a tap, not speech: nothing worth paying to transcribe
        self._phone_turns += 1
        rng = np.random.default_rng(zlib.crc32(f"{self._session_id}:{self._phone_turns}".encode()))
        result = await asyncio.to_thread(
            self._phone_line.apply,
            pcm,
            rate=self._client_sample_rate,
            snr_db=self.config.snr_db,
            loss_pct=self.config.loss_pct or 0.0,
            rng=rng,
        )
        await self._send(
            HeardAudioMessage(
                sample_rate=result.clip.sample_rate,
                pcm_b64=base64.b64encode(result.clip.pcm).decode("ascii"),
                snr_db=self.config.snr_db if result.noise else None,
                loss_pct=self.config.loss_pct or 0.0,
                noise=result.noise,
            )
        )
        return await self._batch_stt.transcribe(result.clip, language=self._stt_language)

    async def _handle_stop(self) -> None:
        """Push-to-talk released. This is end of speech: the latency clock starts here."""
        if self.state is not State.LISTENING or (self._live is None and self._buffer is None):
            log.warning("stop ignored while %s", self.state)
            # Say where things stand, e.g. after a start that failed to open Prisma, so
            # the page does not wait for a turn that is never coming.
            await self._set_state(self.state)
            return

        speech_end = time.perf_counter()
        if self._pushed_bytes / 2 / self._client_sample_rate < MIN_UTTERANCE_S:
            # A tap, not speech. Nothing is worth paying to transcribe, and Prisma would
            # wait out its 15 s timeout for speech that never comes (D-047).
            self._buffer = None
            self._release_stt()
            await self._send(NoticeMessage(message="Too short: hold the button while you speak."))
            await self._set_state(State.LISTENING)
            return
        await self._set_state(State.TRANSCRIBING)

        try:
            if self._buffer is not None:
                transcript = await self._transcribe_phone_line()
            else:
                transcript = await self._live.finish()
            # Measured before teardown: closing the socket is not part of recognition.
            stt_ms = (time.perf_counter() - speech_end) * 1000
        except Exception as e:
            log.exception("STT failed")
            self._buffer = None
            self._release_stt()
            await self._send(ErrorMessage(message=f"Speech recognition failed: {e}", spoken=True))
            await self._speak(COULD_NOT_HEAR, speech_end=speech_end)
            return
        self._release_stt()

        self._log_audio_arrival(speech_end, transcript, stt_ms)
        text = transcript.text.strip()
        await self._send(TranscriptMessage(text=text))

        if not text:
            # Nothing heard: re-ask without spending an LLM call (HLD §13).
            await self._speak(COULD_NOT_HEAR, speech_end=speech_end, stt_ms=stt_ms)
            return

        # A one-word yes or no to a pending write is answered without Evon (D-006).
        # About half of all write turns end here, saving an LLM round trip each time.
        if self._executor is not None and self._executor.has_pending:
            if (answer := self._executor.handle_user_text(text)) is not None:
                self._remember("user", text)
                self._remember("assistant", answer.speech)
                if answer.ledger_changed:
                    await self._send_ledger()
                await self._speak(answer.speech, speech_end=speech_end, stt_ms=stt_ms, llm_ms=0.0)
                return

        await self._set_state(State.THINKING)
        self._remember("user", text)
        llm_start = time.perf_counter()
        try:
            reply = await self._llm.complete(self._history, tools=self._tools)
        except Exception as e:
            log.exception("LLM failed")
            await self._send(ErrorMessage(message=f"Evon failed: {e}", spoken=True))
            await self._speak(SOMETHING_BROKE, speech_end=speech_end, stt_ms=stt_ms)
            return
        llm_ms = (time.perf_counter() - llm_start) * 1000

        answer = await self._act_on(reply, text)
        self._remember("assistant", answer)
        await self._speak(answer, speech_end=speech_end, stt_ms=stt_ms, llm_ms=llm_ms)

    async def _act_on(self, reply: LLMReply, utterance: str) -> str:
        """Run whatever Evon proposed through the executor, and say the result.

        Evon only ever *proposes* (hard rule 5). Nothing reaches the ledger
        except through `Executor`, which validates, parses the amount in code and
        holds every write until the shop owner agrees.
        """
        if self._executor is None or not reply.tool_calls:
            return (reply.text or "").strip() or SOMETHING_BROKE

        # One tool per turn: the confirmation gate is a conversation, and two
        # pending writes cannot both be read back at once.
        call = reply.tool_calls[0]
        if len(reply.tool_calls) > 1:
            log.info("ignoring %d extra tool call(s) this turn", len(reply.tool_calls) - 1)

        if self._executor.has_pending:
            # Evon proposed something new while a write was waiting; the owner moved on.
            self._executor.cancel()

        # The name check looks at the last few things the owner said, not just this
        # turn: "unka 200 aur likho" names nobody, but the previous turn did.
        recent = [m.content for m in self._history if m.role == "user"][-RECENT_USER_TURNS:]
        result = self._executor.handle(call, utterance=utterance, context=" ".join(recent))
        if result.ledger_changed:
            await self._send_ledger()
        if result.outcome is Outcome.CONFIRMING:
            await self._set_state(State.CONFIRMING)
        return result.speech

    async def send_opening_ledger(self) -> None:
        """Show the book as soon as the page connects, before anything is said."""
        await self._send_ledger()

    async def _send_ledger(self) -> None:
        if self._ledger is None:
            return
        snapshot = self._ledger.snapshot()
        await self._send(LedgerUpdateMessage(**snapshot))

    def _log_audio_arrival(self, speech_end: float, transcript: Any, stt_ms: float) -> None:
        """Why stt_ms is what it is: how audio arrived, versus what Prisma reported."""
        audio_s = self._pushed_bytes / 2 / self._client_sample_rate
        arrival_s = (
            (self._last_push_at - self._first_push_at)
            if self._first_push_at and self._last_push_at
            else 0.0
        )
        lag_ms = (speech_end - self._last_push_at) * 1000 if self._last_push_at else None
        raw = transcript.raw or {}
        log.info(
            "turn stt_ms=%.0f | adapter tail=%s server=%s | audio %.2f s arrived over %.2f s "
            "in %d chunks | stop was %s ms after the last chunk",
            stt_ms,
            f"{raw.get('tail_ms'):.0f}" if raw.get("tail_ms") is not None else "?",
            raw.get("server_latency_ms", "?"),
            audio_s,
            arrival_s,
            self._pushes,
            f"{lag_ms:.0f}" if lag_ms is not None else "?",
        )

    async def _speak(
        self,
        text: str,
        *,
        speech_end: float,
        stt_ms: float | None = None,
        llm_ms: float | None = None,
    ) -> None:
        await self._send(AgentTextMessage(text=text))
        await self._set_state(State.SPEAKING)

        tts_start = time.perf_counter()
        first_byte_at: float | None = None
        try:
            await self._send(AudioStartMessage(sample_rate=self._tts.sample_rate))
            async for chunk in self._tts.synthesize(
                text, language=self._tts_language, voice=self._tts_voice
            ):
                if first_byte_at is None:
                    first_byte_at = time.perf_counter()
                await self._send_bytes(chunk)
            await self._send(AudioEndMessage())
        except Exception as e:
            log.exception("TTS failed")
            # The text was already sent, so the user can read what we could not say.
            await self._send(AudioEndMessage())
            await self._send(ErrorMessage(message=f"Timbre failed: {e}"))

        await self._send(
            LatencyMessage(
                stt_ms=round(stt_ms, 1) if stt_ms is not None else None,
                llm_ms=round(llm_ms, 1) if llm_ms is not None else None,
                tts_first_byte_ms=round((first_byte_at - tts_start) * 1000, 1) if first_byte_at else None,
                total_ms=round((first_byte_at - speech_end) * 1000, 1) if first_byte_at else None,
            )
        )
        await self._set_state(State.LISTENING)

    async def aclose(self) -> None:
        self._release_stt()
        pending = [*self._closing, *([self._warming] if self._warming else [])]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    @property
    def history(self) -> Sequence[Message]:
        return tuple(self._history)
