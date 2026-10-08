"""Per-connection turn state machine. See docs/HLD.md §5 and §6.

Half-duplex and push-to-talk: one turn at a time, driven by `start` / audio /
`stop` from the browser.

    IDLE -> LISTENING -> TRANSCRIBING -> THINKING -> SPEAKING -> LISTENING

M1 handles conversation only. The CONFIRMING state, tools, and ledger writes
arrive in M2.

The session is deliberately ignorant of FastAPI: it takes two send callables
and the three adapters, so the whole turn can be tested with fakes.
"""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Sequence
from contextlib import AsyncExitStack
from typing import Any

from app.adapters.base import LLM, TTS, Message
from app.agent.prompts import SYSTEM_PROMPT
from app.protocol import (
    AgentTextMessage,
    AudioEndMessage,
    AudioStartMessage,
    ConfigMessage,
    ErrorMessage,
    LatencyMessage,
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

        self.state = State.IDLE
        self.config = ConfigMessage()
        self._stack: AsyncExitStack | None = None
        self._live: Any = None
        self._closing: set[asyncio.Task] = set()
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
        """Forward microphone audio to Prisma as it arrives."""
        if self.state is not State.LISTENING or self._live is None:
            return  # frames outside a turn, e.g. a late flush after release
        now = time.perf_counter()
        if self._first_push_at is None:
            self._first_push_at = now
        self._last_push_at = now
        self._pushes += 1
        self._pushed_bytes += len(pcm)
        await self._live.push(pcm)

    async def _handle_config(self, message: ConfigMessage) -> None:
        self.config = message
        if message.phone_mode:
            # Honest about scope: channel/degrade.py lands in M3.
            await self._send(NoticeMessage(message="Phone-line mode is not wired up yet (M3)."))

    async def _handle_start(self) -> None:
        if self.state in (State.TRANSCRIBING, State.THINKING, State.SPEAKING):
            log.warning("start ignored while %s", self.state)
            return
        await self._open_stt()
        self._pushes = self._pushed_bytes = 0
        self._first_push_at = self._last_push_at = None
        await self._set_state(State.LISTENING)

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

    async def _handle_stop(self) -> None:
        """Push-to-talk released. This is end of speech: the latency clock starts here."""
        if self.state is not State.LISTENING or self._live is None:
            log.warning("stop ignored while %s", self.state)
            return

        speech_end = time.perf_counter()
        await self._set_state(State.TRANSCRIBING)

        try:
            transcript = await self._live.finish()
            # Measured before teardown: closing the socket is not part of recognition.
            stt_ms = (time.perf_counter() - speech_end) * 1000
        except Exception as e:
            log.exception("STT failed")
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

        await self._set_state(State.THINKING)
        self._remember("user", text)
        llm_start = time.perf_counter()
        try:
            reply = await self._llm.complete(self._history)
        except Exception as e:
            log.exception("LLM failed")
            await self._send(ErrorMessage(message=f"Evon failed: {e}", spoken=True))
            await self._speak(SOMETHING_BROKE, speech_end=speech_end, stt_ms=stt_ms)
            return
        llm_ms = (time.perf_counter() - llm_start) * 1000

        # M2 replaces this with the executor and the confirmation gate.
        answer = (reply.text or "").strip() or SOMETHING_BROKE
        self._remember("assistant", answer)
        await self._speak(answer, speech_end=speech_end, stt_ms=stt_ms, llm_ms=llm_ms)

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
        if self._closing:
            await asyncio.gather(*self._closing, return_exceptions=True)

    @property
    def history(self) -> Sequence[Message]:
        return tuple(self._history)
