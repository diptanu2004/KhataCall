"""WebSocket message schemas — the single source of truth for the client/server contract.

Client → server: binary PCM16 mono frames at `CLIENT_SAMPLE_RATE`, plus the JSON
messages in `ClientMessage`.
Server → client: the JSON messages in `ServerMessage`, plus binary PCM16 mono TTS
frames at the rate the TTS adapter reports.
"""

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, Field, TypeAdapter

# The browser resamples to this before sending. Prisma realtime accepts 8/16/44.1/48 kHz.
CLIENT_SAMPLE_RATE = 16_000


class State(StrEnum):
    """Half-duplex turn states. See docs/HLD.md §6."""

    IDLE = "IDLE"
    LISTENING = "LISTENING"
    TRANSCRIBING = "TRANSCRIBING"
    THINKING = "THINKING"
    CONFIRMING = "CONFIRMING"  # a write has been read back and waits for "haan"
    SPEAKING = "SPEAKING"


# --- client → server -------------------------------------------------------


class StartMessage(BaseModel):
    """Push-to-talk pressed. Audio frames follow."""

    type: Literal["start"] = "start"


class StopMessage(BaseModel):
    """Push-to-talk released. This is the end of speech, and starts the latency clock."""

    type: Literal["stop"] = "stop"


class ConfigMessage(BaseModel):
    """Phone-line mode: a simulated mobile call from a noisy shop (app/channel/phone.py).

    `snr_db` None means no shop noise. The UI offers only the eval grid's values,
    so every demo setting matches a measured condition.
    """

    type: Literal["config"] = "config"
    phone_mode: bool = False
    snr_db: float | None = Field(default=None, ge=-10, le=40)
    loss_pct: float | None = Field(default=None, ge=0, le=50)


ClientMessage = Annotated[StartMessage | StopMessage | ConfigMessage, Field(discriminator="type")]
client_message = TypeAdapter(ClientMessage)


# --- server → client -------------------------------------------------------


class StateMessage(BaseModel):
    type: Literal["state"] = "state"
    state: State


class TranscriptMessage(BaseModel):
    """What Prisma heard. In phone mode this is the degraded audio's transcript."""

    type: Literal["transcript"] = "transcript"
    text: str


class AgentTextMessage(BaseModel):
    """What the agent is about to say."""

    type: Literal["agent_text"] = "agent_text"
    text: str


class AudioStartMessage(BaseModel):
    """Binary TTS frames follow, until `audio_end`."""

    type: Literal["audio_start"] = "audio_start"
    sample_rate: int


class AudioEndMessage(BaseModel):
    type: Literal["audio_end"] = "audio_end"


class LatencyMessage(BaseModel):
    """Per-stage timing for one turn, all measured from end of speech. See docs/HLD.md §12."""

    type: Literal["latency"] = "latency"
    stt_ms: float | None = None
    llm_ms: float | None = None
    tts_first_byte_ms: float | None = None
    total_ms: float | None = None


class LedgerUpdateMessage(BaseModel):
    """The whole book: sent when the page connects and after every committed write."""

    type: Literal["ledger_update"] = "ledger_update"
    customers: list[dict] = Field(default_factory=list)
    items: list[dict] = Field(default_factory=list)


class HeardAudioMessage(BaseModel):
    """Phone-line mode: the degraded audio Prisma was sent, so the user can hear it.

    PCM16 mono, base64, at `sample_rate` (8 kHz). Sent before the transcript.
    """

    type: Literal["heard_audio"] = "heard_audio"
    sample_rate: int
    pcm_b64: str
    snr_db: float | None = None
    loss_pct: float = 0.0
    noise: str | None = None


class ErrorMessage(BaseModel):
    """A failure the user should know about. `spoken` is true if the agent also said it aloud."""

    type: Literal["error"] = "error"
    message: str
    spoken: bool = False


class NoticeMessage(BaseModel):
    """Non-fatal information for the page: phone-line settings, a press too short to be speech."""

    type: Literal["notice"] = "notice"
    message: str


ServerMessage = Annotated[
    StateMessage
    | TranscriptMessage
    | AgentTextMessage
    | AudioStartMessage
    | AudioEndMessage
    | LatencyMessage
    | LedgerUpdateMessage
    | HeardAudioMessage
    | ErrorMessage
    | NoticeMessage,
    Field(discriminator="type"),
]
server_message = TypeAdapter(ServerMessage)
