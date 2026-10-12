"""Builds the three adapters from settings.

`USE_FAKES=true` swaps in the deterministic fakes, so the UI and the whole turn
can be exercised without a network, an Evon endpoint, or spending credits.
"""

from typing import Any

from app.adapters.base import LLMReply, ToolCall
from app.adapters.evon_llm import EvonLLM
from app.adapters.fakes import FakeLiveSTT, FakeLLM, FakeTTS
from app.adapters.gnani_stt_stream import PrismaStreamingSTT
from app.adapters.gnani_tts import TimbreTTS
from app.config import Settings

# A canned conversation that exercises the whole M2 loop without an API key:
# a write that gets read back and confirmed, a read, and a stock change.
#
# Note there is no scripted reply for "haan". The executor answers a pending
# confirmation in code, without calling the LLM at all (D-006), so the replies
# below line up with the transcripts that actually reach Evon.
FAKE_TRANSCRIPTS = [
    "sharma ji ka dedh sau udhaar likho",
    "haan",
    "sharma ji ka kitna baaki hai",
    "maggi ke bees packet aaye",
    "haan",
    "kya kya khatam ho raha hai",
]
FAKE_REPLIES: list[LLMReply] = [
    LLMReply(
        tool_calls=(
            ToolCall("add_credit", {"customer_name": "Sharma ji", "amount_spoken": "dedh sau"}),
        )
    ),
    LLMReply(tool_calls=(ToolCall("get_balance", {"customer_name": "Sharma ji"}),)),
    LLMReply(
        tool_calls=(ToolCall("update_stock", {"item": "Maggi", "qty_delta_spoken": "bees"}),)
    ),
    LLMReply(tool_calls=(ToolCall("list_low_stock", {}),)),
]


def build_adapters(settings: Settings) -> tuple[Any, Any, Any]:
    """Return (stt, llm, tts). The STT is the *streaming* adapter: the live path needs it."""
    if settings.use_fakes:
        return (
            FakeLiveSTT(FAKE_TRANSCRIPTS * 20),
            FakeLLM(FAKE_REPLIES * 20),
            FakeTTS(sample_rate=settings.tts_sample_rate),
        )

    key = settings.gnani_api_key.get_secret_value()
    llm = (
        FakeLLM(FAKE_REPLIES * 20)
        if settings.fake_llm
        else EvonLLM(
            base_url=settings.evon_base_url,
            api_key=settings.evon_api_key.get_secret_value(),
            model=settings.evon_model,
        )
    )
    return PrismaStreamingSTT(key), llm, TimbreTTS(key, sample_rate=settings.tts_sample_rate)
