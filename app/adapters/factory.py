"""Builds the three adapters from settings.

`USE_FAKES=true` swaps in the deterministic fakes, so the UI and the whole turn
can be exercised without a network, an Evon endpoint, or spending credits.
"""

from typing import Any

from app.adapters.base import LLMReply
from app.adapters.evon_llm import EvonLLM
from app.adapters.fakes import FakeLiveSTT, FakeLLM, FakeTTS
from app.adapters.gnani_stt_stream import PrismaStreamingSTT
from app.adapters.gnani_tts import TimbreTTS
from app.config import Settings

# Enough canned turns to walk through the UI without an API key.
FAKE_TRANSCRIPTS = [
    "sharma ji ka dedh sau udhaar likho",
    "gupta aunty ka kitna baaki hai",
    "maggi ke bees packet aaye",
]
FAKE_REPLIES = [
    "Abhi main sirf baat kar sakta hoon, khata agle hafte.",
    "Khata abhi nahi dekh sakta, maaf kijiye.",
    "Stock abhi nahi likh sakta.",
]


def build_adapters(settings: Settings) -> tuple[Any, Any, Any]:
    """Return (stt, llm, tts). The STT is the *streaming* adapter: the live path needs it."""
    if settings.use_fakes:
        return (
            FakeLiveSTT(FAKE_TRANSCRIPTS * 20),
            FakeLLM([LLMReply(text=t) for t in FAKE_REPLIES * 20]),
            FakeTTS(sample_rate=settings.tts_sample_rate),
        )

    key = settings.gnani_api_key.get_secret_value()
    llm = (
        FakeLLM([LLMReply(text=t) for t in FAKE_REPLIES * 20])
        if settings.fake_llm
        else EvonLLM(
            base_url=settings.evon_base_url,
            api_key=settings.evon_api_key.get_secret_value(),
            model=settings.evon_model,
        )
    )
    return PrismaStreamingSTT(key), llm, TimbreTTS(key, sample_rate=settings.tts_sample_rate)
