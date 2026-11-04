"""The end-to-end latency harness's planning and summary (scripts/e2e_latency.py)."""

import numpy as np
import pytest

from scripts.e2e_latency import EST_REPLY_CHARS, RATE, estimate_inr, plan, summarise_turns


def _take(i: int, intent: str, seconds: float = 2.0) -> dict:
    return {"id": f"{i:03}", "intent": intent, "audio": np.zeros(int(seconds * RATE), dtype=np.float32)}


def test_a_short_plan_still_covers_every_intent():
    takes = [_take(i, "add_credit") for i in range(10)] + [_take(10 + i, "get_balance") for i in range(10)]
    assert {t["intent"] for t in plan(takes, 4)} == {"add_credit", "get_balance"}


def test_the_estimate_counts_a_haan_turn_for_every_write_only():
    takes = [_take(1, "add_credit"), _take(2, "get_balance"), _take(3, "update_stock")]
    turns, audio_s, inr = estimate_inr(takes, haan_s=1.5)
    assert turns == 5  # three requests, two confirmations
    assert audio_s == 3 * 2.0 + 2 * 1.5
    assert inr == pytest.approx(27 * audio_s / 3600 + 27 * turns * EST_REPLY_CHARS / 10_000)


def test_the_summary_keeps_evon_turns_apart_and_skips_failed_ones():
    def row(kind, total, llm):
        return {"kind": kind, "stt_ms": 200.0, "llm_ms": llm, "tts_first_byte_ms": 600.0, "total_ms": total}

    rows = [row("request", 3000.0, 2000.0), row("request", 4000.0, 3000.0), row("confirm", 900.0, 0.0)]
    rows.append({"kind": "request", "stt_ms": 200.0, "llm_ms": None, "tts_first_byte_ms": None, "total_ms": None})
    summary = summarise_turns(rows)
    assert summary["request"]["total_ms"]["n"] == 2
    assert summary["request"]["llm_ms"]["max"] == 3000.0
    assert summary["confirm"]["total_ms"]["p50"] == 900.0
