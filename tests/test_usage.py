"""The spend meter. Credits are a hard, non-renewable cap, so the arithmetic is tested."""

import json

import pytest

from app.adapters.usage import (
    read_log,
    record_stt,
    record_tts,
    stt_cost_inr,
    tts_cost_inr,
)


@pytest.fixture
def log(tmp_path, monkeypatch):
    path = tmp_path / "usage.jsonl"
    monkeypatch.setenv("KHATACALL_USAGE_LOG", str(path))
    return path


def test_published_rates():
    assert stt_cost_inr(3600) == pytest.approx(27.0)      # Rs 27 per hour of audio
    assert tts_cost_inr(10_000) == pytest.approx(27.0)    # Rs 27 per 10k characters


def test_cost_is_proportional_not_per_call():
    """Observed: 5 credits for ~28 STT and 23 TTS calls, so nothing is rounded up."""
    assert stt_cost_inr(4) == pytest.approx(stt_cost_inr(2) * 2)
    assert tts_cost_inr(60) == pytest.approx(tts_cost_inr(10) * 6)
    assert stt_cost_inr(4) == pytest.approx(0.03, abs=0.001)
    assert tts_cost_inr(60) == pytest.approx(0.162, abs=0.001)


def test_a_typical_turn_costs_about_a_fifth_of_a_credit():
    turn = stt_cost_inr(4) + tts_cost_inr(60)
    assert turn == pytest.approx(0.192, abs=0.005)
    # TTS dominates: it is worth caching templated read-backs (D-018).
    assert tts_cost_inr(60) > stt_cost_inr(4) * 5


def test_records_are_appended_and_readable(log):
    record_stt(3.52, model="prisma-v2.5-stream", note="stream")
    record_tts(57, model="timbre-v2.5")

    lines = log.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["service"] == "stt"

    records = read_log(log)
    assert [r.service for r in records] == ["stt", "tts"]
    assert records[0].units == 3.52
    assert records[1].cost_inr == pytest.approx(tts_cost_inr(57))


def test_metering_can_be_switched_off(monkeypatch, tmp_path):
    monkeypatch.setenv("KHATACALL_USAGE_LOG", "off")
    record_stt(10, model="m")
    assert list(tmp_path.iterdir()) == []


def test_an_unwritable_log_never_breaks_a_call(monkeypatch, tmp_path):
    blocked = tmp_path / "file.txt"
    blocked.write_text("not a directory")
    monkeypatch.setenv("KHATACALL_USAGE_LOG", str(blocked / "usage.jsonl"))
    assert record_stt(1.0, model="m").cost_inr > 0  # returns normally despite the OSError


def test_corrupt_lines_are_skipped(log):
    log.write_text('{"not": "a record"}\nnonsense\n', encoding="utf-8")
    record_tts(10, model="m")
    assert len(read_log(log)) == 1
