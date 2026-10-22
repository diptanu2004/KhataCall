"""The eval runner's cache keys and scoring. No network: the stages that call out are not run here."""

import asyncio

import numpy as np

from app.channel.degrade import Condition
from eval.conditions import by_id
from eval.run_eval import Pacer, amount_heard, degraded_clip, error_rates, stt_cache_path, summarise, task_cache_key


def take(seconds=1.5, uid="001"):
    t = np.arange(int(seconds * 16_000)) / 16_000
    return {"id": uid, "audio": (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)}


BANK = {"freesound_1": np.random.default_rng(0).standard_normal(16_000 * 5).astype(np.float32) * 0.05}


def test_the_same_take_and_condition_hit_the_same_cache_entry():
    a, _ = degraded_clip(take(), by_id("mob_worst"), BANK)
    b, _ = degraded_clip(take(), by_id("mob_worst"), BANK)
    assert stt_cache_path(a, "mob_worst") == stt_cache_path(b, "mob_worst")


def test_a_different_condition_or_model_is_a_different_entry():
    clip, _ = degraded_clip(take(), by_id("mob"), BANK)
    assert stt_cache_path(clip, "mob") != stt_cache_path(clip, "mob_loss1")
    assert stt_cache_path(clip, "mob") != stt_cache_path(clip, "mob", model_id="prisma-v3")


def test_degraded_clips_are_at_the_condition_rate():
    assert degraded_clip(take(), Condition("clean"), BANK)[0].sample_rate == 16_000
    clip, noise = degraded_clip(take(), by_id("mob_snr10"), BANK)
    assert clip.sample_rate == 8_000 and noise == "freesound_1"


def test_a_prompt_change_invalidates_the_agent_cache():
    assert task_cache_key("sharma ji", "evon", "aaa") != task_cache_key("sharma ji", "evon", "bbb")
    assert "/" not in task_cache_key("x", "gnani/evon", "aaa").parent.parent.name


def test_empty_hypothesis_is_all_errors_not_a_crash():
    assert error_rates(["sharma ji"], [""])["wer"] == 1.0


def test_summary_counts_outcomes():
    rows = [
        {"text": "dedh sau", "transcript": "150", "stt_ms": 500, "outcome": "correct"},
        {"text": "dhai sau", "transcript": "", "stt_ms": 700, "outcome": "caught"},
    ]
    s = summarise(rows)
    assert s["normalised"]["wer"] == 0.5
    assert s["empty_transcripts"] == 1
    assert s["task"] == {"n": 2, "correct": 1, "caught": 1, "wrong": 0, "accuracy": 0.5, "wrong_rate": 0.0}


def test_pacer_spreads_requests(monkeypatch):
    slept = []

    async def fake_sleep(s):
        slept.append(s)

    monkeypatch.setattr("eval.run_eval.asyncio.sleep", fake_sleep)
    async def three():
        pacer = Pacer(60)
        for _ in range(3):
            await pacer.wait()

    asyncio.run(three())
    assert len(slept) == 2 and all(0.9 < s <= 2.0 for s in slept)


def test_amount_heard_separates_kept_changed_and_lost():
    row = {"amount_paise": "35000"}
    assert amount_heard(row | {"transcript": "yadav ji ka sadhe teen sau likho"}) == "kept"
    assert amount_heard(row | {"transcript": "yadav ji ka teen sau likho"}) == "changed"
    assert amount_heard(row | {"transcript": "yadav ji ka"}) == "lost"
    assert amount_heard({"amount_paise": "", "transcript": "cheeni kitni hai"}) is None
