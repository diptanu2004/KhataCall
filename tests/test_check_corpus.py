"""Corpus quality checks, on synthetic audio. These decide what gets re-recorded."""

import wave

import numpy as np
import pytest

from scripts.check_corpus import EXPECTED_RATE, measure, problems

RNG = np.random.default_rng(0)


def write_wav(path, audio, rate=EXPECTED_RATE):
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return path


def speech(seconds=3.0, level=0.25):
    """A stand-in for a spoken line: a tone burst with a little noise on top."""
    t = np.linspace(0, seconds, int(seconds * EXPECTED_RATE), endpoint=False)
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)
    return (level * envelope * np.sin(2 * np.pi * 180 * t)).astype("float32")


def take(tmp_path, audio, name="001", rate=EXPECTED_RATE):
    return measure(write_wav(tmp_path / f"{name}.wav", audio, rate), name)


def padded(audio, lead_s=0.4, trail_s=0.4, floor=0.0):
    """Speech with silence either side, as a real take has."""
    def pad(seconds):
        n = int(seconds * EXPECTED_RATE)
        return (RNG.normal(0, floor, n) if floor else np.zeros(n)).astype("float32")

    return np.concatenate([pad(lead_s), audio, pad(trail_s)])


# --- a good take -----------------------------------------------------------


def test_a_clean_take_has_no_problems(tmp_path):
    assert problems(take(tmp_path, padded(speech(), floor=0.0005))) == []


def test_measures_duration_and_peak(tmp_path):
    t = take(tmp_path, padded(speech(seconds=2.0, level=0.5), 0.5, 0.5, floor=0.0005))
    assert t.seconds == pytest.approx(3.0, abs=0.05)
    assert t.peak_dbfs == pytest.approx(-6.0, abs=1.5)


# --- the noise gate --------------------------------------------------------
# Windows voice processing zeroes near-silence. Averaging those frames into the
# noise floor reported 166 dB SNR, which is not a number any microphone can
# produce. The floor is measured from frames that contain something.


def test_digital_silence_does_not_inflate_snr(tmp_path):
    gated = padded(speech(), floor=0.0)  # exact zeros either side, as the driver writes
    t = take(tmp_path, gated)
    assert t.gated_frames > 0, "the fixture should contain zeroed frames"
    assert 20 < t.snr_db < 100, f"implausible SNR {t.snr_db}"


def test_gating_only_around_the_speech_is_accepted(tmp_path):
    t = take(tmp_path, padded(speech(), floor=0.0))
    assert t.gated_inside_speech == 0
    assert problems(t) == []


def test_gating_inside_the_speech_is_flagged(tmp_path):
    """A gate that eats a soft word ending would be blamed on the channel later."""
    audio = speech()
    hole = slice(int(1.0 * EXPECTED_RATE), int(1.3 * EXPECTED_RATE))
    audio[hole] = 0.0
    t = take(tmp_path, padded(audio, floor=0.0))

    assert t.gated_inside_speech > 0
    assert any("inside the speech" in p for p in problems(t))


# --- things worth re-recording ---------------------------------------------


def test_clipping_is_flagged(tmp_path):
    t = take(tmp_path, padded(speech(level=1.0) * 3, floor=0.0005))
    assert any("clipped" in p for p in problems(t))


def test_a_very_quiet_take_is_flagged(tmp_path):
    t = take(tmp_path, padded(speech(level=0.005), floor=0.0005))
    assert any("quiet" in p for p in problems(t))


def test_a_truncated_take_is_flagged(tmp_path):
    """Speech running right to the file edge means a word was cut off."""
    t = take(tmp_path, speech(seconds=2.0))  # no silence either side
    assert any("cut off" in p for p in problems(t))


def test_a_too_short_take_is_flagged(tmp_path):
    assert any("only" in p for p in problems(take(tmp_path, speech(seconds=0.3))))


def test_a_wrong_sample_rate_is_flagged(tmp_path):
    t = take(tmp_path, padded(speech(), floor=0.0005), rate=8000)
    assert any("8000 Hz" in p for p in problems(t))


def test_a_noisy_take_is_flagged(tmp_path):
    """At 20 dB SNR the eval's own 20 dB condition would measure the room."""
    noisy = padded(speech(level=0.08), floor=0.0) + RNG.normal(0, 0.02, int(3.8 * EXPECTED_RATE)).astype("float32")
    t = take(tmp_path, noisy.astype("float32"))
    assert any("SNR" in p for p in problems(t))


def test_excessive_silence_is_flagged(tmp_path):
    t = take(tmp_path, padded(speech(seconds=1.0), lead_s=3.0, trail_s=0.3, floor=0.0005))
    assert any("lead" in p for p in problems(t))
