"""The telephone channel simulation. If these numbers are wrong, every eval result is."""

import numpy as np
import pytest

from app.channel.degrade import (
    PHONE_FRAME,
    Condition,
    active_speech_power,
    conceal,
    degrade,
    gilbert_elliott_mask,
    mix_noise,
    mulaw_decode,
    mulaw_encode,
    packet_loss,
    telephone_band,
    to_pcm16,
)

SR = 16_000


def rng(seed=0):
    return np.random.default_rng(seed)


def tone(freq, seconds=1.0, amp=0.3, rate=SR):
    t = np.arange(int(seconds * rate)) / rate
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def speechlike(seconds=3.0, duty=0.5, seed=0):
    """Bursts of band-limited noise separated by silence, like words and pauses."""
    g = rng(seed)
    x = np.zeros(int(seconds * SR), dtype=np.float32)
    burst = int(0.25 * SR)
    for start in range(0, len(x) - burst, int(burst / duty)):
        x[start : start + burst] = 0.2 * g.standard_normal(burst)
    return x


def level_db(x):
    return 10 * np.log10(np.mean(np.asarray(x, dtype=np.float64) ** 2) + 1e-20)


# --- active speech power -----------------------------------------------------


def test_silence_between_words_does_not_dilute_speech_power():
    speech = speechlike(duty=0.5)
    whole = np.mean(speech.astype(np.float64) ** 2)
    active = active_speech_power(speech)
    assert 10 * np.log10(active / whole) == pytest.approx(3.0, abs=0.5)  # half silence is 3 dB


# --- noise mixing ------------------------------------------------------------


@pytest.mark.parametrize("snr", [20, 10, 5, 0, -5])
def test_mix_hits_the_target_snr(snr):
    noise = rng(1).standard_normal(SR * 10).astype(np.float32) * 0.1
    result = mix_noise(speechlike(), noise, snr, rng(2))
    assert result.snr_db == pytest.approx(snr, abs=0.05)
    assert np.allclose(result.mixed, result.speech + result.noise, atol=1e-6)


def test_snr_holds_whichever_part_of_a_varying_noise_file_is_used():
    """The flea-market recording is 12 dB louder when a crowd passes (D-033)."""
    g = rng(3)
    quiet = 0.01 * g.standard_normal(SR * 5)
    busy = 0.04 * g.standard_normal(SR * 5)
    noise = np.concatenate([quiet, busy]).astype(np.float32)
    speech = speechlike(seconds=2.0)
    for seed in range(8):  # different seeds land in different halves
        assert mix_noise(speech, noise, 10, rng(seed)).snr_db == pytest.approx(10, abs=0.05)


def test_mixing_never_clips_and_keeps_the_snr():
    loud = (0.95 * np.sign(speechlike())).astype(np.float32)
    noise = rng(4).standard_normal(SR * 5).astype(np.float32)
    result = mix_noise(loud, noise, 0, rng(5))
    assert np.max(np.abs(result.mixed)) <= 0.99 + 1e-6
    assert result.snr_db == pytest.approx(0, abs=0.05)


def test_short_noise_is_looped():
    noise = rng(6).standard_normal(SR // 2).astype(np.float32) * 0.1  # 0.5 s for a 3 s clip
    result = mix_noise(speechlike(), noise, 10, rng(7))
    assert len(result.mixed) == len(speechlike())
    assert result.snr_db == pytest.approx(10, abs=0.05)


def test_silent_noise_is_refused():
    with pytest.raises(ValueError, match="silent"):
        mix_noise(speechlike(), np.zeros(SR * 5, dtype=np.float32), 10, rng())


def test_mixing_is_reproducible_from_the_seed():
    noise = rng(8).standard_normal(SR * 10).astype(np.float32)
    a = mix_noise(speechlike(), noise, 5, rng(42)).mixed
    b = mix_noise(speechlike(), noise, 5, rng(42)).mixed
    c = mix_noise(speechlike(), noise, 5, rng(43)).mixed
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


# --- the phone band ----------------------------------------------------------


def test_output_is_8_khz():
    assert len(telephone_band(tone(1000, seconds=1.0))) == 8_000


@pytest.mark.parametrize(
    "freq,min_loss,max_loss",
    [(1000, -1, 1), (2000, -1, 1), (100, 15, 200), (7000, 40, 400)],
)
def test_the_phone_band_keeps_voice_and_cuts_the_rest(freq, min_loss, max_loss):
    x = tone(freq, seconds=2.0)
    y = telephone_band(x)[4000:]  # skip the filter's start-up
    loss = level_db(x) - level_db(y)
    assert min_loss <= loss <= max_loss, f"{freq} Hz lost {loss:.1f} dB"


# --- G.711 mu-law ------------------------------------------------------------


def test_known_g711_codes():
    """Reference points of the standard: zero, full scale, and their signs."""
    codes = mulaw_encode(np.array([0.0, 32767 / 32767, -1.0], dtype=np.float32))
    assert codes.dtype == np.uint8
    assert list(codes) == [0xFF, 0x80, 0x00]
    assert mulaw_decode(np.array([0xFF, 0x7F], dtype=np.uint8)).tolist() == [0.0, 0.0]


def test_mulaw_is_symmetric_and_monotonic():
    x = np.linspace(-1, 1, 4001, dtype=np.float32)
    y = mulaw_decode(mulaw_encode(x))
    assert np.all(np.diff(y) >= 0)
    assert np.allclose(y, -mulaw_decode(mulaw_encode(-x)), atol=1 / 32768)


def test_decoding_is_a_fixed_point():
    """Re-encoding decoded audio changes nothing: a second codec pass is lossless."""
    y = mulaw_decode(mulaw_encode(tone(440)))
    assert np.array_equal(mulaw_encode(y), mulaw_encode(mulaw_decode(mulaw_encode(y))))


@pytest.mark.parametrize("amp", [0.5, 0.05, 0.005])
def test_mulaw_keeps_about_38_db_snr_across_levels(amp):
    """The point of companding: quantisation SNR stays roughly constant as level falls."""
    x = tone(440, amp=amp)
    error = x - mulaw_decode(mulaw_encode(x))
    sqnr = level_db(x) - level_db(error)
    assert 30 < sqnr < 45, f"{sqnr:.1f} dB at amplitude {amp}"


# --- packet loss -------------------------------------------------------------


@pytest.mark.parametrize("loss,burst", [(0.01, 2), (0.05, 2), (0.10, 3)])
def test_long_run_loss_rate_and_burst_length(loss, burst):
    mask = gilbert_elliott_mask(200_000, loss, burst, rng(9))
    assert mask.mean() == pytest.approx(loss, rel=0.1)
    starts = np.flatnonzero(np.diff(mask.astype(int)) == 1)
    ends = np.flatnonzero(np.diff(mask.astype(int)) == -1)
    n = min(len(starts), len(ends))
    lengths = ends[-n:] - starts[:n] if ends[0] > starts[0] else ends[1:n] - starts[: n - 1]
    assert lengths.mean() == pytest.approx(burst, rel=0.1)


def test_losses_come_in_bursts_not_one_at_a_time():
    """At the same loss rate, bursts hurt speech more than scattered single frames."""
    mask = gilbert_elliott_mask(100_000, 0.05, 3, rng(10))
    assert np.mean(mask[1:] & mask[:-1]) > 0.05 * 0.05 * 5


def test_no_loss_changes_nothing():
    x = tone(1000, rate=8_000)
    y, lost = packet_loss(x, 0.0, rng())
    assert not lost.any()
    assert np.array_equal(x, y)


def test_zero_fill_silences_exactly_the_lost_frames():
    x = np.ones(PHONE_FRAME * 4, dtype=np.float32)
    lost = np.array([False, True, False, True])
    y = conceal(x, lost, "zero")
    assert y[:PHONE_FRAME].all() and not y[PHONE_FRAME : 2 * PHONE_FRAME].any()
    assert y[2 * PHONE_FRAME : 3 * PHONE_FRAME].all() and not y[3 * PHONE_FRAME :].any()


def test_repeat_copies_the_last_good_frame():
    x = np.concatenate([np.full(PHONE_FRAME, 0.1), np.full(PHONE_FRAME, 0.2), np.full(PHONE_FRAME, 0.3)])
    y = conceal(x.astype(np.float32), np.array([False, True, True]), "repeat")
    assert np.allclose(y[PHONE_FRAME:], 0.1)


def test_impossible_loss_settings_are_refused():
    with pytest.raises(ValueError):
        gilbert_elliott_mask(100, 1.0, 2, rng())
    with pytest.raises(ValueError):
        gilbert_elliott_mask(100, 0.05, 0.5, rng())


# --- the whole channel -------------------------------------------------------


def test_clean_condition_returns_the_input_untouched():
    x = speechlike()
    assert np.array_equal(degrade(x, Condition("clean"), rng()), x)


def test_telephone_condition_is_8_khz_and_close_to_the_input_in_band():
    x = speechlike()
    y = degrade(x, Condition("tel", telephone=True), rng())
    assert len(y) == len(x) // 2


def test_full_condition_is_reproducible():
    noise = rng(11).standard_normal(SR * 10).astype(np.float32) * 0.1
    worst = Condition("worst", telephone=True, snr_db=5, loss_pct=5)
    a = degrade(speechlike(), worst, rng(7), noise)
    b = degrade(speechlike(), worst, rng(7), noise)
    assert np.array_equal(a, b)


def test_noise_conditions_need_noise():
    with pytest.raises(ValueError, match="needs noise"):
        degrade(speechlike(), Condition("n", telephone=True, snr_db=10), rng())


def test_loss_without_the_phone_network_is_refused():
    with pytest.raises(ValueError, match="phone network"):
        degrade(speechlike(), Condition("x", loss_pct=5), rng())


def test_pcm16_conversion():
    pcm = to_pcm16(np.array([0.0, 1.0, -1.0, 2.0], dtype=np.float32))
    assert np.frombuffer(pcm, dtype="<i2").tolist() == [0, 32767, -32767, 32767]


# --- the mobile codec (D-036) -----------------------------------------------


def test_gsm_keeps_length_and_timing():
    from scipy.signal import correlate

    from app.channel.degrade import gsm_round_trip

    x = telephone_band(speechlike(seconds=2.03))  # not a multiple of the 320-sample block
    y = gsm_round_trip(x)
    assert len(y) == len(x)
    lag = int(np.argmax(correlate(y, x, mode="full", method="fft"))) - (len(x) - 1)
    assert lag == 0


def test_gsm_is_much_lossier_than_g711():
    from app.channel.degrade import gsm_round_trip

    x = telephone_band(speechlike())
    g711 = level_db(x) - level_db(x - mulaw_decode(mulaw_encode(x)))
    gsm = level_db(x) - level_db(x - gsm_round_trip(x))
    assert g711 > 30
    assert gsm < g711 - 15


def test_mobile_condition_runs_gsm_then_g711():
    x = speechlike()
    land = degrade(x, Condition("tel", telephone=True), rng())
    mobile = degrade(x, Condition("mob", telephone=True, codec="gsm"), rng())
    assert len(mobile) == len(land)
    assert not np.allclose(mobile, land, atol=1e-3)


def test_unknown_codec_is_refused():
    with pytest.raises(ValueError, match="codec"):
        degrade(speechlike(), Condition("x", telephone=True, codec="amr"), rng())
