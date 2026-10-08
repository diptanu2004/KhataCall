"""Mic calibration logic. The hardware parts are not tested; the judgements are.

A recording device that is already narrowband would bake the telephone channel
into the "clean" corpus, which is exactly what the eval is supposed to add
itself at known settings. Getting these warnings wrong silently invalidates
every eval number, so they are worth pinning down.
"""

import math

import numpy as np
import pytest

from scripts.calibrate import (
    Reading,
    active_speech_rms,
    dbfs,
    device_warning,
    rms,
    verdicts,
)


def device(name, rate=48_000, channels=1):
    return {"name": name, "default_samplerate": rate, "max_input_channels": channels}


# --- device suitability ----------------------------------------------------


def test_a_normal_wideband_mic_is_fine():
    assert device_warning(device("Microphone Array (Realtek(R) Audio)", 48_000)) is None
    assert device_warning(device("Microphone (USB Audio Device)", 44_100)) is None


def test_telephone_rate_devices_are_rejected():
    warning = device_warning(device("Input (Nothing Phone (3a))", 8_000))
    assert warning is not None and "8000 Hz" in warning


def test_bluetooth_is_rejected_even_when_the_driver_reports_a_high_rate():
    """MME reports 44100 for the same buds WASAPI shows at 16000."""
    warning = device_warning(device("Headset (realme Buds T310)", 44_100))
    assert warning is not None
    assert "hands-free" in warning.lower() or "bluetooth" in warning.lower()


def test_hands_free_driver_names_are_rejected():
    name = r"Headset (@System32\drivers\bthhfenum.sys,#2;%1 Hands-Free%0 ;(JBL RACE TWS))"
    assert device_warning(device(name, 16_000)) is not None


def test_loopback_and_virtual_devices_are_rejected():
    for name in ("Stereo Mix (Realtek HD Audio Stereo input)", "Microsoft Sound Mapper - Input"):
        assert "not a microphone" in device_warning(device(name))


def test_multiline_driver_names_are_handled():
    """Some Windows driver names contain newlines."""
    assert device_warning(device("Headset (@System32\ndrivers,#2;Hands-Free)", 16_000)) is not None


# --- level maths -----------------------------------------------------------


def test_dbfs_reference_points():
    assert dbfs(1.0) == pytest.approx(0.0)
    assert dbfs(0.5) == pytest.approx(-6.02, abs=0.01)
    assert dbfs(0.001) == pytest.approx(-60.0, abs=0.01)
    assert dbfs(0.0) == -math.inf


def test_rms_of_a_sine_is_amplitude_over_root_two():
    t = np.linspace(0, 1, 16_000, endpoint=False)
    assert rms(np.sin(2 * np.pi * 220 * t).astype("float32")) == pytest.approx(0.7071, abs=0.001)


def test_active_speech_ignores_the_pauses():
    """Silence between words must not drag the measured speech level down."""
    loud = np.full(16_000, 0.5, dtype="float32")
    silence = np.zeros(16_000, dtype="float32")
    speech = np.concatenate([loud, silence])

    assert rms(speech) == pytest.approx(0.3536, abs=0.001)  # dragged down by the gap
    assert active_speech_rms(speech) == pytest.approx(0.5, abs=0.01)  # the speech itself


# --- verdicts --------------------------------------------------------------


def good_reading(**overrides) -> Reading:
    base = dict(
        noise_dbfs=-60.0, speech_peak_dbfs=-8.0, speech_rms_dbfs=-18.0, dc_offset=0.0, clipped_samples=0
    )
    return Reading(**(base | overrides))


def test_a_quiet_room_with_healthy_levels_passes():
    notes, ok = verdicts(good_reading())
    assert ok
    assert any("good" in n for n in notes)


def test_clipping_fails():
    _, ok = verdicts(good_reading(clipped_samples=120, speech_peak_dbfs=0.0))
    assert not ok


def test_too_quiet_fails():
    _, ok = verdicts(good_reading(speech_peak_dbfs=-30.0))
    assert not ok


def test_a_noisy_room_fails():
    _, ok = verdicts(good_reading(noise_dbfs=-30.0))
    assert not ok


def test_snr_near_the_evals_quietest_condition_fails():
    """At 20 dB SNR the 'clean' baseline is really a 20 dB recording."""
    notes, ok = verdicts(good_reading(noise_dbfs=-40.0, speech_rms_dbfs=-18.0))  # 22 dB
    assert not ok
    assert any("20 dB" in n for n in notes)


def test_dc_offset_is_flagged_without_failing_the_run():
    notes, ok = verdicts(good_reading(dc_offset=0.05))
    assert ok
    assert any("DC offset" in n for n in notes)
