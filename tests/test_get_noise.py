"""Importing background noise: conversion, provenance, and refusing to guess a licence."""

from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

import scripts.get_noise as gn

RNG = np.random.default_rng(0)


@pytest.fixture
def noise_dirs(tmp_path, monkeypatch):
    noise = tmp_path / "noise"
    monkeypatch.setattr(gn, "NOISE_DIR", noise)
    monkeypatch.setattr(gn, "INCOMING", noise / "incoming")
    monkeypatch.setattr(gn, "SOURCES", noise / "SOURCES.md")
    monkeypatch.setattr(gn, "ROOT", tmp_path)
    (noise / "incoming").mkdir(parents=True)
    return noise


def write(path, seconds=30.0, rate=48_000, channels=2):
    data = (0.1 * RNG.standard_normal((int(seconds * rate), channels))).astype("float32")
    sf.write(path, data, rate)


def run(**overrides):
    return gn.import_incoming(SimpleNamespace(**({"dry_run": False, "source": None, "license": None} | overrides)))


def test_a_freesound_download_is_converted_and_traced(noise_dirs):
    write(noise_dirs / "incoming" / "516655__soundensemble__delhi-atmo.flac")

    assert run() == 0

    out = noise_dirs / "freesound_516655.wav"
    info = sf.info(out)
    assert (info.samplerate, info.channels, info.subtype) == (16_000, 1, "PCM_16")
    assert info.duration == pytest.approx(30.0, abs=0.01)
    sources = (noise_dirs / "SOURCES.md").read_text(encoding="utf-8")
    assert "freesound.org/s/516655" in sources
    assert "CC0 1.0" in sources


def test_an_unknown_freesound_id_needs_an_explicit_licence(noise_dirs):
    """A filename cannot prove a licence, so the importer will not guess one."""
    write(noise_dirs / "incoming" / "999999__someone__street.wav")

    assert run() == 1
    assert not (noise_dirs / "freesound_999999.wav").exists()

    assert run(license="CC BY 4.0") == 0
    assert "CC BY 4.0" in (noise_dirs / "SOURCES.md").read_text(encoding="utf-8")


def test_a_non_freesound_file_needs_source_and_licence(noise_dirs):
    write(noise_dirs / "incoming" / "my street.wav", rate=44_100, channels=1)
    assert run() == 1
    assert run(source="own recording, Kolkata", license="own recording") == 0
    assert (noise_dirs / "my_street.wav").exists()


def test_a_too_short_clip_is_refused(noise_dirs):
    write(noise_dirs / "incoming" / "516655__soundensemble__short.wav", seconds=5)
    assert run() == 1
    assert not (noise_dirs / "freesound_516655.wav").exists()


def test_dry_run_writes_nothing(noise_dirs):
    write(noise_dirs / "incoming" / "263635__alcappuccino__market.flac")
    assert run(dry_run=True) == 0
    assert not (noise_dirs / "freesound_263635.wav").exists()
    assert not (noise_dirs / "SOURCES.md").exists()


def test_reimporting_keeps_one_row_per_file(noise_dirs):
    write(noise_dirs / "incoming" / "516655__soundensemble__delhi.flac")
    run()
    run()
    rows = [l for l in (noise_dirs / "SOURCES.md").read_text(encoding="utf-8").splitlines() if l.startswith("| `")]
    assert len(rows) == 1


def test_downmix_and_resample_preserve_the_level():
    rate = 48_000
    t = np.arange(rate * 2) / rate
    tone = 0.3 * np.sin(2 * np.pi * 440 * t)
    stereo = np.stack([tone, tone], axis=1)
    out = gn.to_mono_16k(stereo, rate)
    assert len(out) == 32_000
    assert np.sqrt(np.mean(out**2)) == pytest.approx(0.3 / np.sqrt(2), rel=0.02)
