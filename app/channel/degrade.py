"""Simulate a phone call from a noisy shop, one stage at a time.

Shared by the eval harness and the demo's "phone line mode", so what the demo
plays is what the eval measured. Order matters, because it follows the
physical path of the sound:

    1. shop noise mixed in at a target SNR      (in the shop, before the phone)
    2. band-limit to 300-3400 Hz, resample to 8 kHz      (the phone channel)
    3. G.711 mu-law encode and decode                    (the network codec)
    4. bursty packet loss on 20 ms frames, with concealment   (the network)

Every stage is a pure function of its inputs and an explicit numpy Generator,
so a condition reproduces exactly from its seed. Audio is float32 in [-1, 1].

Two measurement choices, both deliberate:

- **SNR uses active speech power**, not whole-clip power. A take is part
  silence; averaging the gaps in would understate the speech level and make
  every condition noisier than its label.
- **SNR uses the power of the noise segment actually used**, not the noise
  file's average. The flea-market recording varies 12 dB between quiet and
  busy stretches (D-033), so a file-wide figure would mislabel conditions.
"""

from dataclasses import dataclass
from math import gcd

import numpy as np
from scipy.signal import butter, resample_poly, sosfilt

__all__ = [
    "Condition",
    "active_speech_power",
    "conceal",
    "degrade",
    "gilbert_elliott_mask",
    "gsm_round_trip",
    "mix_noise",
    "mulaw_decode",
    "mulaw_encode",
    "packet_loss",
    "telephone_band",
    "to_pcm16",
]

WIDEBAND_RATE = 16_000
PHONE_RATE = 8_000
FRAME_MS = 20
PHONE_FRAME = PHONE_RATE * FRAME_MS // 1000  # 160 samples
EPS = 1e-12


# --- 1. noise ----------------------------------------------------------------


def active_speech_power(x: np.ndarray, rate: int = WIDEBAND_RATE, frame_ms: int = FRAME_MS) -> float:
    """Mean power over the frames that contain speech.

    A simplified take on ITU-T P.56: frames within 30 dB of the loudest
    (95th percentile) frame count as active. Plain RMS over a take that is
    half silence would read 3 dB low.
    """
    frame = max(1, rate * frame_ms // 1000)
    usable = x[: len(x) - len(x) % frame].astype(np.float64)
    if len(usable) == 0:
        return float(np.mean(x.astype(np.float64) ** 2)) if len(x) else 0.0
    powers = (usable.reshape(-1, frame) ** 2).mean(axis=1)
    levels = 10 * np.log10(powers + EPS)
    active = powers[levels > np.percentile(levels, 95) - 30]
    return float(active.mean()) if len(active) else float(powers.mean())


@dataclass(frozen=True)
class Mix:
    mixed: np.ndarray
    speech: np.ndarray  # the speech as it sits in `mixed`, after any headroom scaling
    noise: np.ndarray  # likewise for the noise
    snr_db: float  # measured, from the two components


def mix_noise(speech: np.ndarray, noise: np.ndarray, snr_db: float, rng: np.random.Generator) -> Mix:
    """Add a random stretch of `noise` to `speech` at exactly `snr_db`."""
    if len(noise) == 0:
        raise ValueError("empty noise")
    n = len(speech)
    if len(noise) >= n:
        start = int(rng.integers(0, len(noise) - n + 1))
        segment = noise[start : start + n]
    else:  # loop a short noise file from a random offset
        reps = -(-(n + len(noise)) // len(noise))
        start = int(rng.integers(0, len(noise)))
        segment = np.tile(noise, reps)[start : start + n]
    segment = segment.astype(np.float64)

    speech_power = active_speech_power(speech)
    noise_power = float(np.mean(segment**2))
    if noise_power < EPS:
        raise ValueError("the noise segment is silent; trim silence from the noise file")
    gain = np.sqrt(speech_power / (noise_power * 10 ** (snr_db / 10)))
    s, v = speech.astype(np.float64), segment * gain

    # Keep the sum inside full scale by scaling both together, which leaves SNR unchanged.
    peak = float(np.max(np.abs(s + v))) if n else 0.0
    if peak > 0.99:
        s, v = s * (0.99 / peak), v * (0.99 / peak)
    measured = 10 * np.log10(active_speech_power(s) / (np.mean(v**2) + EPS))
    return Mix((s + v).astype(np.float32), s.astype(np.float32), v.astype(np.float32), float(measured))


# --- 2. the phone band -------------------------------------------------------


def telephone_band(x: np.ndarray, rate: int = WIDEBAND_RATE) -> np.ndarray:
    """300-3400 Hz band-pass, then resample to 8 kHz.

    The filter is causal (a real channel cannot look ahead). The resampler has
    its own anti-aliasing filter; the band-pass already sits below 4 kHz.
    """
    sos = butter(4, [300, 3400], btype="bandpass", fs=rate, output="sos")
    filtered = sosfilt(sos, x.astype(np.float64))
    divisor = gcd(rate, PHONE_RATE)
    return resample_poly(filtered, PHONE_RATE // divisor, rate // divisor).astype(np.float32)


# --- 3. G.711 mu-law -------------------------------------------------------
# The segmented G.711 algorithm on 14-bit magnitudes, not the continuous
# mu-law curve: 8 segments of 16 steps. (audioop, which used to do this, is
# gone from Python 3.13.)

_BIAS = 0x84
_CLIP = 32635


def mulaw_encode(x: np.ndarray) -> np.ndarray:
    """Float audio in [-1, 1] to G.711 mu-law bytes."""
    pcm = np.clip(np.round(np.asarray(x, dtype=np.float64) * 32767), -32768, 32767).astype(np.int32)
    sign = np.where(pcm < 0, 0x80, 0)
    magnitude = np.minimum(np.abs(pcm), _CLIP) + _BIAS
    exponent = np.clip(np.floor(np.log2(magnitude)).astype(np.int32) - 7, 0, 7)
    mantissa = (magnitude >> (exponent + 3)) & 0x0F
    return (~(sign | (exponent << 4) | mantissa) & 0xFF).astype(np.uint8)


def mulaw_decode(codes: np.ndarray) -> np.ndarray:
    """G.711 mu-law bytes back to float audio."""
    c = ~np.asarray(codes, dtype=np.int32) & 0xFF
    exponent = (c >> 4) & 0x07
    mantissa = c & 0x0F
    magnitude = (((mantissa << 3) + _BIAS) << exponent) - _BIAS
    return (np.where(c & 0x80, -magnitude, magnitude) / 32768.0).astype(np.float32)


def gsm_round_trip(x: np.ndarray) -> np.ndarray:
    """GSM 06.10 full-rate (13 kbps), the 2G mobile speech codec, at 8 kHz.

    G.711 alone is landline quality and sounded too kind for a shop owner
    calling from a mobile (D-036). This is libsndfile's real GSM 06.10 codec,
    not an imitation: measured zero delay and a waveform SNR of 8.9 dB against
    G.711's ~38. Output is trimmed to the input length (GSM-in-WAV pads to
    whole 320-sample blocks).
    """
    import io

    import soundfile as sf

    buffer = io.BytesIO()
    sf.write(buffer, np.clip(x, -1.0, 1.0), PHONE_RATE, format="WAV", subtype="GSM610")
    buffer.seek(0)
    y, _ = sf.read(buffer, dtype="float32")
    if len(y) < len(x):
        y = np.pad(y, (0, len(x) - len(y)))
    return y[: len(x)].astype(np.float32)


# --- 4. packet loss --------------------------------------------------------


def gilbert_elliott_mask(n_frames: int, loss_rate: float, mean_burst: float, rng: np.random.Generator) -> np.ndarray:
    """True for each lost frame. A two-state Markov chain: losses come in bursts.

    In the bad state every frame is lost, in the good state none (the simple
    Gilbert model). With p = P(good -> bad) and r = P(bad -> good), the long-run
    loss rate is p / (p + r) and the mean burst is 1 / r frames.
    """
    if not 0 <= loss_rate < 1:
        raise ValueError("loss_rate must be in [0, 1)")
    if loss_rate == 0 or n_frames == 0:
        return np.zeros(n_frames, dtype=bool)
    if mean_burst < 1:
        raise ValueError("mean_burst is in frames and must be at least 1")
    r = 1.0 / mean_burst
    p = loss_rate * r / (1 - loss_rate)
    if p > 1:
        raise ValueError("that loss rate is impossible with bursts this short")
    draws = rng.random(n_frames)
    lost = np.zeros(n_frames, dtype=bool)
    bad = draws[0] < loss_rate  # start in the stationary distribution
    for i in range(n_frames):
        if i:
            bad = draws[i] < (1 - r) if bad else draws[i] < p
        lost[i] = bad
    return lost


def conceal(x: np.ndarray, lost: np.ndarray, how: str = "zero", frame: int = PHONE_FRAME) -> np.ndarray:
    """Fill lost frames: silence, or a copy of the last frame that arrived."""
    if how not in ("zero", "repeat"):
        raise ValueError("concealment is 'zero' or 'repeat'")
    out = x.copy()
    last_good = np.zeros(frame, dtype=x.dtype)
    for i, is_lost in enumerate(lost):
        chunk = out[i * frame : (i + 1) * frame]
        if is_lost:
            chunk[:] = 0 if how == "zero" else last_good[: len(chunk)]
        elif len(chunk) == frame:
            last_good = chunk.copy()
    return out


def packet_loss(
    x: np.ndarray, loss_rate: float, rng: np.random.Generator, *, mean_burst: float = 2.0, how: str = "zero"
) -> tuple[np.ndarray, np.ndarray]:
    """Drop 20 ms frames with bursty loss; return (audio, lost-frame mask)."""
    n_frames = -(-len(x) // PHONE_FRAME)
    lost = gilbert_elliott_mask(n_frames, loss_rate, mean_burst, rng)
    return conceal(x, lost, how), lost


# --- the whole channel -----------------------------------------------------


@dataclass(frozen=True)
class Condition:
    """One channel condition. All-default is the clean, untouched wideband signal."""

    id: str
    telephone: bool = False
    snr_db: float | None = None
    loss_pct: float = 0.0
    mean_burst: float = 2.0
    concealment: str = "zero"
    # "g711": landline / VoIP. "gsm": a mobile call - GSM 06.10 on the radio leg,
    # then G.711 in the core network, the tandem a real mobile-to-PSTN call goes through.
    codec: str = "g711"

    @property
    def output_rate(self) -> int:
        return PHONE_RATE if self.telephone else WIDEBAND_RATE


def degrade(
    speech: np.ndarray,
    condition: Condition,
    rng: np.random.Generator,
    noise: np.ndarray | None = None,
    rate: int = WIDEBAND_RATE,
) -> np.ndarray:
    """Apply a condition to 16 kHz speech. Returns audio at `condition.output_rate`."""
    x = speech.astype(np.float32)
    if condition.snr_db is not None:
        if noise is None:
            raise ValueError(f"condition {condition.id} needs noise")
        x = mix_noise(x, noise, condition.snr_db, rng).mixed
    if not condition.telephone:
        if condition.loss_pct:
            raise ValueError("packet loss is a property of the phone network; set telephone=True")
        return x
    x = telephone_band(x, rate)
    if condition.codec == "gsm":
        x = gsm_round_trip(x)
    elif condition.codec != "g711":
        raise ValueError(f"unknown codec {condition.codec!r}")
    x = mulaw_decode(mulaw_encode(x))
    if condition.loss_pct:
        x, _ = packet_loss(x, condition.loss_pct / 100, rng, mean_burst=condition.mean_burst, how=condition.concealment)
    return x


def to_pcm16(x: np.ndarray) -> bytes:
    """Float audio to little-endian PCM16 bytes, the format the STT adapters take."""
    return (np.clip(x, -1.0, 1.0) * 32767).round().astype("<i2").tobytes()
