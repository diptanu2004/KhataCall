"""Phone-line mode for the live demo: the eval's mobile channel, applied to one utterance.

What the demo plays is what the eval measured. The condition is built the same
way as the eval's `mob_*` grid (GSM 06.10 then G.711, D-036), and the degraded
8 kHz audio goes to Prisma REST, the same endpoint the eval used (D-037).

It works on the whole utterance, after push-to-talk is released, rather than
frame by frame: the SNR is set against the utterance's active speech power,
which is not known until the owner stops speaking.
"""

from dataclasses import dataclass

import numpy as np

from app.adapters.base import AudioClip
from app.channel.degrade import Condition, degrade, to_pcm16

__all__ = ["MIN_UTTERANCE_S", "PhoneLine", "PhoneResult"]

# A press shorter than this is a slip of the finger, not speech; it is not sent.
MIN_UTTERANCE_S = 0.2


@dataclass(frozen=True)
class PhoneResult:
    clip: AudioClip  # degraded, at 8 kHz: what Prisma hears
    noise: str | None  # which noise recording was mixed in


class PhoneLine:
    def __init__(self, noise_bank: dict[str, np.ndarray]) -> None:
        self._bank = noise_bank

    @property
    def has_noise(self) -> bool:
        return bool(self._bank)

    def apply(
        self,
        pcm16: bytes,
        *,
        rate: int,
        snr_db: float | None,
        loss_pct: float,
        rng: np.random.Generator,
    ) -> PhoneResult:
        """Degrade one utterance of PCM16 mono. Noise is skipped if there is no bank."""
        speech = np.frombuffer(pcm16, dtype="<i2").astype(np.float32) / 32768
        noise_name, noise = None, None
        if snr_db is not None and self._bank:
            names = sorted(self._bank)
            noise_name = names[int(rng.integers(len(names)))]
            noise = self._bank[noise_name]
        condition = Condition(
            "live",
            telephone=True,
            codec="gsm",
            snr_db=snr_db if noise is not None else None,
            loss_pct=loss_pct,
        )
        degraded = degrade(speech, condition, rng, noise, rate)
        return PhoneResult(AudioClip(to_pcm16(degraded), condition.output_rate), noise_name)
