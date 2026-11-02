"""The channel conditions the eval measures, and how each clip's randomness is seeded.

Twelve conditions: clean wideband audio; a landline reference (G.711); and a
mobile call (GSM then G.711, D-036) bare, with packet loss, with Indian shop
noise, and the two together.

Randomness is seeded per (utterance, condition), so any degraded clip can be
regenerated exactly, and changing one condition never reshuffles another. The
noise file is chosen per utterance from the bank, so every condition sees the
same mix of Delhi, Trivandrum and flea-market noise.
"""

import zlib
from pathlib import Path

import numpy as np

from app.channel.degrade import Condition
from app.channel.noise import load_noise_bank as _load_noise_bank

__all__ = ["CONDITIONS", "NOISE_DIR", "by_id", "load_noise_bank", "noise_for", "rng_for"]

NOISE_DIR = Path(__file__).resolve().parent / "noise"

CONDITIONS: tuple[Condition, ...] = (
    Condition("clean"),
    # Landline: G.711 only. Kept as a reference; it sounded too kind for a shop (D-036).
    Condition("landline", telephone=True),
    # Mobile: GSM 06.10 then G.711, the path of a real mobile-to-landline call.
    Condition("mob", telephone=True, codec="gsm"),
    Condition("mob_loss1", telephone=True, codec="gsm", loss_pct=1),
    Condition("mob_loss5", telephone=True, codec="gsm", loss_pct=5),
    Condition("mob_loss10", telephone=True, codec="gsm", loss_pct=10),
    Condition("mob_snr20", telephone=True, codec="gsm", snr_db=20),
    Condition("mob_snr10", telephone=True, codec="gsm", snr_db=10),
    Condition("mob_snr5", telephone=True, codec="gsm", snr_db=5),
    Condition("mob_snr0", telephone=True, codec="gsm", snr_db=0),
    Condition("mob_worst", telephone=True, codec="gsm", snr_db=5, loss_pct=5),
    Condition("mob_harsh", telephone=True, codec="gsm", snr_db=0, loss_pct=10),
)


def by_id(condition_id: str) -> Condition:
    for condition in CONDITIONS:
        if condition.id == condition_id:
            return condition
    raise KeyError(f"no condition {condition_id!r}; known: {', '.join(c.id for c in CONDITIONS)}")


def rng_for(utterance_id: str, condition_id: str) -> np.random.Generator:
    """A generator that depends only on this utterance and this condition.

    crc32 rather than hash(): Python salts str hashes per process, which would
    give every run different noise and packet losses.
    """
    return np.random.default_rng(zlib.crc32(f"{utterance_id}:{condition_id}".encode()))


def load_noise_bank(directory: Path = NOISE_DIR, pattern: str = "freesound_*.wav") -> dict[str, np.ndarray]:
    """The imported noise files, as float32 16 kHz mono, by file stem."""
    return _load_noise_bank(directory, pattern)


def noise_for(utterance_id: str, bank: dict[str, np.ndarray]) -> tuple[str, np.ndarray]:
    """Which noise file this utterance gets, the same in every condition."""
    if not bank:
        raise ValueError("no noise files; see docs/NOISE.md")
    names = sorted(bank)
    name = names[zlib.crc32(utterance_id.encode()) % len(names)]
    return name, bank[name]
