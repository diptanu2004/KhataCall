"""The channel conditions the eval measures, and how each clip's randomness is seeded.

Ten conditions (CLAUDE.md §5): clean wideband audio, the bare phone channel,
the phone with packet loss, the phone with shop noise, and the worst case.

Randomness is seeded per (utterance, condition), so any degraded clip can be
regenerated exactly, and changing one condition never reshuffles another. The
noise file is chosen per utterance from the bank, so every condition sees the
same mix of Delhi, Trivandrum and flea-market noise.
"""

import zlib
from pathlib import Path

import numpy as np
import soundfile as sf

from app.channel.degrade import Condition

__all__ = ["CONDITIONS", "NOISE_DIR", "by_id", "load_noise_bank", "noise_for", "rng_for"]

NOISE_DIR = Path(__file__).resolve().parent / "noise"

CONDITIONS: tuple[Condition, ...] = (
    Condition("clean"),
    Condition("tel", telephone=True),
    Condition("tel_loss1", telephone=True, loss_pct=1),
    Condition("tel_loss5", telephone=True, loss_pct=5),
    Condition("tel_loss10", telephone=True, loss_pct=10),
    Condition("tel_snr20", telephone=True, snr_db=20),
    Condition("tel_snr10", telephone=True, snr_db=10),
    Condition("tel_snr5", telephone=True, snr_db=5),
    Condition("tel_snr0", telephone=True, snr_db=0),
    Condition("worst", telephone=True, snr_db=5, loss_pct=5),
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
    bank = {}
    for path in sorted(directory.glob(pattern)):
        audio, rate = sf.read(path, dtype="float32")
        if rate != 16_000 or audio.ndim != 1:
            raise ValueError(f"{path.name} is not 16 kHz mono; re-import it with scripts.get_noise")
        bank[path.stem] = audio
    return bank


def noise_for(utterance_id: str, bank: dict[str, np.ndarray]) -> tuple[str, np.ndarray]:
    """Which noise file this utterance gets, the same in every condition."""
    if not bank:
        raise ValueError("no noise files; see docs/NOISE.md")
    names = sorted(bank)
    name = names[zlib.crc32(utterance_id.encode()) % len(names)]
    return name, bank[name]
