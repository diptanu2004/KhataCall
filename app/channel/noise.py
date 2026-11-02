"""The background-noise bank: imported Indian street recordings (docs/NOISE.md).

Shared by the eval (eval/conditions.py) and the demo's phone-line mode, so both
mix in the same shop noise.
"""

from pathlib import Path

import numpy as np
import soundfile as sf

__all__ = ["load_noise_bank"]


def load_noise_bank(directory: Path, pattern: str = "freesound_*.wav") -> dict[str, np.ndarray]:
    """The imported noise files, as float32 16 kHz mono, by file stem. Empty if none."""
    bank = {}
    for path in sorted(Path(directory).glob(pattern)):
        audio, rate = sf.read(path, dtype="float32")
        if rate != 16_000 or audio.ndim != 1:
            raise ValueError(f"{path.name} is not 16 kHz mono; re-import it with scripts.get_noise")
        bank[path.stem] = audio
    return bank
