"""Prepare the background noise the eval mixes into the corpus.

    uv run python -m scripts.get_noise                 # import everything in eval/noise/incoming/
    uv run python -m scripts.get_noise --dry-run
    uv run python -m scripts.get_noise --demand PCAFETER    # optional reference noise, slow

**Primary source: Indian street and market recordings from Freesound**, licensed
CC0 (DECISIONS.md D-033). Download the original files from the Freesound pages
(a free account is needed), drop them into eval/noise/incoming/ unrenamed, and
run this. Freesound names downloads "<id>__<uploader>__<title>", so each one
is traced back to its page automatically.

Each file is converted to what the eval expects: mono, 16 kHz, 16-bit WAV,
with no other change - the eval sets the level itself, per condition, from the
target SNR. The source and licence of every file go into eval/noise/SOURCES.md,
which is committed; the audio is gitignored.

**Optional reference: DEMAND** (Thiemann, Ito and Vincent, 2013; CC BY 4.0;
DOI 10.5281/zenodo.1227121), the noise half of a standard speech-enhancement
benchmark. Zenodo served it at about 55 KB/s when tried, roughly half an hour
per environment, so it is opt-in.

No Gnani API calls are made here.
"""

import argparse
import re
import sys
import tempfile
import zipfile
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
NOISE_DIR = ROOT / "eval" / "noise"
INCOMING = NOISE_DIR / "incoming"
SOURCES = NOISE_DIR / "SOURCES.md"
TARGET_RATE = 16_000
MIN_SECONDS = 20.0
AUDIO_SUFFIXES = {".wav", ".flac", ".ogg", ".mp3", ".aif", ".aiff"}

# Freesound sounds whose CC0 licence was checked (search filtered to CC0, Oct 22).
# Anything else needs --license, because a filename cannot prove a licence.
KNOWN_FREESOUND = {
    516655: "CC0 1.0",  # SoundEnsemble: Delhi street, traffic, horns, Hindi voices, bazaar
    263635: "CC0 1.0",  # Alcappuccino: Trivandrum street market, voices, tuktuks, honking
    552692: "CC0 1.0",  # nickcase: flea market in India
    451528: "CC0 1.0",  # kyles: traffic near a covered sidewalk
}
FREESOUND_NAME = re.compile(r"^(\d+)__([^_]+(?:_[^_]+)*?)__(.+)$")

DEMAND_RECORD = "https://zenodo.org/api/records/1227121"
DEMAND_CITATION = (
    "DEMAND, Thiemann, Ito and Vincent (2013), CC BY 4.0, "
    "[doi:10.5281/zenodo.1227121](https://doi.org/10.5281/zenodo.1227121)"
)

HEADER = """# Background noise sources

The eval mixes these recordings into the corpus at controlled signal-to-noise
ratios. Each was converted to mono 16 kHz 16-bit WAV and otherwise left as
recorded; the eval sets its level per condition. The audio is not committed:
recreate it with `uv run python -m scripts.get_noise` (see its docstring).

| File | Source | Licence | Description |
|---|---|---|---|
"""


def to_mono_16k(samples: np.ndarray, rate: int) -> np.ndarray:
    from scipy.signal import resample_poly

    mono = samples.mean(axis=1) if samples.ndim > 1 else samples
    if rate != TARGET_RATE:
        divisor = gcd(rate, TARGET_RATE)
        mono = resample_poly(mono, TARGET_RATE // divisor, rate // divisor)
    return np.clip(mono, -1.0, 1.0).astype(np.float32)


SILENT_EDGE_DBFS = -45.0
EDGE_WINDOW_MS = 100


def trim_silent_edges(mono: np.ndarray, rate: int = TARGET_RATE) -> tuple[np.ndarray, float, float]:
    """Cut near-silent stretches from the start and end; return (audio, cut_start_s, cut_end_s).

    Recordings often fade in or out. The flea-market clip ends in a full second of
    digital silence. The eval cuts random stretches from these files and scales each
    to a target SNR, so a cut landing in silence would make a "10 dB" condition
    almost noise-free - or, scaled up, pure amplified hiss.
    """
    window = max(1, rate * EDGE_WINDOW_MS // 1000)
    usable = mono[: len(mono) - len(mono) % window]
    if not len(usable):
        return mono, 0.0, 0.0
    energy = np.sqrt((usable.reshape(-1, window).astype(np.float64) ** 2).mean(axis=1))
    loud = np.flatnonzero(20 * np.log10(energy + 1e-12) > SILENT_EDGE_DBFS)
    if not len(loud):
        return mono[:0], len(mono) / rate, 0.0
    start, end = loud[0] * window, (loud[-1] + 1) * window
    return mono[start:end], start / rate, (len(mono) - end) / rate


def describe(path: Path) -> tuple[str, str, str, str]:
    """(output name, source, licence, description) for an incoming file."""
    stem = path.stem
    match = FREESOUND_NAME.match(stem)
    if match:
        sound_id, uploader, title = int(match[1]), match[2], match[3].replace("_", " ")
        return (
            f"freesound_{sound_id}.wav",
            f"[freesound.org/s/{sound_id}](https://freesound.org/s/{sound_id}/)",
            KNOWN_FREESOUND.get(sound_id, ""),
            f"{uploader}: {title}"[:90],
        )
    slug = re.sub(r"[^a-z0-9]+", "_", stem.lower()).strip("_")[:40]
    return f"{slug}.wav", "", "", stem[:90]


def read_rows() -> dict[str, str]:
    if not SOURCES.exists():
        return {}
    rows = {}
    for line in SOURCES.read_text(encoding="utf-8").splitlines():
        if line.startswith("| `"):
            rows[line.split("`")[1]] = line
    return rows


def write_rows(rows: dict[str, str]) -> None:
    SOURCES.write_text(HEADER + "\n".join(rows[k] for k in sorted(rows)) + "\n", encoding="utf-8")


def import_incoming(args) -> int:
    INCOMING.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in INCOMING.iterdir() if p.suffix.lower() in AUDIO_SUFFIXES)
    if not files:
        print(f"Nothing in {INCOMING.relative_to(ROOT)}.")
        print("Download recordings from the Freesound pages listed in docs/NOISE.md, put them there, and rerun.")
        return 1

    rows = read_rows()
    failed = False
    for path in files:
        name, source, licence, description = describe(path)
        licence = licence or args.license or ""
        source = source or args.source or ""
        if not licence or not source:
            print(f"  {path.name}: unknown source or licence. Pass --source URL --license NAME for it.")
            failed = True
            continue
        info = sf.info(path)
        print(f"  {path.name[:60]:60} {info.samplerate} Hz x{info.channels}, {info.duration:5.1f} s")
        if info.duration < MIN_SECONDS:
            print(f"    too short ({info.duration:.0f} s); want at least {MIN_SECONDS:.0f} s of noise")
            failed = True
            continue
        if args.dry_run:
            print(f"    -> {name}  ({licence})")
            continue
        samples, rate = sf.read(path, dtype="float32", always_2d=True)
        mono, cut_start, cut_end = trim_silent_edges(to_mono_16k(samples, rate))
        if len(mono) < MIN_SECONDS * TARGET_RATE:
            print(f"    only {len(mono) / TARGET_RATE:.0f} s left after trimming silent edges, skipped")
            failed = True
            continue
        sf.write(NOISE_DIR / name, mono, TARGET_RATE, subtype="PCM_16")
        rows[name] = f"| `{name}` | {source} | {licence} | {description} |"
        trimmed = f", trimmed {cut_start:.1f} s + {cut_end:.1f} s of silent edges" if cut_start or cut_end else ""
        print(f"    -> {name}  ({len(mono) / TARGET_RATE:.0f} s at 16 kHz mono{trimmed})")

    if not args.dry_run:
        write_rows(rows)
        print(f"sources recorded in {SOURCES.relative_to(ROOT)}")
    return 1 if failed else 0


def fetch_demand(envs: list[str], dry_run: bool) -> int:
    import httpx

    with httpx.Client() as client:
        index = {f["key"]: f for f in client.get(DEMAND_RECORD, timeout=30).raise_for_status().json()["files"]}
        rows = read_rows()
        for env in (e.upper() for e in envs):
            entry = index.get(f"{env}_16k.zip")
            if entry is None:
                print(f"unknown DEMAND environment {env}", file=sys.stderr)
                return 2
            name = f"demand_{env.lower()}.wav"
            print(f"  {env}: {entry['size'] / 1e6:.0f} MB to download -> {name}")
            if dry_run or (NOISE_DIR / name).exists():
                continue
            with tempfile.TemporaryFile() as tmp:
                done = 0
                with client.stream("GET", entry["links"]["self"], timeout=120, follow_redirects=True) as r:
                    r.raise_for_status()
                    for chunk in r.iter_bytes(1 << 20):
                        tmp.write(chunk)
                        done += len(chunk)
                        print(f"\r    {done / 1e6:6.1f} / {entry['size'] / 1e6:.1f} MB", end="", flush=True)
                print()
                tmp.seek(0)
                with zipfile.ZipFile(tmp) as z:
                    member = next(n for n in z.namelist() if n.lower().endswith("ch01.wav"))
                    (NOISE_DIR / name).write_bytes(z.read(member))
            rows[name] = f"| `{name}` | {DEMAND_CITATION} | CC BY 4.0 | {env}, channel 1 |"
        if not dry_run:
            write_rows(rows)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--source", help="source URL, for a file not downloaded from Freesound")
    parser.add_argument("--license", help="licence, for a file whose licence is not already known")
    parser.add_argument("--demand", nargs="+", metavar="ENV", help="also fetch DEMAND environments (slow)")
    args = parser.parse_args()
    NOISE_DIR.mkdir(parents=True, exist_ok=True)
    if args.demand:
        return fetch_demand(args.demand, args.dry_run)
    return import_incoming(args)


if __name__ == "__main__":
    raise SystemExit(main())
