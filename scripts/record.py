"""Record the eval corpus, one line at a time.

    uv run python -m scripts.record                 # every line not yet recorded
    uv run python -m scripts.record --speaker a     # only this speaker's share
    uv run python -m scripts.record --ids 004,017   # specific lines, e.g. to redo
    uv run python -m scripts.record --limit 40      # at most 40 this session, every intent represented
    uv run python -m scripts.record --limit 40 --dry-run   # show which 40, record nothing
    uv run python -m scripts.record --noise         # ambient noise for eval/noise/
    uv run python -m scripts.record --list-devices

Run `uv run python -m scripts.calibrate` first: it picks the input device,
checks the room is quiet enough, and saves the choice for this script.

Press Enter to start a take, Enter again to stop. The take is checked for
clipping and for being too quiet, then you can keep it or redo it. Already
recorded lines are skipped, so the session can be stopped and resumed.

**Record somewhere quiet.** The eval adds shop noise and the telephone channel
synthetically (`channel/degrade.py`), so every condition is controlled and
reproducible. A noisy source recording silently destroys that: the "clean"
baseline stops being clean, and every degraded condition is measured from the
wrong starting point.

No Gnani API calls are made here, so recording costs nothing.
"""

import argparse
import csv
import sys
import wave
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from scripts.calibrate import dbfs, load_config, record_until_enter

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_CSV = ROOT / "eval" / "corpus" / "scripts.csv"
AUDIO_DIR = ROOT / "eval" / "corpus" / "audio"
NOISE_DIR = ROOT / "eval" / "noise"

SAMPLE_RATE = 16_000  # what the browser sends, and what Prisma takes natively
CLIP_CEILING = 0.98  # fraction of full scale that counts as clipped
QUIET_RMS = 0.01  # below this the take is probably unusable


def describe(audio: np.ndarray) -> tuple[str, bool]:
    """Return a one-line verdict, and whether the take looks usable."""
    if audio.size == 0:
        return "nothing recorded", False
    peak = float(np.abs(audio).max())
    rms = float(np.sqrt(np.mean(audio**2)))
    seconds = audio.size / SAMPLE_RATE
    notes = []
    ok = True
    if peak >= CLIP_CEILING:
        notes.append("CLIPPED - move back from the mic")
        ok = False
    if rms < QUIET_RMS:
        notes.append("very quiet - move closer or raise the input gain")
        ok = False
    if seconds < 0.4:
        notes.append("very short - did it cut off?")
        ok = False
    verdict = f"{seconds:.1f}s  peak {dbfs(peak):5.1f} dBFS  rms {dbfs(rms):5.1f} dBFS"
    return verdict + ("  <- " + "; ".join(notes) if notes else "  ok"), ok


def save(audio: np.ndarray, path: Path) -> None:
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())


def take_until_happy(label: str, text: str, out: Path, device: int | None = None) -> bool:
    """Record, review, redo. Returns False if the user wants to quit."""
    while True:
        print(f"\n{label}  {text}")
        choice = input("  Enter = record, s = skip, q = quit > ").strip().lower()
        if choice == "q":
            return False
        if choice == "s":
            return True

        audio = record_until_enter("  recording... Enter to stop > ", device)
        verdict, ok = describe(audio)
        print(f"  {verdict}")
        if not ok:
            again = input("  r = redo, k = keep anyway, s = skip > ").strip().lower()
            if again == "s":
                return True
            if again != "k":
                continue
        else:
            again = input("  Enter = keep, r = redo, s = skip > ").strip().lower()
            if again == "s":
                return True
            if again == "r":
                continue
        save(audio, out)
        print(f"  saved {out.name}")
        return True


def spread_across_intents(rows: list[dict]) -> list[dict]:
    """Order lines so that any prefix has roughly the corpus's mix of intents.

    The file is grouped by intent, so a session stopped after 40 lines would
    otherwise record nothing but credit entries, and a partial held-out set
    would have no payments or stock lines at all. Each row is placed by its
    relative position within its own intent; file order breaks ties.
    """
    by_intent: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        by_intent[row["intent"]].append(i)
    position = {}
    for indices in by_intent.values():
        for rank, i in enumerate(indices):
            position[i] = (rank + 0.5) / len(indices)
    return [rows[i] for i in sorted(range(len(rows)), key=lambda i: (position[i], i))]


def device_still_matches(index: int, saved_name: str) -> bool:
    """Windows renumbers audio devices; the saved index must still be the same mic."""
    try:
        import sounddevice as sd

        return sd.query_devices(index)["name"] == saved_name
    except Exception:
        return False


def calibrated_device() -> int | None:
    config = load_config()
    if not config:
        print("No mic calibration found. Run this first, it takes a minute:")
        print("    uv run python -m scripts.calibrate\n")
        if input("Carry on with the system default anyway? [y/N] > ").strip().lower() != "y":
            raise SystemExit(0)
        return None
    name = config.get("device_name") or "system default"
    if config.get("device") is not None and config.get("device_name"):
        if not device_still_matches(config["device"], config["device_name"]):
            raise SystemExit(
                f"Device {config['device']} is no longer {config['device_name']!r} (Windows renumbered "
                "the devices). Run: uv run python -m scripts.calibrate"
            )
    print(f"Using {name}  (noise floor {config.get('noise_floor_dbfs')} dBFS, SNR {config.get('snr_db')} dB)")
    if not config.get("usable", True):
        print("  That calibration did not pass. Recording now risks a corpus the eval cannot use.")
        if input("  Carry on anyway? [y/N] > ").strip().lower() != "y":
            raise SystemExit("Run: uv run python -m scripts.calibrate")
    return config.get("device")


def record_corpus(args) -> int:
    if not SCRIPTS_CSV.exists():
        print(f"{SCRIPTS_CSV} not found", file=sys.stderr)
        return 2
    rows = list(csv.DictReader(SCRIPTS_CSV.open(encoding="utf-8")))

    wanted = {i.strip() for i in args.ids.split(",")} if args.ids else None
    todo = []
    for row in rows:
        if wanted is not None and row["id"] not in wanted:
            continue
        if wanted is None and args.speaker and row["speaker"] != args.speaker:
            continue
        if wanted is None and not args.redo and (AUDIO_DIR / f"{row['id']}.wav").exists():
            continue
        todo.append(row)
    if wanted is None:
        todo = spread_across_intents(todo)
    if args.limit:
        todo = todo[: args.limit]

    done = sum(1 for row in rows if (AUDIO_DIR / f"{row['id']}.wav").exists())
    print(f"{done}/{len(rows)} lines already recorded. {len(todo)} to do in this session.")
    if not todo:
        print("Nothing to record.")
        return 0
    mix = Counter(row["intent"] for row in todo)
    print("  this session: " + ", ".join(f"{intent} {n}" for intent, n in mix.most_common()))
    if args.dry_run:
        for row in todo:
            print(f"  {row['id']}  {row['intent']:15} {row['text']}")
        return 0
    print("Read each line naturally, as you would say it in a shop. Quiet room, no background music.")
    device = calibrated_device()

    for n, row in enumerate(todo, start=1):
        label = f"[{n}/{len(todo)}] {row['id']} ({row['intent']}, speaker {row['speaker']})"
        if not take_until_happy(label, row["text"], AUDIO_DIR / f"{row['id']}.wav", device):
            print("\nStopped. Run again to carry on where you left off.")
            break

    done = sum(1 for row in rows if (AUDIO_DIR / f"{row['id']}.wav").exists())
    print(f"\n{done}/{len(rows)} recorded.")
    return 0


def record_noise(args) -> int:
    """Background noise for the eval's SNR conditions."""
    print("Record 2-3 minutes of the kind of noise a shop has: a fan, traffic, chatter, a TV.")
    print("Several short clips are better than one long one. No speech you care about.")
    device = calibrated_device()
    n = 1
    while True:
        while (NOISE_DIR / f"noise_{n:02d}.wav").exists():
            n += 1
        label = f"[noise_{n:02d}]"
        if not take_until_happy(label, "(ambient noise - stay quiet)", NOISE_DIR / f"noise_{n:02d}.wav", device):
            break
        if input("  another clip? Enter = yes, q = done > ").strip().lower() == "q":
            break
    existing = sorted(NOISE_DIR.glob("*.wav"))
    total = 0.0
    for path in existing:
        with wave.open(str(path)) as w:
            total += w.getnframes() / w.getframerate()
    print(f"\n{len(existing)} noise clips, {total / 60:.1f} minutes total.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--speaker", help="record only this speaker's share (a, b, c)")
    parser.add_argument("--ids", help="comma-separated ids to record, e.g. 004,017")
    parser.add_argument("--redo", action="store_true", help="include lines already recorded")
    parser.add_argument("--limit", type=int, help="record at most N lines this session, spread across intents")
    parser.add_argument("--dry-run", action="store_true", help="list the lines this session would record")
    parser.add_argument("--noise", action="store_true", help="record ambient noise instead")
    parser.add_argument("--list-devices", action="store_true")
    args = parser.parse_args()

    if args.list_devices:
        from scripts.calibrate import print_devices

        print_devices()
        return 0
    return record_noise(args) if args.noise else record_corpus(args)


if __name__ == "__main__":
    raise SystemExit(main())
