"""Check the recorded corpus before it is trusted as ground truth.

    uv run python -m scripts.check_corpus              # summary plus anything suspect
    uv run python -m scripts.check_corpus --all        # every file, one line each
    uv run python -m scripts.check_corpus --speaker a
    uv run python -m scripts.check_corpus --redo       # print a scripts.record --ids command

Calibration checks the setup once. This checks what actually came out, take by
take: a lorry going past during take 94 is invisible until the eval produces a
number nobody can explain.

Every check exists because of how the eval works (DECISIONS.md D-021). The
harness mixes noise at known SNRs and applies the telephone channel on top of
these files, so each one has to start clean and intact:

  clipping          distortion the degradation cannot undo, and it is not the
                    distortion we mean to measure
  low SNR           if a take sits near 20 dB, the eval's 20 dB condition is
                    measuring the room instead of the added noise
  truncation        a clipped-off first or last word becomes a transcription
                    error that gets blamed on the channel
  silence           long gaps inflate audio-seconds, which is what Gnani bills

No Gnani API calls are made here.
"""

import argparse
import csv
import sys
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_CSV = ROOT / "eval" / "corpus" / "scripts.csv"
AUDIO_DIR = ROOT / "eval" / "corpus" / "audio"

EXPECTED_RATE = 16_000
CLIP_LEVEL = 0.99
MAX_CLIPPED_SAMPLES = 3  # a few samples at the ceiling is nothing; a run of them is not
MIN_SECONDS = 0.7
MAX_SECONDS = 12.0
MIN_SNR_DB = 25.0  # calibrate.py asks 30 of the setup; allow a little slack per take
MIN_PEAK_DBFS = -30.0
EDGE_MS = 60  # a word cut off at the very start or end shows up here
EDGE_ACTIVE_RATIO = 0.35  # speech this loud at the very edge means it was truncated
MAX_LEAD_TRAIL_S = 2.0

FLOOR = 1e-10


def dbfs(x: float) -> float:
    return 20 * np.log10(max(float(x), FLOOR))


@dataclass
class Take:
    id: str
    path: Path
    seconds: float
    rate: int
    peak_dbfs: float
    speech_dbfs: float
    noise_dbfs: float
    clipped: int
    lead_s: float
    trail_s: float
    edge_energy: float
    # Windows voice processing gates near-silence to digital zero. Measured on this
    # corpus: 6.6-16.1% of frames per file, every one of them outside the speech span.
    gated_frames: int
    gated_inside_speech: int
    total_frames: int

    @property
    def snr_db(self) -> float:
        return self.speech_dbfs - self.noise_dbfs

    @property
    def gated_fraction(self) -> float:
        return self.gated_frames / self.total_frames if self.total_frames else 0.0


def frame_energies(audio: np.ndarray, frame_ms: int = 20) -> np.ndarray:
    frame = max(1, EXPECTED_RATE * frame_ms // 1000)
    usable = audio[: audio.size - audio.size % frame]
    if usable.size < frame:
        return np.array([np.sqrt(np.mean(audio**2))]) if audio.size else np.array([0.0])
    return np.sqrt((usable.reshape(-1, frame).astype(np.float64) ** 2).mean(axis=1))


def measure(path: Path, take_id: str) -> Take:
    with wave.open(str(path)) as w:
        rate = w.getframerate()
        channels = w.getnchannels()
        raw = w.readframes(w.getnframes())
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    energies = frame_energies(audio)
    # Windows voice processing zeroes near-silence outright. Those frames are not a
    # noise measurement, they are the absence of one, and averaging them in reports a
    # physically impossible SNR (166 dB, as this script first did). Measure the floor
    # from the quietest frames that actually contain something.
    nonzero = energies[energies > 0]
    quiet = np.sort(nonzero)[: max(1, len(nonzero) // 10)] if nonzero.size else np.array([0.0])
    loud = energies[energies >= np.median(energies)]
    noise = float(quiet.mean())
    speech = float(loud.mean()) if loud.size else 0.0

    # Where speech starts and stops, using a threshold between the two levels.
    threshold = max(noise * 4, speech * 0.08)
    active = np.flatnonzero(energies > threshold)
    frame_s = 0.02
    lead = float(active[0] * frame_s) if active.size else 0.0
    trail = float((len(energies) - 1 - active[-1]) * frame_s) if active.size else 0.0

    silent = energies == 0
    inside_speech = (
        int(silent[active[0] : active[-1] + 1].sum()) if active.size else 0
    )

    edge = max(1, EXPECTED_RATE * EDGE_MS // 1000)
    edge_rms = max(
        float(np.sqrt(np.mean(audio[:edge] ** 2))), float(np.sqrt(np.mean(audio[-edge:] ** 2)))
    ) if audio.size >= edge else 0.0

    return Take(
        id=take_id,
        path=path,
        seconds=audio.size / rate if rate else 0.0,
        rate=rate,
        peak_dbfs=dbfs(float(np.abs(audio).max()) if audio.size else 0.0),
        speech_dbfs=dbfs(speech),
        noise_dbfs=dbfs(noise),
        clipped=int(np.sum(np.abs(audio) >= CLIP_LEVEL)),
        lead_s=lead,
        trail_s=trail,
        edge_energy=edge_rms / speech if speech > 0 else 0.0,
        gated_frames=int(silent.sum()),
        gated_inside_speech=inside_speech,
        total_frames=len(energies),
    )


def problems(t: Take) -> list[str]:
    """Reasons to redo this take. Empty means it is fine."""
    out = []
    if t.rate != EXPECTED_RATE:
        out.append(f"{t.rate} Hz, expected {EXPECTED_RATE}")
    if t.seconds < MIN_SECONDS:
        out.append(f"only {t.seconds:.1f}s")
    elif t.seconds > MAX_SECONDS:
        out.append(f"{t.seconds:.1f}s is unusually long")
    if t.clipped > MAX_CLIPPED_SAMPLES:
        out.append(f"clipped ({t.clipped} samples)")
    if t.peak_dbfs < MIN_PEAK_DBFS:
        out.append(f"very quiet (peak {t.peak_dbfs:.0f} dBFS)")
    if t.snr_db < MIN_SNR_DB:
        out.append(f"SNR {t.snr_db:.0f} dB - too close to the eval's own conditions")
    if t.edge_energy > EDGE_ACTIVE_RATIO:
        out.append("speech at the very start or end - likely cut off")
    if t.gated_inside_speech:
        # Trimming the silence around a take is harmless. Zeroing frames between the
        # first and last word means the noise gate ate part of a word - a quiet
        # fricative or a soft ending - and that error would be blamed on the channel.
        out.append(
            f"noise gate cut {t.gated_inside_speech * 20} ms from inside the speech"
        )
    if t.lead_s > MAX_LEAD_TRAIL_S or t.trail_s > MAX_LEAD_TRAIL_S:
        out.append(f"{t.lead_s:.1f}s lead / {t.trail_s:.1f}s trail of silence")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all", action="store_true", help="list every take, not just the suspect ones")
    parser.add_argument("--speaker", help="only this speaker's lines")
    parser.add_argument("--redo", action="store_true", help="print a scripts.record command for the bad takes")
    args = parser.parse_args()

    if not SCRIPTS_CSV.exists():
        print(f"{SCRIPTS_CSV} not found", file=sys.stderr)
        return 2
    rows = [r for r in csv.DictReader(SCRIPTS_CSV.open(encoding="utf-8"))]
    if args.speaker:
        rows = [r for r in rows if r["speaker"] == args.speaker]

    takes, missing = [], []
    for row in rows:
        path = AUDIO_DIR / f"{row['id']}.wav"
        if path.exists():
            takes.append((row, measure(path, row["id"])))
        else:
            missing.append(row)

    scope = f"speaker {args.speaker}" if args.speaker else "corpus"
    print(f"{len(takes)}/{len(rows)} recorded for the {scope}.\n")
    if not takes:
        return 1

    flagged = [(row, t, p) for row, t, p in ((r, t, problems(t)) for r, t in takes) if p]

    if args.all:
        print(f"{'id':>4}  {'secs':>5}  {'peak':>6}  {'SNR':>6}  {'lead':>5} {'trail':>5}  text")
        for row, take in takes:
            mark = "!" if problems(take) else " "
            print(
                f"{take.id:>4}{mark} {take.seconds:5.1f}  {take.peak_dbfs:6.1f}  {take.snr_db:6.1f}  "
                f"{take.lead_s:5.1f} {take.trail_s:5.1f}  {row['text'][:44]}"
            )
        print()

    if flagged:
        print(f"{len(flagged)} take(s) worth redoing:")
        for row, take, issues in flagged:
            print(f"  {take.id}  {row['text'][:48]}")
            for issue in issues:
                print(f"        {issue}")
        print()

    seconds = [t.seconds for _, t in takes]
    snrs = [t.snr_db for _, t in takes]
    peaks = [t.peak_dbfs for _, t in takes]
    total = sum(seconds)
    print("summary")
    print(f"  duration   median {np.median(seconds):4.1f}s   range {min(seconds):.1f}-{max(seconds):.1f}s   total {total / 60:.1f} min")
    print(f"  peak       median {np.median(peaks):5.1f} dBFS  range {min(peaks):.0f} to {max(peaks):.0f}")
    print(f"  SNR        median {np.median(snrs):5.1f} dB    worst {min(snrs):.0f}")

    gated = [t for _, t in takes if t.gated_frames]
    if gated:
        fractions = [t.gated_fraction for t in gated]
        eaten = sum(t.gated_inside_speech for t in gated)
        print(
            f"  noise gate on {len(gated)}/{len(takes)} takes, "
            f"{np.median(fractions) * 100:.1f}% of frames zeroed (median)"
        )
        print(
            f"             {'none of it inside speech - only the silence around it' if not eaten else f'{eaten * 20} ms CUT FROM INSIDE SPEECH'}"
        )
    # Billing follows audio-seconds (D-020), so the corpus duration sets the eval cost.
    per_condition = total / 3600 * 27.0
    print(f"  eval cost  Rs {per_condition:.2f} per condition, Rs {per_condition * 10:.2f} for a 10-condition baseline")

    if missing:
        print(f"\n  {len(missing)} not yet recorded: {', '.join(r['id'] for r in missing[:12])}" + (" ..." if len(missing) > 12 else ""))

    if flagged and args.redo:
        print(f"\nuv run python -m scripts.record --ids {','.join(t.id for _, t, _ in flagged)}")
    elif flagged:
        print("\nRe-run with --redo for a command to re-record those.")
    else:
        print("\nNo problems found.")
    return 1 if flagged else 0


if __name__ == "__main__":
    raise SystemExit(main())
