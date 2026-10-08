"""Pick an input device and check the room is quiet enough to record the corpus.

    uv run python -m scripts.calibrate            # full check, saves the chosen mic
    uv run python -m scripts.calibrate --monitor  # live level meter while you adjust
    uv run python -m scripts.calibrate --list     # just show the input devices

Run this once before recording, and again if you move room or change mic.
`scripts/record.py` reads the device this saves.

**Why the room matters.** The eval adds shop noise at known SNRs (20, 10, 5,
0 dB) on top of the recordings. That only measures anything if the recordings
start far quieter than the quietest condition. If the room sits at 20 dB SNR,
then the "clean" baseline is really a 20 dB recording, and the 20 dB condition
is indistinguishable from it — the whole curve flattens and the result says
nothing. So this script measures the room's own noise floor and says plainly
whether it is good enough.

No Gnani API calls are made here.
"""

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "scripts" / "out" / "mic.json"

SAMPLE_RATE = 16_000
SILENCE_SECONDS = 4.0

# A sentence from the corpus, so the level test matches what will be recorded.
TEST_SENTENCE = "Sharma ji ka dedh sau udhaar likho"

# Targets, in dBFS unless stated.
PEAK_CLIPPING = -1.0  # at or above this, assume clipping
PEAK_IDEAL = (-12.0, -3.0)
PEAK_TOO_QUIET = -24.0
NOISE_FLOOR_GOOD = -55.0
NOISE_FLOOR_OK = -45.0
# The eval's quietest added-noise condition is 20 dB SNR. The source needs a
# good margin above that, or the added noise stops being what is measured.
SNR_GOOD = 40.0
SNR_OK = 30.0
SNR_POOR = 20.0

FLOOR = 1e-10

# A corpus recording must start wideband and uncompressed. Anything that is already
# narrowband has the telephone channel baked in - which is exactly what the eval is
# supposed to add synthetically, at known settings.
MIN_USABLE_SAMPLE_RATE = 32_000
# "hands-free" and the Bluetooth HFP driver name are definitive. "headset (" is how
# Windows names paired Bluetooth audio, so treat it as Bluetooth too.
BLUETOOTH_CERTAIN = ("hands-free", "bthhfenum", "hf audio")
BLUETOOTH_LIKELY = ("headset (", "airpods", "buds", "tws")
LOOPBACK_HINTS = ("stereo mix", "what u hear", "pc speaker", "sound mapper")


def device_warning(device: dict) -> str | None:
    """Why this device would spoil the corpus, if it would.

    Note that MME and DirectSound report a nominal 44100 Hz for Bluetooth mics that
    actually capture at 8 or 16 kHz in hands-free mode - the same device shows its
    real rate only under WASAPI. So the name is trusted over the reported rate.
    """
    name = " ".join(device["name"].split()).lower()
    rate = float(device.get("default_samplerate") or 0)

    if any(hint in name for hint in LOOPBACK_HINTS):
        return "not a microphone (loopback or virtual device)"
    if rate and rate <= 8_000:
        return f"{rate:.0f} Hz - telephone quality already; the eval adds that itself"
    if any(hint in name for hint in BLUETOOTH_CERTAIN):
        return (
            "Bluetooth hands-free - narrowband and compressed, whatever rate is reported. "
            "Use a wired mic or the laptop's own mic"
        )
    if any(hint in name for hint in BLUETOOTH_LIKELY):
        return (
            "looks like Bluetooth - its mic captures in hands-free mode (8-16 kHz, compressed) "
            "even when the driver reports more. Prefer a wired mic"
        )
    if rate and rate < MIN_USABLE_SAMPLE_RATE:
        return f"{rate:.0f} Hz - narrower than the corpus needs"
    return None


def _sounddevice():
    try:
        import sounddevice
    except OSError as e:
        raise SystemExit(f"Could not load the audio backend: {e}") from e
    return sounddevice


def dbfs(amplitude: float) -> float:
    """Amplitude (0-1 full scale) as dBFS. Digital silence returns -inf."""
    return 20.0 * math.log10(max(float(amplitude), FLOOR)) if amplitude > 0 else -math.inf


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))) if audio.size else 0.0


def active_speech_rms(audio: np.ndarray, frame_ms: int = 20) -> float:
    """RMS of the loudest half of the frames, so pauses do not drag the level down.

    The same idea as measuring SNR on active speech in channel/degrade.py: the
    gaps between words are not signal.
    """
    frame = max(1, SAMPLE_RATE * frame_ms // 1000)
    usable = audio[: audio.size - audio.size % frame]
    if usable.size < frame:
        return rms(audio)
    frames = usable.reshape(-1, frame)
    energies = np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1))
    loud = energies[energies >= np.median(energies)]
    return float(loud.mean()) if loud.size else rms(audio)


MIN_TAKE_SECONDS = 0.8
# Below this the take cannot be speech, so report it as a mistake rather than
# dressing it up as a verdict about the room.
MIN_PLAUSIBLE_SNR_DB = 6.0


def take_problem(speech: np.ndarray, noise_rms: float) -> str | None:
    """Why this take is not a usable measurement of someone speaking, if it is not."""
    seconds = speech.size / SAMPLE_RATE
    if seconds < MIN_TAKE_SECONDS:
        return (
            f"only {seconds:.1f}s captured - the first Enter starts the recording, "
            "the second stops it."
        )
    if dbfs(active_speech_rms(speech)) - dbfs(noise_rms) < MIN_PLAUSIBLE_SNR_DB:
        return "that is barely above the room noise - was the line read aloud, and is the right mic selected?"
    return None


@dataclass
class Reading:
    noise_dbfs: float
    speech_peak_dbfs: float
    speech_rms_dbfs: float
    dc_offset: float
    clipped_samples: int

    @property
    def snr_db(self) -> float:
        return self.speech_rms_dbfs - self.noise_dbfs


def verdicts(r: Reading) -> tuple[list[str], bool]:
    """Human-readable findings, and whether the setup is good enough to record."""
    notes: list[str] = []
    ok = True

    if r.noise_dbfs <= NOISE_FLOOR_GOOD:
        notes.append(f"Room noise {r.noise_dbfs:.0f} dBFS - quiet.")
    elif r.noise_dbfs <= NOISE_FLOOR_OK:
        notes.append(f"Room noise {r.noise_dbfs:.0f} dBFS - acceptable, quieter would be better.")
    else:
        notes.append(
            f"Room noise {r.noise_dbfs:.0f} dBFS - too loud. Turn off fans, AC and anything humming."
        )
        ok = False

    if r.clipped_samples:
        notes.append(f"CLIPPING on {r.clipped_samples} samples - lower the input gain or move back.")
        ok = False
    elif r.speech_peak_dbfs >= PEAK_CLIPPING:
        notes.append(f"Peak {r.speech_peak_dbfs:.0f} dBFS - on the edge of clipping. Back off a little.")
        ok = False
    elif r.speech_peak_dbfs < PEAK_TOO_QUIET:
        notes.append(f"Peak {r.speech_peak_dbfs:.0f} dBFS - too quiet. Move closer or raise the gain.")
        ok = False
    elif PEAK_IDEAL[0] <= r.speech_peak_dbfs <= PEAK_IDEAL[1]:
        notes.append(f"Speech peak {r.speech_peak_dbfs:.0f} dBFS - good.")
    else:
        notes.append(f"Speech peak {r.speech_peak_dbfs:.0f} dBFS - usable, {PEAK_IDEAL[0]:.0f} to {PEAK_IDEAL[1]:.0f} is ideal.")

    if r.snr_db >= SNR_GOOD:
        notes.append(f"Signal-to-noise {round(r.snr_db):d} dB - plenty of headroom for the eval conditions.")
    elif r.snr_db >= SNR_OK:
        notes.append(
            f"Signal-to-noise {round(r.snr_db):d} dB - workable, but the 20 dB eval condition will be "
            "a little muddied by the room."
        )
    elif r.snr_db >= SNR_POOR:
        notes.append(
            f"Signal-to-noise {round(r.snr_db):d} dB - too close to the eval's quietest condition (20 dB). "
            "The 'clean' baseline would not be clean."
        )
        ok = False
    else:
        notes.append(f"Signal-to-noise {round(r.snr_db):d} dB - unusable for the corpus.")
        ok = False

    if abs(r.dc_offset) > 0.01:
        notes.append(f"DC offset {r.dc_offset:+.3f} - unusual; try another device or driver.")

    return notes, ok


# --- device handling -------------------------------------------------------


def input_devices() -> list[tuple[int, dict]]:
    sd = _sounddevice()
    host_apis = sd.query_hostapis()
    out = []
    for index, device in enumerate(sd.query_devices()):
        if device["max_input_channels"] > 0:
            device = dict(device)
            device["host_api_name"] = host_apis[device["hostapi"]]["name"]
            out.append((index, device))
    return out


def print_devices() -> None:
    sd = _sounddevice()
    try:
        default_in = sd.default.device[0]
    except (TypeError, IndexError):
        default_in = None
    print("Input devices:")
    for index, device in input_devices():
        mark = " (system default)" if index == default_in else ""
        name = " ".join(device["name"].split())  # some driver names span lines
        print(
            f"  [{index:2}] {name}  -  {device['host_api_name']}, "
            f"{device['max_input_channels']} ch, {device['default_samplerate']:.0f} Hz{mark}"
        )
        if (warning := device_warning(device)) is not None:
            print(f"       avoid: {warning}")


def choose_device(preset: int | None) -> int | None:
    devices = input_devices()
    if not devices:
        raise SystemExit("No input devices found. Plug in a microphone and try again.")
    if preset is not None:
        if preset not in {i for i, _ in devices}:
            raise SystemExit(f"Device {preset} is not an input device. Run with --list.")
        return preset

    print_devices()
    good = [(i, d) for i, d in devices if device_warning(d) is None]
    if good:
        best = max(good, key=lambda pair: float(pair[1].get("default_samplerate") or 0))
        name = " ".join(best[1]["name"].split())
        print(f"\nSuggested: [{best[0]}] {name} ({best[1]['default_samplerate']:.0f} Hz)")
    print("\nA wired or built-in mic beats a Bluetooth one here: Bluetooth mics capture in")
    print("hands-free mode, which is narrowband and compressed - the very degradation the")
    print("eval is supposed to add itself, at known settings.")

    raw = input("Device number (Enter for the system default) > ").strip()
    if not raw:
        return None
    if not raw.isdigit() or int(raw) not in {i for i, _ in devices}:
        raise SystemExit(f"{raw!r} is not one of the listed device numbers.")

    chosen = int(raw)
    if (warning := device_warning(next(d for i, d in devices if i == chosen))) is not None:
        print(f"\n  Warning: {warning}.")
        if input("  Use it anyway? [y/N] > ").strip().lower() != "y":
            raise SystemExit("Nothing changed. Run again and pick another device.")
    return chosen


def save_config(device: int | None, reading: Reading, usable: bool) -> None:
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    name = None
    if device is not None:
        name = next((d["name"] for i, d in input_devices() if i == device), None)
    CONFIG.write_text(
        json.dumps(
            {
                "device": device,
                "device_name": name,
                "sample_rate": SAMPLE_RATE,
                "calibrated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "noise_floor_dbfs": round(reading.noise_dbfs, 1),
                "speech_peak_dbfs": round(reading.speech_peak_dbfs, 1),
                "snr_db": round(reading.snr_db, 1),
                "usable": usable,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved to {CONFIG}. scripts/record.py will use this device.")


def load_config() -> dict:
    if not CONFIG.exists():
        return {}
    try:
        return json.loads(CONFIG.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


# --- capture ---------------------------------------------------------------


def record_seconds(seconds: float, device: int | None) -> np.ndarray:
    sd = _sounddevice()
    audio = sd.rec(
        int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32", device=device
    )
    sd.wait()
    return audio.ravel()


def record_until_enter(prompt: str, device: int | None) -> np.ndarray:
    sd = _sounddevice()
    chunks: list[np.ndarray] = []
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        device=device,
        callback=lambda indata, *_: chunks.append(indata.copy()),
    ):
        input(prompt)
    return np.concatenate(chunks).ravel() if chunks else np.zeros(0, dtype="float32")


def countdown(seconds: float) -> None:
    for remaining in range(int(seconds), 0, -1):
        print(f"\r  {remaining} ", end="", flush=True)
        time.sleep(1)
    print("\r    ", end="\r", flush=True)


def monitor(device: int | None) -> int:
    """Live level meter, so the mic can be positioned before any take."""
    sd = _sounddevice()
    print("Live level meter. Speak normally. Ctrl+C to stop.\n")
    width = 44

    def show(indata, *_):
        peak = float(np.abs(indata).max())
        level = dbfs(peak)
        filled = 0 if level == -math.inf else max(0, min(width, int((level + 60) / 60 * width)))
        if level >= PEAK_CLIPPING:
            tag = "CLIP"
        elif level >= PEAK_IDEAL[0]:
            tag = "good"
        elif level >= PEAK_TOO_QUIET:
            tag = "low "
        else:
            tag = "    "
        bar = "#" * filled + "-" * (width - filled)
        shown = "-inf" if level == -math.inf else f"{level:5.1f}"
        print(f"\r  [{bar}] {shown} dBFS  {tag}", end="", flush=True)

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32", device=device, callback=show
        ):
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n\nAim for peaks between -12 and -3 dBFS while speaking normally.")
    return 0


def calibrate(device: int | None) -> int:
    print("\nStep 1 of 2: the room's noise floor.")
    print(f"Stay silent for {SILENCE_SECONDS:.0f} seconds. Do not type, breathe on the mic, or move.")
    input("Enter when ready > ")
    countdown(2)
    print("  listening...", flush=True)
    silence = record_seconds(SILENCE_SECONDS, device)
    noise_rms = rms(silence)
    print(f"  noise floor {dbfs(noise_rms):.1f} dBFS")

    print("\nStep 2 of 2: your speaking level.")
    print(f'Read this aloud, as you would in a shop:\n\n    "{TEST_SENTENCE}"\n')

    # Two separate prompts. With one, the Enter meant to start the take stops it instead,
    # and the room gets measured against itself for a nonsensical 0 dB SNR.
    while True:
        input("  Enter to START recording > ")
        speech = record_until_enter("  recording... read the line aloud, then Enter to STOP > ", device)
        if (problem := take_problem(speech, noise_rms)) is None:
            break
        print(f"  {problem}")
        if input("  Try again? [Y/n] > ").strip().lower() == "n":
            return 1

    reading = Reading(
        noise_dbfs=dbfs(noise_rms),
        speech_peak_dbfs=dbfs(float(np.abs(speech).max())),
        speech_rms_dbfs=dbfs(active_speech_rms(speech)),
        dc_offset=float(speech.mean()),
        clipped_samples=int(np.sum(np.abs(speech) >= 0.999)),
    )

    print("\n--- results ---")
    notes, ok = verdicts(reading)
    for note in notes:
        print(f"  {note}")

    save_config(device, reading, ok)
    if ok:
        print("\nGood to record. Run:  uv run python -m scripts.record")
        return 0

    print("\nFix the points above and run this again.")
    print("Tip: --monitor gives a live meter while you move the mic or change the gain.")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true", help="list input devices and exit")
    parser.add_argument("--monitor", action="store_true", help="live level meter")
    parser.add_argument("--device", type=int, help="device number, skipping the prompt")
    args = parser.parse_args()

    if args.list:
        print_devices()
        return 0

    device = args.device if args.device is not None else load_config().get("device")
    if args.monitor:
        return monitor(device)

    return calibrate(choose_device(args.device))


if __name__ == "__main__":
    raise SystemExit(main())
