"""Single real calls to Timbre and Prisma, to prove the adapters work (milestone M0).

    uv run python -m scripts.smoke tts            # 1 Timbre call  -> scripts/out/smoke_tts.wav
    uv run python -m scripts.smoke stt            # 1 Prisma call  on scripts/out/smoke_tts.wav
    uv run python -m scripts.smoke stt my.wav     # 1 Prisma call  on your own recording
    uv run python -m scripts.smoke both           # 2 calls: TTS, then STT on its output

Each subcommand makes exactly the number of API calls shown. Nothing is retried.
"""

import argparse
import asyncio
import sys
import time
import wave
from pathlib import Path

import numpy as np
import soundfile as sf

from app.adapters.base import AudioClip
from app.adapters.gnani_common import GnaniAPIError
from app.adapters.gnani_stt import PrismaSTT
from app.adapters.gnani_tts import TimbreTTS
from app.config import get_settings

OUT_DIR = Path(__file__).parent / "out"
DEFAULT_WAV = OUT_DIR / "smoke_tts.wav"
DEFAULT_TEXT = "Sharma ji ke khate mein dedh sau rupaye udhaar likh doon?"


def load_clip(path: Path) -> AudioClip:
    samples, rate = sf.read(path, dtype="int16", always_2d=True)
    mono = samples.mean(axis=1).astype(np.int16) if samples.shape[1] > 1 else samples[:, 0]
    return AudioClip(pcm=mono.tobytes(), sample_rate=rate)


async def run_tts(text: str, out: Path) -> None:
    s = get_settings()
    tts = TimbreTTS(s.gnani_api_key.get_secret_value(), sample_rate=s.tts_sample_rate)
    print(f"Timbre: voice={s.tts_voice} language={s.tts_language} rate={s.tts_sample_rate}")
    print(f"  text: {text}")

    start = time.perf_counter()
    first_chunk_ms = None
    pcm = bytearray()
    async for chunk in tts.synthesize(text, language=s.tts_language, voice=s.tts_voice):
        if first_chunk_ms is None:
            first_chunk_ms = (time.perf_counter() - start) * 1000
        pcm += chunk
    total_ms = (time.perf_counter() - start) * 1000

    out.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(tts.sample_rate)
        w.writeframes(bytes(pcm))
    seconds = len(pcm) / 2 / tts.sample_rate
    print(f"  first chunk {first_chunk_ms:.0f} ms, done {total_ms:.0f} ms, {seconds:.2f} s of audio -> {out}")


async def run_stt(path: Path) -> None:
    s = get_settings()
    clip = load_clip(path)
    stt = PrismaSTT(s.gnani_api_key.get_secret_value(), base_url=s.gnani_base_url)
    print(f"Prisma: language={s.stt_language} file={path} ({clip.duration_s:.2f} s @ {clip.sample_rate} Hz)")
    try:
        result = await stt.transcribe(clip, language=s.stt_language)
    finally:
        await stt.aclose()
    print(f"  {result.latency_ms:.0f} ms, request_id={result.raw.get('request_id')}")
    print(f"  transcript: {result.text}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("what", choices=["tts", "stt", "both"])
    parser.add_argument("wav", nargs="?", type=Path, default=DEFAULT_WAV, help="input for stt / output for tts")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    args = parser.parse_args()

    try:
        if args.what in ("tts", "both"):
            asyncio.run(run_tts(args.text, args.wav))
        if args.what in ("stt", "both"):
            if not args.wav.exists():
                print(f"{args.wav} not found; run `tts` first or pass a WAV path", file=sys.stderr)
                return 2
            asyncio.run(run_stt(args.wav))
    except GnaniAPIError as e:
        print(f"Gnani API error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
