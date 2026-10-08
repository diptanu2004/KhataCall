"""Measure warm latency of Timbre TTS and Prisma STT.

    uv run python -m scripts.latency --dry-run        # print planned call count, spend nothing
    uv run python -m scripts.latency --n 5            # 5 calls per service
    uv run python -m scripts.latency --n 5 --only stt_stream

Every call is a real API call. Results print per sample so a cold first call
stays visible, and summary stats are given both with and without it.

What is measured:
  TTS  first_chunk_ms  connect + synthesis until the first audio byte arrives
       total_ms        until the 'complete' event
  STT        total_ms  request sent until the transcript is parsed (REST)
  STT_STREAM total_ms  last frame sent until the transcript arrives ("tail")

REST and streaming are not measured the same way, because the pipelines differ.
REST can only start after push-to-talk is released, so its whole duration is
felt. Streaming sends audio while the owner is still speaking, so only the tail
after the last frame is felt. Both numbers are therefore "time the owner waits
after they stop talking", which is what docs/HLD.md §12 budgets.

The headline number for the demo is TTS first_chunk_ms + STT total_ms: the two
Gnani legs of the end-of-speech -> first-audio path in docs/HLD.md §12.
"""

import argparse
import asyncio
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from app.adapters.gnani_common import GnaniAPIError
from app.adapters.gnani_stt import PrismaSTT
from app.adapters.gnani_stt_stream import PrismaStreamingSTT
from app.adapters.usage import stt_cost_inr, tts_cost_inr
from app.adapters.gnani_tts import TimbreTTS
from app.config import get_settings
from scripts.smoke import DEFAULT_WAV, load_clip

OUT_DIR = Path(__file__).parent / "out"

# Realistic agent replies: short read-backs and confirmations, which is what the demo speaks.
TTS_TEXTS = [
    "Sharma ji ke khate mein dedh sau rupaye udhaar likh doon?",
    "Likh diya. Sharma ji ka total baaki barah sau chalis rupaye.",
    "Gupta aunty ka do hazaar paanch sau udhaar likh doon?",
    "Maggi ke pandrah packet bache hain.",
    "Theek hai, cancel kar diya.",
]


@dataclass
class Sample:
    index: int
    service: str
    first_chunk_ms: float | None
    total_ms: float
    audio_s: float
    detail: str


def summarise(values: list[float]) -> dict[str, float]:
    s = sorted(values)
    return {
        "n": len(s),
        "min": round(s[0], 1),
        "p50": round(statistics.median(s), 1),
        # With n < 10 this is the near-worst sample, not a real p90. Labelled honestly.
        "p90": round(s[min(len(s) - 1, int(0.9 * (len(s) - 1) + 0.5))], 1),
        "max": round(s[-1], 1),
    }


async def measure_tts(n: int) -> list[Sample]:
    s = get_settings()
    tts = TimbreTTS(s.gnani_api_key.get_secret_value(), sample_rate=s.tts_sample_rate)
    samples: list[Sample] = []
    for i in range(n):
        text = TTS_TEXTS[i % len(TTS_TEXTS)]
        start = time.perf_counter()
        first_ms: float | None = None
        total_bytes = 0
        async for chunk in tts.synthesize(text, language=s.tts_language, voice=s.tts_voice):
            if first_ms is None:
                first_ms = (time.perf_counter() - start) * 1000
            total_bytes += len(chunk)
        total_ms = (time.perf_counter() - start) * 1000
        audio_s = total_bytes / 2 / tts.sample_rate
        samples.append(Sample(i, "tts", round(first_ms or 0.0, 1), round(total_ms, 1), round(audio_s, 2), text[:40]))
        print(f"  tts  #{i}  first_chunk {first_ms:7.0f} ms  total {total_ms:7.0f} ms  audio {audio_s:4.2f} s")
    return samples


async def measure_stt_stream(n: int, wav: Path) -> list[Sample]:
    s = get_settings()
    clip = load_clip(wav)
    stt = PrismaStreamingSTT(s.gnani_api_key.get_secret_value())
    samples: list[Sample] = []
    for i in range(n):
        result = await stt.transcribe(clip, language=s.stt_language)
        tail = result.latency_ms or 0.0
        vad = result.raw.get("last_frame_to_vad_end_ms")
        samples.append(Sample(i, "stt_stream", None, round(tail, 1), round(clip.duration_s, 2), result.text[:40]))
        vad_note = f"  (vad end at +{vad:.0f} ms)" if vad is not None else ""
        print(f"  strm #{i}  tail  {tail:7.0f} ms  segments={result.raw['num_segments']}{vad_note}")
    return samples


async def measure_stt(n: int, wav: Path) -> list[Sample]:
    s = get_settings()
    clip = load_clip(wav)
    stt = PrismaSTT(s.gnani_api_key.get_secret_value(), base_url=s.gnani_base_url)
    samples: list[Sample] = []
    try:
        for i in range(n):
            result = await stt.transcribe(clip, language=s.stt_language)
            ms = result.latency_ms or 0.0
            samples.append(Sample(i, "stt", None, round(ms, 1), round(clip.duration_s, 2), result.text[:40]))
            print(f"  stt  #{i}  total {ms:7.0f} ms  ({ms / clip.duration_s:5.0f} ms per audio-second)")
    finally:
        await stt.aclose()
    return samples


def report(samples: list[Sample]) -> dict:
    out: dict = {"measured_at": datetime.now(UTC).isoformat(timespec="seconds"), "stats": {}}
    for service in ("tts", "stt", "stt_stream"):
        rows = [s for s in samples if s.service == service]
        if not rows:
            continue
        metrics = {"total_ms": [r.total_ms for r in rows]}
        if service == "tts":
            metrics["first_chunk_ms"] = [r.first_chunk_ms for r in rows if r.first_chunk_ms is not None]
        block: dict = {}
        for name, values in metrics.items():
            block[name] = summarise(values)
            if len(values) > 1:
                block[name + "_excluding_first"] = summarise(values[1:])
        out["stats"][service] = block
    out["samples"] = [asdict(s) for s in samples]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=5, help="calls per service (default 5)")
    parser.add_argument(
        "--only", choices=["tts", "stt", "stt_stream"], help="measure one service only"
    )
    parser.add_argument("--wav", type=Path, default=DEFAULT_WAV, help="input clip for STT")
    parser.add_argument("--max-calls", type=int, default=20, help="hard cap on total API calls")
    parser.add_argument("--dry-run", action="store_true", help="print the planned call count and exit")
    args = parser.parse_args()

    services = [args.only] if args.only else ["tts", "stt", "stt_stream"]
    planned = args.n * len(services)

    # Billing follows audio seconds and characters, not calls, so estimate those.
    clip_s = 4.0
    if args.wav.exists():
        try:
            info = sf.info(args.wav)
            clip_s = info.frames / info.samplerate
        except Exception:
            pass
    cost = 0.0
    for service in services:
        if service == "tts":
            cost += sum(tts_cost_inr(len(TTS_TEXTS[i % len(TTS_TEXTS)])) for i in range(args.n))
        else:
            cost += args.n * stt_cost_inr(clip_s)
    print(
        f"Planned API calls: {planned} ({args.n} x {len(services)} service(s): {', '.join(services)})"
        f"  ~Rs {cost:.2f}"
    )
    if args.dry_run:
        return 0
    if planned > args.max_calls:
        print(f"Refusing: {planned} calls exceeds --max-calls {args.max_calls}", file=sys.stderr)
        return 2
    if {"stt", "stt_stream"} & set(services) and not args.wav.exists():
        print(f"{args.wav} not found; run `scripts.smoke tts` first", file=sys.stderr)
        return 2

    samples: list[Sample] = []
    try:
        if "tts" in services:
            samples += asyncio.run(measure_tts(args.n))
        if "stt" in services:
            samples += asyncio.run(measure_stt(args.n, args.wav))
        if "stt_stream" in services:
            samples += asyncio.run(measure_stt_stream(args.n, args.wav))
    except GnaniAPIError as e:
        print(f"Gnani API error after {len(samples)} samples: {e}", file=sys.stderr)
    finally:
        if samples:
            result = report(samples)
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            path = OUT_DIR / "latency.json"
            path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            print("\n" + json.dumps(result["stats"], indent=2))
            print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
