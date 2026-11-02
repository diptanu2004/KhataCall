"""Run the recorded corpus through every channel condition and score it.

    uv run python -m eval.run_eval --dry-run                 # plan and price it, no calls
    uv run python -m eval.run_eval --max-calls 800           # STT, then the agent if Evon is up
    uv run python -m eval.run_eval --stages stt              # transcripts only
    uv run python -m eval.run_eval --stages task             # re-score cached transcripts with Evon
    uv run python -m eval.run_eval --conditions clean,mob_snr5

Two stages, each cached on disk so nothing is ever paid for twice (hard rule 3):

  stt   each take is degraded (eval/conditions.py, app/channel/degrade.py) and
        sent to Prisma REST - the same endpoint the live loop uses (D-003).
        Cache key: (sha256 of the degraded audio, condition id, Prisma model id).
        Costs Gnani credits: Rs 27 per hour of audio.
  task  each transcript goes to Evon with the agent's prompt and tools, and the
        proposed call is scored with eval/metrics.py - the executor's own
        parsing. Cache key: (sha256 of the transcript, Evon model, prompt+tools).
        Evon is self-hosted, so this stage spends no credits.

Errors are never cached: a failed call is retried on the next run.
"""

import argparse
import asyncio
import csv
import hashlib
import json
import statistics
import sys
import time
from collections import defaultdict
from datetime import date
from pathlib import Path

import jiwer
import numpy as np
import soundfile as sf

from app.adapters.base import AudioClip, Message
from app.adapters.usage import MAX_REQUESTS_PER_MINUTE, stt_cost_inr
from app.channel.degrade import Condition, degrade, to_pcm16
from eval.conditions import CONDITIONS, by_id, load_noise_bank, noise_for, rng_for
from app.agent.fixes import ALL, NONE
from eval.metrics import Outcome, score_call
from eval.normalize import devanagari_fraction, normalise_text, raw_text

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "eval" / "corpus" / "scripts.csv"
AUDIO_DIR = ROOT / "eval" / "corpus" / "audio"
CACHE_DIR = ROOT / "eval" / "cache"
# The 60 takes the M4 fixes were designed from (D-042). Everything else is held out.
DEV_IDS_FILE = ROOT / "eval" / "corpus" / "dev_ids.txt"
RESULTS_DIR = ROOT / "eval" / "results"

STT_MODEL_ID = "prisma-v2.5"  # PrismaSTT.model_id; duplicated so --dry-run needs no API key
# Below the published 60/min, so a retry or the live demo running alongside never trips it.
REQUESTS_PER_MINUTE = MAX_REQUESTS_PER_MINUTE - 10
# Prisma REST allows one active request per key, despite the published 20 concurrent
# sessions: 4 at a time drew HTTP 429 "Only 1 active request is allowed at a time" (D-037).
STT_CONCURRENCY = 1


# --- planning ----------------------------------------------------------------


def load_takes(audio_dir: Path = AUDIO_DIR) -> list[dict]:
    """Corpus rows that have a recording, with the audio loaded."""
    rows = list(csv.DictReader(CORPUS.open(encoding="utf-8")))
    takes = []
    for row in rows:
        path = audio_dir / f"{row['id']}.wav"
        if path.exists():
            audio, rate = sf.read(path, dtype="float32")
            if rate != 16_000 or audio.ndim != 1:
                raise ValueError(f"{path.name} is not 16 kHz mono; check it with scripts.check_corpus")
            takes.append({**row, "audio": audio})
    return takes


def split_of(utterance_id: str, dev_ids: set[str]) -> str:
    return "dev" if utterance_id in dev_ids else "heldout"


def load_dev_ids() -> set[str]:
    return set(DEV_IDS_FILE.read_text(encoding="utf-8").split()) if DEV_IDS_FILE.exists() else set()


def summarise_by_condition(rows: list[dict], conditions) -> dict:
    by_condition: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_condition[r["condition"]].append(r)
    return {c.id: summarise(by_condition[c.id]) for c in conditions if by_condition[c.id]}


def print_table(title: str, summary: dict) -> None:
    print(f"\n{title}")
    print(
        f"{'condition':12} {'n':>3}  {'WER raw':>7} {'WER norm':>8} {'CER norm':>8} {'empty':>5}  "
        f"{'amount ok':>9} {'changed':>7} {'lost':>4}  {'before':>13} {'after fixes':>13}"
    )
    for cid, s in summary.items():
        task, before = s.get("task"), s.get("task_before")

        def cell(t):
            return f"{t['accuracy']:6.1%}/{t['wrong_rate']:5.1%}" if t else f"{'-':>12}"

        print(
            f"{cid:12} {s['n']:3d}  {s['raw']['wer']:7.1%} {s['normalised']['wer']:8.1%} {s['normalised']['cer']:8.1%} "
            f"{s['empty_transcripts']:5d}  "
            + (f"{s['amount']['kept_rate']:9.0%} {s['amount']['changed']:7d} {s['amount']['lost']:4d}  " if "amount" in s else " " * 24)
            + f"{cell(before):>13} {cell(task):>13}"
        )
    print("  (task cells: correct/wrong)")


def known_customers() -> list[str]:
    rows = csv.DictReader(CORPUS.open(encoding="utf-8"))
    return sorted({r["customer"] for r in rows if r["customer"]})


def known_items() -> list[str]:
    rows = csv.DictReader(CORPUS.open(encoding="utf-8"))
    return sorted({r["item"] for r in rows if r["item"]})


def degraded_clip(take: dict, condition: Condition, bank: dict[str, np.ndarray]) -> tuple[AudioClip, str | None]:
    noise_name, noise = (None, None)
    if condition.snr_db is not None:
        noise_name, noise = noise_for(take["id"], bank)
    audio = degrade(take["audio"], condition, rng_for(take["id"], condition.id), noise)
    return AudioClip(to_pcm16(audio), condition.output_rate), noise_name


def stt_cache_path(clip: AudioClip, condition_id: str, model_id: str = STT_MODEL_ID) -> Path:
    return CACHE_DIR / "stt" / model_id / condition_id / f"{clip.sha256()}.json"


def task_cache_key(transcript: str, model: str, prompt_sha: str) -> Path:
    digest = hashlib.sha256(transcript.encode("utf-8")).hexdigest()
    safe_model = model.replace("/", "_")
    return CACHE_DIR / "task" / safe_model / prompt_sha / f"{digest}.json"


def prompt_fingerprint() -> str:
    """Changes whenever the prompt or the tool schemas do, so a fix re-runs the agent."""
    from app.agent.prompts import SYSTEM_PROMPT
    from app.agent.tools import TOOL_SCHEMAS

    blob = SYSTEM_PROMPT + json.dumps(TOOL_SCHEMAS, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


# --- pacing ------------------------------------------------------------------


class Pacer:
    """Starts at most `per_minute` requests a minute, spread evenly."""

    def __init__(self, per_minute: int) -> None:
        self._gap = 60.0 / per_minute
        self._next = 0.0
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self._lock:
            now = time.monotonic()
            delay = self._next - now
            self._next = max(now, self._next) + self._gap
        if delay > 0:
            await asyncio.sleep(delay)


# --- stage 1: speech to text -------------------------------------------------


async def run_stt(jobs: list[dict], args, settings) -> int:
    """Transcribe the uncached jobs. Returns the number of real calls made."""
    from app.adapters.gnani_stt import PrismaSTT

    stt = PrismaSTT(settings.gnani_api_key.get_secret_value(), base_url=settings.gnani_base_url)
    pacer, gate = Pacer(REQUESTS_PER_MINUTE), asyncio.Semaphore(STT_CONCURRENCY)
    done = failed = 0

    async def one(job: dict) -> None:
        nonlocal done, failed
        async with gate:
            await pacer.wait()
            try:
                result = await stt.transcribe(job["clip"], language=settings.stt_language)
            except Exception as e:  # noqa: BLE001 - one bad clip must not stop the run
                failed += 1
                print(f"  ! {job['id']} {job['condition']}: {type(e).__name__}: {e}")
                return
            write_json(
                job["cache"],
                {
                    "id": job["id"],
                    "condition": job["condition"],
                    "model": STT_MODEL_ID,
                    "noise": job["noise"],
                    "audio_s": round(job["clip"].duration_s, 3),
                    "text": result.text,
                    "latency_ms": round(result.latency_ms or 0),
                },
            )
            done += 1
            if done % 25 == 0 or done == len(jobs):
                print(f"  stt {done}/{len(jobs)}" + (f"  ({failed} failed)" if failed else ""))

    try:
        await asyncio.gather(*(one(j) for j in jobs))
    finally:
        await stt.aclose()
    return done + failed


# --- stage 2: the agent ------------------------------------------------------


async def evon_reachable(settings) -> str | None:
    import httpx

    root = settings.evon_base_url.rstrip("/").removesuffix("/v1")
    key = settings.evon_api_key.get_secret_value()
    try:
        async with httpx.AsyncClient(timeout=10, headers={"Authorization": f"Bearer {key}"} if key else {}) as c:
            r = await c.get(f"{root}/health")
        return None if r.status_code == 200 else f"health returned {r.status_code}"
    except httpx.HTTPError as e:
        return f"{type(e).__name__}"


async def run_task(transcripts: list[str], settings, prompt_sha: str) -> None:
    """Ask Evon for a tool call on each distinct uncached transcript."""
    from app.adapters.evon_llm import EvonError, EvonLLM
    from app.agent.prompts import SYSTEM_PROMPT
    from app.agent.tools import TOOL_SCHEMAS

    llm = EvonLLM(
        base_url=settings.evon_base_url,
        api_key=settings.evon_api_key.get_secret_value(),
        model=settings.evon_model,
        timeout_s=120,
    )
    system = Message("system", SYSTEM_PROMPT)
    try:
        for n, text in enumerate(transcripts, 1):
            start = time.perf_counter()
            try:
                reply = await llm.complete([system, Message("user", text)], tools=TOOL_SCHEMAS)
            except EvonError as e:
                print(f"  ! evon on {text[:40]!r}: {e}")
                continue
            call = reply.tool_calls[0] if reply.tool_calls else None
            write_json(
                task_cache_key(text, settings.evon_model, prompt_sha),
                {
                    "transcript": text,
                    "tool": call.name if call else None,
                    "arguments": call.arguments if call else None,
                    "text_reply": reply.text,
                    "ms": round((time.perf_counter() - start) * 1000),
                },
            )
            if n % 25 == 0 or n == len(transcripts):
                print(f"  evon {n}/{len(transcripts)}")
    finally:
        await llm.aclose()


# --- scoring -----------------------------------------------------------------


def error_rates(refs: list[str], hyps: list[str]) -> dict:
    # jiwer refuses empty references; an empty hypothesis is just all deletions.
    pairs = [(r, h) for r, h in zip(refs, hyps, strict=True) if r.strip()]
    if not pairs:
        return {"wer": None, "cer": None}
    r, h = zip(*pairs, strict=True)
    return {"wer": round(jiwer.wer(list(r), list(h)), 4), "cer": round(jiwer.cer(list(r), list(h)), 4)}


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def amount_heard(row: dict) -> str | None:
    """Did the spoken amount survive transcription? kept / changed / lost, None if no amount.

    Agent-independent: it asks whether the right number is anywhere in the
    transcript, after the same normalisation as WER. "changed" is the dangerous
    one - a different number the agent would read back with confidence.
    """
    if not row.get("amount_paise"):
        return None
    want = int(row["amount_paise"]) / 100
    heard = [float(t) for t in normalise_text(row["transcript"]).split() if t.replace(".", "", 1).isdigit()]
    return "kept" if want in heard else "changed" if heard else "lost"


def summarise(rows: list[dict]) -> dict:
    refs_raw = [raw_text(r["text"]) for r in rows]
    hyps_raw = [raw_text(r["transcript"]) for r in rows]
    refs_norm = [normalise_text(r["text"]) for r in rows]
    hyps_norm = [normalise_text(r["transcript"]) for r in rows]
    latencies = [r["stt_ms"] for r in rows if r.get("stt_ms")]
    out = {
        "n": len(rows),
        "raw": error_rates(refs_raw, hyps_raw),
        "normalised": error_rates(refs_norm, hyps_norm),
        "empty_transcripts": sum(1 for r in rows if not r["transcript"].strip()),
        "devanagari_fraction": round(statistics.fmean(devanagari_fraction(r["transcript"]) for r in rows), 4),
        "stt_ms_p50": statistics.median(latencies) if latencies else None,
        "stt_ms_p90": percentile(latencies, 0.9),
    }
    heard = [h for h in (amount_heard(r) for r in rows) if h]
    if heard:
        out["amount"] = {"n": len(heard), **{k: heard.count(k) for k in ("kept", "changed", "lost")}}
        out["amount"]["kept_rate"] = round(out["amount"]["kept"] / len(heard), 4)
    # "task" is the product as it runs (fixes on); "task_before" the same proposals
    # with the M4 fixes off. Rows scored before the switch existed have only "outcome".
    for key, field in (("task", "outcome"), ("task_before", "outcome_before")):
        scored = [r for r in rows if field in r]
        if scored:
            counts = {o.value: sum(r[field] == o.value for r in scored) for o in Outcome}
            out[key] = {
                "n": len(scored),
                **counts,
                "accuracy": round(counts["correct"] / len(scored), 4),
                "wrong_rate": round(counts["wrong"] / len(scored), 4),
            }
    return out


# --- main --------------------------------------------------------------------


async def run(args) -> int:
    from app.config import get_settings

    settings = get_settings()
    conditions = [by_id(c) for c in args.conditions.split(",")] if args.conditions else list(CONDITIONS)
    takes = load_takes()
    if args.ids:
        wanted = set(args.ids.split(","))
        takes = [t for t in takes if t["id"] in wanted]
    if not takes:
        print("No recordings found in eval/corpus/audio. Record some with scripts.record.")
        return 1
    bank = load_noise_bank() if any(c.snr_db is not None for c in conditions) else {}

    # Plan: degrade everything (cheap, local) and see what is already cached.
    jobs, cached = [], 0
    for condition in conditions:
        for take in takes:
            clip, noise_name = degraded_clip(take, condition, bank)
            path = stt_cache_path(clip, condition.id)
            job = {"id": take["id"], "condition": condition.id, "clip": clip, "noise": noise_name, "cache": path}
            if path.exists():
                cached += 1
            else:
                jobs.append(job)
    seconds = sum(j["clip"].duration_s for j in jobs)
    print(
        f"{len(takes)} takes x {len(conditions)} conditions = {len(takes) * len(conditions)} transcripts; "
        f"{cached} cached, {len(jobs)} to fetch ({seconds / 60:.1f} min of audio, Rs {stt_cost_inr(seconds):.2f})"
    )

    if "stt" in args.stages and jobs:
        if len(jobs) > args.max_calls:
            print(f"  --max-calls {args.max_calls}: fetching only the first {args.max_calls}")
            jobs = jobs[: args.max_calls]
        if args.dry_run:
            minutes = len(jobs) / REQUESTS_PER_MINUTE
            print(f"  dry run: would make {len(jobs)} Prisma calls, about {minutes:.0f} min at {REQUESTS_PER_MINUTE}/min")
        else:
            if not settings.gnani_api_key.get_secret_value():
                print("GNANI_API_KEY is not set in .env")
                return 1
            print(f"  calling Prisma ({REQUESTS_PER_MINUTE}/min, {STT_CONCURRENCY} at a time)...")
            await run_stt(jobs, args, settings)

    # Gather every transcript that exists, for whatever was planned.
    rows = []
    take_by_id = {t["id"]: t for t in takes}
    for condition in conditions:
        for take in takes:
            clip, _ = degraded_clip(take, condition, bank)
            hit = read_json(stt_cache_path(clip, condition.id))
            if hit:
                rows.append({k: v for k, v in take.items() if k != "audio"} | {
                    "condition": condition.id,
                    "noise": hit.get("noise"),
                    "transcript": hit["text"],
                    "stt_ms": hit.get("latency_ms"),
                })

    prompt_sha = prompt_fingerprint()
    if "task" in args.stages and rows:
        wanted = sorted({r["transcript"] for r in rows
                         if not task_cache_key(r["transcript"], settings.evon_model, prompt_sha).exists()})
        print(f"\nagent: {len(set(r['transcript'] for r in rows))} distinct transcripts, {len(wanted)} not yet sent to Evon")
        if wanted and not args.dry_run:
            problem = await evon_reachable(settings)
            if problem:
                print(f"  Evon is not reachable ({problem}); skipping the task stage. See docs/EVON_KAGGLE.md.")
            else:
                await run_task(wanted, settings, prompt_sha)
        known, stock = known_customers(), known_items()
        for r in rows:
            hit = read_json(task_cache_key(r["transcript"], settings.evon_model, prompt_sha))
            if hit:
                # Both sides of the M4 comparison come from this one cached proposal:
                # every fix acts after Evon, so "before" costs nothing to re-score (D-042).
                for side, fixes in (("before", NONE), ("after", ALL)):
                    s = score_call(take_by_id[r["id"]], hit["tool"], hit["arguments"],
                                   known_customers=known, known_items=stock,
                                   fixes=fixes, utterance=r["transcript"])
                    r[f"outcome_{side}"], r[f"reasons_{side}"] = s.outcome.value, list(s.reasons)
                    if side == "after":
                        r["fix_applied"] = s.details.get("fix_applied")
                r.update(outcome=r["outcome_after"], reasons=r["reasons_after"],
                         proposal={"tool": hit["tool"], "arguments": hit["arguments"]}, evon_ms=hit.get("ms"))

    if args.dry_run or not rows:
        return 0

    dev_ids = load_dev_ids()
    for r in rows:
        r["split"] = split_of(r["id"], dev_ids)
    summary = summarise_by_condition(rows, conditions)
    # Held out is the honest number: the fixes were designed by reading the dev set's failures.
    splits = {
        name: summarise_by_condition([r for r in rows if r["split"] == name], conditions)
        for name in ("heldout", "dev")
        if any(r["split"] == name for r in rows)
    }
    for name, title in (("heldout", "HELD OUT (never used to design a fix)"), ("dev", "DEV (the fixes were designed from these)")):
        if name in splits:
            print_table(f"{title}: {len({r['id'] for r in rows if r['split'] == name})} takes", splits[name])

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{args.label}_{date.today().isoformat()}.json"
    write_json(out, {
        "label": args.label,
        "stt_model": STT_MODEL_ID,
        "evon_model": settings.evon_model,
        "prompt_sha": prompt_sha,
        "conditions": {c.id: c.__dict__ for c in conditions},
        "summary": summary,
        "splits": splits,
        "rows": rows,
    })
    print(f"\nwrote {out.relative_to(ROOT)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="plan and price; make no calls")
    parser.add_argument("--max-calls", type=int, default=200, help="cap on real Prisma calls this run (default 200)")
    parser.add_argument("--stages", default="stt,task", help="comma list: stt, task (default both)")
    parser.add_argument("--conditions", help="comma list of condition ids (default all)")
    parser.add_argument("--ids", help="comma list of utterance ids (default every recorded take)")
    parser.add_argument("--label", default="baseline", help="results file prefix (default baseline)")
    args = parser.parse_args()
    args.stages = set(args.stages.split(","))
    try:
        return asyncio.run(run(args))
    except KeyboardInterrupt:
        print("\ninterrupted; everything fetched so far is cached", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
