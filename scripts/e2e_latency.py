"""End-to-end latency with all three real models: corpus takes replayed through /ws.

    uv run python -m scripts.e2e_latency --dry-run      # takes, turns and estimated rupees
    uv run python -m scripts.e2e_latency --n 30         # measure; starts its own server
    uv run python -m scripts.e2e_latency --url ws://127.0.0.1:8000/ws   # a server you started

What is measured is what the owner waits: from releasing push-to-talk to the
first byte of the reply's audio, split by stage exactly as the UI shows it (the
session's own `latency` message, app/pipeline/session.py). Nothing is timed
here that the product does not time itself.

Each take is one short call: a fresh connection, the take as the first turn,
and, when the agent reads a write back, a "haan" turn (scripts/out/turn_haan.wav,
Timbre's voice, D-032) so the confirmation path, which skips Evon (D-006), is
measured as well. Audio is paced at real time, like a microphone.

The script starts its own server on a scratch, freshly seeded ledger, so the
demo database is never written. Before spending anything it checks Evon is up:
with Evon down every turn would still pay for Prisma and Timbre. USE_FAKES=true
runs the whole thing offline, for checking the script itself.

Spend: Prisma bills audio seconds, Timbre bills characters (D-020); both are
metered by the adapters to scripts/out/usage.jsonl as usual.
"""

import argparse
import asyncio
import json
import os
import socket
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf
import websockets

from app.adapters.usage import stt_cost_inr, tts_cost_inr
from app.config import get_settings
from app.ledger.db import connect
from app.ledger.repo import Ledger
from eval.run_eval import load_takes
from scripts.evon_check import probe
from scripts.latency import summarise
from scripts.record import spread_across_intents
from scripts.seed import seed

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "scripts" / "out"
RESULTS_DIR = ROOT / "eval" / "results"
HAAN_WAV = OUT_DIR / "turn_haan.wav"
RATE = 16_000
CHUNK_MS = 20
WRITE_INTENTS = {"add_credit", "record_payment", "update_stock"}
# Agent lines are read-backs and short answers; 60 characters is a generous average.
EST_REPLY_CHARS = 60
STAGES = ("stt_ms", "llm_ms", "tts_first_byte_ms", "total_ms")


def plan(takes: list[dict], n: int) -> list[dict]:
    """The first n takes in an order that keeps every prefix's intent mix (as record.py)."""
    return spread_across_intents(takes)[:n]


def estimate_inr(planned: list[dict], haan_s: float) -> tuple[int, float, float]:
    """(turns, audio seconds, rupees), assuming every write is read back and confirmed."""
    confirms = sum(1 for t in planned if t["intent"] in WRITE_INTENTS)
    turns = len(planned) + confirms
    audio_s = sum(len(t["audio"]) / RATE for t in planned) + confirms * haan_s
    return turns, audio_s, stt_cost_inr(audio_s) + tts_cost_inr(turns * EST_REPLY_CHARS)


def summarise_turns(rows: list[dict]) -> dict:
    """p50/p90 per stage, separately for turns that went to Evon and those that did not."""
    out = {}
    for kind in ("request", "confirm"):
        done = [r for r in rows if r["kind"] == kind and r.get("total_ms") is not None]
        out[kind] = {
            stage: summarise(values)
            for stage in STAGES
            if (values := [r[stage] for r in done if r.get(stage) is not None])
        }
    return out


def _pcm(audio: np.ndarray) -> bytes:
    return (np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes()


async def _turn(ws, events: asyncio.Queue, pcm: bytes) -> dict:
    """One push-to-talk turn. Returns what the server said and how long each stage took."""
    seen: dict = {"states": []}

    async def until(match, timeout: float) -> None:
        while True:
            event = await asyncio.wait_for(events.get(), timeout)
            match event["type"]:
                case "state":
                    seen["states"].append(event["state"])
                case "transcript":
                    seen["transcript"] = event["text"]
                case "agent_text":
                    seen["agent"] = event["text"]
                case "error":
                    seen["error"] = event["message"]
                case "latency":
                    seen.update({k: event.get(k) for k in STAGES})
            if match(event):
                return

    def listening(e: dict) -> bool:
        return e["type"] == "state" and e["state"] == "LISTENING"

    await ws.send(json.dumps({"type": "start"}))
    await until(listening, 30)  # the Prisma socket is open; do not stream before it is
    seen["states"].clear()
    chunk = 2 * RATE * CHUNK_MS // 1000
    next_send = time.perf_counter()
    for i in range(0, len(pcm), chunk):
        await ws.send(pcm[i : i + chunk])
        next_send += CHUNK_MS / 1000
        if (delay := next_send - time.perf_counter()) > 0:
            await asyncio.sleep(delay)
    await ws.send(json.dumps({"type": "stop"}))
    await until(lambda e: e["type"] == "latency", 120)
    await until(listening, 30)
    return seen


async def _call(url: str, take: dict, haan: bytes, budget: list[int]) -> list[dict]:
    """One take as its own call; a "haan" follows any read-back. Spends from `budget`."""
    rows = []
    async with websockets.connect(url, max_size=None) as ws:
        events: asyncio.Queue = asyncio.Queue()

        async def receive() -> None:
            async for message in ws:
                if not isinstance(message, bytes):
                    await events.put(json.loads(message))

        receiver = asyncio.create_task(receive())
        try:
            for kind, pcm in (("request", _pcm(take["audio"])), ("confirm", haan)):
                if budget[0] <= 0:
                    break
                budget[0] -= 1
                seen = await _turn(ws, events, pcm)
                rows.append(
                    {
                        "id": take["id"],
                        "intent": take["intent"],
                        "kind": kind,
                        "audio_s": round(len(pcm) / 2 / RATE, 2),
                        "transcript": seen.get("transcript"),
                        "agent": seen.get("agent"),
                        "error": seen.get("error"),
                        **{k: seen.get(k) for k in STAGES},
                    }
                )
                if "CONFIRMING" not in seen["states"]:
                    break  # nothing was read back, so there is nothing to say yes to
        finally:
            receiver.cancel()
    return rows


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _start_server(db: Path) -> tuple[subprocess.Popen, str]:
    if db.exists():
        db.unlink()
    seed(Ledger(connect(str(db))))
    port = _free_port()
    env = {**os.environ, "DB_PATH": str(db)}
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(port), "--log-level", "warning"],
        cwd=ROOT,
        env=env,
    )
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"http://127.0.0.1:{port}/healthz", timeout=2).status_code == 200:
                return server, f"ws://127.0.0.1:{port}/ws"
        except httpx.HTTPError:
            pass
        if server.poll() is not None:
            raise SystemExit("the server exited during startup; see its output above")
        time.sleep(0.5)
    server.terminate()
    raise SystemExit("the server did not come up within 60 s")


def _print(summary: dict, rows: list[dict]) -> None:
    for kind, label in (("request", "request turns (Prisma -> Evon -> Timbre)"), ("confirm", "'haan' turns (no Evon)")):
        stages = summary[kind]
        if not stages:
            continue
        print(f"\n{label}, n={stages['total_ms']['n'] if 'total_ms' in stages else 0}")
        print(f"  {'stage':18} {'p50':>8} {'p90':>8} {'max':>8}")
        for stage, s in stages.items():
            print(f"  {stage:18} {s['p50']:8.0f} {s['p90']:8.0f} {s['max']:8.0f}")
    failed = [r for r in rows if r["total_ms"] is None or r["error"]]
    if failed:
        print(f"\n{len(failed)} turn(s) with an error or no audio:")
        for r in failed:
            print(f"  {r['id']} {r['kind']}: {r['error']}")


async def run(args) -> int:
    settings = get_settings()
    takes = plan(load_takes(), args.n)
    haan_audio, rate = sf.read(HAAN_WAV, dtype="float32")
    if rate != RATE or haan_audio.ndim != 1:
        raise SystemExit(f"{HAAN_WAV} must be 16 kHz mono")
    haan = _pcm(haan_audio)

    turns, audio_s, inr = estimate_inr(takes, len(haan_audio) / RATE)
    turns = min(turns, args.max_calls)
    print(
        f"{len(takes)} takes, up to {turns} turns (--max-calls {args.max_calls}), "
        f"{audio_s:.0f} s of audio: about Rs {inr:.2f} at most"
        + (" -- fakes, so Rs 0" if settings.use_fakes else "")
    )
    if args.dry_run:
        return 0

    if not (settings.use_fakes or settings.fake_llm):
        found = await probe(settings.evon_base_url, settings.evon_api_key.get_secret_value())
        if "error" in found:
            print(f"Evon is not reachable, so nothing was spent: {found['error']}")
            print("Start the Kaggle notebook and update EVON_BASE_URL in .env (docs/EVON_KAGGLE.md).")
            return 2

    server = None
    url = args.url
    if url is None:
        server, url = _start_server(OUT_DIR / "e2e_latency.db")
    rows: list[dict] = []
    budget = [args.max_calls]
    try:
        for i, take in enumerate(takes, start=1):
            if budget[0] <= 0:
                print(f"--max-calls {args.max_calls} reached")
                break
            for r in await _call(url, take, haan, budget):
                rows.append(r)
                print(
                    f"[{i:2}/{len(takes)}] {r['id']} {r['kind']:7} total={r['total_ms']} "
                    f"stt={r['stt_ms']} llm={r['llm_ms']} tts={r['tts_first_byte_ms']}  {r['transcript']!r}",
                    flush=True,
                )
    finally:
        if server is not None:
            server.terminate()
            server.wait(10)

    summary = summarise_turns(rows)
    _print(summary, rows)
    mode = "fakes" if settings.use_fakes else "fake_llm" if settings.fake_llm else "real"
    # Only a real run is a result; a fakes run checks the script and stays local.
    out = (RESULTS_DIR if mode == "real" else OUT_DIR) / f"latency_{datetime.now():%Y-%m-%d}_{mode}.json"
    out.write_text(
        json.dumps(
            {
                "measured_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "mode": mode,
                "evon_base_url_host": httpx.URL(settings.evon_base_url).host,
                "summary": summary,
                "rows": rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {out.relative_to(ROOT)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=30, help="corpus takes to replay (default 30, all intents)")
    parser.add_argument("--max-calls", type=int, default=60, help="hard cap on turns, each one Prisma + Timbre call")
    parser.add_argument("--url", help="use a server already running at this /ws URL")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and its estimated cost, spend nothing")
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
