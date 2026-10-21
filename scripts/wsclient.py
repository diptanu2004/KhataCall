"""Drive the /ws endpoint exactly as the browser does, without a browser.

    uv run python -m scripts.wsclient                                 # default clip
    uv run python -m scripts.wsclient --wav my.wav                     # your own recording
    uv run python -m scripts.wsclient --wav ask.wav --wav haan.wav     # a conversation
    uv run python -m scripts.wsclient --url ws://host/ws

Each --wav is one push-to-talk turn on the same connection, streamed at
real-time cadence. Every server message is printed, ledger changes as
"before -> after", and the agent's spoken replies are written to
scripts/out/ws_reply.wav.

Useful for checking the real server path (uvicorn, protocol, session,
executor) and for confirming a change works before recording a demo. Whether
this spends Gnani credits depends on the server's mode: USE_FAKES=true spends
none.
"""

import argparse
import asyncio
import json
import time
import wave
from collections.abc import Callable
from pathlib import Path

import numpy as np
import soundfile as sf
import websockets

from app.protocol import CLIENT_SAMPLE_RATE
from scripts.smoke import DEFAULT_WAV, OUT_DIR

# 20 ms of audio per send, close to what a browser's capture callback produces.
CHUNK_MS = 20


def load_pcm(path: Path) -> bytes:
    samples, rate = sf.read(path, dtype="int16", always_2d=True)
    mono = samples.mean(axis=1).astype(np.int16) if samples.shape[1] > 1 else samples[:, 0]
    if rate != CLIENT_SAMPLE_RATE:
        raise SystemExit(f"{path} is {rate} Hz; the browser sends {CLIENT_SAMPLE_RATE} Hz")
    return mono.tobytes()


async def run(url: str, wavs: list[Path], out: Path, wait_for_listening: bool) -> int:
    turns = [load_pcm(w) for w in wavs]
    chunk_bytes = 2 * (CLIENT_SAMPLE_RATE * CHUNK_MS // 1000)
    reply = bytearray()
    reply_rate = CLIENT_SAMPLE_RATE
    balances: dict[str, int] = {}
    t0 = time.perf_counter()

    def log(line: str) -> None:
        print(f"[{(time.perf_counter() - t0) * 1000:7.0f} ms] {line}", flush=True)

    async with websockets.connect(url, max_size=None) as ws:
        log(f"connected to {url}")
        events: asyncio.Queue[dict] = asyncio.Queue()

        async def receive() -> None:
            nonlocal reply_rate
            async for message in ws:
                if isinstance(message, bytes):
                    reply.extend(message)
                    continue
                event = json.loads(message)
                match event["type"]:
                    case "state":
                        log(f"state      {event['state']}")
                    case "transcript":
                        log(f"transcript {event['text']!r}")
                    case "agent_text":
                        log(f"agent      {event['text']!r}")
                    case "audio_start":
                        reply_rate = event["sample_rate"]
                    case "audio_end":
                        log(f"audio      {len(reply) / 2 / reply_rate:.2f} s of reply so far")
                    case "latency":
                        log("latency    " + "  ".join(f"{k}={v}" for k, v in event.items() if k != "type"))
                    case "ledger_update":
                        now = {c["name"]: c["balance_paise"] for c in event.get("customers", [])}
                        changed = [
                            f"{name} Rs {balances.get(name, 0) / 100:,.0f} -> Rs {paise / 100:,.0f}"
                            for name, paise in now.items()
                            if balances and balances.get(name) != paise
                        ]
                        summary = f"{len(now)} customers, {len(event.get('items', []))} items"
                        log("ledger     " + ("; ".join(changed) if changed else summary))
                        balances.clear()
                        balances.update(now)
                    case _:
                        log(f"{event['type']:10} {event}")
                await events.put(event)

        async def wait_for(match: Callable[[dict], bool], timeout: float) -> dict:
            while True:
                event = await asyncio.wait_for(events.get(), timeout)
                if match(event):
                    return event

        def listening(e: dict) -> bool:
            return e["type"] == "state" and e["state"] == "LISTENING"

        receiver = asyncio.create_task(receive())
        try:
            for n, (wav, pcm) in enumerate(zip(wavs, turns, strict=True), start=1):
                if len(turns) > 1:
                    log(f"--- turn {n}: {wav.name}")
                await ws.send(json.dumps({"type": "start"}))
                if wait_for_listening:
                    # The server opens the Prisma socket on "start". Streaming before it is
                    # ready would queue a backlog and then burst it, which the docs forbid.
                    await wait_for(listening, 30)

                next_send = time.perf_counter()  # pace at real time, like a microphone
                for i in range(0, len(pcm), chunk_bytes):
                    await ws.send(pcm[i : i + chunk_bytes])
                    next_send += CHUNK_MS / 1000
                    if (delay := next_send - time.perf_counter()) > 0:
                        await asyncio.sleep(delay)
                log(f"sent {len(pcm) / 2 / CLIENT_SAMPLE_RATE:.2f} s of audio; releasing push-to-talk")
                await ws.send(json.dumps({"type": "stop"}))

                await wait_for(lambda e: e["type"] == "latency", 120)
                # The turn ends by going back to LISTENING; take it so the next turn's
                # wait sees a fresh one.
                await wait_for(listening, 30)
        except TimeoutError:
            log("timed out waiting for the server")
            return 1
        finally:
            receiver.cancel()

    if reply:
        out.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(out), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(reply_rate)
            w.writeframes(bytes(reply))
        log(f"wrote {out}")
    else:
        log("no audio came back")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="ws://127.0.0.1:8000/ws")
    parser.add_argument("--wav", type=Path, action="append", help="one turn; repeat for a conversation")
    parser.add_argument("--out", type=Path, default=OUT_DIR / "ws_reply.wav")
    parser.add_argument("--no-wait", action="store_true", help="stream audio without waiting for LISTENING")
    args = parser.parse_args()
    wavs = args.wav or [DEFAULT_WAV]
    for wav in wavs:
        if not wav.exists():
            print(f"{wav} not found; make one with: uv run python -m scripts.smoke tts {wav} --text '...'")
            return 2
    return asyncio.run(run(args.url, wavs, args.out, not args.no_wait))


if __name__ == "__main__":
    raise SystemExit(main())
