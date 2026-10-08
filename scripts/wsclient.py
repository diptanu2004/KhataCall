"""Drive the /ws endpoint exactly as the browser does, without a browser.

    uv run python -m scripts.wsclient                      # default clip
    uv run python -m scripts.wsclient --wav my.wav          # your own recording
    uv run python -m scripts.wsclient --url ws://host/ws

Streams a WAV at real-time cadence, then prints every server message and
writes the agent's spoken reply to scripts/out/ws_reply.wav.

Useful for checking the real server path (uvicorn, protocol, session) and for
confirming a change works before recording a demo. Whether this spends Gnani
credits depends on the server's mode: USE_FAKES=true spends none.
"""

import argparse
import asyncio
import json
import time
import wave
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


async def run(url: str, wav: Path, out: Path, wait_for_listening: bool) -> int:
    pcm = load_pcm(wav)
    chunk_bytes = 2 * (CLIENT_SAMPLE_RATE * CHUNK_MS // 1000)
    reply = bytearray()
    reply_rate = CLIENT_SAMPLE_RATE
    t0 = time.perf_counter()

    def log(line: str) -> None:
        print(f"[{(time.perf_counter() - t0) * 1000:7.0f} ms] {line}", flush=True)

    async with websockets.connect(url, max_size=None) as ws:
        log(f"connected to {url}")

        async def receive(listening: asyncio.Event) -> None:
            nonlocal reply_rate
            async for message in ws:
                if isinstance(message, bytes):
                    reply.extend(message)
                    continue
                event = json.loads(message)
                match event["type"]:
                    case "state":
                        log(f"state      {event['state']}")
                        if event["state"] == "LISTENING":
                            listening.set()
                    case "transcript":
                        log(f"transcript {event['text']!r}")
                    case "agent_text":
                        log(f"agent      {event['text']!r}")
                    case "audio_start":
                        reply_rate = event["sample_rate"]
                        log(f"audio      starting at {reply_rate} Hz")
                    case "audio_end":
                        log(f"audio      ended, {len(reply) / 2 / reply_rate:.2f} s")
                    case "latency":
                        log("latency    " + "  ".join(f"{k}={v}" for k, v in event.items() if k != "type"))
                        return
                    case _:
                        log(f"{event['type']:10} {event}")

        listening = asyncio.Event()
        receiver = asyncio.create_task(receive(listening))
        await ws.send(json.dumps({"type": "start"}))

        if wait_for_listening:
            # The server opens the Prisma socket on "start". Streaming before it is ready
            # would queue a backlog and then burst it, which the docs warn against.
            await asyncio.wait_for(listening.wait(), timeout=30)
            log("server is LISTENING; streaming audio")

        # Pace at real time, like a microphone.
        next_send = time.perf_counter()
        for i in range(0, len(pcm), chunk_bytes):
            await ws.send(pcm[i : i + chunk_bytes])
            next_send += CHUNK_MS / 1000
            if (delay := next_send - time.perf_counter()) > 0:
                await asyncio.sleep(delay)
        log(f"sent {len(pcm) / 2 / CLIENT_SAMPLE_RATE:.2f} s of audio; releasing push-to-talk")
        await ws.send(json.dumps({"type": "stop"}))

        try:
            await asyncio.wait_for(receiver, timeout=60)
        except TimeoutError:
            log("timed out waiting for the turn to finish")
            return 1

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
    parser.add_argument("--wav", type=Path, default=DEFAULT_WAV)
    parser.add_argument("--out", type=Path, default=OUT_DIR / "ws_reply.wav")
    parser.add_argument("--no-wait", action="store_true", help="stream audio without waiting for LISTENING")
    args = parser.parse_args()
    if not args.wav.exists():
        print(f"{args.wav} not found; run `scripts.smoke tts` first")
        return 2
    return asyncio.run(run(args.url, args.wav, args.out, not args.no_wait))


if __name__ == "__main__":
    raise SystemExit(main())
