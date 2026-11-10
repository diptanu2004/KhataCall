"""Drive the real page in a phone-sized Chrome, with real touch events.

    uv run python -m scripts.ui_check            # fakes: layout + touch flow, no credits
    uv run python -m scripts.ui_check --real     # real Prisma, Evon and Timbre (about Rs 0.20)

The owners this is for will use a phone, so this checks what a phone does:

  layout   no sideways scrolling at 320-1280 px, the mic button on screen at
           every scroll position, nothing hidden behind it, 16 px selects (iOS
           zooms on smaller), 44 px touch targets
  touch    a hold that begins the conversation, "haan" committing to the ledger,
           a finger sliding off the button mid-sentence, a tap that is too short,
           the app going to the background
  https    a phone on a plain http:// link (not localhost) is told why the mic
           cannot start
  playback a reply in Timbre's 16-128 ms chunks plays as one clean stream at
           44.1 and 48 kHz, all of it, once (D-050)

Touches go through Chrome's DevTools protocol as real input, so the browser's
own rules apply: on a phone the touch that begins a hold is not a user gesture,
and audio can only start from a tap. That rule is what broke the first press
before Nov 8 (D-047), and only a real touch shows it.

`--real` feeds take 001 and a "haan" into Chrome's fake microphone, so the
browser's capture and downsampling reach real Prisma: the transcript shows
whether the audio survived. It needs Evon up, like the demo.

Uses the Chrome already installed (set CHROME to override). Screenshots go to
scripts/out/ui/. Exit status 1 if any check fails.
"""

import argparse
import asyncio
import base64
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf
import websockets

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "scripts" / "out" / "ui"
CHROME_PATHS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
]
SIZES = [(320, 568), (360, 740), (390, 844), (412, 915), (844, 390), (768, 1024)]

failures: list[str] = []


def check(ok: bool, what: str, detail: object = "") -> None:
    print(f"  {'ok  ' if ok else 'FAIL'} {what}" + (f"  ({detail})" if detail != "" else ""), flush=True)
    if not ok:
        failures.append(what)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Tab:
    """Just enough of the DevTools protocol: evaluate, emulate a phone, touch, screenshot."""

    def __init__(self, ws) -> None:
        self.ws, self.n, self.waiting, self.errors = ws, 0, {}, []

    async def pump(self) -> None:
        async for raw in self.ws:
            m = json.loads(raw)
            if "id" in m:
                self.waiting.pop(m["id"]).set_result(m)
            elif m.get("method") == "Runtime.exceptionThrown":
                d = m["params"]["exceptionDetails"]
                self.errors.append(d.get("exception", {}).get("description") or d.get("text"))

    async def send(self, method: str, **params):
        self.n += 1
        future = asyncio.get_running_loop().create_future()
        self.waiting[self.n] = future
        await self.ws.send(json.dumps({"id": self.n, "method": method, "params": params}))
        reply = await asyncio.wait_for(future, 30)
        if "error" in reply:
            raise RuntimeError(f"{method}: {reply['error']}")
        return reply["result"]

    async def js(self, expression: str):
        reply = await self.send("Runtime.evaluate", expression=expression, returnByValue=True, awaitPromise=True)
        if "exceptionDetails" in reply:
            raise RuntimeError(reply["exceptionDetails"])
        return reply["result"].get("value")

    async def open(self, url: str, width: int, height: int, *, mobile: bool = True) -> None:
        await self.send("Emulation.setDeviceMetricsOverride", width=width, height=height,
                        deviceScaleFactor=2, mobile=mobile)
        await self.send("Emulation.setTouchEmulationEnabled", enabled=mobile, maxTouchPoints=5 if mobile else 1)
        await self.send("Page.navigate", url=url)
        for _ in range(50):
            await asyncio.sleep(0.2)
            if await self.js("typeof connected !== 'undefined' && connected"):
                return
        raise RuntimeError(f"the page at {url} never connected")

    async def shot(self, name: str) -> None:
        OUT.mkdir(parents=True, exist_ok=True)
        data = (await self.send("Page.captureScreenshot", format="png"))["data"]
        (OUT / f"{name}.png").write_bytes(base64.b64decode(data))

    async def mic_centre(self) -> tuple[float, float]:
        r = await self.js("(() => { const r = mic.getBoundingClientRect(); return [r.x + r.width / 2, r.y + r.height / 2]; })()")
        return r[0], r[1]

    async def touch(self, kind: str, x: float = 0, y: float = 0) -> None:
        points = [] if kind == "touchEnd" else [{"x": x, "y": y}]
        await self.send("Input.dispatchTouchEvent", type=kind, touchPoints=points)

    async def tap(self) -> None:
        x, y = await self.mic_centre()
        await self.touch("touchStart", x, y)
        await asyncio.sleep(0.05)
        await self.touch("touchEnd")

    async def hold(self, seconds: float, *, slide_px: float = 0) -> None:
        x, y = await self.mic_centre()
        await self.touch("touchStart", x, y)
        if slide_px:
            await asyncio.sleep(seconds / 2)
            await self.touch("touchMove", x, y - slide_px)  # off the button, still pressing
            await asyncio.sleep(seconds / 2)
        else:
            await asyncio.sleep(seconds)
        await self.touch("touchEnd")

    async def wait_for(self, expression: str, timeout: float = 20.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if value := await self.js(expression):
                return value
            await asyncio.sleep(0.2)
        return None


SNAP = """({label: mic.textContent, holding, micReady, ctx: audioCtx && audioCtx.state,
          turns: [...document.querySelectorAll('.turn .said')].map(e => e.textContent),
          hint: document.getElementById('hint').textContent,
          alert: document.getElementById('alert').hidden ? null : document.getElementById('alert').textContent,
          sharma: [...document.querySelectorAll('#customers .row')].map(r => r.textContent).find(t => t.startsWith('Sharma ji')),
          changed: document.querySelectorAll('#customers .row.changed').length})"""

LAYOUT = """(() => {
  const de = document.documentElement, m = mic.getBoundingClientRect(), dock = document.querySelector('.dock').getBoundingClientRect();
  const small = [...document.querySelectorAll('button, summary, select, .switch')]
    .filter(e => e.offsetParent !== null).map(e => [e.id || e.className || e.tagName, Math.round(e.getBoundingClientRect().height)])
    .filter(([, h]) => h < 44 && h > 0);
  return {scrollW: de.scrollWidth, innerW: innerWidth, innerH: innerHeight, micTop: m.top, micBottom: m.bottom,
          dockTop: dock.top, selectPx: parseFloat(getComputedStyle(document.getElementById('snr')).fontSize),
          callout: getComputedStyle(mic).webkitTouchCallout, select: getComputedStyle(mic).userSelect, small};
})()"""


async def layout_checks(tab: Tab, url: str) -> None:
    print("\nlayout")
    for width, height in [*SIZES, (1280, 800)]:
        mobile = width < 1000
        await tab.open(url, width, height, mobile=mobile)
        # The ledger and the eval summary arrive after the page connects and lengthen it;
        # measure once they are in, scrolling and measuring in the same frame.
        await tab.wait_for("document.querySelectorAll('#customers .row').length > 0", 10)
        await tab.js("document.querySelectorAll('details').forEach(d => d.open = true); null")
        await asyncio.sleep(0.3)
        top = await tab.js(LAYOUT)
        bottom = await tab.js("""new Promise(done => {
          scrollTo(0, document.documentElement.scrollHeight);
          requestAnimationFrame(() => requestAnimationFrame(() => done(%s)));
        })""" % LAYOUT)
        last = await tab.js("document.querySelector('main').lastElementChild.getBoundingClientRect().bottom")
        size = f"{width}x{height}"
        check(top["scrollW"] <= top["innerW"], f"{size}: no sideways scrolling", f"{top['scrollW']} vs {top['innerW']}")
        check(0 <= top["micTop"] and top["micBottom"] <= top["innerH"], f"{size}: mic on screen at the top")
        check(0 <= bottom["micTop"] and bottom["micBottom"] <= bottom["innerH"], f"{size}: mic on screen at the bottom")
        check(last <= bottom["dockTop"] + 1, f"{size}: nothing hidden behind the dock", f"{last:.0f} vs {bottom['dockTop']:.0f}")
        if mobile:
            check(top["selectPx"] >= 16, f"{size}: selects are 16 px (no iOS zoom)", top["selectPx"])
            check(not top["small"], f"{size}: touch targets at least 44 px", top["small"])
        await tab.js("scrollTo(0, 0); document.querySelectorAll('details').forEach(d => d.open = false); null")
        await tab.shot(f"{size}")
    check(top["select"] == "none", "the mic button cannot be text-selected by a long press")


async def touch_checks(tab: Tab, url: str) -> None:
    print("\ntouch, on a 390x844 phone (fake models)")
    await tab.open(url, 390, 844)
    s = await tab.js(SNAP)
    check(s["label"] == "Tap to turn on the mic", "before anything, the button asks for one tap", s["label"])

    await tab.hold(1.0)  # what an owner does first: press and talk
    s = await tab.js(SNAP)
    check(not s["holding"] and not s["turns"], "a hold before the mic is on records nothing and sticks nowhere", s)

    await tab.tap()
    await tab.wait_for("micReady && audioCtx.state === 'running'", 10)
    s = await tab.js(SNAP)
    check(s["micReady"] and s["ctx"] == "running", "one tap turns the mic on and starts audio", s)
    check(s["label"] == "Hold to speak", "then the button says hold to speak", s["label"])
    await tab.shot("phone_ready")

    x, y = await tab.mic_centre()
    await tab.touch("touchStart", x, y)
    await asyncio.sleep(0.6)
    during = await tab.js(SNAP)
    await tab.shot("phone_listening")
    await asyncio.sleep(0.6)
    await tab.touch("touchEnd")
    check(during["holding"] and during["label"].startswith("Listening"), "while held, it is listening", during["label"])
    s = await tab.wait_for(f"(() => {{ const s = {SNAP}; return s.turns.length >= 2 && s.label === 'Hold to speak' && s; }})()")
    check(bool(s) and "150 rupaye" in s["turns"][0], "the first hold is heard and read back", s and s["turns"][:2])

    before = s and s["sharma"]
    await tab.hold(1.0)
    s = await tab.wait_for(f"(() => {{ const s = {SNAP}; return s.turns.length >= 4 && s.label === 'Hold to speak' && s; }})()")
    check(bool(s) and s["sharma"] != before and s["changed"] == 1,
          "'haan' commits, and the changed row flashes", s and (before, s["sharma"]))
    await tab.shot("phone_committed")

    await tab.hold(1.0, slide_px=250)
    s = await tab.wait_for(f"(() => {{ const s = {SNAP}; return s.turns.length >= 6 && s.label === 'Hold to speak' && s; }})()")
    check(bool(s), "a finger that slides off the button still finishes the sentence", s and s["turns"][:2])

    turns = len(s["turns"]) if s else 0
    await tab.tap()
    await asyncio.sleep(0.8)
    s = await tab.js(SNAP)
    check(len(s["turns"]) == turns and "Hold the button" in s["hint"] and s["label"] == "Hold to speak",
          "a tap that is too short gets a hint, not a 15 s wait", s["hint"])

    await tab.js("""Object.defineProperty(document, 'hidden', {configurable: true, get: () => true});
                    document.dispatchEvent(new Event('visibilitychange')); null""")
    s = await tab.js(SNAP)
    check(not s["micReady"] and s["label"] == "Tap to turn on the mic", "going to the background turns the mic off", s["label"])
    check(not tab.errors, "no script errors", tab.errors)


async def https_check(tab: Tab, port: int) -> None:
    print("\na phone on a plain http:// address")
    await tab.open(f"http://khatacall.test:{port}/", 390, 844)
    await tab.tap()
    await asyncio.sleep(0.5)
    s = await tab.js(SNAP)
    check(bool(s["alert"]) and "https" in s["alert"] and not s["micReady"], "it is told the mic needs https", s["alert"])
    await tab.shot("phone_http")


async def downsample_check(tab: Tab) -> None:
    print("\ndownsampling 48 kHz to 16 kHz")
    db = await tab.js("""(() => {
      const n = 48000, f = 12000, x = new Float32Array(n);
      for (let i = 0; i < n; i++) x[i] = 0.5 * Math.sin(2 * Math.PI * f * i / 48000 + 0.3);
      const rms = (a) => Math.sqrt(a.reduce((s, v) => s + (v / 32767) ** 2, 0) / a.length);
      const picked = Int16Array.from({length: n / 3}, (_, i) => x[i * 3] * 32767);
      return [20 * Math.log10(rms(downsample(x, 3)) / 0.3536), 20 * Math.log10(rms(picked) / 0.3536)];
    })()""")
    check(db[0] < -6, "a 12 kHz tone (above what 16 kHz can carry) is suppressed, not folded into speech",
          f"{db[0]:.1f} dB, against {db[1]:.1f} dB for picking every third sample")


PLAYBACK = """(async (rate, speed) => {
  // A 440 Hz tone at 16 kHz, cut the way Timbre cuts a reply (Nov 10: 73 chunks,
  // 16-128 ms, median 32 ms), played at a phone's output rate. Scored by how far the
  // output is from a pure 440 Hz tone: clicks and gaps at the joins are the error.
  const fs = 16000, f = 440, n = fs;
  const pcm = Int16Array.from({length: n}, (_, i) => Math.round(16383 * Math.sin(2 * Math.PI * f * i / fs)));
  const ms = [80, 16, 16, 16, 32, 16, 80, 32, 48, 32, 32, 32, 16, 128];
  const chunks = [];
  for (let at = 0, j = 0; at < n; j++) {
    const len = (ms[j % ms.length] * fs) / 1000;
    chunks.push(pcm.slice(at, at + len));
    at += len;
  }
  const snr = (out, from, to) => {
    let ss = 0, sc = 0, cc = 0, ys = 0, yc = 0;
    const w = 2 * Math.PI * f / rate;
    for (let i = from; i < to; i++) {
      const s = Math.sin(w * i), c = Math.cos(w * i);
      ss += s * s; sc += s * c; cc += c * c; ys += out[i] * s; yc += out[i] * c;
    }
    const det = ss * cc - sc * sc, a = (ys * cc - yc * sc) / det, b = (yc * ss - ys * sc) / det;
    let sig = 0, err = 0;
    for (let i = from; i < to; i++) {
      const m = a * Math.sin(w * i) + b * Math.cos(w * i);
      sig += m * m; err += (out[i] - m) ** 2;
    }
    return 10 * Math.log10(sig / err);
  };

  // Before D-050: one AudioBufferSource per chunk, scheduled back to back.
  const off = new OfflineAudioContext(1, Math.ceil(1.2 * rate), rate);
  let head = 0.06;
  for (const c of chunks) {
    const buf = off.createBuffer(1, c.length, fs);
    buf.getChannelData(0).set(Float32Array.from(c, (v) => v / 32768));
    const src = off.createBufferSource();
    src.buffer = buf;
    src.connect(off.destination);
    src.start(head);
    head += buf.duration;
  }
  const old = (await off.startRendering()).getChannelData(0);

  // Now: the page's own player, on a simulated clock. Chunks arrive at `speed` times
  // real time (Timbre: ~10x; a throttled tunnel: below 1), and the output is pulled
  // in 2048-frame blocks as a real audio device would, one block per 2048 frames of time.
  const saved = [audioCtx, playNode, performance.now];
  let clock = 0;  // ms
  performance.now = () => clock;
  audioCtx = {sampleRate: rate};
  playNode = {};  // so playChunk does not build a real node; fillOutput is driven here
  playRate = fs; playBuf = new Float32Array(0); playPos = 0; playing = false; replyEnded = false;
  replyFirstAt = 0; replyAudioS = 0; replyRefills = 0;
  const out = [];
  const pullUntil = (ms) => {
    while ((out.length / rate) * 1000 < ms) { const o = new Float32Array(2048); fillOutput(o); out.push(...o); }
  };
  for (const c of chunks) {
    clock += (c.length / fs) * 1000 / speed;
    pullUntil(clock);
    playChunk(c.buffer);
  }
  replyEnded = true;
  pullUntil(clock + 1500);
  [audioCtx, playNode, performance.now] = saved;

  const first = out.findIndex((v) => v !== 0);
  let last = out.length - 1;
  while (last > 0 && out[last] === 0) last--;
  // Pauses inside the reply: runs of 10 ms or more of exact silence between first and last.
  let pauses = 0, zeros = 0;
  for (let i = first; i <= last; i++) {
    if (out[i] === 0) zeros++;
    else { if (zeros >= rate / 100) pauses++; zeros = 0; }
  }
  return {
    old: snr(old, Math.round(0.2 * rate), Math.round(1.0 * rate)),
    now: snr(out, first + Math.round(0.1 * rate), first + Math.round(0.9 * rate)),
    seconds: (last - first + 1) / rate,
    pauses,
    startsAfter: first / rate,
  };
})"""


async def playback_check(tab: Tab) -> None:
    print("\nplaying a reply streamed in Timbre's chunks (D-050)")
    for rate in (44100, 48000):
        r = await tab.js(f"{PLAYBACK}({rate}, 10)")
        check(r["now"] > 40 and r["pauses"] == 0, f"at {rate} Hz the reply plays as one clean stream",
              f"{r['now']:.1f} dB from a pure tone, against {r['old']:.1f} dB one buffer per chunk; "
              f"{r['pauses']} pauses, starts after {r['startsAfter'] * 1000:.0f} ms")
        check(abs(r["seconds"] - 1.0) < 0.01, f"at {rate} Hz all of it plays, once", f"{r['seconds']:.3f} s of 1 s")
    slow = await tab.js(f"{PLAYBACK}(48000, 0.5)")
    check(slow["pauses"] == 0 and abs(slow["seconds"] - 1.0) < 0.01,
          "on a link at half real time it plays whole, once it has arrived, not in bursts",
          f"{slow['pauses']} pauses, starts after {slow['startsAfter']:.2f} s")


async def real_checks(tab: Tab, url: str, wav: Path) -> None:
    print("\nreal models: take 001 then 'haan', through the browser's microphone")
    await tab.open(url, 390, 844)
    await tab.tap()
    await tab.wait_for("micReady", 10)
    mic_on = time.monotonic()  # the fake microphone's file starts playing as the mic opens
    await asyncio.sleep(0.8)
    await tab.hold(4.6)  # the take is at 1.0-5.2 s in the file
    s = await tab.wait_for(f"(() => {{ const s = {SNAP}; return s.turns.length >= 2 && s.label === 'Hold to speak' && s; }})()", 30)
    check(bool(s), "the request is heard and answered", s and s["turns"][:2])
    if s:
        print(f"       heard: {s['turns'][1]!r}\n       said:  {s['turns'][0]!r}")
    latency = await tab.js("document.getElementById('l-total').textContent")
    print(f"       end of speech to first audio: {latency}")
    await tab.shot("real_readback")
    await asyncio.sleep(max(0.0, mic_on + 13.7 - time.monotonic()))  # "haan" is at 14.0-15.4 s
    before = s and s["sharma"]
    await tab.hold(2.2)
    s = await tab.wait_for(f"(() => {{ const s = {SNAP}; return s.turns.length >= 4 && s.label === 'Hold to speak' && s; }})()", 30)
    check(bool(s) and s["sharma"] != before, "'haan' commits on the real ledger path", s and (before, s["sharma"], s["turns"][:2]))
    await tab.shot("real_committed")
    check(not tab.errors, "no script errors", tab.errors)


def real_wav(tmp: Path) -> Path:
    """1 s of quiet, take 001, quiet, then "haan" at 14 s: what the fake mic will play."""
    take, rate = sf.read(ROOT / "eval" / "corpus" / "audio" / "001.wav", dtype="float32")
    haan, haan_rate = sf.read(ROOT / "scripts" / "out" / "turn_haan.wav", dtype="float32")
    if rate != 16_000 or haan_rate != 16_000:
        raise SystemExit("expected 16 kHz recordings")
    out = np.zeros(int(18 * rate), dtype=np.float32)
    out[rate : rate + len(take)] = take
    out[14 * rate : 14 * rate + len(haan)] = haan
    path = tmp / "mic.wav"
    sf.write(path, out, rate, subtype="PCM_16")
    return path


def find_chrome() -> str:
    for path in [os.environ.get("CHROME", ""), *CHROME_PATHS]:
        if path and Path(path).exists():
            return path
    raise SystemExit("Chrome not found; set CHROME to its path")


async def run(args) -> int:
    tmp = Path(tempfile.mkdtemp(prefix="khatacall-ui-"))
    port, debug_port = free_port(), free_port()
    env = {**os.environ, "DB_PATH": str(tmp / "ui.db"), "PYTHONIOENCODING": "utf-8"}
    env["USE_FAKES"] = "false" if args.real else "true"
    subprocess.run([sys.executable, "-m", "scripts.seed", "--db", env["DB_PATH"]], env=env, cwd=ROOT,
                   capture_output=True, check=True)
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(port), "--log-level", "warning"],
        env=env, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=None if args.real else subprocess.DEVNULL,
    )
    flags = [
        "--headless=new", f"--remote-debugging-port={debug_port}", f"--user-data-dir={tmp / 'chrome'}",
        "--no-first-run", "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
        "--autoplay-policy=document-user-activation-required",  # what a phone enforces
        f"--host-resolver-rules=MAP khatacall.test 127.0.0.1",  # a non-localhost http:// address
    ]
    if args.real:
        flags.append(f"--use-file-for-fake-audio-capture={real_wav(tmp)}%noloop")
    chrome = subprocess.Popen([find_chrome(), *flags, "about:blank"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(120):
            try:
                if (httpx.get(f"http://127.0.0.1:{port}/healthz", timeout=1).status_code == 200
                        and httpx.get(f"http://127.0.0.1:{debug_port}/json/version", timeout=1).status_code == 200):
                    break
            except httpx.HTTPError:
                pass
            time.sleep(0.5)
        target = httpx.put(f"http://127.0.0.1:{debug_port}/json/new?about:blank").json()
        async with websockets.connect(target["webSocketDebuggerUrl"], max_size=None) as ws:
            tab = Tab(ws)
            pump = asyncio.create_task(tab.pump())
            await tab.send("Page.enable")
            await tab.send("Runtime.enable")
            url = f"http://127.0.0.1:{port}/"
            if args.real:
                await real_checks(tab, url, tmp / "mic.wav")
            else:
                await layout_checks(tab, url)
                await touch_checks(tab, url)
                await downsample_check(tab)
                await playback_check(tab)
                await https_check(tab, port)
            pump.cancel()
    finally:
        chrome.terminate()
        server.terminate()
        chrome.wait(10)
        server.wait(10)
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{'all checks passed' if not failures else f'{len(failures)} failed: ' + '; '.join(failures)}")
    print(f"screenshots in {OUT.relative_to(ROOT)}")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--real", action="store_true", help="real Prisma, Evon and Timbre (about Rs 0.20; needs Evon up)")
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
