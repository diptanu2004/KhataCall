"""Generate deploy/kaggle_evon.ipynb.

    uv run python deploy/make_kaggle_notebook.py

The notebook is generated rather than hand-edited so it stays reviewable as
plain Python in a diff, and so a malformed cell is caught here (every code
cell is parsed before writing) instead of twenty minutes into a Kaggle run.
"""

import ast
import json
import shutil
import subprocess
from pathlib import Path

OUT = Path(__file__).with_name("kaggle_evon.ipynb")


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text.strip("\n").splitlines(keepends=True),
    }


CELLS = [
    md(
        """
# KhataCall: serve Gnani Evon v3.3 on Kaggle

Runs Evon v3.3 (30B MoE, `nemotron_h`) with **llama.cpp** on Kaggle's two T4 GPUs
and exposes it through a **Cloudflare quick tunnel** as an OpenAI-compatible API.
KhataCall's `app/adapters/evon_llm.py` talks to it unchanged.

**Before you press Run All**, in the right-hand panel:
1. **Accelerator: GPU T4 x2** (not P100 - one 16 GB card is too small)
2. **Internet: On** (needs a phone-verified Kaggle account)

**The first run takes 45-75 minutes**, almost all of it step 4 compiling
llama.cpp's 191 CUDA files on Kaggle's 4 CPU cores. The 25 GB model downloads
at the same time. Every step that waits prints a line at least once a minute,
so if a step goes completely silent for several minutes, something is wrong.

**Later sessions can skip the build.** Step 4 leaves the compiled
`llama-server` in `/kaggle/working/llama-bin/`. Save it as a private Kaggle
dataset once (see the note after step 4), attach it next time with
**Add Input**, and step 4 picks it up in seconds.

At the end, step 8 prints three lines to paste into your local `.env`.

Why llama.cpp and not vLLM: the official BF16 weights are ~60 GB and the two
T4s have 32 GB between them. A 4-bit GGUF fits. See DECISIONS.md D-027.
"""
    ),
    code(
        """
# 1. What do we have? Two T4s, and where is the disk space.
import os, shutil, subprocess, sys, time, json, re, secrets, urllib.request

print(subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv"],
                     capture_output=True, text=True).stdout)

# /kaggle/working is capped at 20 GB, too small for a 25 GB model. Use the
# largest scratch directory instead. It is not saved after the session, which is fine.
candidates = [d for d in ("/kaggle/tmp", "/tmp", "/root") if os.path.isdir(d)]
for d in candidates + ["/kaggle/working"]:
    print(f"{d:16} {shutil.disk_usage(d).free / 1e9:6.1f} GB free")
WORK = max(candidates, key=lambda d: shutil.disk_usage(d).free) + "/evon"
os.makedirs(WORK, exist_ok=True)
os.environ["WORK"] = WORK
print("\\nusing", WORK)
"""
    ),
    code(
        """
# 2. Settings. Change QUANT if the model does not fit (see step 6).
REPO = "luminenio/gnani-evon-v3.3-30B-A3B-GGUF"  # community GGUF of gnani/gnani-evon-v3.3-30B-A3B
QUANT = "Q4_K_M"  # 24.7 GB. If loading runs out of GPU memory, use "IQ4_XS" (18.4 GB).
SIZE_GB = {"Q4_K_M": 24.7, "Q4_K_S": 22.1, "IQ4_XS": 18.4, "Q3_K_M": 20.0}.get(QUANT, 0)
FILE = f"gnani-evon-v3.3-30B-A3B-{QUANT}.gguf"
MODEL_PATH = f"{WORK}/{FILE}"
ALIAS = "gnani-evon-v3.3"  # what to put in EVON_MODEL
CTX = 8192                 # KhataCall's longest prompt is a few thousand tokens
PORT = 8080

# The tunnel URL is public. This key is what stops anyone else using your GPUs.
API_KEY = secrets.token_urlsafe(24)
print("settings ready:", FILE)
"""
    ),
    code(
        """
# 3. Start the model download in the background. It runs while llama.cpp builds.
# Use Kaggle's preinstalled huggingface_hub as-is. Upgrading it to 2.x breaks the
# preinstalled transformers, datasets and gradio, and 2.x dropped the old
# hf_transfer extra anyway; the bundled version already downloads quickly.
download = subprocess.Popen(
    [sys.executable, "-c",
     "import huggingface_hub as h; print('huggingface_hub', h.__version__, flush=True);"
     f"h.hf_hub_download({REPO!r}, {FILE!r}, local_dir={WORK!r}); print('done', flush=True)"],
    stdout=open(f"{WORK}/download.log", "w"), stderr=subprocess.STDOUT,
)
print("downloading", FILE, "in the background, pid", download.pid)
"""
    ),
    code(
        """
%%bash
# 4. Get a llama-server for the T4: from an attached dataset if you saved one,
#    otherwise build it. The first build takes 30-60 minutes.
#
# CMAKE_CUDA_ARCHITECTURES=75: real T4 machine code, and one architecture instead
#   of a dozen. (The prebuilt release binary carries the T4 only as 75-virtual PTX,
#   which the driver would JIT-compile on first use in every session - D-030.)
# GGML_CUDA_NO_VMM=ON: Kaggle's CUDA toolkit has no libcuda.so for CMake to link
#   ("CUDA::cuda_driver ... target was not found"). Only VMM, a memory-pool
#   optimisation, needs it; ggml falls back to its ordinary pool allocator.
# BUILD_SHARED_LIBS=OFF: one self-contained binary, so it can be saved and reused.
#
# The build runs DETACHED. Interrupting this cell does not stop or damage it, and
# running the cell again just resumes watching. It prints a line every minute.
set -e
POLL=${POLL:-60}
INPUT=${KAGGLE_INPUT:-/kaggle/input}
SAVE=${KAGGLE_SAVE:-/kaggle/working/llama-bin}
BIN="$WORK/llama-bin"
cd "$WORK"

# A build saved from an earlier session and attached with Add Input? It is the
# llama-server binary plus any shared libraries saved beside it; LD_LIBRARY_PATH
# lets the binary find them (step 6 sets it the same way).
if [ ! -x "$BIN/llama-server" ]; then
    cached=$(find "$INPUT" -name llama-server -type f 2>/dev/null | head -1)
    if [ -n "$cached" ]; then
        mkdir -p "$BIN" && cp -r "$(dirname "$cached")"/. "$BIN"/ && chmod +x "$BIN/llama-server"
        echo "found a saved build: $(dirname "$cached")"
    fi
fi
if [ -x "$BIN/llama-server" ] && LD_LIBRARY_PATH="$BIN:${LD_LIBRARY_PATH:-}" "$BIN/llama-server" --version > /dev/null 2>&1; then
    LD_LIBRARY_PATH="$BIN:${LD_LIBRARY_PATH:-}" "$BIN/llama-server" --version 2>&1 | head -1
    echo "ready - no build needed"
    exit 0
fi
rm -rf "$BIN"  # missing, or present but will not run here: build it

[ -d llama.cpp ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp
cd llama.cpp

started() { cat build.start 2>/dev/null || date +%s; }
minutes() { echo $(( ($(date +%s) - $(started)) / 60 )); }

watch_build() {
    while kill -0 "$(cat build.pid 2>/dev/null)" 2>/dev/null; do
        sleep "$POLL"
        pct=$(grep -oE '\\[ *[0-9]+%\\]|\\[[0-9]+/[0-9]+\\]' build.log 2>/dev/null | tail -1)
        file=$(grep -oE 'Building [A-Z]+ object [^ ]+' build.log 2>/dev/null | tail -1 | sed 's#.*/##')
        echo "  $(minutes) min  $pct  $file"
    done
}

# A build with no pid file was started by an older copy of this cell. Adopt it
# rather than deleting the directory it is still writing into.
if [ ! -s build.pid ]; then
    orphan=$(pgrep -of "cmake --build build" || true)
    if [ -n "$orphan" ]; then
        echo "$orphan" > build.pid
        [ -s build.start ] || date +%s > build.start
    fi
fi

if [ -x build/bin/llama-server ] && ! kill -0 "$(cat build.pid 2>/dev/null)" 2>/dev/null; then
    echo "build already finished"
elif kill -0 "$(cat build.pid 2>/dev/null)" 2>/dev/null; then
    echo "a build is already running (pid $(cat build.pid)); watching it"
    watch_build
else
    nvcc --version | tail -1
    command -v cmake > /dev/null || pip install -q cmake
    cmake --version | head -1

    # A configure left running by an older copy of this cell would race with this one.
    pkill -f "cmake -B build" 2> /dev/null || true
    rm -rf build build.pid

    # Configure in the background as well, printing its latest line every 15 s, so a
    # slow configure is visible instead of silent. The 45-minute limit only catches a
    # true hang: a configure on Kaggle has run silently for a long time and still
    # succeeded, and a tight limit would have killed it (D-030).
    # LLAMA_USE_PREBUILT_UI=OFF: skip downloading the server's web UI; KhataCall only
    #   uses the API, and llama.cpp builds fine without it.
    # LLAMA_BUILD_TESTS/EXAMPLES=OFF: less to configure; only llama-server is built.
    timeout "${CONFIGURE_TIMEOUT:-2700}" cmake -B build \\
        -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DGGML_CUDA_NO_VMM=ON \\
        -DBUILD_SHARED_LIBS=OFF -DLLAMA_USE_PREBUILT_UI=OFF \\
        -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \\
        -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release > cmake.log 2>&1 &
    cfg=$!
    cfg_start=$(date +%s)
    while kill -0 "$cfg" 2> /dev/null; do
        sleep "${POLL_CONFIGURE:-15}"
        echo "  configuring, $(( $(date +%s) - cfg_start )) s: $(tail -1 cmake.log 2> /dev/null | cut -c1-100)"
    done
    if ! wait "$cfg"; then
        echo "CONFIGURE FAILED (or hit the time limit). The end of cmake.log:"
        tail -40 cmake.log
        exit 1
    fi
    echo "configured in $(( $(date +%s) - cfg_start )) s"
    date +%s > build.start
    # setsid puts the build in its own session, out of reach of the cell's interrupt.
    # The child records its own pid, then becomes cmake via exec, keeping that pid.
    setsid bash -c 'echo $$ > build.pid; exec cmake --build build --target llama-server -j "$(nproc)"' \\
        > build.log 2>&1 < /dev/null &
    for _ in $(seq 1 20); do [ -s build.pid ] && break; sleep 0.5; done
    echo "build started (pid $(cat build.pid)), first progress line in a minute"
    watch_build
fi

if [ ! -x build/bin/llama-server ]; then
    echo "BUILD FAILED"
    grep -E "error" build.log | head -20
    tail -20 build.log
    exit 1
fi
echo "built in $(minutes) min: $PWD/build/bin/llama-server"

# Keep it for later sessions: the binary, plus whichever of llama.cpp's own shared
# libraries it loads (a build from an older copy of this cell has some). ldd lists
# exactly those, so nothing is missed and nothing extra is copied. Then one zip.
# Saving is a convenience: if any of it fails, the build above still stands.
set +e
rm -rf "$SAVE" && mkdir -p "$SAVE" && cp build/bin/llama-server "$SAVE/"
ldd build/bin/llama-server 2>/dev/null | awk -v dir="$PWD/build/" 'index($3, dir) == 1 {print $1, $3}' |
    while read -r name path; do cp -L "$path" "$SAVE/$name"; done
if ( cd "$(dirname "$SAVE")" && python3 -m zipfile -c "$(basename "$SAVE").zip" "$(basename "$SAVE")" ); then
    echo "saved for reuse: $(ls "$SAVE" | tr '\\n' ' ')"
    echo "  zipped as $SAVE.zip ($(du -m "$SAVE.zip" | cut -f1) MB) - see the note below to keep it"
else
    echo "(could not zip the saved copy; the folder $SAVE is still there to download)"
fi
exit 0
"""
    ),
    md(
        """
**Keep the build for next time (once, optional, saves 30-60 minutes per session).**
Step 4 ends by writing `/kaggle/working/llama-bin.zip`: the llama-server binary
plus any shared libraries it needs. Download it from the **Output** section of
the right-hand panel. Then on kaggle.com: **Create, New Dataset**, upload the
zip (Kaggle unpacks it), keep it private. In every later session, before Run All,
use **Add Input** to attach that dataset; step 4 finds it under `/kaggle/input`,
test-runs it, and skips the build. If a Kaggle image update ever stops it
running, step 4 notices and builds again.
"""
    ),
    code(
        """
# 5. Wait for the download to finish.
def downloaded_gb():
    # The file only appears at MODEL_PATH when complete; until then the bytes sit
    # in a *.incomplete file under WORK/.cache, so count those too.
    total = 0
    for root, _, files in os.walk(WORK):
        if "llama.cpp" in root:
            continue
        total += sum(os.path.getsize(os.path.join(root, f))
                     for f in files if f.endswith((".gguf", ".incomplete")))
    return total / 1e9

while download.poll() is None:
    print(f"\\r  {downloaded_gb():5.1f} of ~{SIZE_GB} GB", end="", flush=True)
    time.sleep(10)
print()
assert download.returncode == 0, open(f"{WORK}/download.log").read()[-2000:]
print("downloaded", MODEL_PATH, f"{os.path.getsize(MODEL_PATH) / 1e9:.1f} GB")
"""
    ),
    code(
        """
# 6. Start llama-server across both GPUs. Loading takes a few minutes; this prints
#    the server's latest log line every 15 seconds while it does.
#   -ngl 99    every layer on the GPUs (split across both by default)
#   --jinja    use the model's own chat template; REQUIRED for tool calls
#   --api-key  the key from step 2
# A saved build (step 4, from Add Input) if there is one, else this session's build.
# LD_LIBRARY_PATH lets a saved binary find the shared libraries saved beside it.
SAVED = f"{WORK}/llama-bin"
BIN = next(p for p in (f"{SAVED}/llama-server", f"{WORK}/llama.cpp/build/bin/llama-server")
           if os.path.exists(p))
server_env = {**os.environ, "LD_LIBRARY_PATH": f"{SAVED}:{os.environ.get('LD_LIBRARY_PATH', '')}"}
print("running", BIN)
server = subprocess.Popen(
    [BIN,
     "-m", MODEL_PATH, "--alias", ALIAS,
     "--host", "127.0.0.1", "--port", str(PORT),
     "-ngl", "99", "-c", str(CTX), "--parallel", "1",
     "--jinja", "--api-key", API_KEY],
    env=server_env, stdout=open(f"{WORK}/server.log", "w"), stderr=subprocess.STDOUT,
)

def healthy():
    try:
        # /health answers 503 while the model loads, and 200 once it is ready.
        return urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5).status == 200
    except Exception:
        return False

start, shown = time.time(), 0.0
while time.time() - start < 20 * 60 and server.poll() is None and not healthy():
    if time.time() - shown >= 15:
        lines = open(f"{WORK}/server.log").read().strip().splitlines()
        print(f"  {time.time() - start:4.0f} s  {lines[-1][:110] if lines else '(starting)'}", flush=True)
        shown = time.time()
    time.sleep(5)

log = open(f"{WORK}/server.log").read()
if not healthy():
    print(log[-3000:])
    if "out of memory" in log.lower():
        raise SystemExit("Out of GPU memory. Set QUANT = 'IQ4_XS' in step 2 and re-run from step 2.")
    raise SystemExit("llama-server is not answering. The log above says why.")
print(f"llama-server is up after {time.time() - start:.0f} s")
print(subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv"],
                     capture_output=True, text=True).stdout)
"""
    ),
    code(
        """
# 7. One tool call, locally, before exposing anything.
body = {
    "model": ALIAS,
    "messages": [
        {"role": "system", "content": "Tum ek kirana dukaan ke assistant ho. Amount hamesha shabdon mein do."},
        {"role": "user", "content": "Sharma ji ka dedh sau udhaar likho"},
    ],
    "tools": [{"type": "function", "function": {
        "name": "add_credit", "description": "Record credit against a customer",
        "parameters": {"type": "object", "properties": {
            "customer_name": {"type": "string"}, "amount_spoken": {"type": "string"}},
            "required": ["customer_name", "amount_spoken"]}}}],
    "chat_template_kwargs": {"enable_thinking": False},
    "max_tokens": 200, "temperature": 0.2,
}
request = urllib.request.Request(
    f"http://127.0.0.1:{PORT}/v1/chat/completions", data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"})
start = time.time()
reply = json.load(urllib.request.urlopen(request, timeout=180))
message = reply["choices"][0]["message"]
print(f"{time.time() - start:.1f} s")
print("tool_calls:", json.dumps(message.get("tool_calls"), ensure_ascii=False, indent=1))
print("content:", repr(message.get("content")))
# Good: one add_credit call with amount_spoken "dedh sau", and empty content.
# No tool_calls but JSON in content: run evon_check with --tool-mode prompt.
"""
    ),
    code(
        """
# 8. Expose it through a Cloudflare quick tunnel and print the .env lines.
if not os.path.exists(f"{WORK}/cloudflared"):
    urllib.request.urlretrieve(
        "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
        f"{WORK}/cloudflared")
    os.chmod(f"{WORK}/cloudflared", 0o755)

tunnel = subprocess.Popen(
    [f"{WORK}/cloudflared", "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{PORT}"],
    stdout=open(f"{WORK}/tunnel.log", "w"), stderr=subprocess.STDOUT)

url = None
for _ in range(60):
    found = re.search(r"https://[a-z0-9-]+\\.trycloudflare\\.com", open(f"{WORK}/tunnel.log").read())
    if found:
        url = found.group(0)
        break
    time.sleep(2)
assert url, open(f"{WORK}/tunnel.log").read()[-2000:]

print("Paste these into your local .env. They change every session.\\n")
print(f"EVON_BASE_URL={url}/v1")
print(f"EVON_API_KEY={API_KEY}")
print(f"EVON_MODEL={ALIAS}")
print("\\nThen run:  uv run python -m scripts.evon_check")
"""
    ),
    code(
        """
# 9. Keep this cell running while you use Evon. It keeps the session active and
# shows the server is alive. Interrupt it, then Stop Session, when you are done:
# the free quota is 30 GPU-hours a week and an idle server still counts.
while True:
    try:
        ok = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5).status == 200
    except Exception:
        ok = False
    last = open(f"{WORK}/server.log").read().strip().splitlines()[-1:] or [""]
    print(time.strftime("%H:%M:%S"), "healthy" if ok else "DOWN", "|", last[0][:100], flush=True)
    time.sleep(60)
"""
    ),
]


def check_bash(index: int, source: str) -> None:
    """`bash -n` the body of a %%bash cell, if bash is available here."""
    bash = shutil.which("bash")
    if bash is None:
        print(f"  (no bash on this machine; cell {index} not syntax-checked)")
        return
    body = source.split("\n", 1)[1] if "\n" in source else ""
    result = subprocess.run([bash, "-n"], input=body, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"cell {index} is not valid bash:\n{result.stderr}")


def main() -> None:
    for i, cell in enumerate(CELLS):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if source.startswith("%%bash"):
            # A cell magic must be the first line of its cell, or Jupyter treats
            # the whole cell as Python and fails on the first shell line.
            check_bash(i, source)
            continue
        if "%%" in source:
            raise SystemExit(f"cell {i}: a cell magic must be on the first line")
        try:
            ast.parse(source)
        except SyntaxError as e:
            raise SystemExit(f"cell {i} does not parse: {e}") from e

    notebook = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    code_cells = sum(c["cell_type"] == "code" for c in CELLS)
    print(f"wrote {OUT.name}: {len(CELLS)} cells ({code_cells} code), all Python cells parse")


if __name__ == "__main__":
    main()
