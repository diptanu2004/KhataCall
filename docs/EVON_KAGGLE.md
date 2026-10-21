# Running Evon on Kaggle

Evon v3.3 is not on Gnani's API platform, so KhataCall serves it itself on
Kaggle's free GPUs (DECISIONS.md D-001, D-027, D-030). **The first session takes
45-75 minutes**, nearly all of it compiling llama.cpp. Save the compiled binary
once (step 4's note) and later sessions take about 10 minutes.

## What you need

- A Kaggle account with a **verified phone number**. Without it, notebooks
  cannot reach the internet, and both the model download and the tunnel need it.
- Nothing from Hugging Face. The weights are **not gated** (`gated: False` on
  the Hub API, ~35K downloads), despite the challenge page saying to request
  them. No token, no approval.

## Steps

**1. Generate the notebook** (already committed; regenerate if you change it):

```bash
uv run python deploy/make_kaggle_notebook.py
```

**2. Import it into Kaggle.** kaggle.com → Create → New Notebook → File →
Import Notebook → upload `deploy/kaggle_evon.ipynb`.

**3. Set the session up.** In the right-hand panel, under Session options:
- **Accelerator: GPU T4 x2.** Not P100: one 16 GB card cannot hold the model.
- **Internet: On.**

**4. Run All.** Then wait. Roughly:

| Step | Time | What you see while it works |
|---|---|---|
| 3 | starts at once | the 24.7 GB download begins in the background |
| 4 | **30-60 min** first time, seconds with a saved binary | first `configuring, 45 s: <cmake's latest line>` every 15 s (1-3 min usually, much longer seen; stopped only at 45), then `12 min  [ 34%]  mmq-instance-q4_k.cu.o` every minute |
| 5 | usually done by then | `18.3 of ~24.7 GB`, updating every 10 s |
| 6 | a few minutes | `45 s  <llama-server log line>`, every 15 s |
| 7 | seconds to a minute | one test tool call, inside Kaggle |
| 8 | seconds | opens the tunnel and prints your `.env` lines |

**5. Read step 7's output before anything else.** You want one `add_credit`
tool call with `amount_spoken: "dedh sau"` and empty content. That single line
answers whether Evon does native tool calling under llama.cpp.

**6. Copy step 8's three lines into your local `.env`:**

```
EVON_BASE_URL=https://something-random.trycloudflare.com/v1
EVON_API_KEY=...
EVON_MODEL=gnani-evon-v3.3
```

**7. Leave step 9 running.** It keeps the session alive and prints a heartbeat.

**8. On your laptop, check it:**

```bash
uv run python -m scripts.evon_check            # 12 corpus sentences, all 6 intents
uv run python -m scripts.evon_check --n 180    # the whole corpus, ~5-10 min
```

**9. Use it:**

```bash
uv run uvicorn app.main:app      # real Prisma, real Evon, real Timbre
```

**10. Stop when done:** interrupt step 9, then Stop Session. The free quota is
30 GPU-hours a week, and an idle server still uses it.

## Does it look stuck?

Every waiting step prints at least once a minute. **A step is only stuck if its
output stops changing for several minutes.** A step that is merely slow:

- **Step 4, `configuring` for a long time.** Slow but possible: one Kaggle
  configure ran silently for a long stretch and then succeeded. The lines show
  what it is doing; it only gives up at 45 minutes, printing the end of
  `cmake.log`.
- **Step 4, percentage not moving.** Normal. llama.cpp's CUDA backend is 191
  files, and some single files take several minutes in `nvcc`; the percentage
  only moves when a file finishes. The minute counter is what matters: if it
  keeps rising, the build is alive.
- **Step 4 is safe to interrupt.** The build runs detached. Interrupting the
  cell stops the *watching*, not the build, and running the cell again resumes
  watching it. Interrupt it if you need to run something else in the meantime.
- **Steps 5 and 6 are safe to interrupt too:** the download and the server are
  background processes. Re-run the cell to resume waiting.

To look underneath any of them, interrupt the cell and run:

```python
print(open(f"{WORK}/download.log").read()[-800:])            # step 3/5
print(open(f"{WORK}/llama.cpp/build.log").read()[-1500:])    # step 4
print(open(f"{WORK}/server.log").read()[-1500:])             # step 6
```

## Keeping the build for next time

Step 4 ends by writing `/kaggle/working/llama-bin.zip`: the binary plus any
shared libraries it needs (found with `ldd`, so exactly those). Download it from
the notebook's Output panel and upload it as a **private Kaggle dataset**
(Create, New Dataset; Kaggle unpacks the zip). In later
sessions, attach it with **Add Input** before Run All; step 4 finds it under
`/kaggle/input`, test-runs it, and skips the build. If a Kaggle image update ever
stops it running, step 4 notices and builds again.

## Reading `evon_check`

It sends corpus sentences as **text**, as if transcription were perfect, and
scores what Evon proposes with the same code the executor uses. So its task
accuracy is the ceiling for the whole system; the audio eval can only come in
lower, and the gap is what the phone line costs.

| Line | Good | If not |
|---|---|---|
| chat template mentions tools | yes | llama-server is missing `--jinja` |
| thinking visible in the reply | no | see troubleshooting below |
| task accuracy | high | look at the per-row reasons |
| WRONG | as low as possible | **the important one**: confident wrong values |
| digits converted | 0 | the prompt is not holding hard rule 7 |
| latency p50 | under ~2 s | the budget assumed 0.8 s; see below |

Results also land in `scripts/out/evon_check.json`.

## Troubleshooting

**Step 4 fails with `CUDA::cuda_driver ... target was not found`.** Your copy of
the notebook predates the fix: add `-DGGML_CUDA_NO_VMM=ON` to the `cmake -B build`
line, or re-import the current `deploy/kaggle_evon.ipynb`. Kaggle's CUDA toolkit
has no `libcuda.so` for CMake to link, and only the VMM memory pool needs it
(D-030).

**Step 6 runs out of GPU memory.** Set `QUANT = "IQ4_XS"` in step 2 (18.4 GB
instead of 24.7) and re-run from step 2. Also, the build directory survives
within a session, so a rerun skips most of step 4.

**Step 7 returns text, not a tool call.** Either the template is not being used
(check `--jinja` is in step 6) or the model writes calls in a format
llama.cpp does not parse. Try the prompt-based fallback, which needs no
server support:

```bash
uv run python -m scripts.evon_check --tool-mode prompt
```

If that scores well, set the adapter to prompt mode. It is the documented
fallback in CLAUDE.md §7.

**`<think>` shows up in replies.** KhataCall already sends
`chat_template_kwargs: {"enable_thinking": false}` with every request. If the
template ignores it, add `"--reasoning-budget", "0"` to the server arguments in
step 6.

**`evon_check` says NOT REACHABLE.** The tunnel URL changes every session, so
`.env` is stale, or the session stopped. Check step 9 is still printing.

**401 from evon_check.** `EVON_API_KEY` does not match the one step 8 printed.

**Slow.** The tunnel adds a round trip to Kaggle's region, and T4s are old. The
first request of a session is slower again while the prompt is cached. Measure
before deciding; `evon_check` reports p50 and p90.

## Security

The tunnel URL is public; the API key is what keeps strangers off your GPUs.
Do not commit `.env`, and do not paste the URL and key together anywhere public.
