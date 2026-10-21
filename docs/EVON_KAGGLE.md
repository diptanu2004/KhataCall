# Running Evon on Kaggle

Evon v3.3 is not on Gnani's API platform, so KhataCall serves it itself on
Kaggle's free GPUs (DECISIONS.md D-001, D-027). This takes about 30 minutes the
first time, and about 10 for each later session.

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

| Step | Time | What happens |
|---|---|---|
| 3 | starts at once | 24.7 GB download begins in the background |
| 4 | ~15 min | llama.cpp builds with CUDA for the T4 (compute 7.5) |
| 5 | usually done by then | waits for the download |
| 6 | ~2 min | loads the model across both GPUs |
| 7 | seconds | one test tool call, inside Kaggle |
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
