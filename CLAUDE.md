# CLAUDE.md — KhataCall

Voice agent that lets a kirana shop owner manage credit (udhaar) and stock by
speaking Hinglish over a phone-quality line. Built for Gnani AI's Great Indian
AI Internship Challenge, plus an evaluation harness that measures how the
pipeline holds up on degraded telephone audio.

**Hard deadline: Nov 10, 2026. Internal submission target: Fri Nov 6**
(Diwali is Sun Nov 8). Feature freeze: end of Sun Nov 1.

---

## 1. Scope (read before adding anything)

We are building **production-shaped, not production-scaled** software.
Judged on: a 60-second demo, a written explanation, and individual interviews.

**In scope**
- One FastAPI service, started with one command.
- Streaming audio over WebSocket between browser and server.
- Pipeline: Gnani Prisma (STT) → Gnani Evon (LLM, tool calls) → Gnani Timbre (TTS).
- Ledger + stock in SQLite.
- Telephone-channel degradation module, shared by the eval harness and a
  "phone line mode" toggle in the demo UI.
- Eval harness producing error-rate, task-accuracy, and latency results + charts.

**Out of scope — do not build, even if it seems quick**
Auth, multi-tenancy, Postgres, queues, Docker/K8s orchestration, horizontal
scaling, real phone numbers (Exotel/Twilio), admin dashboards, barge-in.
These go in the README's "Path to production" section as prose only.
If a task seems to need one of these, stop and ask.

---

## 2. Hard rules

1. **Gnani models only in the product path.** STT = Prisma, TTS = Timbre,
   LLM = Evon. Never substitute another provider's model in `app/` without
   asking first. Fakes are fine in tests.
2. **Never guess Gnani API details.** Endpoints, auth headers, audio formats,
   sample rates, streaming protocol, confidence fields: read
   https://docs.gnani.ai (or ask). Anything unverified gets a `# VERIFY:` comment.
   All Gnani-specific code lives in `app/adapters/` only.
3. **Credits are finite.** Any code path that calls Gnani APIs in bulk must
   support `--dry-run` (prints planned call count), `--max-calls N`, and must
   cache results on disk keyed by (audio sha256, condition id, model id).
   Never re-spend credits on a cached result.
4. **Synthetic data only.** Made-up customer names, no real phone numbers,
   Aadhaar, PAN, account numbers, or recorded real calls — in code, fixtures,
   corpus, logs, or demo. (Contest rule.)
5. **The LLM never writes to the database.** Evon proposes tool calls; the
   executor validates them and every write passes the confirmation gate (§5).
6. **Money is integer paise.** No floats for amounts anywhere.
7. **Amounts are parsed by code, not by the LLM.** Tool args carry the amount
   as spoken (`amount_spoken`); `nlu/numbers.py` converts it deterministically.
8. **Secrets** live in `.env` (gitignored). Keep `.env.example` current.
9. Tests must never hit the network. Use `adapters/fakes.py`.
10. Log every non-trivial design decision with its reason (and numbers, if any)
    in `DECISIONS.md` — this is interview prep.

---

## 3. Stack

- Python 3.11+, managed with `uv`
- FastAPI + uvicorn, native WebSockets
- Pydantic v2 for tool arguments and config
- SQLite via stdlib `sqlite3` with a thin repository layer
- numpy, scipy (`signal.resample_poly`, filters), soundfile
- rapidfuzz (customer name matching), jiwer (WER/CER)
- matplotlib (eval charts)
- pytest
- Frontend: one static `index.html`, vanilla JS, Web Audio API. No build step.

## 4. Commands

```bash
uv sync                                         # install
uv run uvicorn app.main:app --reload            # run server → http://localhost:8000
uv run pytest -q                                # tests (offline)
uv run python -m scripts.smoke both             # 2 real API calls: Timbre TTS, then Prisma STT on its output
uv run python -m scripts.latency --dry-run      # count planned calls
uv run python -m scripts.latency --n 5          # measure warm Gnani latency (10 real calls)
uv run python -m eval.run_eval --dry-run        # count planned API calls
uv run python -m eval.run_eval --max-calls 200  # run eval with a hard cap
uv run python -m eval.report                    # build charts from results
```
Keep this section accurate when commands change.

## 5. Architecture

```
khatacall/
  CLAUDE.md  README.md  DECISIONS.md  pyproject.toml  .env.example
  app/
    main.py              # FastAPI app, static files, /ws endpoint
    config.py            # settings from env
    protocol.py          # WebSocket message schemas
    pipeline/
      session.py         # per-call state machine
    adapters/
      base.py            # STT / LLM / TTS Protocols
      gnani_stt.py       # Prisma
      gnani_tts.py       # Timbre
      evon_llm.py        # Evon (hosted or self-hosted, see §7)
      fakes.py           # deterministic fakes
    agent/
      prompts.py         # system prompt (Hinglish shop assistant)
      tools.py           # tool schemas + Pydantic arg models
      executor.py        # validation, name resolution, confirmation gate
    nlu/
      numbers.py         # Hinglish/Hindi/English spoken numbers → paise
      names.py           # honorific stripping + fuzzy match
    ledger/
      schema.sql  db.py  repo.py
    channel/
      degrade.py         # telephone channel simulation (shared)
    static/index.html
  eval/
    corpus/scripts.csv   # ground-truth labels (committed)
    corpus/audio/        # recordings (gitignored if large)
    noise/               # self-recorded background noise
    conditions.py  normalize.py  metrics.py  run_eval.py  report.py
    cache/  results/
  tests/
```

### Session state machine (half-duplex, push-to-talk)
`IDLE → LISTENING → TRANSCRIBING → THINKING → [CONFIRMING] → SPEAKING → LISTENING`
Push-to-talk first (reliable for the demo). STT runs on the buffered utterance
via Prisma REST on release (DECISIONS.md D-003). Prisma's realtime API does
report end of speech (a `processing` event from its VAD) but requires
real-time frame cadence; switching to it is an M4 latency experiment.

### WebSocket protocol (`app/protocol.py`)
- Client → server: binary PCM16 mono frames (16 kHz unless docs say otherwise);
  JSON `{type: "start" | "stop" | "config", phone_mode?, snr_db?, loss_pct?}`
- Server → client: JSON `{type: "state" | "transcript" | "agent_text" |
  "ledger_update" | "latency", ...}` and binary TTS audio frames
  (format per Timbre docs).

### Agent tools (`agent/tools.py`)
| Tool | Args | Writes? |
|---|---|---|
| `add_credit` | customer_name, amount_spoken, note? | yes |
| `record_payment` | customer_name, amount_spoken | yes |
| `get_balance` | customer_name | no |
| `update_stock` | item, qty_delta_spoken | yes |
| `get_stock` | item | no |
| `list_low_stock` | threshold? | no |

### Confirmation gate (`agent/executor.py`)
1. Validate args (Pydantic) → parse amount via `nlu/numbers.py` → resolve the name.
2. Write tools create a *pending action*; the agent reads it back
   ("Sharma ji ke khate mein ₹350 udhaar likh doon?").
3. Yes → commit. No → cancel. Unclear → re-ask once, then cancel.
4. Unknown customer (match score below threshold) → ask
   "Naya customer banaun?" before creating one.
5. Prisma exposes **no confidence score** (neither REST nor realtime). Confirm
   the amount digit by digit when phone mode is on, or when `numbers.py` used
   a fallback parse path (DECISIONS.md D-004).

### Number parsing (`nlu/numbers.py`) — must be heavily unit-tested
Hindi 1–99 are irregular → lookup table. Multipliers: sau, hazaar, lakh,
crore. Fractions: sava (+¼), sadhe (+½), paune (−¼), dedh (1.5), dhai (2.5).
**Romanized spelling is not stable** — Prisma returned `rupay` for "rupaye",
`udhar` for "udhaar", `du` for "doon" (D-012). Match on a normalized key
(collapse doubled vowels, trailing `h`), never on exact strings, and list
variants in every lookup table.
Also handle English words, digits, mixed forms ("2 hazaar 500", "dedh sau"),
"rupaye/rupees/rs", and ambiguity → return `None` (agent re-asks), never guess.

### Channel degradation (`channel/degrade.py`)
Order matters — noise happens in the shop, before the phone network:
1. Mix background noise at target SNR (computed on active-speech power)
2. Band-limit to 300–3400 Hz, resample to 8 kHz (`resample_poly`)
3. G.711 μ-law encode/decode (implement in numpy; `audioop` is gone in 3.13)
4. Packet loss on 20 ms frames (160 samples), Gilbert-Elliott bursty model;
   concealment configurable: zero-fill or repeat-last-frame
No final resampling stage: Prisma accepts 8 kHz natively (realtime
`x-sample-rate: 8000`; REST converts to 16 kHz server-side). DECISIONS.md D-007.

All randomness seeded. Each stage is a pure function, testable on its own.

### Eval harness (`eval/`)
- `scripts.csv` columns: `id, text, intent, customer, amount_paise, item, qty, speaker`
- Condition grid (~10): clean; tel-only; tel + loss {1, 5, 10}%;
  tel + noise SNR {20, 10, 5, 0} dB; worst case (SNR 5 + 5% loss)
- Metrics:
  - CER/WER after normalization (`normalize.py`: lowercase, strip punctuation,
    numerals → digits, Devanagari ↔ romanized unification). Document its limits.
  - **Task accuracy (headline metric):** intent, customer, and amount all correct
  - Latency: end of speech → first TTS byte, p50/p90, broken down by stage
- Output: `results/*.json` + charts (task accuracy vs SNR, vs loss; before vs after fixes)

---

## 6. Milestones

Mark tasks `[x]` as they are completed. `[human]` = Diptanu does this, not Claude Code.

### M0 — Unblock (Tue Oct 6 – Wed Oct 7)
- [x] [human] Register for the contest; read the IP/ownership terms before accepting
- [ ] [human] Request Evon v3.3 weights on Hugging Face
- [ ] [human] Ask on Gnani's Discord: is a hosted Evon endpoint available to participants? Does it support tool/function calling?
- [ ] [human] Find out how credits are counted (per call / per audio-second)
- [x] Scaffold the repo, pyproject, `.env.example`, adapter Protocols, fakes
- [x] Smoke scripts: one Prisma transcription, one Timbre synthesis (verified against docs)

**Done when:** a real Prisma call and a real Timbre call succeed from the adapters.
✅ Both succeeded Oct 7 (`uv run python -m scripts.smoke both`). Findings: D-012
(romanized output, variable spelling), D-013 (cold starts), D-014 (error shapes).

### M1 — End-to-end loop (Thu Oct 8 – Tue Oct 13)
- [x] **Decision by Fri Oct 9:** Evon hosted vs self-hosted → **Kaggle, self-hosted** (D-001)
- [ ] Evon adapter (see §7)
- [ ] `/ws` endpoint + session state machine (no tools yet; plain replies)
- [ ] `index.html`: push-to-talk, transcript, agent text, audio playback
- [ ] Per-stage latency timing, sent to the client
- [ ] [human] Write ~100 synthetic utterances into `scripts.csv`; start recording (self + 1–2 friends)

**Done when:** speaking a Hinglish sentence produces a spoken Hinglish reply in the browser.

### M2 — Useful agent (Wed Oct 14 – Tue Oct 20)
- [ ] Ledger schema + repo (customers, transactions, stock)
- [ ] `nlu/numbers.py` + tests (aim for 50+ cases)
- [ ] `nlu/names.py` + tests (honorifics: ji, bhaiya, didi, aunty, uncle)
- [ ] Tools, executor, confirmation gate
- [ ] Live ledger panel in the UI (`ledger_update` messages)
- [ ] Seed script with synthetic customers/items
- [ ] [human] Build-in-public post #1 (LinkedIn/X with contest hashtags)

**Done when:** "Sharma ji ka dedh sau udhaar likho" → read-back → "haan" → the ledger updates live.

### M3 — Channel + eval baseline (Wed Oct 21 – Tue Oct 27)
- [ ] `channel/degrade.py` + tests (SNR correctness, μ-law round trip, loss rate statistics)
- [ ] Eval runner with cache, `--dry-run`, `--max-calls`
- [ ] STT-level metrics across the condition grid
- [ ] Task-level eval: run transcripts through the agent with the fake ledger, score task accuracy
- [ ] Phone-line mode toggle in the UI (SNR + loss sliders) using the same `degrade.py`
- [ ] Baseline charts committed to `eval/results/`

**Done when:** baseline numbers exist for every condition, with the API call count known.

### M4 — Fix what eval exposed (Wed Oct 28 – Sun Nov 1) → FEATURE FREEZE
- [ ] Rank failure modes from the baseline (record in DECISIONS.md)
- [ ] Fixes, likely: re-ask on low confidence, digit-by-digit amount confirmation,
      numbers parser gaps, name matching thresholds
- [ ] Latency: streaming STT and TTS where supported; target < 1.5 s end of speech → first audio
- [ ] Re-run eval → before/after charts

**Done when:** a before/after task-accuracy chart shows improvement on the degraded conditions.

### M5 — Package and ship (Mon Nov 2 – Fri Nov 6)
- [ ] README: what it is, architecture diagram, which Gnani model is used where,
      setup in < 5 minutes, eval method + results, synthetic-data note,
      known limitations, **Path to production** (concurrent calls, telephony
      integration, per-shop tenancy on Postgres, latency budgets)
- [ ] Clean demo script + seeded demo database
- [ ] [human] Record the 60-second demo video (noisy-line credit entry → confirm → ledger updates → eval chart)
- [ ] [human] Main LinkedIn/X post by **Nov 3–4**; nominate it along with post #1
- [ ] [human] Submit by **Fri Nov 6**: demo, written explanation, models used, post links

Nov 7–10 is buffer only. No new features.

**If time slips:** cut M4 fixes first. Never cut the eval baseline or M5 packaging.

---

## 7. Gnani integration

Verified API facts (base URL, auth, endpoints, audio formats) are in
`docs/HLD.md` §7, with links to the doc pages. Re-check the docs before
changing adapter behaviour. Key ones:
- Base URL `https://api.vachana.ai`, header `X-API-Key-ID`.
- Hinglish STT uses `language_code` / `lang_code` = **`en-IN`** (docs: code-mixed
  English-Hindi is supported via en-IN). Timbre has a `hi-en` language option.
- Timbre returns audio as base64 chunks inside JSON, not raw binary.

### Evon

Evon v3.3 is not on Gnani's API platform: open weights on Hugging Face (by
request), 30B MoE (~3.5B active), `nemotron_h` architecture. Hosting is
decision D-001. Build `evon_llm.py` against an
OpenAI-compatible chat-completions interface; that works for both paths:
- **Hosted:** use Gnani's endpoint as documented (`# VERIFY:` everything).
- **Self-hosted:** 4-bit quantized weights served with vLLM (or similar) on a
  rented/free GPU, exposing an OpenAI-compatible server.

If Evon lacks native tool calling: put the tool JSON schemas in the system
prompt, require a single JSON object response, parse it with Pydantic, and
re-prompt once on a parse failure. Keep this behind the same adapter interface.

## 8. Conventions

- Type hints everywhere; small pure functions in `nlu/` and `channel/`
- Tests live next to the behaviour they cover in `tests/`; name them for the behaviour
- Small commits with clear messages; one milestone task per commit where possible
- Prefer the simplest thing that works for the demo; if unsure whether
  something is in scope, check §1, then ask
