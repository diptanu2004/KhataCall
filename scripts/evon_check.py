"""Does Evon work, and how well? One command, against whatever EVON_BASE_URL points at.

    uv run python -m scripts.evon_check              # probe + 12 corpus sentences
    uv run python -m scripts.evon_check --n 180      # the whole corpus, as text
    uv run python -m scripts.evon_check --tool-mode prompt

It sends corpus sentences *as text* - a perfect transcription - and scores the
tool call Evon proposes with the same functions the executor uses
(eval/metrics.py). That makes this the ceiling for the whole pipeline: audio
can only make it worse. The gap between this and the audio eval is what the
telephone channel costs.

What it checks, in order, and why:
  reachable        the Kaggle tunnel is up and the key is right
  tool template    llama-server only parses tool calls with --jinja and a
                   template that supports them; without it every reply is text
  thinking off     Evon is a reasoning model, and visible <think> blocks would
                   be spoken aloud and wreck latency (D-001, chat_template_kwargs)
  task accuracy    intent + customer + amount, split into correct / caught /
                   wrong (see eval/metrics.py for why the split matters)
  digit conversion the model turning "dedh sau" into "150", which hard rule 7
                   forbids even when the digits happen to be right
  latency          per call, p50/p90 - Evon is the one unmeasured term in the
                   latency budget (docs/HLD.md §12, risk R7)

Evon is self-hosted, so this spends no Gnani credits.
"""

import argparse
import asyncio
import csv
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import httpx

from app.adapters.base import Message
from app.adapters.evon_llm import EvonError, EvonLLM
from app.agent.prompts import SYSTEM_PROMPT
from app.agent.tools import TOOL_SCHEMAS
from app.config import get_settings
from eval.metrics import Outcome, score_call

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "eval" / "corpus" / "scripts.csv"
OUT = ROOT / "scripts" / "out" / "evon_check.json"


def sample(rows: list[dict], n: int) -> list[dict]:
    """Round-robin across intents, so a small sample still covers all six."""
    by_intent: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_intent[r["intent"]].append(r)
    picked: list[dict] = []
    while len(picked) < n and any(by_intent.values()):
        for intent in list(by_intent):
            if by_intent[intent] and len(picked) < n:
                picked.append(by_intent[intent].pop(0))
    return sorted(picked, key=lambda r: r["id"])


async def probe(base_url: str, api_key: str) -> dict:
    """Is the server up, and will it parse tool calls?"""
    root = base_url.rstrip("/").removesuffix("/v1")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    found: dict = {"base_url": base_url}
    async with httpx.AsyncClient(timeout=20, headers=headers) as client:
        try:
            health = await client.get(f"{root}/health")
            found["health"] = health.status_code
        except httpx.HTTPError as e:
            found["error"] = f"cannot reach {root}: {e}"
            return found
        try:
            models = (await client.get(f"{base_url.rstrip('/')}/models")).json()
            found["models"] = [m.get("id") for m in models.get("data", [])]
        except (httpx.HTTPError, ValueError):
            pass
        try:
            # llama-server only. Tool calls need a template that knows about tools.
            props = (await client.get(f"{root}/props")).json()
            template = props.get("chat_template") or ""
            found["template_supports_tools"] = "tool" in template.lower()
            found["template_reads_enable_thinking"] = "enable_thinking" in template
        except (httpx.HTTPError, ValueError):
            pass
    return found


def describe(tool: str | None, args: dict | None, text: str | None) -> str:
    if tool is None:
        return f"text: {(text or '').strip()[:50]!r}"
    shown = ", ".join(str(v) for v in (args or {}).values())
    return f"{tool}({shown})"


async def run(args) -> int:
    settings = get_settings()
    base_url = args.base_url or settings.evon_base_url
    api_key = settings.evon_api_key.get_secret_value()
    model = args.model or settings.evon_model

    print(f"Evon at {base_url}  (model {model}, tool mode {args.tool_mode})\n")
    found = await probe(base_url, api_key)
    if "error" in found:
        print(f"  NOT REACHABLE: {found['error']}")
        print("  Is the Kaggle notebook running, and are EVON_BASE_URL / EVON_API_KEY in .env current?")
        print("  The tunnel URL changes every session. See docs/EVON_KAGGLE.md.")
        return 2
    print(f"  health {found.get('health')}   models {found.get('models')}")
    if "template_supports_tools" in found:
        tools_ok = found["template_supports_tools"]
        print(f"  chat template mentions tools: {'yes' if tools_ok else 'NO - start llama-server with --jinja'}")
        print(f"  template reads enable_thinking: {'yes' if found['template_reads_enable_thinking'] else 'no'}")

    llm = EvonLLM(base_url=base_url, api_key=api_key, model=model, tool_mode=args.tool_mode, timeout_s=120)
    system = Message("system", SYSTEM_PROMPT)

    # A plain turn first: does it answer, in Hinglish, without visible thinking?
    start = time.perf_counter()
    try:
        reply = await llm.complete([system, Message("user", "Namaste, aap kaun ho?")])
    except EvonError as e:
        print(f"\n  FAILED on a plain message: {e}")
        if e.status == 401:
            print("  401 means the API key is wrong. Copy EVON_API_KEY from the notebook output.")
        await llm.aclose()
        return 1
    plain_ms = (time.perf_counter() - start) * 1000
    plain_text = reply.text or ""
    leak = "<think>" in plain_text or "</think>" in plain_text
    print(f"\n  plain reply {plain_ms:.0f} ms: {plain_text.strip()[:80]!r}")
    print(f"  thinking visible in the reply: {'YES - check chat_template_kwargs' if leak else 'no'}")

    rows = list(csv.DictReader(CORPUS.open(encoding="utf-8")))
    known = sorted({r["customer"] for r in rows if r["customer"]})
    picked = rows if args.n >= len(rows) else sample(rows, args.n)

    print(f"\n  {'id':>4}  {'intent':15} {'outcome':8} {'ms':>6}  proposal")
    results, latencies = [], []
    for r in picked:
        start = time.perf_counter()
        try:
            reply = await llm.complete([system, Message("user", r["text"])], tools=TOOL_SCHEMAS)
        except EvonError as e:
            print(f"  {r['id']:>4}  {r['intent']:15} ERROR    {e}")
            results.append({"id": r["id"], "error": str(e)})
            continue
        ms = (time.perf_counter() - start) * 1000
        latencies.append(ms)
        call = reply.tool_calls[0] if reply.tool_calls else None
        s = score_call(r, call.name if call else None, call.arguments if call else None, known_customers=known)
        leaked = "<think>" in (reply.text or "")
        mark = {"correct": "  ", "caught": " ?", "wrong": " !"}[s.outcome.value]
        print(
            f"{mark}{r['id']:>4}  {r['intent']:15} {s.outcome.value:8} {ms:6.0f}  "
            f"{describe(call.name if call else None, call.arguments if call else None, reply.text)}"
            + (f"   <- {'; '.join(s.reasons)}" if s.reasons else "")
            + ("   [digits]" if s.converted_to_digits else "")
        )
        results.append(
            {
                "id": r["id"],
                "text": r["text"],
                "intent": r["intent"],
                "outcome": s.outcome.value,
                "reasons": list(s.reasons),
                "converted_to_digits": s.converted_to_digits,
                "thinking_leaked": leaked,
                "ms": round(ms),
                "proposal": {"tool": call.name if call else None, "arguments": call.arguments if call else None},
                "text_reply": reply.text,
            }
        )
    await llm.aclose()

    scored = [x for x in results if "outcome" in x]
    if not scored:
        print("\n  Nothing scored.")
        return 1
    counts = {o.value: sum(1 for x in scored if x["outcome"] == o.value) for o in Outcome}
    n = len(scored)
    print(f"\nsummary  n={n}   (text in, so this is the ceiling the audio eval cannot beat)")
    print(f"  task accuracy  {counts['correct'] / n:6.1%}   correct {counts['correct']}")
    print(f"  caught         {counts['caught'] / n:6.1%}   would ask again - safe")
    print(f"  WRONG          {counts['wrong'] / n:6.1%}   would read back a wrong value")
    print(f"  digits converted  {sum(x['converted_to_digits'] for x in scored)}   (hard rule 7)")
    print(f"  thinking leaked   {sum(x['thinking_leaked'] for x in scored)}")
    if latencies:
        ordered = sorted(latencies)
        p90 = ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))]
        print(f"  latency  p50 {statistics.median(ordered) / 1000:.2f} s   p90 {p90 / 1000:.2f} s   (budget: 0.8 s)")

    by_intent: dict[str, list[str]] = defaultdict(list)
    for x in scored:
        by_intent[x["intent"]].append(x["outcome"])
    print("  by intent: " + ",  ".join(
        f"{k} {sum(o == 'correct' for o in v)}/{len(v)}" for k, v in sorted(by_intent.items())
    ))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "base_url": base_url,
                "model": model,
                "tool_mode": args.tool_mode,
                "probe": found,
                "plain_reply_ms": round(plain_ms),
                "counts": counts,
                "results": results,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {OUT}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=12, help="corpus sentences to try (default 12, all 6 intents)")
    parser.add_argument("--tool-mode", choices=["native", "prompt"], default="native")
    parser.add_argument("--base-url", help="override EVON_BASE_URL")
    parser.add_argument("--model", help="override EVON_MODEL")
    args = parser.parse_args()
    try:
        return asyncio.run(run(args))
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
