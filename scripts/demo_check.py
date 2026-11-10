"""The demo's lines, checked against real Evon as text: no audio, no Gnani credits.

    uv run python -m scripts.demo_check              # every scenario
    uv run python -m scripts.demo_check --only bill  # the scenarios whose name contains "bill"

Each scenario starts from a fresh copy of the demo shop (scripts/seed.py) and
runs turn by turn the way a call does (app/pipeline/session.py): while an entry
waits for a yes, the reply is answered in code; anything else goes to Evon, and
Evon's proposal goes through the real executor. What KhataCall would say is
compared with what docs/DEMO.md promises.

tests/test_demo_scenarios.py runs the same scenarios offline, with each turn's
expected proposal standing in for Evon, so the replies are pinned by tests. This
script is the other half: does real Evon propose those calls? Evon is
self-hosted, so it costs nothing, but it does need Evon up (docs/EVON_KAGGLE.md).
"""

import argparse
import asyncio
import re
import sys
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from app.adapters.base import Message, ToolCall
from app.agent.executor import Executor, Reply
from app.agent.prompts import SYSTEM_PROMPT
from app.agent.tools import TOOL_SCHEMAS
from app.ledger.db import connect
from app.ledger.repo import Ledger
from scripts.seed import seed


@dataclass(frozen=True)
class Turn:
    say: str  # what the owner says
    expect: str  # what KhataCall should say back
    tool: str | None = None  # the call Evon should propose; None when code answers (a yes or no)
    args: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Scenario:
    name: str
    turns: tuple[Turn, ...]
    phone_line: bool = False  # phone-line mode: amounts are spelled out


def _credit(name: str, amount: str, **extra: str) -> dict[str, str]:
    return {"customer_name": name, "amount_spoken": amount, **extra}


# Balances in the demo shop (scripts/seed.py): Sharma ji 400, Gupta aunty settled,
# Verma bhaiya 350, Yadav ji 1,500, Joshi uncle 100 paid ahead, Chauhan bhai 2,500,
# Mehra ji 200. Stock: Maggi 12 and tel 9 below their thresholds, Parle-G 48, chini 30.
SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        "udhaar on a noisy phone line",
        (
            Turn("Sharma ji ka dedh sau udhaar likho",
                 "Sharma ji ke khate mein ek paanch shunya, 150 rupaye udhaar likh doon?",
                 "add_credit", _credit("Sharma ji", "dedh sau")),
            Turn("haan", "Likh diya. Sharma ji ka ab 550 rupaye baaki hai."),
        ),
        phone_line=True,
    ),
    Scenario(
        "a bill paid in part, the rest on credit",
        (
            # What real Evon proposes (Nov 10): the bill only. The part paid is read from
            # the owner's words in code (fix 6, D-049).
            Turn("Mehra ji ka bill saadhe chhe sau ka hua, paanch sau diye, baaki khate mein likh do",
                 "Mehra ji ka bill 650 rupaye, 500 rupaye mile. Baaki 150 rupaye udhaar likh doon?",
                 "add_credit", _credit("Mehra ji", "saadhe chhe sau")),
            Turn("haan ji", "Likh diya. Mehra ji ka ab 350 rupaye baaki hai."),
        ),
    ),
    Scenario(
        "a bill paid in part, said the other way round, on a phone line",
        (
            Turn("Mehra ji ne paanch sau diye, bill saadhe chhe sau ka tha, baaki khate mein likh do",
                 "Mehra ji ka bill 650 rupaye, 500 rupaye mile. Baaki ek paanch shunya, 150 rupaye udhaar likh doon?",
                 "add_credit", _credit("Mehra ji", "saadhe chhe sau")),
            Turn("haan", "Likh diya. Mehra ji ka ab 350 rupaye baaki hai."),
        ),
        phone_line=True,
    ),
    Scenario(
        "how much a customer owes",
        (Turn("Yadav ji ka kitna baaki hai?", "Yadav ji ka 1,500 rupaye baaki hai.",
              "get_balance", {"customer_name": "Yadav ji"}),),
    ),
    Scenario(
        "a settled account",
        (Turn("Gupta aunty ka hisaab batao", "Gupta aunty ka hisaab chukta hai.",
              "get_balance", {"customer_name": "Gupta aunty"}),),
    ),
    Scenario(
        "a customer who paid ahead",
        (Turn("Joshi uncle ka kitna baaki hai?", "Joshi uncle ke 100 rupaye jama hain.",
              "get_balance", {"customer_name": "Joshi uncle"}),),
    ),
    Scenario(
        "a payment",
        (
            Turn("Chauhan bhai ne ek hazaar diye", "Chauhan bhai ke khate mein 1,000 rupaye jama kar doon?",
                 "record_payment", _credit("Chauhan bhai", "ek hazaar")),
            Turn("haan", "Likh diya. Chauhan bhai ka ab 1,500 rupaye baaki hai."),
        ),
    ),
    Scenario(
        "saying no writes nothing",
        (
            Turn("Verma bhaiya ka do sau likho", "Verma bhaiya ke khate mein 200 rupaye udhaar likh doon?",
                 "add_credit", _credit("Verma bhaiya", "do sau")),
            Turn("nahi", "Theek hai, nahi likha."),
            Turn("Verma bhaiya ka kitna baaki hai?", "Verma bhaiya ka 350 rupaye baaki hai.",
                 "get_balance", {"customer_name": "Verma bhaiya"}),
        ),
    ),
    Scenario(
        "a new customer",
        (
            Turn("Kapoor ji ka teen sau udhaar likho",
                 "Kapoor ji naya customer hai. Kapoor ji ke khate mein 300 rupaye udhaar likh doon?",
                 "add_credit", _credit("Kapoor ji", "teen sau")),
            Turn("haan", "Likh diya. Kapoor ji ka ab 300 rupaye baaki hai."),
        ),
    ),
    Scenario(
        "stock left",
        (Turn("Maggi kitni bachi hai?", "Maggi 12 packet bachi hai.", "get_stock", {"item": "Maggi"}),),
    ),
    Scenario(
        "what is running low",
        (Turn("Kya kya khatam hone wala hai?", "Ye kam hain: tel 9 litre, Maggi 12 packet.", "list_low_stock", {}),),
    ),
    Scenario(
        "stock arriving",
        (
            Turn("Parle-G ke bees packet aaye", "Parle-G ke 20 packet aaye, likh doon?",
                 "update_stock", {"item": "Parle-G", "qty_delta_spoken": "bees"}),
            Turn("haan", "Likh diya. Parle-G ab 68 packet."),
        ),
    ),
    Scenario(
        "stock sold",
        (
            Turn("Paanch kilo chini bik gayi", "chini ke 5 kilo nikal gaye, likh doon?",
                 "update_stock", {"item": "chini", "qty_delta_spoken": "minus paanch"}),
            Turn("haan", "Likh diya. chini ab 25 kilo."),
        ),
    ),
)


def demo_shop() -> Ledger:
    """A fresh, in-memory copy of the shop `scripts.seed` builds."""
    ledger = Ledger(connect(":memory:"))
    seed(ledger)
    return ledger


Propose = Callable[[Turn], Awaitable[ToolCall | str | None]]


@dataclass
class Step:
    turn: Turn
    proposal: ToolCall | str | None  # what Evon proposed (text if it answered in words); None: not asked
    said: str  # what KhataCall said

    @property
    def ok(self) -> bool:
        return same_words(self.said, self.turn.expect)


def same_words(a: str, b: str) -> bool:
    """Equal ignoring case and punctuation: "chini" and "Chini" read back the same."""
    def key(s: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", s.lower())
    return key(a) == key(b)


async def play(scenario: Scenario, propose: Propose) -> tuple[list[Step], Ledger]:
    """Run one scenario the way a call runs it, on a fresh demo shop."""
    ledger = demo_shop()
    executor = Executor(ledger, session_id="demo", spell_amounts=scenario.phone_line)
    heard: list[str] = []
    steps = []
    for turn in scenario.turns:
        heard.append(turn.say)
        reply: Reply | None = executor.handle_user_text(turn.say) if executor.has_pending else None
        proposal: ToolCall | str | None = None
        if reply is not None:
            said = reply.speech
        else:
            proposal = await propose(turn)
            if isinstance(proposal, ToolCall):
                # The session passes the last few turns as context for the name check.
                said = executor.handle(proposal, utterance=turn.say, context=" ".join(heard[-3:])).speech
            else:
                said = proposal or ""
        steps.append(Step(turn, proposal, said))
    return steps, ledger


def describe(proposal: ToolCall | str | None) -> str:
    if proposal is None:
        return "answered in code"
    if isinstance(proposal, str):
        return f"text: {proposal.strip()[:60]!r}"
    return f"{proposal.name}({', '.join(f'{k}={v!r}' for k, v in proposal.arguments.items())})"


async def run(args) -> int:
    from app.adapters.evon_llm import EvonError, EvonLLM
    from app.config import get_settings
    from scripts.evon_check import probe

    settings = get_settings()
    api_key = settings.evon_api_key.get_secret_value()
    found = await probe(settings.evon_base_url, api_key)
    if "error" in found or found.get("health") != 200:
        # A dead Kaggle tunnel still answers: Cloudflare serves a 530 error page.
        print(f"Evon is not up (health: {found.get('health', 'unreachable')}). Start it on Kaggle and")
        print("update EVON_BASE_URL in .env (docs/EVON_KAGGLE.md).")
        return 2

    llm = EvonLLM(base_url=settings.evon_base_url, api_key=api_key, model=settings.evon_model, timeout_s=120)
    system = Message("system", SYSTEM_PROMPT)

    async def propose(turn: Turn) -> ToolCall | str | None:
        reply = await llm.complete([system, Message("user", turn.say)], tools=TOOL_SCHEMAS)
        return reply.tool_calls[0] if reply.tool_calls else (reply.text or "")

    picked = [s for s in SCENARIOS if args.only.lower() in s.name]
    failed = 0
    try:
        for scenario in picked:
            print(f"\n{scenario.name}{'  [phone line]' if scenario.phone_line else ''}")
            start = time.perf_counter()
            try:
                steps, _ = await play(scenario, propose)
            except EvonError as e:
                print(f"  ERROR from Evon: {e}")
                failed += 1
                continue
            for step in steps:
                print(f"  {'ok ' if step.ok else 'NO '} owner: {step.turn.say}")
                print(f"       Evon:  {describe(step.proposal)}")
                print(f"       said:  {step.said}")
                if not step.ok:
                    print(f"       want:  {step.turn.expect}")
            failed += not all(step.ok for step in steps)
            print(f"       ({(time.perf_counter() - start):.1f} s)")
    finally:
        await llm.aclose()

    print(f"\n{len(picked) - failed} of {len(picked)} scenarios as the demo script expects.")
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", default="", help="run only scenarios whose name contains this")
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
