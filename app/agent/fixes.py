"""The M4 fixes, behind one switch (D-040).

Each fix answers a failure the eval measured (D-031, D-038, D-039), and every one
of them acts *after* Evon, in deterministic code. So the eval scores "before" and
"after" from the same cached Evon proposals and the same commit, at no cost, and
the switch is an object passed explicitly rather than a global.

    fix 1  payment guard     "Verma ji ne teen sau diye" proposed as credit -> payment
    fix 2  number spellings  "ikkes", "unteen", "nao"; "mile" as filler
    fix 3  unit quantities   "paanch kilo", "do bori" parse as 5 and 2
    fix 4  heard check       the number Evon passes must be one the owner said
    fix 5  name check        the customer Evon names must be one the owner said
    fix 6  part-paid bill    "bill saadhe chhe sau, paanch sau diye, baaki likh do" -> 150 (D-049)

The product runs with everything on (`ALL`); the eval's "before" uses `NONE`.
"""

import re
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Any, Literal

from rapidfuzz import fuzz

from app.nlu.names import strip_honorifics
from app.nlu.numbers import heard_values, normalise, number_spans

__all__ = [
    "ALL", "NONE", "Fixes", "apply_to_proposal", "filled", "money_direction", "name_was_heard", "part_paid",
    "was_heard",
]


@dataclass(frozen=True)
class Fixes:
    payment_guard: bool = True
    number_spellings: bool = True
    unit_quantities: bool = True
    heard_check: bool = True
    name_check: bool = True
    part_paid: bool = True


ALL = Fixes()
NONE = replace(
    ALL, payment_guard=False, number_spellings=False, unit_quantities=False, heard_check=False, name_check=False,
    part_paid=False,
)


# --- fix 1: the payment guard ------------------------------------------------
#
# Evon books "X ne ... diye" (X gave) as credit about a fifth of the time, and
# Prisma makes it worse by writing "diye" as "die" (D-039). A payment booked as
# credit moves the balance the wrong way by twice the amount, so the cue is read
# in code. The grammar is the agent role: "Sharma ji NE diye" (he paid us) versus
# "Sharma ji KO diye" (we lent him) and "Sharma ji KA ... likho" (put on his tab).
#
# Words are compared after `normalise`, so "diye"/"die"/"diyee" and "wapas"/
# "vapas" each meet one entry.

_GIVE_AFTER_NE = frozenset(
    normalise(w)
    for w in "de diye die diya dia diyo diyi jama vapas wapas lautaye lautaya chukaye chukaya chuka bhare bhara".split()
)
# Cues that say "we received money" with no subject marker needed.
_RECEIVED = frozenset(normalise(w) for w in "jama chukta chukaya chukaye mile mila vasool vasul".split())
_CREDIT = frozenset(normalise(w) for w in "udhaar udhar udhari".split())
_WORD = re.compile(r"[a-z]+")


def money_direction(utterance: str) -> Literal["payment", "credit"] | None:
    """Which way the money moves, from grammar alone; None unless it is unambiguous."""
    words = [normalise(w) for w in _WORD.findall(utterance.lower())]
    after_ne = words[words.index("ne") + 1 :] if "ne" in words else []
    payment = any(w in _GIVE_AFTER_NE for w in after_ne) or any(w in _RECEIVED for w in words)
    credit = any(w in _CREDIT for w in words)
    if payment and not credit:
        return "payment"
    if credit and not payment:
        return "credit"
    return None


# --- fix 4: the heard check ----------------------------------------------------
#
# Hard rule 7 says amounts are parsed by code from the words *as spoken*. This
# makes the executor hold Evon to it: the value in amount_spoken (or the stock
# quantity) must be readable from the transcript itself. It catches a fragment
# with no number in it (heavy noise, D-039), and Evon inventing one ("saath
# sachet" passed as "one sachet", D-031). Names are deliberately not checked:
# in a conversation Evon may rightly take the name from an earlier turn.


def was_heard(value: Fraction, utterance: str, *, spelling_fixes: bool = True) -> bool:
    return abs(value) in heard_values(utterance, spelling_fixes=spelling_fixes)


# --- fix 5: the name check -----------------------------------------------------
#
# The twin of fix 4, for names. When noise swallows the name, Evon supplies one -
# and it supplies "Sharma", the example in its own system prompt (D-043, D-044:
# 65 wrong rows on the held-out set). Every name word Evon passes must therefore
# resemble a word the owner said. Measured on all 180 takes x 12 conditions
# (D-045): proposals that resolved to the right customer all scored >= 85,
# because Evon copies names from the transcript, misspellings and all; the ones
# that resolved to a wrong customer all scored < 75. 80 sits between.
#
# "Said" includes the last few user turns, so "unka 200 aur likho" after a turn
# that named Sharma ji still passes in a real conversation.

NAME_HEARD = 80.0


def name_was_heard(name: str, context: str) -> bool:
    words = [w for w in (normalise(t) for t in _WORD.findall(context.lower())) if w]
    parts = [w for w in (normalise(t) for t in strip_honorifics(name).split()) if w]
    return all(any(fuzz.ratio(part, w) >= NAME_HEARD for w in words) for part in parts)


def filled(value: Any) -> bool:
    """An optional argument Evon actually filled ("None" and "" are how it says it did not)."""
    return isinstance(value, str) and value.strip().lower() not in ("", "none", "null", "nil", "undefined")


# --- fix 6: a bill paid in part --------------------------------------------------
#
# "Mehra ji ka bill saadhe chhe sau ka hua, paanch sau diye, baaki khate mein likh
# do" puts 150 on credit, a number nobody said. Evon makes one tool call, so it
# would have to do the subtraction, and hard rule 7 forbids that. Asked to pass the
# part paid in an argument of its own, it never did: 0 of 36 tries across three
# prompts, and twice it passed the part paid as the bill (D-049). So code reads both
# amounts from the transcript, and acts only on the clear pattern:
#   - a word for "the rest" (baaki) and one that puts it on the book (bill, likh, khate);
#   - exactly two amounts, and exactly one of them right before a word for paying;
#   - "baaki" not followed by an amount: "baaki dedh sau likh do" names the credit itself.
# Anything else is Evon's proposal as before, and the read-back is the net.

_REST = frozenset(normalise(w) for w in "baaki baki baqi bacha bache bachi bakaya".split())
_ON_THE_BOOK = frozenset(
    normalise(w)
    for w in (
        "bill total saamaan saman likh likho likhna likhiye likhdo khate khata khaate udhaar udhar udhari "
        "daal daalo dalo jod jodo chadha chadhao chadhado"
    ).split()
)
_PAYING = frozenset(normalise(w) for w in "de diye die diya dia diyo mile mila jama".split())
_BETWEEN = frozenset(normalise(w) for w in "rupaye rupay rupaya rupaiya rupees rs ka ke ki only".split())


def part_paid(utterance: str, *, spelling_fixes: bool = True) -> tuple[str, str] | None:
    """(the bill, the part paid), as spoken, when the owner put the rest of a bill on credit."""
    tokens, spans = number_spans(utterance, spelling_fixes=spelling_fixes)
    words = [normalise(t) for t in tokens]
    if not any(w in _REST for w in words) or not any(w in _ON_THE_BOOK for w in words):
        return None
    spans = [(s, e) for s, e in spans if words[s:e] != ["do"]]  # "likh do": the verb, not 2
    if len(spans) != 2 or any(s > 0 and words[s - 1] in _REST for s, _ in spans):
        return None

    def paying_follows(end: int) -> bool:
        while end < len(words) and words[end] in _BETWEEN:
            end += 1
        return end < len(words) and words[end] in _PAYING

    paid = [span for span in spans if paying_follows(span[1])]
    if len(paid) != 1:
        return None
    bill = next(span for span in spans if span != paid[0])
    return " ".join(tokens[bill[0]:bill[1]]), " ".join(tokens[paid[0][0]:paid[0][1]])


def apply_to_proposal(
    fixes: Fixes, utterance: str, tool: str | None, arguments: dict[str, Any] | None
) -> tuple[str | None, dict[str, Any] | None, str | None]:
    """Correct Evon's proposal where a fix applies. Returns (tool, arguments, what changed).

    Only ever swaps between the two money tools, keeping the customer, or fills a
    part-paid bill's two amounts from the owner's own words. The read-back still
    asks the owner before anything is written, so a wrong correction is caught
    exactly as a wrong proposal would be.
    """
    if tool not in ("add_credit", "record_payment") or arguments is None:
        return tool, arguments, None
    if fixes.part_paid and (found := part_paid(utterance, spelling_fixes=fixes.number_spellings)):
        bill, paid = found
        kept = {k: v for k, v in arguments.items() if k == "customer_name" or (k == "note" and filled(v))}
        changed = f"part paid: {tool}({arguments.get('amount_spoken')!r}) -> bill {bill!r}, paid {paid!r}"
        return "add_credit", {**kept, "amount_spoken": bill, "paid_spoken": paid}, changed
    if not fixes.payment_guard:
        return tool, arguments, None
    if tool == "add_credit" and filled(arguments.get("paid_spoken")):
        # A bill paid in part: "ne paanch sau diye" is the part paid, already in the
        # proposal. Swapping to a payment would drop it and book the whole bill as paid.
        return tool, arguments, None
    direction = money_direction(utterance)
    wanted = {"payment": "record_payment", "credit": "add_credit"}.get(direction or "")
    if wanted is None or wanted == tool:
        return tool, arguments, None
    kept = {k: v for k, v in arguments.items() if k in ("customer_name", "amount_spoken")}
    return wanted, kept, f"payment guard: {tool} -> {wanted}"
