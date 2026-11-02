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

The product runs with everything on (`ALL`); the eval's "before" uses `NONE`.
"""

import re
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Any, Literal

from rapidfuzz import fuzz

from app.nlu.names import strip_honorifics
from app.nlu.numbers import heard_values, normalise

__all__ = ["ALL", "NONE", "Fixes", "apply_to_proposal", "money_direction", "name_was_heard", "was_heard"]


@dataclass(frozen=True)
class Fixes:
    payment_guard: bool = True
    number_spellings: bool = True
    unit_quantities: bool = True
    heard_check: bool = True
    name_check: bool = True


ALL = Fixes()
NONE = replace(
    ALL, payment_guard=False, number_spellings=False, unit_quantities=False, heard_check=False, name_check=False
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


def apply_to_proposal(
    fixes: Fixes, utterance: str, tool: str | None, arguments: dict[str, Any] | None
) -> tuple[str | None, dict[str, Any] | None, str | None]:
    """Correct Evon's proposal where a fix applies. Returns (tool, arguments, what changed).

    Only ever swaps between the two money tools, keeping the customer and the
    amount as spoken. The read-back still asks the owner before anything is
    written, so a wrong swap is caught exactly as a wrong proposal would be.
    """
    if not fixes.payment_guard or tool not in ("add_credit", "record_payment") or arguments is None:
        return tool, arguments, None
    direction = money_direction(utterance)
    wanted = {"payment": "record_payment", "credit": "add_credit"}.get(direction or "")
    if wanted is None or wanted == tool:
        return tool, arguments, None
    kept = {k: v for k, v in arguments.items() if k in ("customer_name", "amount_spoken")}
    return wanted, kept, f"payment guard: {tool} -> {wanted}"
