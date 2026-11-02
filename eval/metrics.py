"""Scoring a proposed tool call against the corpus's ground truth.

Task accuracy is the headline metric (CLAUDE.md §5): intent, customer and amount
all correct. But "incorrect" hides two very different outcomes, and the eval
reports them separately:

    correct    the agent would read back the right thing
    caught     it would ask again instead: an unparseable amount, an ambiguous
               or unknown name, no tool call. Annoying, but nothing wrong can
               be written, because nothing reaches the read-back.
    wrong      it would read back a *wrong* value with confidence: a different
               amount, a different customer, a different intent. The shop
               owner's "haan" is then the only thing between the error and the
               book.

`wrong` is the number that matters for a ledger, and a system that converts
`wrong` into `caught` has improved even if task accuracy has not moved.

Scoring calls the same functions the executor calls - `parse_amount_paise`,
`parse_signed_quantity`, `match_name` - so a row scores correct exactly when the
running agent would have proposed the right write. A reimplementation here
would drift from the product and measure something else.
"""

from dataclasses import dataclass, field
from fractions import Fraction
from enum import StrEnum
from typing import Any

from pydantic import ValidationError

from app.agent.fixes import ALL, Fixes, apply_to_proposal, name_was_heard, was_heard
from app.agent.tools import args_model_for
from app.nlu.names import Decision, has_real_name, match_name
from app.nlu.numbers import normalise, parse_amount_paise, parse_number, parse_signed_quantity

__all__ = ["Outcome", "Score", "score_call"]


class Outcome(StrEnum):
    CORRECT = "correct"
    CAUGHT = "caught"
    WRONG = "wrong"


@dataclass(frozen=True)
class Score:
    outcome: Outcome
    intent_ok: bool
    customer_ok: bool | None = None  # None: this intent has no customer
    amount_ok: bool | None = None
    item_ok: bool | None = None
    qty_ok: bool | None = None
    # Why it was not correct, in words, for the failure-mode ranking in M4.
    reasons: tuple[str, ...] = ()
    # The model turned spoken words into digits, which hard rule 7 forbids even
    # when the digits happen to be right.
    converted_to_digits: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def correct(self) -> bool:
        return self.outcome is Outcome.CORRECT


def _item_key(name: str) -> str:
    return normalise(name.replace(" ", ""))


def _has_digits(text: str) -> bool:
    return any(ch.isdigit() for ch in text)


def score_call(
    row: dict[str, str],
    tool_name: str | None,
    arguments: dict[str, Any] | None,
    *,
    known_customers: list[str],
    known_items: list[str] | None = None,
    fixes: Fixes = ALL,
    utterance: str | None = None,
) -> Score:
    """Score one proposal against one corpus row.

    `row` is a line of eval/corpus/scripts.csv. `tool_name` is None when the
    model replied in text instead of calling a tool. `known_customers` is the
    ledger the executor would resolve names against. `known_items`, if given, is
    the stock list: a misheard item that matches none of it is read back as
    "naya item hai", so it counts as caught, like an unknown customer.
    """
    # The fixes run first, as in the executor: `utterance` is the transcript the
    # agent saw (the corpus text, when scoring text input).
    fix_applied = None
    if utterance is not None:
        tool_name, arguments, fix_applied = apply_to_proposal(fixes, utterance, tool_name, arguments)
    spelling = fixes.number_spellings

    args = arguments or {}
    expected = row["intent"]
    reasons: list[str] = []
    caught: list[str] = []
    details: dict[str, Any] = {"tool": tool_name, "arguments": args, "fix_applied": fix_applied}

    if tool_name is None:
        return Score(Outcome.CAUGHT, intent_ok=False, reasons=("no tool call",), details=details)

    # Validate exactly as the executor does. Arguments it would reject make it ask
    # again, and normalisation it applies (e.g. "None" -> absent) must apply here too.
    model = args_model_for(tool_name)
    if model is None:
        return Score(Outcome.CAUGHT, intent_ok=False, reasons=(f"unknown tool {tool_name}",), details=details)
    try:
        args = model.model_validate(args).model_dump()
    except ValidationError:
        return Score(
            Outcome.CAUGHT, intent_ok=tool_name == expected, reasons=("arguments rejected",), details=details
        )

    intent_ok = tool_name == expected
    if not intent_ok:
        reasons.append(f"intent {tool_name} instead of {expected}")

    customer_ok = amount_ok = item_ok = qty_ok = None
    converted = False

    # --- customer: resolved the way the executor resolves it ---
    if row.get("customer"):
        spoken = str(args.get("customer_name") or "")
        unheard = (
            fixes.name_check
            and utterance is not None
            and has_real_name(spoken)
            and not name_was_heard(spoken, utterance)
        )
        result = match_name(spoken, known_customers) if spoken and not unheard else None
        details["customer_decision"] = "not heard" if unheard else (result.decision.value if result else "missing")
        if unheard:
            customer_ok = False
            caught.append(f"customer not heard: {spoken}")  # the executor asks for the name
        elif result is None:
            customer_ok = False
            caught.append("no customer name")
        elif result.decision is Decision.MATCH:
            customer_ok = normalise(result.name or "") == normalise(row["customer"])
            if not customer_ok:
                reasons.append(f"customer {result.name} instead of {row['customer']}")
        else:
            customer_ok = False
            caught.append(f"customer {result.decision.value}")

    # --- amount: parsed the way the executor parses it ---
    if row.get("amount_paise"):
        spoken = str(args.get("amount_spoken") or "")
        paise = parse_amount_paise(spoken, spelling_fixes=spelling) if spoken else None
        details["amount_paise"] = paise
        converted = _has_digits(spoken) and not _has_digits(row["text"])
        heard_check = fixes.heard_check and utterance is not None
        if paise is None:
            amount_ok = False
            caught.append("amount unparseable")
        elif heard_check and not was_heard(Fraction(paise, 100), utterance, spelling_fixes=spelling):
            amount_ok = False
            caught.append(f"amount not heard: {spoken}")  # the executor asks again
        else:
            amount_ok = paise == int(row["amount_paise"])
            if not amount_ok:
                reasons.append(f"amount {paise} instead of {row['amount_paise']}")

    # --- stock ---
    if row.get("item"):
        spoken = str(args.get("item") or "")
        item_ok = bool(spoken) and _item_key(spoken) == _item_key(row["item"])
        stock = {_item_key(i) for i in known_items or ()}
        if item_ok:
            pass
        elif spoken and known_items is not None and _item_key(spoken) not in stock:
            caught.append(f"item new: {spoken}")  # the executor flags "naya item hai"
        else:
            reasons.append(f"item {spoken or 'missing'} instead of {row['item']}")

    if expected == "update_stock" and row.get("qty"):
        spoken = str(args.get("qty_delta_spoken") or "")
        qty = (
            parse_signed_quantity(spoken, spelling_fixes=spelling, unit_words=fixes.unit_quantities)
            if spoken
            else None
        )
        details["qty"] = str(qty) if qty is not None else None
        if qty is None:
            qty_ok = False
            caught.append("quantity unparseable")
        elif fixes.heard_check and utterance is not None and not was_heard(qty, utterance, spelling_fixes=spelling):
            qty_ok = False
            caught.append(f"quantity not heard: {spoken}")
        else:
            qty_ok = float(qty) == float(row["qty"])
            if not qty_ok:
                reasons.append(f"quantity {qty} instead of {row['qty']}")

    if expected == "list_low_stock":
        spoken = args.get("threshold_spoken")
        expected_threshold = row.get("qty") or None
        if expected_threshold is None:
            invented = parse_number(str(spoken), spelling_fixes=spelling) if spoken else None
            qty_ok = not spoken
            if invented is not None:
                reasons.append(f"invented a threshold: {spoken}")
            elif spoken:
                caught.append(f"unparseable threshold {spoken!r}")  # the executor would ask
        else:
            value = parse_number(str(spoken), spelling_fixes=spelling) if spoken else None
            qty_ok = value is not None and float(value) == float(expected_threshold)
            if value is None:
                caught.append("threshold missing or unparseable")
            elif not qty_ok:
                reasons.append(f"threshold {value} instead of {expected_threshold}")

    checks = [c for c in (intent_ok, customer_ok, amount_ok, item_ok, qty_ok) if c is not None]
    if all(checks):
        outcome = Outcome.CORRECT
    elif reasons:
        # Any confident wrong value is dangerous, even if something else was caught.
        outcome = Outcome.WRONG
    else:
        outcome = Outcome.CAUGHT

    return Score(
        outcome=outcome,
        intent_ok=intent_ok,
        customer_ok=customer_ok,
        amount_ok=amount_ok,
        item_ok=item_ok,
        qty_ok=qty_ok,
        reasons=tuple(reasons + caught),
        converted_to_digits=converted,
        details=details,
    )
