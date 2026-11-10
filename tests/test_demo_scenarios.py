"""The demo's lines, turn by turn through the executor, on the shop the demo uses.

Each turn's expected proposal stands in for Evon, so every reply docs/DEMO.md
promises is pinned here, word for word. `scripts.demo_check` sends the same
lines to real Evon to check that it proposes these calls.
"""

import asyncio

import pytest

from app.adapters.base import ToolCall
from scripts.demo_check import SCENARIOS, Scenario, Turn, play


async def as_scripted(turn: Turn) -> ToolCall | None:
    return ToolCall(turn.tool, dict(turn.args)) if turn.tool else None


def run(scenario: Scenario):
    return asyncio.run(play(scenario, as_scripted))


def by_name(fragment: str) -> Scenario:
    return next(s for s in SCENARIOS if fragment in s.name)


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
def test_the_demo_says_what_the_script_promises(scenario):
    steps, _ = run(scenario)
    assert [step.said for step in steps] == [step.turn.expect for step in steps]


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
def test_yes_and_no_are_answered_in_code_and_everything_else_by_evon(scenario):
    steps, _ = run(scenario)
    for step in steps:
        asked_evon = step.proposal is not None
        assert asked_evon == (step.turn.tool is not None), step.turn.say


def test_a_bill_paid_in_part_writes_only_the_rest_and_keeps_the_bill_as_the_note():
    _, ledger = run(by_name("bill paid in part, the rest"))
    mehra = ledger.find_customer("Mehra ji")
    assert ledger.balance_paise(mehra.id) == 20_000 + 15_000
    entry = ledger.entries(mehra.id)[0]
    assert (entry["kind"], entry["amount_paise"]) == ("credit", 15_000)
    assert entry["note"] == "bill 650 rupaye, 500 rupaye mile"


def test_a_part_paid_bill_said_with_ne_diye_is_not_turned_into_a_payment():
    # The payment guard (fix 1) reads "ne ... diye" as a payment. Fix 6 reads the bill
    # and the part paid from the words first; a swap would book the whole Rs 650 as paid.
    _, ledger = run(by_name("said the other way round"))
    assert ledger.balance_paise(ledger.find_customer("Mehra ji").id) == 35_000


def test_saying_no_leaves_the_book_as_it_was():
    _, ledger = run(by_name("saying no"))
    assert ledger.balance_paise(ledger.find_customer("Verma bhaiya").id) == 35_000


def test_a_payment_lowers_the_balance():
    _, ledger = run(by_name("a payment"))
    assert ledger.balance_paise(ledger.find_customer("Chauhan bhai").id) == 150_000


def test_reading_balances_and_stock_changes_nothing():
    for name in ("how much", "settled", "paid ahead", "stock left", "running low"):
        _, ledger = run(by_name(name))
        fresh = asyncio.run(play(Scenario("nothing", ()), as_scripted))[1]
        assert ledger.snapshot() == fresh.snapshot(), name


def test_stock_moves_both_ways():
    _, arrived = run(by_name("arriving"))
    _, sold = run(by_name("sold"))
    assert arrived.find_item("Parle-G").qty == 48 + 20
    assert sold.find_item("chini").qty == 30 - 5
