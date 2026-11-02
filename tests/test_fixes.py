"""The M4 fixes and their switch (app/agent/fixes.py). Each case is a measured failure."""

import pytest

from app.adapters.base import ToolCall
from app.agent.executor import Executor, Outcome
from app.agent.fixes import ALL, NONE, apply_to_proposal, money_direction
from app.ledger.db import connect
from app.ledger.repo import Ledger
from app.nlu.numbers import parse_amount_paise

# --- fix 1: which way the money moves ------------------------------------------


@pytest.mark.parametrize(
    "utterance",
    [
        "Verma bhaiya ne saadhe teen sau diye",  # D-031 #1, as text
        "verma bhaiya ne sadhe teen sau die",  # D-039: Prisma's spelling
        "Ahuja ji ne dedh hazaar de diye",
        "Nigam ji ne ikkees rupaye wapas kiye",
        "Tiwari ji ne sava hazaar jama kiye",
        "Sharma ji ka hisaab chukta, aath sau mile",
    ],
)
def test_payment_grammar_is_read_as_payment(utterance):
    assert money_direction(utterance) == "payment"


@pytest.mark.parametrize(
    "utterance",
    ["Sharma ji ka dedh sau udhaar likho", "nigam bhaiya ka fifty rupees udhar likho"],
)
def test_credit_grammar_is_read_as_credit(utterance):
    assert money_direction(utterance) == "credit"


@pytest.mark.parametrize(
    "utterance",
    [
        "Sharma ji ko paanch sau de diye",  # KO: we gave him money - a loan, not a payment
        "Sharma ji ne paanch sau ka saamaan liya",  # NE, but he *took* goods
        "Gupta aunty ke khate mein dhai sau likh do",  # no cue at all
        "Sharma ji ne do sau diye, baaki udhaar",  # both cues: leave it to Evon and the owner
    ],
)
def test_no_cue_or_mixed_cues_leave_the_proposal_alone(utterance):
    assert money_direction(utterance) is None


def test_the_guard_swaps_credit_for_payment_and_keeps_the_amount_as_spoken():
    tool, args, changed = apply_to_proposal(
        ALL,
        "verma bhaiya ne sadhe teen sau die",
        "add_credit",
        {"customer_name": "verma bhaiya", "amount_spoken": "sadhe teen sau", "note": "x"},
    )
    assert tool == "record_payment"
    assert args == {"customer_name": "verma bhaiya", "amount_spoken": "sadhe teen sau"}
    assert "payment guard" in changed


def test_the_guard_never_touches_other_tools_or_runs_when_switched_off():
    utterance = "verma bhaiya ne sadhe teen sau die"
    assert apply_to_proposal(ALL, utterance, "get_balance", {"customer_name": "verma"})[2] is None
    assert apply_to_proposal(NONE, utterance, "add_credit", {"customer_name": "verma"})[0] == "add_credit"


# --- fix 2: spellings the real models produced ---------------------------------


@pytest.mark.parametrize(
    "spoken,paise",
    [("ikkes", 2100), ("unteen", 2900), ("nao sau chalis", 94000), ("aath sau mile", 80000)],
)
def test_measured_spellings_parse_with_the_fix_and_not_without(spoken, paise):
    assert parse_amount_paise(spoken) == paise
    assert parse_amount_paise(spoken, spelling_fixes=False) is None


@pytest.mark.parametrize("spoken,paise", [("ten", 1000), ("teen", 300), ("saat", 700), ("saath", 6000)])
def test_the_fix_keeps_close_numbers_apart(spoken, paise):
    assert parse_amount_paise(spoken) == paise


# --- the product path: the executor applies the guard before the read-back -------


@pytest.fixture
def executor(tmp_path):
    ledger = Ledger(connect(str(tmp_path / "k.db")))
    ledger.add_customer("Verma")
    return Executor(ledger)


def test_the_executor_reads_back_a_payment_when_evon_proposed_credit(executor):
    call = ToolCall("add_credit", {"customer_name": "Verma bhaiya", "amount_spoken": "sadhe teen sau"})
    reply = executor.handle(call, utterance="verma bhaiya ne sadhe teen sau die")
    assert reply.outcome is Outcome.CONFIRMING
    assert "jama kar doon" in reply.speech and "udhaar" not in reply.speech


def test_the_executor_with_fixes_off_behaves_as_before(executor):
    executor.fixes = NONE
    call = ToolCall("add_credit", {"customer_name": "Verma bhaiya", "amount_spoken": "sadhe teen sau"})
    reply = executor.handle(call, utterance="verma bhaiya ne sadhe teen sau die")
    assert "udhaar likh doon" in reply.speech
