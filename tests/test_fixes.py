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


# --- fix 3: quantities carry a unit --------------------------------------------


@pytest.mark.parametrize(
    "spoken,qty",
    [("paanch kilo", 5), ("do bori", 2), ("paanch jar", 5), ("minus chhe kilo", -6), ("bees packet", 20)],
)
def test_measured_unit_quantities_parse_with_the_fix_and_not_without(spoken, qty):
    from app.nlu.numbers import parse_signed_quantity

    assert parse_signed_quantity(spoken) == qty
    assert parse_signed_quantity(spoken, unit_words=False) is None


def test_units_are_for_quantities_only_never_money():
    assert parse_amount_paise("paanch kilo") is None


def test_dozen_is_still_a_multiplier_not_a_unit():
    from app.nlu.numbers import parse_signed_quantity

    assert parse_signed_quantity("do darjan") == 24


# --- fix 4: the number must be one the owner said ---------------------------------


@pytest.mark.parametrize(
    "value,utterance",
    [
        (150, "sharma ji ka dedh sau udhar likho"),
        (350, "teen sau rupaye pachaas"),  # a filler inside the number
        (150, "sharma ji ka 150 udhar likho"),  # Prisma wrote digits
        (5, "chini k paanch kilo aaye"),
        (21, "nigam ji ne ikkes rupaye wapas kiye"),  # fix 2's spelling counts as heard
    ],
)
def test_numbers_the_owner_said_are_heard(value, utterance):
    from fractions import Fraction

    from app.agent.fixes import was_heard

    assert was_heard(Fraction(value), utterance)


@pytest.mark.parametrize(
    "value,utterance",
    [
        (150, "sharma jitah"),  # D-039: a fragment; Evon filled in the prompt's own example
        (1, "shampoo ke saath sachet aaye"),  # D-031: "saath" (60) passed as "one"
        (20, "namak ke beech packet aa gaye"),  # a plausible repair, still not what was heard
    ],
)
def test_numbers_the_owner_did_not_say_are_not(value, utterance):
    from fractions import Fraction

    from app.agent.fixes import was_heard

    assert not was_heard(Fraction(value), utterance)


def test_the_executor_asks_again_rather_than_take_an_unheard_amount(executor):
    call = ToolCall("add_credit", {"customer_name": "Verma bhaiya", "amount_spoken": "dedh sau"})
    reply = executor.handle(call, utterance="verma bhaiya")
    assert reply.outcome is not Outcome.CONFIRMING
    assert "Kitne rupaye" in reply.speech


def test_without_an_utterance_the_heard_check_does_not_apply(executor):
    """Callers that have no transcript (tests, tools) keep the old behaviour."""
    call = ToolCall("add_credit", {"customer_name": "Verma bhaiya", "amount_spoken": "dedh sau"})
    assert executor.handle(call).outcome is Outcome.CONFIRMING


# --- fix 5: the customer must be one the owner named -------------------------------


@pytest.mark.parametrize(
    "name,heard",
    [
        ("Sharma ji", "sharma ji ka dedh sau udhar likho"),
        ("Vadma bhaiya", "vadma bhaiya ne chaar sau die"),  # Evon copies Prisma's spelling
        ("Tiwadi deedi", "tiwadi deedi ka hisaab batao"),
        ("Sharma ji", "sharma jitah"),  # the name was heard, even if the rest was not
    ],
)
def test_names_the_owner_said_pass(name, heard):
    from app.agent.fixes import name_was_heard

    assert name_was_heard(name, heard)


@pytest.mark.parametrize(
    "name,heard",
    [
        ("Sharma ji", "theek hai dus hazaar jama"),  # D-044: the prompt's example name, supplied
        ("Sharma ji", "pachpan rupay udhar likho"),
        ("Verma ji", "sharma ji ka assi rupay"),  # a different customer is not the same name
    ],
)
def test_names_the_owner_did_not_say_are_refused(name, heard):
    from app.agent.fixes import name_was_heard

    assert not name_was_heard(name, heard)


def test_the_executor_asks_for_the_name_rather_than_supply_one(executor):
    call = ToolCall("add_credit", {"customer_name": "Verma bhaiya", "amount_spoken": "pachpan"})
    reply = executor.handle(call, utterance="pachpan rupay udhar likho")
    assert reply.outcome is not Outcome.CONFIRMING
    assert "Naam phir se" in reply.speech


def test_a_name_from_an_earlier_turn_still_counts(executor):
    """ "unka 200 aur likho" names nobody, but the turn before did. """
    call = ToolCall("add_credit", {"customer_name": "Verma bhaiya", "amount_spoken": "do sau"})
    reply = executor.handle(
        call, utterance="unka do sau aur likho", context="verma bhaiya ka hisaab batao unka do sau aur likho"
    )
    assert reply.outcome is Outcome.CONFIRMING


def test_a_bare_honorific_is_left_to_the_resolver(executor):
    call = ToolCall("add_credit", {"customer_name": "ji", "amount_spoken": "do sau"})
    reply = executor.handle(call, utterance="ji do sau likho")
    assert "Naam phir se" not in reply.speech
