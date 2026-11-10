"""The confirmation gate: the only path from a model's proposal to the ledger.

Most of these tests are about what does *not* happen. The LLM never writes
(hard rule 5), nothing is written before the shop owner agrees, and anything
uncertain becomes a question instead of a default.
"""

from dataclasses import replace

import pytest

from app.adapters.base import ToolCall
from app.agent.executor import Executor, Outcome, rupees, spell_digits
from app.agent.fixes import ALL
from app.ledger.db import connect
from app.ledger.repo import Ledger


@pytest.fixture
def ledger():
    return Ledger(connect(":memory:"))


@pytest.fixture
def seeded(ledger):
    sharma = ledger.add_customer("Sharma ji")
    ledger.add_customer("Gupta aunty")
    ledger.add_credit(sharma.id, 100_000)
    maggi = ledger.add_item("Maggi", unit="packet", low_threshold=5)
    ledger.apply_movement(maggi.id, 12)
    return ledger


@pytest.fixture
def agent(seeded):
    return Executor(seeded, session_id="call-1")


def call(name: str, **arguments) -> ToolCall:
    return ToolCall(name=name, arguments=arguments)


def credit(customer="Sharma ji", amount="dedh sau"):
    return call("add_credit", customer_name=customer, amount_spoken=amount)


# --- the headline path ------------------------------------------------------


def test_a_credit_is_read_back_then_committed(agent, seeded):
    sharma = seeded.find_customer("Sharma ji")

    asked = agent.handle(credit())
    assert asked.outcome is Outcome.CONFIRMING
    assert "150 rupaye" in asked.speech
    assert "Sharma ji" in asked.speech
    assert seeded.balance_paise(sharma.id) == 100_000, "nothing may be written before yes"

    done = agent.handle_user_text("haan")
    assert done.outcome is Outcome.COMMITTED
    assert done.ledger_changed
    assert seeded.balance_paise(sharma.id) == 115_000
    assert "1,150 rupaye" in done.speech


def test_saying_no_writes_nothing(agent, seeded):
    sharma = seeded.find_customer("Sharma ji")
    agent.handle(credit())

    cancelled = agent.handle_user_text("nahi")

    assert cancelled.outcome is Outcome.CANCELLED
    assert not cancelled.ledger_changed
    assert seeded.balance_paise(sharma.id) == 100_000
    assert not agent.has_pending


def test_a_payment_reduces_the_balance(agent, seeded):
    sharma = seeded.find_customer("Sharma ji")
    agent.handle(call("record_payment", customer_name="Sharma ji", amount_spoken="saadhe teen sau"))
    agent.handle_user_text("haan")
    assert seeded.balance_paise(sharma.id) == 100_000 - 35_000


def test_the_read_back_comes_from_the_parsed_amount_not_the_words(agent):
    """D-005: what the owner hears is the number that will be written."""
    asked = agent.handle(credit(amount="one fifty"))
    assert "150 rupaye" in asked.speech
    assert "one fifty" not in asked.speech


# --- yes and no -------------------------------------------------------------


@pytest.mark.parametrize("reply", ["haan", "han", "haan ji", "ji", "yes", "theek hai", "likh do", "ok"])
def test_agreement_is_recognised(agent, reply):
    agent.handle(credit())
    assert agent.handle_user_text(reply).outcome is Outcome.COMMITTED


@pytest.mark.parametrize("reply", ["nahi", "nahin", "no", "cancel", "rehne do", "mat karo"])
def test_refusal_is_recognised(agent, reply):
    agent.handle(credit())
    assert agent.handle_user_text(reply).outcome is Outcome.CANCELLED


def test_a_refusal_that_contains_an_agreement_phrase_is_still_a_refusal(agent):
    """"rehne do" ends in "do"; "mat likho" contains "likho". Neither is a yes."""
    for reply in ("rehne do", "mat likho", "abhi mat karo"):
        agent.handle(credit())
        assert agent.handle_user_text(reply).outcome is Outcome.CANCELLED, reply


def test_a_contradictory_reply_asks_again(agent):
    """"nahi likh do" says both things. Guessing either way is a coin flip."""
    agent.handle(credit())
    assert agent.handle_user_text("nahi likh do").outcome is Outcome.CONFIRMING


def test_an_unclear_reply_is_asked_once_more_then_cancelled(agent, seeded):
    """Never commit on a reply we could not read. Cancelling loses nothing."""
    sharma = seeded.find_customer("Sharma ji")
    agent.handle(credit())

    again = agent.handle_user_text("matlab")
    assert again.outcome is Outcome.CONFIRMING
    assert "Haan ya nahi" in again.speech
    assert agent.has_pending

    given_up = agent.handle_user_text("pata nahi")
    assert given_up.outcome is Outcome.CANCELLED
    assert seeded.balance_paise(sharma.id) == 100_000


def test_a_reply_containing_both_yes_and_no_is_not_guessed(agent):
    agent.handle(credit())
    assert agent.handle_user_text("haan nahi ruko").outcome is Outcome.CONFIRMING


def test_replies_are_ignored_when_nothing_is_pending(agent):
    assert agent.handle_user_text("haan") is None


def test_a_new_instruction_replaces_a_pending_one(agent, seeded):
    """The owner moved on. The old proposal must not survive to be committed."""
    agent.handle(credit(amount="dedh sau"))
    agent.handle(credit(amount="do sau"))

    agent.handle_user_text("haan")

    assert seeded.balance_paise(seeded.find_customer("Sharma ji").id) == 100_000 + 20_000


# --- amounts ----------------------------------------------------------------


def test_an_unparseable_amount_is_asked_about_not_guessed(agent):
    reply = agent.handle(credit(amount="thoda sa"))
    assert reply.outcome is Outcome.ANSWERED
    assert "Kitne rupaye" in reply.speech
    assert not agent.has_pending


def test_phone_mode_spells_the_amount_out(seeded):
    """No STT confidence exists to key off, so phone mode always spells (D-004)."""
    agent = Executor(seeded, spell_amounts=True)
    asked = agent.handle(credit())
    assert "ek paanch shunya" in asked.speech
    assert "150 rupaye" in asked.speech


# --- a bill paid in part: both amounts read from the owner's words, code subtracts ---

PART_PAID = "Sharma ji ka bill saadhe chhe sau ka hua, paanch sau diye, baaki likh do"


def part_paid(bill="saadhe chhe sau", paid="paanch sau"):
    return call("add_credit", customer_name="Sharma ji", amount_spoken=bill, paid_spoken=paid)


@pytest.fixture
def without_fix_6(seeded):
    """The executor's own checks on a part paid, with fix 6 not filling it from the words."""
    return Executor(seeded, fixes=replace(ALL, part_paid=False))


@pytest.mark.parametrize(
    "proposed",
    [
        credit(amount="saadhe chhe sau"),  # what real Evon proposes: the bill only (Nov 10)
        credit(amount="paanch sau"),  # the part paid passed as the bill
        call("record_payment", customer_name="Sharma ji", amount_spoken="paanch sau"),
    ],
    ids=["bill only", "part paid as the bill", "proposed as a payment"],
)
def test_whatever_evon_makes_of_a_part_paid_bill_the_read_back_comes_from_the_owners_words(agent, proposed):
    asked = agent.handle(proposed, utterance=PART_PAID)
    assert asked.speech == "Sharma ji ka bill 650 rupaye, 500 rupaye mile. Baaki 150 rupaye udhaar likh doon?"


def test_a_bill_paid_in_part_puts_only_the_rest_on_credit(agent, seeded):
    sharma = seeded.find_customer("Sharma ji")
    asked = agent.handle(part_paid(), utterance=PART_PAID)
    assert asked.outcome is Outcome.CONFIRMING
    assert asked.speech == "Sharma ji ka bill 650 rupaye, 500 rupaye mile. Baaki 150 rupaye udhaar likh doon?"
    assert seeded.balance_paise(sharma.id) == 100_000, "nothing may be written before yes"

    assert agent.handle_user_text("haan").outcome is Outcome.COMMITTED
    assert seeded.balance_paise(sharma.id) == 115_000
    assert seeded.entries(sharma.id)[0]["note"] == "bill 650 rupaye, 500 rupaye mile"


def test_phone_mode_spells_out_the_amount_that_will_be_written(seeded):
    asked = Executor(seeded, spell_amounts=True).handle(part_paid(), utterance=PART_PAID)
    assert "Baaki ek paanch shunya, 150 rupaye udhaar" in asked.speech


def test_the_part_paid_must_be_one_the_owner_said(without_fix_6):
    reply = without_fix_6.handle(part_paid(paid="teen sau"), utterance=PART_PAID)
    assert reply.speech == "Kitne diye? Amount samajh nahi aaya."
    assert not without_fix_6.has_pending


@pytest.mark.parametrize("bill, paid", [("paanch sau", "paanch sau"), ("paanch sau", "saadhe chhe sau")])
def test_paying_the_whole_bill_or_more_is_asked_again_not_subtracted(without_fix_6, bill, paid):
    reply = without_fix_6.handle(part_paid(bill=bill, paid=paid), utterance=PART_PAID)
    assert reply.speech == "Bill kitne ka tha, aur kitne diye? Phir se boliye."
    assert not without_fix_6.has_pending


def test_a_part_paid_bill_where_more_was_paid_than_owed_is_asked_again(agent):
    said = "Sharma ji ka bill paanch sau ka tha, saat sau diye, baaki likh do"
    reply = agent.handle(credit(amount="paanch sau"), utterance=said)
    assert reply.speech == "Bill kitne ka tha, aur kitne diye? Phir se boliye."


def test_a_null_word_for_the_part_paid_is_an_ordinary_credit(agent):
    assert agent.handle(part_paid(bill="dedh sau", paid="None")).speech == (
        "Sharma ji ke khate mein 150 rupaye udhaar likh doon?"
    )


def test_spell_digits_and_rupees():
    assert spell_digits(15_000) == "ek paanch shunya"
    assert rupees(15_000) == "150 rupaye"
    assert rupees(115_000) == "1,150 rupaye"


# --- customers --------------------------------------------------------------


def test_a_new_customer_is_flagged_in_the_read_back(agent, seeded):
    asked = agent.handle(credit(customer="Chaturvedi ji"))

    assert "naya customer" in asked.speech
    assert seeded.find_customer("Chaturvedi ji") is None, "not created before yes"

    agent.handle_user_text("haan")
    created = seeded.find_customer("Chaturvedi ji")
    assert created is not None and created.balance_paise == 15_000


def test_refusing_a_new_customer_creates_nobody(agent, seeded):
    agent.handle(credit(customer="Chaturvedi ji"))
    agent.handle_user_text("nahi")
    assert seeded.find_customer("Chaturvedi ji") is None


def test_an_ambiguous_customer_is_asked_about(ledger):
    """Two real people, equally close. Picking one would debit the wrong person."""
    ledger.add_customer("Sharma ji")
    ledger.add_customer("Varma ji")
    agent = Executor(ledger)

    reply = agent.handle(credit(customer="arma"))

    assert reply.outcome is Outcome.ANSWERED
    assert "Sharma ji" in reply.speech and "Varma ji" in reply.speech
    assert not agent.has_pending, "nothing may be pending on an unresolved name"


def test_an_honorific_variant_reaches_the_same_customer(agent, seeded):
    agent.handle(credit(customer="Sharma bhaiya"))
    agent.handle_user_text("haan")
    assert seeded.balance_paise(seeded.find_customer("Sharma ji").id) == 115_000


# --- reads ------------------------------------------------------------------


def test_balance_is_reported(agent):
    reply = agent.handle(call("get_balance", customer_name="Sharma ji"))
    assert "1,000 rupaye" in reply.speech
    assert reply.outcome is Outcome.ANSWERED


def test_a_settled_customer_is_reported_as_settled(ledger):
    ledger.add_customer("Sharma ji")
    reply = Executor(ledger).handle(call("get_balance", customer_name="Sharma ji"))
    assert "chukta" in reply.speech


def test_balance_of_an_unknown_customer_does_not_create_them(agent, seeded):
    reply = agent.handle(call("get_balance", customer_name="Chaturvedi ji"))
    assert "khata nahi hai" in reply.speech
    assert seeded.find_customer("Chaturvedi ji") is None


def test_a_read_never_leaves_anything_pending(agent):
    agent.handle(call("get_balance", customer_name="Sharma ji"))
    assert not agent.has_pending


def test_stock_is_reported(agent):
    assert "12 packet" in agent.handle(call("get_stock", item="Maggi")).speech


def test_low_stock_lists_what_is_running_out(seeded):
    atta = seeded.add_item("atta", low_threshold=10)
    seeded.apply_movement(atta.id, 2)
    reply = Executor(seeded).handle(call("list_low_stock"))
    assert "atta" in reply.speech and "Maggi" not in reply.speech


def test_low_stock_accepts_a_spoken_threshold(seeded):
    reply = Executor(seeded).handle(call("list_low_stock", threshold_spoken="pandrah"))
    assert "Maggi" in reply.speech


def test_low_stock_with_nothing_low(ledger):
    assert "Sab theek hai" in Executor(ledger).handle(call("list_low_stock")).speech


# --- stock writes -----------------------------------------------------------


def test_stock_arriving_is_confirmed_then_committed(agent, seeded):
    asked = agent.handle(call("update_stock", item="Maggi", qty_delta_spoken="bees"))
    assert "20 packet" in asked.speech and "aaye" in asked.speech
    assert seeded.find_item("Maggi").qty == 12

    agent.handle_user_text("haan")
    assert seeded.find_item("Maggi").qty == 32


def test_stock_going_out_is_negative(agent, seeded):
    agent.handle(call("update_stock", item="Maggi", qty_delta_spoken="minus paanch"))
    agent.handle_user_text("haan")
    assert seeded.find_item("Maggi").qty == 7


def test_a_new_item_is_flagged_in_the_read_back(agent, seeded):
    asked = agent.handle(call("update_stock", item="Lays", qty_delta_spoken="das"))
    assert "naya item" in asked.speech
    assert seeded.find_item("Lays") is None

    agent.handle_user_text("haan")
    assert seeded.find_item("Lays").qty == 10


def test_stock_cannot_leave_an_item_we_do_not_have(agent):
    reply = agent.handle(call("update_stock", item="Lays", qty_delta_spoken="minus das"))
    assert "stock list mein nahi" in reply.speech
    assert not agent.has_pending


def test_fractional_quantities_are_refused_not_rounded(agent):
    """Loose goods are a known gap (D-024, risk R6), not a rounding opportunity."""
    reply = agent.handle(call("update_stock", item="chini", qty_delta_spoken="saadhe teen"))
    assert "poore number" in reply.speech
    assert not agent.has_pending


def test_dedh_darjan_is_eighteen(agent, seeded):
    agent.handle(call("update_stock", item="Maggi", qty_delta_spoken="dedh darjan"))
    agent.handle_user_text("haan")
    assert seeded.find_item("Maggi").qty == 12 + 18


def test_overselling_is_refused_at_commit(agent, seeded):
    agent.handle(call("update_stock", item="Maggi", qty_delta_spoken="minus sau"))
    reply = agent.handle_user_text("haan")
    assert not reply.ledger_changed
    assert seeded.find_item("Maggi").qty == 12


# --- bad input --------------------------------------------------------------


def test_an_unknown_tool_is_declined(agent):
    reply = agent.handle(call("delete_everything", target="all"))
    assert reply.outcome is Outcome.ANSWERED
    assert not agent.has_pending


def test_malformed_arguments_are_declined(agent):
    for bad in (call("add_credit"), call("add_credit", customer_name="", amount_spoken="dedh sau")):
        reply = agent.handle(bad)
        assert reply.outcome is Outcome.ANSWERED
        assert not agent.has_pending


def test_the_session_is_recorded_on_the_entry(agent, seeded):
    agent.handle(credit())
    agent.handle_user_text("haan")
    entry = seeded.entries(seeded.find_customer("Sharma ji").id)[0]
    assert entry["session_id"] == "call-1"


def test_a_none_string_threshold_lists_everything_low(seeded):
    """Real Evon sent threshold_spoken="None"; that must not trigger "kitne se kam?"."""
    reply = Executor(seeded).handle(call("list_low_stock", threshold_spoken="None"))
    assert "Kitne se kam" not in reply.speech


# --- a yes must be only a yes (D-047) ------------------------------------------


@pytest.mark.parametrize(
    "reply",
    [
        "Gupta ji ka do sau likho",  # a new instruction naming someone else
        "sharma ji",  # a fragment: "ji" alone is a yes, a name with it is not
        "sahi amount do sau hai",  # a correction
        "ok gupta ji ka paanch sau",
    ],
)
def test_a_sentence_containing_a_yes_word_does_not_commit(agent, seeded, reply):
    sharma = seeded.find_customer("Sharma ji")
    agent.handle(credit())
    assert agent.handle_user_text(reply).outcome is Outcome.CONFIRMING  # asked again
    assert seeded.balance_paise(sharma.id) == 100_000


@pytest.mark.parametrize("reply", ["ji", "haan ji likh do", "theek hai bhai", "ji haan kar do", "ok"])
def test_plain_agreement_commits(agent, reply):
    agent.handle(credit())
    assert agent.handle_user_text(reply).outcome is Outcome.COMMITTED


@pytest.mark.parametrize("reply", ["ji nahi", "nahi ji"])
def test_a_polite_no_cancels(agent, reply):
    agent.handle(credit())
    assert agent.handle_user_text(reply).outcome is Outcome.CANCELLED


def test_large_amounts_are_read_back_in_lakhs(agent):
    reply = agent.handle(credit(amount="dedh lakh"))
    assert "1,50,000 rupaye" in reply.speech


def test_a_new_customer_is_written_with_a_capital_whatever_evon_typed(agent, seeded):
    agent.handle(credit(customer="mishra ji"))
    agent.handle_user_text("haan")
    assert "Mishra ji" in seeded.customer_names()


def test_one_near_miss_asks_for_the_name_rather_than_offering_a_choice_of_one(agent):
    """ "Shrama" is close to Sharma ji but not close enough to assume (D-023)."""
    reply = agent.handle(credit(customer="Shrama"))
    assert reply.speech == "Sharma ji? Naam phir se boliye."
    assert not agent.has_pending
