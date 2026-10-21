"""Scoring a proposed tool call. The eval's numbers are only as honest as this."""

import pytest

from eval.metrics import Outcome, score_call

KNOWN = ["Sharma", "Verma", "Gupta", "Yadav", "Varma"]


def row(**overrides) -> dict[str, str]:
    base = {
        "id": "001",
        "text": "Sharma ji ka dedh sau udhaar likho",
        "intent": "add_credit",
        "customer": "Sharma",
        "amount_paise": "15000",
        "item": "",
        "qty": "",
        "speaker": "a",
    }
    return base | overrides


def credit(customer="Sharma ji", amount="dedh sau"):
    return "add_credit", {"customer_name": customer, "amount_spoken": amount}


def score(r, call):
    name, args = call if call else (None, None)
    return score_call(r, name, args, known_customers=KNOWN)


# --- correct ----------------------------------------------------------------


def test_the_right_proposal_is_correct():
    s = score(row(), credit())
    assert s.outcome is Outcome.CORRECT
    assert (s.intent_ok, s.customer_ok, s.amount_ok) == (True, True, True)


def test_an_honorific_variant_still_reaches_the_right_customer():
    assert score(row(), credit(customer="Sharma bhaiya")).correct


def test_digits_that_parse_to_the_right_amount_are_correct_but_flagged():
    """Rule 7 says the model should not convert. The right number still lands,
    so it is correct - but the conversion is counted, because it is luck."""
    s = score(row(), credit(amount="150"))
    assert s.correct
    assert s.converted_to_digits


def test_digits_the_owner_actually_said_are_not_a_conversion():
    r = row(text="Tiwari ji ka 2 hazaar 500 udhaar", customer="Sharma", amount_paise="250000")
    s = score(r, credit(amount="2 hazaar 500"))
    assert s.correct
    assert not s.converted_to_digits


# --- caught: the agent would ask, nothing wrong is written --------------------


def test_no_tool_call_is_caught():
    s = score(row(), None)
    assert s.outcome is Outcome.CAUGHT
    assert "no tool call" in s.reasons


def test_an_unparseable_amount_is_caught():
    s = score(row(), credit(amount="thoda sa"))
    assert s.outcome is Outcome.CAUGHT
    assert "amount unparseable" in s.reasons


def test_an_ambiguous_customer_is_caught_not_wrong():
    """'arma' is equally close to Sharma and Varma: the executor asks which."""
    s = score(row(), credit(customer="arma ji"))
    assert s.outcome is Outcome.CAUGHT


def test_an_unknown_customer_is_caught():
    """The executor would flag a new customer in the read-back."""
    assert score(row(), credit(customer="Chaturvedi ji")).outcome is Outcome.CAUGHT


# --- wrong: a confident bad value reaches the read-back -----------------------


def test_a_wrong_amount_is_wrong():
    s = score(row(), credit(amount="dhai sau"))
    assert s.outcome is Outcome.WRONG
    assert any("amount 25000" in r for r in s.reasons)


def test_a_wrong_customer_is_wrong():
    s = score(row(), credit(customer="Gupta ji"))
    assert s.outcome is Outcome.WRONG
    assert any("customer" in r for r in s.reasons)


def test_a_wrong_intent_is_wrong():
    """A payment recorded as credit moves the balance the wrong way."""
    s = score(row(), ("record_payment", {"customer_name": "Sharma ji", "amount_spoken": "dedh sau"}))
    assert s.outcome is Outcome.WRONG


def test_wrong_outranks_caught_when_both_happen():
    """A confident wrong customer is dangerous even if the amount was caught."""
    s = score(row(), credit(customer="Gupta ji", amount="thoda sa"))
    assert s.outcome is Outcome.WRONG


# --- reads ------------------------------------------------------------------


def test_a_balance_query_needs_only_intent_and_customer():
    r = row(text="Sharma ji ka kitna baaki hai", intent="get_balance", amount_paise="")
    s = score(r, ("get_balance", {"customer_name": "Sharma ji"}))
    assert s.correct
    assert s.amount_ok is None


# --- stock ------------------------------------------------------------------


def stock_row(**overrides):
    return row(
        text="Maggi ke das packet bik gaye",
        intent="update_stock",
        customer="",
        amount_paise="",
        item="Maggi",
        qty="-10",
    ) | overrides


def test_stock_going_out_must_be_negative():
    assert score(stock_row(), ("update_stock", {"item": "Maggi", "qty_delta_spoken": "minus das"})).correct


def test_stock_with_the_sign_missing_is_wrong():
    """Recording a sale as a delivery moves stock the wrong way by twice the amount."""
    s = score(stock_row(), ("update_stock", {"item": "Maggi", "qty_delta_spoken": "das"}))
    assert s.outcome is Outcome.WRONG


def test_item_names_compare_without_spaces_or_case():
    r = stock_row(item="chai patti", qty="40", text="chai patti ke chaalees packet aaye")
    assert score(r, ("update_stock", {"item": "Chai Patti", "qty_delta_spoken": "chaalees"})).correct


def test_a_fractional_quantity_scores_on_value():
    """Row 160. The executor refuses it (D-024), but the *proposal* can be right."""
    r = stock_row(item="chini", qty="-3.5", text="chini ke saadhe teen kilo bik gaye")
    assert score(r, ("update_stock", {"item": "chini", "qty_delta_spoken": "minus saadhe teen"})).correct


# --- low stock --------------------------------------------------------------


def low_row(qty=""):
    return row(text="kya kya khatam hone wala hai", intent="list_low_stock", customer="", amount_paise="", qty=qty)


def test_low_stock_without_a_threshold():
    assert score(low_row(), ("list_low_stock", {})).correct


def test_an_invented_threshold_is_wrong():
    assert score(low_row(), ("list_low_stock", {"threshold_spoken": "das"})).outcome is Outcome.WRONG


def test_a_spoken_threshold_must_match():
    assert score(low_row("5"), ("list_low_stock", {"threshold_spoken": "paanch"})).correct
    assert score(low_row("5"), ("list_low_stock", {"threshold_spoken": "das"})).outcome is Outcome.WRONG


@pytest.mark.parametrize("missing", [{}, {"threshold_spoken": ""}])
def test_a_missing_spoken_threshold_is_caught(missing):
    assert score(low_row("5"), ("list_low_stock", missing)).outcome is Outcome.CAUGHT
