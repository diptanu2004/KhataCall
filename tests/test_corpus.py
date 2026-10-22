"""The eval corpus is ground truth. If it is wrong, every number we report is wrong."""

import csv
from collections import Counter
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "eval" / "corpus" / "scripts.csv"

INTENTS = {"add_credit", "record_payment", "get_balance", "update_stock", "get_stock", "list_low_stock"}
NEEDS_AMOUNT = {"add_credit", "record_payment"}
NEEDS_CUSTOMER = {"add_credit", "record_payment", "get_balance"}
NEEDS_ITEM = {"update_stock", "get_stock"}
HONORIFICS = ("ji", "bhaiya", "didi", "aunty", "uncle", "bhai")


@pytest.fixture(scope="module")
def rows():
    return list(csv.DictReader(SCRIPTS.open(encoding="utf-8")))


def test_ids_are_unique_and_sequential(rows):
    assert [r["id"] for r in rows] == [f"{i:03d}" for i in range(1, len(rows) + 1)]


def test_every_row_has_text_and_a_known_intent(rows):
    for r in rows:
        assert r["text"].strip(), r["id"]
        assert r["intent"] in INTENTS, f"{r['id']}: {r['intent']}"


def test_required_label_columns_are_populated(rows):
    for r in rows:
        if r["intent"] in NEEDS_AMOUNT:
            assert r["amount_paise"], f"{r['id']} needs an amount"
            assert int(r["amount_paise"]) > 0
        if r["intent"] in NEEDS_CUSTOMER:
            assert r["customer"], f"{r['id']} needs a customer"
        if r["intent"] in NEEDS_ITEM:
            assert r["item"], f"{r['id']} needs an item"


def test_amounts_are_whole_paise_and_a_whole_number_of_rupees(rows):
    """Nobody says a paise amount aloud; a stray decimal means a label slipped."""
    for r in rows:
        if r["amount_paise"]:
            paise = int(r["amount_paise"])
            assert paise % 100 == 0, f"{r['id']}: {paise} paise is not whole rupees"


def test_read_only_intents_carry_no_amount(rows):
    for r in rows:
        if r["intent"] in {"get_balance", "get_stock", "list_low_stock"}:
            assert not r["amount_paise"], f"{r['id']} should have no amount"


def test_customer_labels_have_no_honorific(rows):
    """names.py strips honorifics, so the ground truth is the stripped form."""
    for r in rows:
        if r["customer"]:
            last = r["customer"].lower().split()[-1]
            assert last not in HONORIFICS, f"{r['id']}: {r['customer']!r} still has an honorific"


def test_spoken_customer_appears_in_the_text(rows):
    for r in rows:
        if r["customer"]:
            assert r["customer"].lower() in r["text"].lower(), f"{r['id']}: {r['customer']!r} not in text"


def test_every_honorific_is_exercised(rows):
    """names.py must strip all of these, so each needs coverage."""
    text = " ".join(r["text"].lower() for r in rows)
    for honorific in HONORIFICS:
        assert f" {honorific} " in text or f" {honorific}," in text, f"no utterance uses {honorific!r}"


def test_the_hard_number_forms_are_covered(rows):
    """The fraction words and multipliers numbers.py has to get right."""
    text = " ".join(r["text"].lower() for r in rows)
    for form in ("dedh", "dhai", "sava", "saadhe", "paune", "sau", "hazaar", "lakh"):
        assert form in text, f"no utterance uses {form!r}"


def test_digits_and_english_numerals_are_covered(rows):
    """Prisma may return digits, and the shop owner may just speak English."""
    texts = [r["text"].lower() for r in rows]
    assert any(any(ch.isdigit() for ch in t) for t in texts)
    assert any("hundred" in t or "thousand" in t or "fifty" in t for t in texts)


def test_stock_rows_cover_both_directions(rows):
    deltas = [float(r["qty"]) for r in rows if r["intent"] == "update_stock" and r["qty"]]
    assert any(d > 0 for d in deltas), "no stock received"
    assert any(d < 0 for d in deltas), "no stock sold"


def test_the_corpus_is_big_enough_and_balanced(rows):
    counts = Counter(r["intent"] for r in rows)
    assert len(rows) >= 150, "corpus too small to say anything about accuracy"
    assert set(counts) == INTENTS, f"missing intents: {INTENTS - set(counts)}"
    # The money paths are where an error actually costs the shopkeeper.
    assert counts["add_credit"] + counts["record_payment"] >= len(rows) * 0.5


def test_every_row_names_its_speaker(rows):
    """One speaker recorded the whole corpus (D-034). The column must still be
    filled, so a second voice can be added later and reported separately."""
    assert all(r["speaker"].strip() for r in rows)
    assert Counter(r["speaker"] for r in rows) == {"a": len(rows)}


def test_no_real_identifiers_appear(rows):
    """Contest rule: synthetic data only. Catches a pasted phone number or ID."""
    for r in rows:
        digits = "".join(ch for ch in r["text"] if ch.isdigit())
        assert len(digits) <= 6, f"{r['id']}: suspiciously long number {digits!r}"
        assert "@" not in r["text"]
