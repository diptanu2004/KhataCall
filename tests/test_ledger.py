"""The ledger: the thing that has to be right when everything else is uncertain."""

import sqlite3

import pytest

from app.ledger.db import connect
from app.ledger.repo import Ledger, LedgerError


@pytest.fixture
def ledger():
    return Ledger(connect(":memory:"))


@pytest.fixture
def sharma(ledger):
    return ledger.add_customer("Sharma ji")


# --- customers --------------------------------------------------------------


def test_a_new_customer_starts_at_zero(ledger):
    customer = ledger.add_customer("Sharma ji")
    assert customer.display_name == "Sharma ji"
    assert ledger.balance_paise(customer.id) == 0


def test_a_customer_is_found_whatever_honorific_is_used(ledger, sharma):
    """The book has one Sharma, however the owner addresses him today."""
    for spoken in ("Sharma ji", "Sharma", "Sharma bhaiya", "sharma"):
        found = ledger.find_customer(spoken)
        assert found is not None and found.id == sharma.id, spoken


def test_the_same_customer_cannot_be_added_twice(ledger, sharma):
    with pytest.raises(LedgerError, match="already in the khata"):
        ledger.add_customer("Sharma bhaiya")


def test_customer_names_are_listed_for_matching(ledger):
    ledger.add_customer("Sharma ji")
    ledger.add_customer("Gupta aunty")
    assert ledger.customer_names() == ["Gupta aunty", "Sharma ji"]


def test_an_empty_name_is_refused(ledger):
    for name in ("", "   ", "ji"):
        with pytest.raises(LedgerError):
            ledger.add_customer(name)


# --- balances ---------------------------------------------------------------


def test_credit_increases_the_balance(ledger, sharma):
    assert ledger.add_credit(sharma.id, 15_000) == 15_000
    assert ledger.add_credit(sharma.id, 25_000) == 40_000


def test_payment_reduces_the_balance(ledger, sharma):
    ledger.add_credit(sharma.id, 50_000)
    assert ledger.record_payment(sharma.id, 20_000) == 30_000


def test_a_customer_can_be_in_credit(ledger, sharma):
    """Paying more than owed is an overpayment, not an error."""
    ledger.add_credit(sharma.id, 10_000)
    assert ledger.record_payment(sharma.id, 15_000) == -5_000


def test_the_balance_is_computed_not_stored(ledger, sharma):
    """Deleting an entry must change the balance, or the two could drift (D-009)."""
    ledger.add_credit(sharma.id, 15_000)
    ledger.add_credit(sharma.id, 10_000)
    assert ledger.balance_paise(sharma.id) == 25_000

    ledger._db.execute("DELETE FROM ledger_entries WHERE amount_paise = 10000")
    ledger._db.commit()
    assert ledger.balance_paise(sharma.id) == 15_000


def test_balances_of_different_customers_do_not_mix(ledger):
    a = ledger.add_customer("Sharma ji")
    b = ledger.add_customer("Gupta aunty")
    ledger.add_credit(a.id, 15_000)
    ledger.add_credit(b.id, 90_000)
    assert (ledger.balance_paise(a.id), ledger.balance_paise(b.id)) == (15_000, 90_000)


# --- money is integer paise (hard rule 6) -----------------------------------


@pytest.mark.parametrize("amount", [0, -1, -15_000])
def test_non_positive_amounts_are_refused(ledger, sharma, amount):
    with pytest.raises(LedgerError):
        ledger.add_credit(sharma.id, amount)


@pytest.mark.parametrize("amount", [150.0, 150.5, "150", True])
def test_only_whole_paise_integers_are_accepted(ledger, sharma, amount):
    """A float here is how rounding errors get into somebody's book."""
    with pytest.raises(LedgerError):
        ledger.add_credit(sharma.id, amount)


def test_the_database_itself_refuses_a_bad_amount(ledger, sharma):
    """Belt and braces: the CHECK holds even if a caller bypasses the repository."""
    with pytest.raises(sqlite3.IntegrityError):
        ledger._db.execute(
            "INSERT INTO ledger_entries (customer_id, kind, amount_paise, created_at)"
            " VALUES (?, 'credit', 0, 'now')",
            (sharma.id,),
        )


def test_the_database_refuses_an_unknown_kind(ledger, sharma):
    with pytest.raises(sqlite3.IntegrityError):
        ledger._db.execute(
            "INSERT INTO ledger_entries (customer_id, kind, amount_paise, created_at)"
            " VALUES (?, 'refund', 100, 'now')",
            (sharma.id,),
        )


def test_an_entry_cannot_belong_to_nobody(ledger):
    """Foreign keys are off by default in SQLite; db.connect turns them on."""
    with pytest.raises(sqlite3.IntegrityError):
        ledger._db.execute(
            "INSERT INTO ledger_entries (customer_id, kind, amount_paise, created_at)"
            " VALUES (999, 'credit', 100, 'now')"
        )


def test_entries_are_listed_newest_first(ledger, sharma):
    ledger.add_credit(sharma.id, 15_000, note="chawal")
    ledger.record_payment(sharma.id, 5_000)
    entries = ledger.entries(sharma.id)
    assert [e["kind"] for e in entries] == ["payment", "credit"]
    assert entries[1]["note"] == "chawal"


def test_the_session_that_made_a_write_is_recorded(ledger, sharma):
    ledger.add_credit(sharma.id, 15_000, session_id="call-7")
    assert ledger.entries(sharma.id)[0]["session_id"] == "call-7"


# --- stock ------------------------------------------------------------------


@pytest.fixture
def maggi(ledger):
    return ledger.add_item("Maggi", unit="packet", low_threshold=5)


def test_stock_arrives_and_leaves(ledger, maggi):
    assert ledger.apply_movement(maggi.id, 20) == 20
    assert ledger.apply_movement(maggi.id, -8) == 12


def test_stock_cannot_go_negative(ledger, maggi):
    ledger.apply_movement(maggi.id, 3)
    with pytest.raises(LedgerError, match="only 3 packet"):
        ledger.apply_movement(maggi.id, -10)
    assert ledger.find_item("Maggi").qty == 3, "the failed movement must leave no trace"


def test_quantity_and_movements_stay_in_step(ledger, maggi):
    for delta in (20, -5, 10, -2):
        ledger.apply_movement(maggi.id, delta)
    total = ledger._db.execute(
        "SELECT SUM(qty_delta) AS total FROM stock_movements WHERE item_id = ?", (maggi.id,)
    ).fetchone()["total"]
    assert ledger.find_item("Maggi").qty == total == 23


def test_a_zero_movement_is_refused(ledger, maggi):
    with pytest.raises(LedgerError, match="would not change anything"):
        ledger.apply_movement(maggi.id, 0)


def test_fractional_stock_is_refused(ledger, maggi):
    """Loose goods are a known gap, not a rounding opportunity (D-024)."""
    with pytest.raises(LedgerError, match="whole units"):
        ledger.apply_movement(maggi.id, 3.5)


def test_an_unknown_item_is_refused(ledger):
    with pytest.raises(LedgerError, match="do not have that item"):
        ledger.apply_movement(999, 5)


def test_items_are_found_despite_spacing_and_case(ledger):
    ledger.add_item("chai patti")
    for spoken in ("chai patti", "Chai Patti", "chaipatti"):
        assert ledger.find_item(spoken) is not None, spoken


def test_low_stock_uses_each_items_own_threshold(ledger):
    maggi = ledger.add_item("Maggi", low_threshold=5)
    atta = ledger.add_item("atta", low_threshold=20)
    ledger.apply_movement(maggi.id, 4)   # below its threshold
    ledger.apply_movement(atta.id, 25)   # above its own, though a bigger number

    assert [i.display_name for i in ledger.low_stock()] == ["Maggi"]


def test_low_stock_accepts_a_spoken_threshold(ledger):
    maggi = ledger.add_item("Maggi", low_threshold=5)
    atta = ledger.add_item("atta", low_threshold=5)
    ledger.apply_movement(maggi.id, 4)
    ledger.apply_movement(atta.id, 9)

    assert [i.display_name for i in ledger.low_stock(10)] == ["Maggi", "atta"]


# --- the UI panel -----------------------------------------------------------


def test_snapshot_shows_customers_by_debt_and_flags_low_stock(ledger):
    a = ledger.add_customer("Sharma ji")
    b = ledger.add_customer("Gupta aunty")
    ledger.add_credit(a.id, 15_000)
    ledger.add_credit(b.id, 90_000)
    maggi = ledger.add_item("Maggi", low_threshold=5)
    ledger.apply_movement(maggi.id, 3)

    snapshot = ledger.snapshot()

    assert [c["name"] for c in snapshot["customers"]] == ["Gupta aunty", "Sharma ji"]
    assert snapshot["customers"][0]["balance_paise"] == 90_000
    assert snapshot["items"][0] == {"name": "Maggi", "qty": 3, "unit": "packet", "low": True}


def test_snapshot_of_an_empty_book(ledger):
    assert ledger.snapshot() == {"customers": [], "items": []}
