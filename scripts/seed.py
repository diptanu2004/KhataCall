"""Fill the ledger with a believable shop, for the demo and for poking at the UI.

    uv run python -m scripts.seed              # create khatacall.db if absent
    uv run python -m scripts.seed --reset      # wipe and recreate it
    uv run python -m scripts.seed --db demo.db

Every name here is invented and every number made up. The contest forbids real
customer data anywhere in the project, including seeds and demos (hard rule 4).

The amounts are chosen to make a demo read well: a few customers with round
balances, one settled, one who has overpaid, and two items already below their
reorder threshold so "kya khatam ho raha hai" has something to answer.
No Gnani API calls are made here.
"""

import argparse
from pathlib import Path

from app.config import get_settings
from app.ledger.db import connect
from app.ledger.repo import Ledger

# (name, [(kind, rupees, note), ...])
CUSTOMERS: list[tuple[str, list[tuple[str, int, str | None]]]] = [
    ("Sharma ji", [("credit", 450, "chawal aur dal"), ("payment", 200, None), ("credit", 150, "chini")]),
    ("Gupta aunty", [("credit", 1200, "mahine ka saamaan"), ("payment", 1200, None)]),  # settled
    ("Verma bhaiya", [("credit", 350, "doodh aur biscuit")]),
    ("Yadav ji", [("credit", 2500, "shaadi ka order"), ("payment", 1000, None)]),
    ("Tiwari didi", [("credit", 175, "sabun"), ("credit", 125, "chai patti")]),
    ("Joshi uncle", [("credit", 300, None), ("payment", 400, None)]),  # overpaid
    ("Rawat ji", [("credit", 80, "namak")]),
    ("Chauhan bhai", [("credit", 5000, "thok order"), ("payment", 2500, None)]),
    ("Mehra ji", [("credit", 200, "doodh aur bread")]),  # the demo's part-paid bill lands here
]

# (name, unit, quantity, reorder threshold)
ITEMS: list[tuple[str, str, int, int]] = [
    ("Parle-G", "packet", 48, 20),
    ("Maggi", "packet", 12, 15),      # below threshold
    ("chini", "kilo", 30, 10),
    ("chawal", "kilo", 55, 20),
    ("atta", "kilo", 25, 20),
    ("tel", "litre", 9, 10),          # below threshold
    ("doodh", "packet", 18, 10),
    ("chai patti", "packet", 22, 10),
    ("namak", "packet", 30, 10),
    ("Colgate", "tube", 14, 8),
    ("sabun", "packet", 26, 10),
    ("Lays", "packet", 35, 15),
]


def seed(ledger: Ledger) -> tuple[int, int]:
    customers = items = 0
    for name, entries in CUSTOMERS:
        if ledger.find_customer(name) is not None:
            continue
        customer = ledger.add_customer(name)
        customers += 1
        for kind, rupees, note in entries:
            paise = rupees * 100
            if kind == "credit":
                ledger.add_credit(customer.id, paise, note=note, session_id="seed")
            else:
                ledger.record_payment(customer.id, paise, session_id="seed")

    for name, unit, qty, threshold in ITEMS:
        if ledger.find_item(name) is not None:
            continue
        item = ledger.add_item(name, unit=unit, low_threshold=threshold)
        items += 1
        if qty:
            ledger.apply_movement(item.id, qty, session_id="seed")
    return customers, items


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", default=None, help="database path (default: DB_PATH from .env)")
    parser.add_argument("--reset", action="store_true", help="delete the database first")
    args = parser.parse_args()

    path = Path(args.db or get_settings().db_path)
    if args.reset and path.exists():
        path.unlink()
        for suffix in ("-wal", "-shm"):
            path.with_name(path.name + suffix).unlink(missing_ok=True)
        print(f"removed {path}")

    database = connect(path)
    try:
        ledger = Ledger(database)
        customers, items = seed(ledger)
        snapshot = ledger.snapshot()
    finally:
        database.close()

    print(f"{path}: added {customers} customer(s) and {items} item(s)\n")
    owed = sum(c["balance_paise"] for c in snapshot["customers"] if c["balance_paise"] > 0)
    print(f"  {len(snapshot['customers'])} customers, Rs {owed / 100:,.0f} outstanding")
    for customer in snapshot["customers"][:4]:
        print(f"    {customer['name']:16} Rs {customer['balance_paise'] / 100:>8,.0f}")
    low = [i for i in snapshot["items"] if i["low"]]
    print(f"  {len(snapshot['items'])} items, {len(low)} below threshold: " + ", ".join(i["name"] for i in low))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
