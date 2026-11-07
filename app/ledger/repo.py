"""Reading and writing the ledger.

Every write goes through here, and `agent/executor.py` is the only caller. The
LLM never reaches this layer (hard rule 5): it proposes a tool call, the
executor validates it, and nothing lands until the shop owner says yes.

Amounts are integer paise throughout (hard rule 6). Nothing in this file
returns a float.
"""

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime

from app.nlu.names import has_real_name, normalise_name
from app.nlu.numbers import normalise

__all__ = [
    "Customer",
    "Item",
    "Ledger",
    "LedgerError",
]


class LedgerError(RuntimeError):
    """A write that the ledger will not make, with a reason worth speaking aloud."""


@dataclass(frozen=True)
class Customer:
    id: int
    display_name: str
    balance_paise: int = 0


@dataclass(frozen=True)
class Item:
    id: int
    display_name: str
    unit: str
    qty: int
    low_threshold: int


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class Ledger:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self._db = connection

    # --- customers ---

    def customer_names(self) -> list[str]:
        """Every customer as the owner says them, for `nlu.names.match_name`."""
        rows = self._db.execute("SELECT display_name FROM customers ORDER BY display_name")
        return [row["display_name"] for row in rows]

    def find_customer(self, display_name: str) -> Customer | None:
        row = self._db.execute(
            "SELECT id, display_name FROM customers WHERE name_norm = ?",
            (normalise_name(display_name),),
        ).fetchone()
        if row is None:
            return None
        return Customer(row["id"], row["display_name"], self.balance_paise(row["id"]))

    def customer_by_id(self, customer_id: int) -> Customer | None:
        row = self._db.execute(
            "SELECT id, display_name FROM customers WHERE id = ?", (customer_id,)
        ).fetchone()
        if row is None:
            return None
        return Customer(row["id"], row["display_name"], self.balance_paise(row["id"]))

    def add_customer(self, display_name: str) -> Customer:
        """Create a customer. The caller must already have asked the owner."""
        display_name = display_name.strip()
        key = normalise_name(display_name)
        # "ji" normalises to a usable key but names nobody, so it would quietly
        # become a customer called "ji" that every later lookup half-matches.
        if not key or not has_real_name(display_name):
            raise LedgerError("that is not a name I can record")
        if self.find_customer(display_name) is not None:
            raise LedgerError(f"{display_name} is already in the khata")
        cursor = self._db.execute(
            "INSERT INTO customers (display_name, name_norm, created_at) VALUES (?, ?, ?)",
            (display_name, key, _now()),
        )
        self._db.commit()
        return Customer(int(cursor.lastrowid), display_name, 0)

    def balance_paise(self, customer_id: int) -> int:
        """What the customer owes: credits minus payments, computed every time (D-009)."""
        row = self._db.execute(
            """
            SELECT COALESCE(SUM(CASE kind WHEN 'credit' THEN amount_paise ELSE -amount_paise END), 0)
                   AS balance
            FROM ledger_entries WHERE customer_id = ?
            """,
            (customer_id,),
        ).fetchone()
        return int(row["balance"])

    # --- ledger entries ---

    def add_credit(
        self, customer_id: int, amount_paise: int, *, note: str | None = None, session_id: str | None = None
    ) -> int:
        """Record udhaar. Returns the customer's new balance in paise."""
        return self._add_entry(customer_id, "credit", amount_paise, note, session_id)

    def record_payment(
        self, customer_id: int, amount_paise: int, *, note: str | None = None, session_id: str | None = None
    ) -> int:
        """Record money received. Returns the customer's new balance in paise."""
        return self._add_entry(customer_id, "payment", amount_paise, note, session_id)

    def _add_entry(
        self, customer_id: int, kind: str, amount_paise: int, note: str | None, session_id: str | None
    ) -> int:
        if not isinstance(amount_paise, int) or isinstance(amount_paise, bool):
            raise LedgerError("amounts must be whole paise")
        if amount_paise <= 0:
            raise LedgerError("an amount has to be more than zero")
        try:
            self._db.execute(
                """
                INSERT INTO ledger_entries (customer_id, kind, amount_paise, note, session_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (customer_id, kind, amount_paise, note, session_id, _now()),
            )
        except sqlite3.IntegrityError as e:
            self._db.rollback()
            raise LedgerError(f"could not write that entry: {e}") from e
        self._db.commit()
        return self.balance_paise(customer_id)

    def entries(self, customer_id: int, limit: int = 20) -> list[sqlite3.Row]:
        return list(
            self._db.execute(
                """
                SELECT id, kind, amount_paise, note, session_id, created_at
                FROM ledger_entries WHERE customer_id = ?
                ORDER BY id DESC LIMIT ?
                """,
                (customer_id, limit),
            )
        )

    # --- stock ---

    def item_names(self) -> list[str]:
        rows = self._db.execute("SELECT display_name FROM items ORDER BY display_name")
        return [row["display_name"] for row in rows]

    def find_item(self, display_name: str) -> Item | None:
        row = self._db.execute(
            "SELECT id, display_name, unit, qty, low_threshold FROM items WHERE name_norm = ?",
            (normalise(display_name.replace(" ", "")),),
        ).fetchone()
        return self._item(row) if row else None

    def item_by_id(self, item_id: int) -> Item | None:
        row = self._db.execute(
            "SELECT id, display_name, unit, qty, low_threshold FROM items WHERE id = ?", (item_id,)
        ).fetchone()
        return self._item(row) if row else None

    def add_item(self, display_name: str, *, unit: str = "packet", low_threshold: int = 5) -> Item:
        display_name = display_name.strip()
        key = normalise(display_name.replace(" ", ""))
        if not key:
            raise LedgerError("that is not an item name I can record")
        if self.find_item(display_name) is not None:
            raise LedgerError(f"{display_name} is already in the stock list")
        cursor = self._db.execute(
            """
            INSERT INTO items (display_name, name_norm, unit, qty, low_threshold, created_at)
            VALUES (?, ?, ?, 0, ?, ?)
            """,
            (display_name, key, unit, low_threshold, _now()),
        )
        self._db.commit()
        return Item(int(cursor.lastrowid), display_name, unit, 0, low_threshold)

    def apply_movement(self, item_id: int, qty_delta: int, *, session_id: str | None = None) -> int:
        """Add or remove stock in one transaction. Returns the new quantity.

        The movement and the running total are written together, so `items.qty`
        can never disagree with the movements that produced it.
        """
        if not isinstance(qty_delta, int) or isinstance(qty_delta, bool):
            raise LedgerError("stock changes have to be whole units")
        if qty_delta == 0:
            raise LedgerError("that would not change anything")

        row = self._db.execute("SELECT display_name, qty, unit FROM items WHERE id = ?", (item_id,)).fetchone()
        if row is None:
            raise LedgerError("I do not have that item")
        new_qty = row["qty"] + qty_delta
        if new_qty < 0:
            # Spoken to the owner as is, so in Hinglish.
            raise LedgerError(
                f"{row['display_name']} ke sirf {row['qty']} {row['unit']} hain, {-qty_delta} nahi nikal sakte"
            )

        try:
            with self._db:  # one transaction: both rows or neither
                self._db.execute(
                    """
                    INSERT INTO stock_movements (item_id, qty_delta, session_id, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (item_id, qty_delta, session_id, _now()),
                )
                self._db.execute("UPDATE items SET qty = ? WHERE id = ?", (new_qty, item_id))
        except sqlite3.IntegrityError as e:
            raise LedgerError(f"could not record that stock change: {e}") from e
        return new_qty

    def low_stock(self, threshold: int | None = None) -> list[Item]:
        """Items at or below their threshold, or below `threshold` if one is given."""
        if threshold is None:
            rows = self._db.execute(
                """
                SELECT id, display_name, unit, qty, low_threshold FROM items
                WHERE qty <= low_threshold ORDER BY qty, display_name
                """
            )
        else:
            rows = self._db.execute(
                """
                SELECT id, display_name, unit, qty, low_threshold FROM items
                WHERE qty <= ? ORDER BY qty, display_name
                """,
                (threshold,),
            )
        return [self._item(row) for row in rows]

    # --- for the UI panel ---

    def snapshot(self) -> dict[str, list[dict]]:
        """Everything the live ledger panel shows, in one read."""
        customers = [
            {
                "name": row["display_name"],
                "balance_paise": self.balance_paise(row["id"]),
            }
            for row in self._db.execute("SELECT id, display_name FROM customers")
        ]
        customers.sort(key=lambda c: (-c["balance_paise"], c["name"]))
        items = [
            {
                "name": item.display_name,
                "qty": item.qty,
                "unit": item.unit,
                "low": item.qty <= item.low_threshold,
            }
            for item in (self._item(row) for row in self._db.execute(
                "SELECT id, display_name, unit, qty, low_threshold FROM items ORDER BY display_name"
            ))
        ]
        return {"customers": customers, "items": items}

    @staticmethod
    def _item(row: sqlite3.Row) -> Item:
        return Item(row["id"], row["display_name"], row["unit"], row["qty"], row["low_threshold"])
