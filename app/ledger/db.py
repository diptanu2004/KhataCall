"""SQLite connection handling.

Thin on purpose: the repository layer in `repo.py` holds the queries, and this
holds only the things that are easy to get wrong once and then never notice.
"""

import sqlite3
from pathlib import Path

SCHEMA = Path(__file__).with_name("schema.sql")


def connect(path: str | Path = ":memory:") -> sqlite3.Connection:
    """Open a connection with the schema applied and the pragmas we rely on."""
    database = str(path)
    if database != ":memory:":
        Path(database).parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    # Off by default in SQLite, and the schema's REFERENCES clauses do nothing without it.
    connection.execute("PRAGMA foreign_keys = ON")
    # The ledger is append-only and read often; WAL keeps a read from blocking a write.
    if database != ":memory:":
        connection.execute("PRAGMA journal_mode = WAL")
    connection.executescript(SCHEMA.read_text(encoding="utf-8"))
    connection.commit()
    return connection
