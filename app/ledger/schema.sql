-- KhataCall ledger and stock.
--
-- Two rules are enforced here rather than in Python, because a constraint the
-- database holds cannot be forgotten by a caller:
--   * amounts are integer paise and strictly positive (hard rule 6)
--   * a ledger entry belongs to a customer that exists
--
-- Balances are not stored. They are SUM(credit) - SUM(payment) over the entries,
-- so the book has one source of truth and cannot drift (DECISIONS.md D-009).

CREATE TABLE IF NOT EXISTS customers (
    id           INTEGER PRIMARY KEY,
    -- As the shop owner says it, honorific and all: "Sharma ji".
    display_name TEXT NOT NULL,
    -- Honorifics stripped and spelling folded, for lookup. See nlu/names.py.
    name_norm    TEXT NOT NULL UNIQUE,
    created_at   TEXT NOT NULL,
    CHECK (length(trim(display_name)) > 0),
    CHECK (length(trim(name_norm)) > 0)
);

CREATE TABLE IF NOT EXISTS ledger_entries (
    id          INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(id) ON DELETE CASCADE,
    -- 'credit' is money the customer owes; 'payment' is money they have given back.
    kind        TEXT NOT NULL CHECK (kind IN ('credit', 'payment')),
    amount_paise INTEGER NOT NULL CHECK (amount_paise > 0),
    note        TEXT,
    -- Which call made this write. An append-only trail the owner could audit.
    session_id  TEXT,
    created_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_entries_customer ON ledger_entries(customer_id);

CREATE TABLE IF NOT EXISTS items (
    id            INTEGER PRIMARY KEY,
    display_name  TEXT NOT NULL,
    name_norm     TEXT NOT NULL UNIQUE,
    -- What one unit is: packet, kilo, litre, bottle. Spoken back to the owner.
    unit          TEXT NOT NULL DEFAULT 'packet',
    -- Whole units only. Loose goods are refused rather than rounded (D-024).
    qty           INTEGER NOT NULL DEFAULT 0 CHECK (qty >= 0),
    low_threshold INTEGER NOT NULL DEFAULT 5 CHECK (low_threshold >= 0),
    created_at    TEXT NOT NULL,
    CHECK (length(trim(display_name)) > 0)
);

CREATE TABLE IF NOT EXISTS stock_movements (
    id         INTEGER PRIMARY KEY,
    item_id    INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
    -- Positive when stock arrives, negative when it is sold. Never zero.
    qty_delta  INTEGER NOT NULL CHECK (qty_delta <> 0),
    session_id TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_movements_item ON stock_movements(item_id);
