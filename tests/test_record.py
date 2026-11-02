"""Recording order: a session stopped early must still cover every intent."""

from collections import Counter

from scripts.record import spread_across_intents


def rows(counts: dict[str, int]) -> list[dict]:
    return [{"id": f"{intent}{i}", "intent": intent} for intent, n in counts.items() for i in range(n)]


def test_a_prefix_keeps_the_intent_mix():
    todo = rows({"add_credit": 40, "record_payment": 30, "update_stock": 20, "get_balance": 16, "get_stock": 8, "list_low_stock": 6})
    first = Counter(r["intent"] for r in spread_across_intents(todo)[:40])
    assert first == {"add_credit": 13, "record_payment": 10, "update_stock": 7, "get_balance": 5, "get_stock": 3, "list_low_stock": 2}


def test_every_line_is_kept_once_and_file_order_holds_within_an_intent():
    todo = rows({"add_credit": 5, "get_stock": 2})
    ordered = spread_across_intents(todo)
    assert sorted(r["id"] for r in ordered) == sorted(r["id"] for r in todo)
    assert [r["id"] for r in ordered if r["intent"] == "add_credit"] == [f"add_credit{i}" for i in range(5)]
