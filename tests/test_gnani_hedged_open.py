"""open_hedged: a slow Gnani WebSocket handshake gets a second attempt alongside it (D-052). No network."""

import asyncio
from contextlib import AsyncExitStack

import pytest

from app.adapters.gnani_common import open_hedged


class Attempt:
    """A connect whose handshake takes `delay` seconds, then opens or fails."""

    def __init__(self, n: int, delay: float, fails: bool, log: list[str]) -> None:
        self.n, self.delay, self.fails, self.log = n, delay, fails, log

    async def __aenter__(self):
        await asyncio.sleep(self.delay)
        if self.fails:
            raise TimeoutError(f"timed out during opening handshake ({self.n})")
        self.log.append(f"open {self.n}")
        return f"socket {self.n}"

    async def __aexit__(self, *exc):
        self.log.append(f"close {self.n}")
        return False


def connects(plan: list[tuple[float, bool]]):
    log: list[str] = []
    made: list[Attempt] = []

    def connect():
        delay, fails = plan[len(made)]
        made.append(Attempt(len(made), delay, fails, log))
        return made[-1]

    return connect, made, log


def run(connect, **kwargs):
    async def go():
        async with AsyncExitStack() as stack:
            ws = await open_hedged(stack, connect, **kwargs)
            await asyncio.sleep(0.3)  # let any late loser open and be closed
            return ws

    return asyncio.run(go())


def test_a_quick_handshake_is_the_only_attempt():
    connect, made, log = connects([(0.0, False)])
    assert run(connect, hedge_after_s=0.1) == "socket 0"
    assert len(made) == 1
    assert log == ["open 0", "close 0"]


def test_a_slow_handshake_gets_a_second_attempt_and_the_first_to_open_wins():
    # Nov 10: one handshake in a handful took 17.6 s, past websockets' 10 s timeout.
    connect, made, log = connects([(0.2, False), (0.01, False)])
    assert run(connect, hedge_after_s=0.05) == "socket 1"
    assert len(made) == 2
    # The loser opened after all and was closed at once; the winner closes with the stack.
    assert log.index("open 0") > log.index("open 1") and "close 0" in log and log[-1] == "close 1"


def test_a_failed_attempt_starts_the_next_at_once():
    connect, made, _ = connects([(0.0, True), (0.0, False)])
    assert run(connect, hedge_after_s=10) == "socket 1"
    assert len(made) == 2


def test_when_every_attempt_fails_the_first_error_is_raised():
    connect, made, _ = connects([(0.0, True), (0.0, True), (0.0, True)])
    with pytest.raises(TimeoutError, match=r"\(0\)"):
        run(connect, hedge_after_s=0.05, attempts=3)
    assert len(made) == 3
