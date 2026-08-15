"""The clock's own policy, with the tick handed over.

What is under test here is what the loop does *between* ticks, because that is where a
generation's cost hides: the tick is one forward and it is visible, and everything around
it runs once per token whether or not anybody asked for it.
"""

import asyncio
import threading
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

from mlx_omnia.server.flow import Clock, Emission, Member, Outlet


def member(loop: asyncio.AbstractEventLoop) -> Member[list[int], int]:
    return Member([], Outlet(loop, asyncio.Queue()), threading.Event())


async def drive(
    tokens: int,
    *,
    room: int,
    waiting: bool,
    counts: dict[str, int],
) -> None:
    loop = asyncio.get_running_loop()
    one = member(loop)

    def tick(members: Sequence[Member[list[int], int]]) -> Sequence[Emission[list[int], int]]:
        counts["ticks"] += 1
        return [Emission(members[0], (counts["ticks"],), counts["ticks"] >= tokens)]

    def rooms() -> int:
        counts["room"] += 1
        return room

    def is_waiting() -> bool:
        counts["waiting"] += 1
        return waiting

    async def join() -> Member[list[int], int] | None:
        counts["join"] += 1
        return None

    clock: Clock[list[int], int] = Clock(
        ThreadPoolExecutor(max_workers=1),
        tick,
        room=rooms,
        waiting=is_waiting,
        join=join,
        on_leave=lambda _: counts.__setitem__("left", counts["left"] + 1),
    )
    await clock.run([one])


def counters() -> dict[str, int]:
    return {"ticks": 0, "room": 0, "waiting": 0, "join": 0, "left": 0}


def test_an_empty_queue_costs_no_reading_of_the_limits() -> None:
    """`room` comes out of the store — two SQLite reads — and asking it once per token put
    them on the decode's critical path, which measured 1 ms a token on Laguna-XS. Nobody
    queued means no admission, and no admission means the size of the group is not a
    question worth asking."""
    counts = counters()
    asyncio.run(drive(8, room=4, waiting=False, counts=counts))
    assert counts["ticks"] == 8
    assert counts["room"] == 0
    assert counts["join"] == 0


def test_a_queued_joiner_still_asks_how_much_room_there_is() -> None:
    """The cheap question first does not mean the expensive one never runs: a queue with
    something in it is exactly when the limit decides whether it may come in."""
    counts = counters()
    asyncio.run(drive(4, room=4, waiting=True, counts=counts))
    assert counts["ticks"] == 4
    assert counts["room"] >= 4
    assert counts["join"] >= 4


def test_a_full_group_reads_the_limit_and_stops_there() -> None:
    """One member and room for one: the limit is read — somebody is queued — and answers
    that there is nowhere to put them. The single `join` is the turn after the member left,
    when the group has room again and the queue is the right thing to look at."""
    counts = counters()
    asyncio.run(drive(4, room=1, waiting=True, counts=counts))
    assert counts["room"] >= 4
    assert counts["join"] <= 1


def test_the_member_leaves_once_when_its_generation_ends() -> None:
    counts = counters()
    asyncio.run(drive(3, room=1, waiting=False, counts=counts))
    assert counts["left"] == 1
