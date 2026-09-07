import os
import time

import pytest

from mygame.simulation import World


@pytest.mark.skipif(os.getenv("MYGAME_PERF") != "1", reason="set MYGAME_PERF=1")
def test_1000_unit_tick_p95_is_under_budget() -> None:
    world = World()
    world.step(40)
    timings = []
    for _ in range(400):
        before = time.perf_counter()
        world.step()
        timings.append((time.perf_counter() - before) * 1000)
    timings.sort()
    assert timings[int(len(timings) * 0.95)] <= 10.0
