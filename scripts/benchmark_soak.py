from __future__ import annotations

import argparse
import json
import os
import tracemalloc
from pathlib import Path

from mygame.ai import LocalStrategicAI
from mygame.protocols import (
    CommandEnvelopeV1,
    CommandSource,
    Faction,
    MovePayloadV1,
    PositionV1,
    SelectionV1,
)
from mygame.simulation import World

HAS_PROC_STATUS = Path("/proc/self/statm").exists()


def resident_bytes() -> int:
    try:
        pages = int(Path("/proc/self/statm").read_text(encoding="ascii").split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        return tracemalloc.get_traced_memory()[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticks", type=int, default=24_000)
    args = parser.parse_args()
    if not HAS_PROC_STATUS:
        tracemalloc.start()
    world = World(seed=20260907)
    world.victory_enabled = False
    world.units.hp[: world.units.count] *= 1000
    world.units.max_hp[: world.units.count] *= 1000
    world.execute(
        CommandEnvelopeV1(
            command_id="soak-player-advance",
            faction=Faction.PLAYER,
            issued_tick=0,
            source=CommandSource.KEYBOARD,
            payload=MovePayloadV1(
                kind="attack_move",
                selection=SelectionV1(group_id=1),
                target=PositionV1(x=3800, y=1152),
            ),
        )
    )
    ai = LocalStrategicAI(world.map.width, world.map.height, config=world.balance.ai["normal"])
    midpoint = max(1, args.ticks // 2)
    middle_memory = 0
    for _ in range(args.ticks):
        if ai.ready(world.tick):
            for command in ai.decide(world.observation(Faction.ENEMY)):
                world.execute(command)
        world.step()
        if world.tick == midpoint:
            middle_memory = resident_bytes()
    final_memory = resident_bytes()
    growth = (final_memory - middle_memory) / max(middle_memory, 1) * 100
    print(
        json.dumps(
            {
                "ticks": world.tick,
                "midpoint_resident_bytes": middle_memory,
                "final_resident_bytes": final_memory,
                "second_half_growth_percent": round(growth, 3),
                "within_five_percent": growth <= 5.0,
            },
            ensure_ascii=False,
        )
    )
    return 0 if growth <= 5.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
