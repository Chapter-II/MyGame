from __future__ import annotations

from mygame.protocols import (
    BuildPayloadV1,
    CommandEnvelopeV1,
    CommandSource,
    Faction,
    MovePayloadV1,
    ObservationSnapshotV1,
    PositionV1,
    RecruitPayloadV1,
    SelectionV1,
)


class LocalStrategicAI:
    """A fair controller that only accepts a filtered observation."""

    def __init__(
        self,
        world_width: int,
        world_height: int,
        difficulty: str = "normal",
        config: dict[str, float | int | bool] | None = None,
    ) -> None:
        self.width = world_width
        self.height = world_height
        self.difficulty = difficulty
        self.config = config or {
            "strategy_interval_ticks": 40,
            "max_simultaneous_tasks": 4,
            "planning_depth": 2,
            "allow_inspire": False,
        }
        self.last_tick = -10_000

    def ready(self, tick: int) -> bool:
        interval = int(self.config["strategy_interval_ticks"])
        return tick != self.last_tick and tick % interval == 0

    def decide(self, observation: ObservationSnapshotV1) -> list[CommandEnvelopeV1]:
        if not self.ready(observation.tick):
            return []
        self.last_tick = observation.tick
        commands: list[CommandEnvelopeV1] = []
        visible = observation.visible_enemies
        if visible:
            priority = next((unit for unit in visible if unit.kind == "commander"), visible[0])
            target = PositionV1(x=priority.x, y=priority.y)
        elif observation.historical_sightings:
            sighting = max(
                observation.historical_sightings, key=lambda item: int(item["last_seen_tick"])
            )
            target = PositionV1(x=float(sighting["x"]), y=float(sighting["y"]))
        else:
            # Spawn zones are public map rules; searching the opposing deployment zone
            # does not require hidden unit coordinates.
            target = PositionV1(x=self.width * 0.08, y=self.height * 0.5)
        assault_groups = [1, 2, 4]
        if observation.tick >= 6000:
            assault_groups.append(0)
        flank_offset = {1: 0.0, 2: -150.0, 4: 150.0, 0: 0.0}
        for group in assault_groups:
            group_target = PositionV1(
                x=target.x,
                y=max(0.0, min(self.height, target.y + flank_offset[group])),
            )
            commands.append(
                CommandEnvelopeV1(
                    command_id=f"ai-{observation.tick}-{group}",
                    faction=Faction.ENEMY,
                    issued_tick=observation.tick,
                    source=CommandSource.LOCAL_AI,
                    payload=MovePayloadV1(
                        kind="attack_move",
                        selection=SelectionV1(group_id=group),
                        target=group_target,
                    ),
                )
            )
        if observation.known_villages:
            village = min(
                observation.known_villages, key=lambda item: abs(float(item["x"]) - self.width / 2)
            )
            engineer_target = PositionV1(x=float(village["x"]), y=float(village["y"]))
        else:
            engineer_target = PositionV1(x=self.width * 0.62, y=self.height * 0.62)
        bridge_exists = any(item["kind"] == "bridge" for item in observation.known_facilities)
        if not bridge_exists:
            commands.append(
                CommandEnvelopeV1(
                    command_id=f"ai-{observation.tick}-bridge",
                    faction=Faction.ENEMY,
                    issued_tick=observation.tick,
                    source=CommandSource.LOCAL_AI,
                    payload=BuildPayloadV1(
                        selection=SelectionV1(group_id=3),
                        facility_kind="bridge",
                        target=PositionV1(x=self.width * 0.5, y=self.height * 0.62),
                    ),
                )
            )
        else:
            commands.append(
                CommandEnvelopeV1(
                    command_id=f"ai-{observation.tick}-3",
                    faction=Faction.ENEMY,
                    issued_tick=observation.tick,
                    source=CommandSource.LOCAL_AI,
                    payload=MovePayloadV1(
                        kind="move",
                        selection=SelectionV1(group_id=3),
                        target=engineer_target,
                    ),
                )
            )
        for village in observation.known_villages:
            if int(village["population"]) <= 0:
                continue
            if any(
                (unit.x - float(village["x"])) ** 2 + (unit.y - float(village["y"])) ** 2 <= 92**2
                for unit in observation.own_units
            ):
                commands.append(
                    CommandEnvelopeV1(
                        command_id=f"ai-{observation.tick}-recruit",
                        faction=Faction.ENEMY,
                        issued_tick=observation.tick,
                        source=CommandSource.LOCAL_AI,
                        payload=RecruitPayloadV1(
                            village_id=int(village["village_id"]),
                            count=min(20, int(village["population"])),
                        ),
                    )
                )
                break
        return commands[: int(self.config["max_simultaneous_tasks"])]
