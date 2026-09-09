from __future__ import annotations

import math
import random
from typing import Literal

import numpy as np

from mygame.protocols import (
    BuildPayloadV1,
    CommandEnvelopeV1,
    CommandSource,
    ConvertPayloadV1,
    Faction,
    FocusFirePayloadV1,
    GuardPayloadV1,
    MovePayloadV1,
    ObservationSnapshotV1,
    PositionV1,
    RecruitPayloadV1,
    SelectionV1,
    TacticalPayloadV1,
)

_DIFFICULTY_PROFILES: dict[str, dict[str, float | int | bool]] = {
    "easy": {
        "strategy_interval_ticks": 60,
        "tactical_interval_ticks": 10,
        "max_simultaneous_tasks": 3,
        "planning_depth": 1,
        "allow_inspire": False,
        "aggression": 0.5,
        "retreat_hp_ratio": 0.15,
        "scout_count": 1,
    },
    "normal": {
        "strategy_interval_ticks": 40,
        "tactical_interval_ticks": 5,
        "max_simultaneous_tasks": 4,
        "planning_depth": 2,
        "allow_inspire": False,
        "aggression": 0.7,
        "retreat_hp_ratio": 0.25,
        "scout_count": 2,
    },
    "hard": {
        "strategy_interval_ticks": 30,
        "tactical_interval_ticks": 4,
        "max_simultaneous_tasks": 5,
        "planning_depth": 3,
        "allow_inspire": True,
        "aggression": 0.9,
        "retreat_hp_ratio": 0.35,
        "scout_count": 3,
    },
}


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
        self.config = config or _DIFFICULTY_PROFILES.get(difficulty, _DIFFICULTY_PROFILES["normal"])
        self.last_strategy_tick = -10_000
        self.last_tactical_tick = -10_000
        self._scout_targets: list[tuple[float, float]] = []
        self._last_known_enemy_pos: tuple[float, float] | None = None
        self._rng = random.Random(42)
        self._commander_dead = False

    def _has_commander(self, obs: ObservationSnapshotV1) -> bool:
        return any(u.kind == "commander" for u in obs.own_units)

    def _strategy_ready(self, tick: int) -> bool:
        interval = int(self.config["strategy_interval_ticks"])
        return tick != self.last_strategy_tick and tick % interval == 0

    def _tactical_ready(self, tick: int) -> bool:
        interval = int(self.config["tactical_interval_ticks"])
        return tick != self.last_tactical_tick and tick % interval == 0

    def ready(self, tick: int) -> bool:
        return self._strategy_ready(tick) or self._tactical_ready(tick)

    def decide(self, observation: ObservationSnapshotV1) -> list[CommandEnvelopeV1]:
        commands: list[CommandEnvelopeV1] = []

        if self._tactical_ready(observation.tick):
            commands.extend(self._tactical_decisions(observation))
            self.last_tactical_tick = observation.tick

        if self._strategy_ready(observation.tick):
            commands.extend(self._strategic_decisions(observation))
            self.last_strategy_tick = observation.tick

        max_tasks = int(self.config["max_simultaneous_tasks"])
        return commands[:max_tasks]

    # ── Strategic layer (every ~2s) ──────────────────────────────

    def _strategic_decisions(self, obs: ObservationSnapshotV1) -> list[CommandEnvelopeV1]:
        commands: list[CommandEnvelopeV1] = []
        target = self._pick_assault_target(obs)

        # Update enemy position memory
        visible = obs.visible_enemies
        if visible:
            cmd = next((u for u in visible if u.kind == "commander"), visible[0])
            self._last_known_enemy_pos = (cmd.x, cmd.y)

        # Assault groups with flanking
        commands.extend(self._order_assault(obs, target))

        # Engineering
        commands.extend(self._order_engineering(obs))

        # Recruit from villages
        commands.extend(self._order_recruit(obs))

        # Convert recruits to useful classes (skip if commander dead — rush units instead)
        if not self._commander_dead:
            commands.extend(self._order_conversions(obs))

        return commands

    def _pick_assault_target(self, obs: ObservationSnapshotV1) -> PositionV1:
        visible = obs.visible_enemies
        if visible:
            priority = next((u for u in visible if u.kind == "commander"), visible[0])
            return PositionV1(x=priority.x, y=priority.y)
        if obs.historical_sightings:
            sighting = max(obs.historical_sightings, key=lambda s: int(s["last_seen_tick"]))
            return PositionV1(x=float(sighting["x"]), y=float(sighting["y"]))
        if self._last_known_enemy_pos:
            return PositionV1(x=self._last_known_enemy_pos[0], y=self._last_known_enemy_pos[1])
        target_x = self.width * (0.92 if obs.faction == Faction.PLAYER else 0.08)
        return PositionV1(x=target_x, y=self.height * 0.5)

    def _order_assault(
        self, obs: ObservationSnapshotV1, target: PositionV1
    ) -> list[CommandEnvelopeV1]:
        commands: list[CommandEnvelopeV1] = []
        assault_groups = [1, 4]
        if obs.tick >= 4000:
            assault_groups.append(0)

        # Scouts go to different flanking position
        commands.append(
            self._move_cmd(obs, 2, self._scout_assault_target(obs, target), kind="attack_move")
        )

        for group in assault_groups:
            offset_y = self._flank_offset(group)
            group_target = PositionV1(
                x=target.x,
                y=max(0.0, min(self.height, target.y + offset_y)),
            )
            commands.append(self._move_cmd(obs, group, group_target, kind="attack_move"))

        # Assassin: try to reach commander directly (stealth approach)
        commander_visible = any(u.kind == "commander" for u in obs.visible_enemies)
        if commander_visible:
            cmd_enemy = next(u for u in obs.visible_enemies if u.kind == "commander")
            commands.append(self._move_cmd(obs, 4, PositionV1(x=cmd_enemy.x, y=cmd_enemy.y)))

        return commands

    def _scout_assault_target(
        self, obs: ObservationSnapshotV1, primary_target: PositionV1
    ) -> PositionV1:
        """Scouts take a wide flanking route to reveal fog."""
        # Offset scouts significantly to the side for recon
        offset = self.height * 0.3 * (1 if obs.tick % 800 < 400 else -1)
        return PositionV1(
            x=max(
                0.0,
                min(
                    self.width,
                    primary_target.x
                    + (-1 if obs.faction == Faction.PLAYER else 1) * self.width * 0.15,
                ),
            ),
            y=max(0.0, min(self.height, primary_target.y + offset)),
        )

    def _flank_offset(self, group: int) -> float:
        offsets = {0: 0.0, 1: 0.0, 2: -180.0, 4: 180.0}
        return offsets.get(group, 0.0)

    def _order_engineering(self, obs: ObservationSnapshotV1) -> list[CommandEnvelopeV1]:
        commands: list[CommandEnvelopeV1] = []
        engineer_target = self._pick_engineer_target(obs)

        own_facilities = [
            facility
            for facility in obs.known_facilities
            if int(facility["faction"]) == int(obs.faction) and not bool(facility["destroyed"])
        ]
        # Do not overwrite the engineers' BUILD order while a project is in progress.
        if any(not bool(facility["complete"]) for facility in own_facilities):
            return commands
        bridge_exists = any(f["kind"] == "bridge" for f in own_facilities)
        boat_exists = any(f["kind"] == "boat" for f in own_facilities)
        tower_exists = any(f["kind"] == "tower" for f in own_facilities)

        # Priority 1: Build bridge if none exists
        if not bridge_exists:
            river_target = self._known_river_target(obs)
            if river_target is None:
                staging_x = self.width * (0.43 if obs.faction == Faction.PLAYER else 0.57)
                return [
                    self._move_cmd(
                        obs,
                        3,
                        PositionV1(x=staging_x, y=self.height * 0.5),
                        kind="move",
                    )
                ]
            commands.append(
                CommandEnvelopeV1(
                    command_id=f"ai-{int(obs.faction)}-{obs.tick}-bridge",
                    faction=obs.faction,
                    issued_tick=obs.tick,
                    source=CommandSource.LOCAL_AI,
                    payload=BuildPayloadV1(
                        selection=SelectionV1(group_id=3),
                        facility_kind="bridge",
                        target=river_target,
                    ),
                )
            )
            return commands

        # Priority 2: Build boat if no boat and river is wide
        if not boat_exists and obs.tick > 2000:
            boat_y = self.height * 0.5 + self._rng.uniform(-self.height * 0.15, self.height * 0.15)
            commands.append(
                CommandEnvelopeV1(
                    command_id=f"ai-{int(obs.faction)}-{obs.tick}-boat",
                    faction=obs.faction,
                    issued_tick=obs.tick,
                    source=CommandSource.LOCAL_AI,
                    payload=BuildPayloadV1(
                        selection=SelectionV1(group_id=3),
                        facility_kind="boat",
                        target=PositionV1(x=self.width * 0.52, y=boat_y),
                    ),
                )
            )
            return commands

        # Priority 3: Build tower near a village for defense
        if not tower_exists and obs.tick > 4000 and obs.known_villages:
            village = min(
                obs.known_villages,
                key=lambda v: abs(float(v["x"]) - self.width * 0.7),
            )
            commands.append(
                CommandEnvelopeV1(
                    command_id=f"ai-{int(obs.faction)}-{obs.tick}-tower",
                    faction=obs.faction,
                    issued_tick=obs.tick,
                    source=CommandSource.LOCAL_AI,
                    payload=BuildPayloadV1(
                        selection=SelectionV1(group_id=3),
                        facility_kind="tower",
                        target=PositionV1(
                            x=float(village["x"]) + 50,
                            y=float(village["y"]),
                        ),
                    ),
                )
            )
            return commands

        # Priority 4: Build road to clear forest for faster movement
        if obs.tick > 6000 and obs.terrain_shape[0] > 0:
            # Build road in the direction of assault
            road_x = self.width * 0.45
            road_y = self.height * 0.5
            commands.append(
                CommandEnvelopeV1(
                    command_id=f"ai-{int(obs.faction)}-{obs.tick}-road",
                    faction=obs.faction,
                    issued_tick=obs.tick,
                    source=CommandSource.LOCAL_AI,
                    payload=BuildPayloadV1(
                        selection=SelectionV1(group_id=3),
                        facility_kind="road",
                        target=PositionV1(x=road_x, y=road_y),
                    ),
                )
            )
            return commands

        # Default: engineers move to nearest village or assist assault
        commands.append(self._move_cmd(obs, 3, engineer_target, kind="move"))
        return commands

    def _known_river_target(self, obs: ObservationSnapshotV1) -> PositionV1 | None:
        if not obs.known_terrain or not obs.terrain_shape[0] or not obs.terrain_shape[1]:
            return None
        terrain = np.frombuffer(obs.known_terrain, dtype=np.uint8).reshape(obs.terrain_shape)
        rows, cols = np.nonzero(terrain == 4)
        if not len(rows):
            return None
        middle_row = obs.terrain_shape[0] / 2
        choice = min(
            range(len(rows)),
            key=lambda index: abs(float(rows[index]) - middle_row),
        )
        tile_width = self.width / obs.terrain_shape[1]
        tile_height = self.height / obs.terrain_shape[0]
        return PositionV1(
            x=(float(cols[choice]) + 0.5) * tile_width,
            y=(float(rows[choice]) + 0.5) * tile_height,
        )

    def _pick_engineer_target(self, obs: ObservationSnapshotV1) -> PositionV1:
        if obs.known_villages:
            village = min(
                obs.known_villages,
                key=lambda v: abs(float(v["x"]) - self.width / 2),
            )
            return PositionV1(x=float(village["x"]), y=float(village["y"]))
        fallback_x = self.width * (0.62 if obs.faction == Faction.PLAYER else 0.38)
        return PositionV1(x=fallback_x, y=self.height * 0.62)

    def _order_recruit(self, obs: ObservationSnapshotV1) -> list[CommandEnvelopeV1]:
        commands: list[CommandEnvelopeV1] = []
        for village in obs.known_villages:
            if int(village["population"]) <= 0:
                continue
            if any(
                (u.x - float(village["x"])) ** 2 + (u.y - float(village["y"])) ** 2 <= 92**2
                for u in obs.own_units
            ):
                commands.append(
                    CommandEnvelopeV1(
                        command_id=f"ai-{int(obs.faction)}-{obs.tick}-recruit",
                        faction=obs.faction,
                        issued_tick=obs.tick,
                        source=CommandSource.LOCAL_AI,
                        payload=RecruitPayloadV1(
                            village_id=int(village["village_id"]),
                            count=min(20, int(village["population"])),
                        ),
                    )
                )
                break
        return commands

    def _order_conversions(self, obs: ObservationSnapshotV1) -> list[CommandEnvelopeV1]:
        """Convert recruits to useful classes based on army composition."""
        commands: list[CommandEnvelopeV1] = []
        recruits = [u for u in obs.own_units if u.kind == "recruit"]
        if not recruits:
            return commands

        # Count current army composition
        counts: dict[str, int] = {}
        for u in obs.own_units:
            counts[u.kind] = counts.get(u.kind, 0) + 1

        target_scouts = int(self.config.get("scout_count", 2))
        current_scouts = counts.get("scout", 0)
        current_engineers = counts.get("engineer", 0)
        current_assassins = counts.get("assassin", 0)

        # Convert to scout if we need more
        if current_scouts < target_scouts and len(recruits) >= 1:
            commands.append(
                CommandEnvelopeV1(
                    command_id=f"ai-{int(obs.faction)}-{obs.tick}-convert-scout",
                    faction=obs.faction,
                    issued_tick=obs.tick,
                    source=CommandSource.LOCAL_AI,
                    payload=ConvertPayloadV1(
                        selection=SelectionV1(count=1, unit_kind="recruit"),
                        unit_kind="scout",
                    ),
                )
            )
            return commands

        # Convert to engineer if we need more
        if current_engineers < 10 and len(recruits) >= 5:
            commands.append(
                CommandEnvelopeV1(
                    command_id=f"ai-{int(obs.faction)}-{obs.tick}-convert-eng",
                    faction=obs.faction,
                    issued_tick=obs.tick,
                    source=CommandSource.LOCAL_AI,
                    payload=ConvertPayloadV1(
                        selection=SelectionV1(count=5, unit_kind="recruit"),
                        unit_kind="engineer",
                    ),
                )
            )
            return commands

        # Convert to assassin for commander assassination
        if current_assassins < 5 and len(recruits) >= 3 and obs.tick > 3000:
            commands.append(
                CommandEnvelopeV1(
                    command_id=f"ai-{int(obs.faction)}-{obs.tick}-convert-assassin",
                    faction=obs.faction,
                    issued_tick=obs.tick,
                    source=CommandSource.LOCAL_AI,
                    payload=ConvertPayloadV1(
                        selection=SelectionV1(count=3, unit_kind="recruit"),
                        unit_kind="assassin",
                    ),
                )
            )
            return commands

        # Default: convert to infantry
        if len(recruits) >= 10:
            commands.append(
                CommandEnvelopeV1(
                    command_id=f"ai-{int(obs.faction)}-{obs.tick}-convert-infantry",
                    faction=obs.faction,
                    issued_tick=obs.tick,
                    source=CommandSource.LOCAL_AI,
                    payload=ConvertPayloadV1(
                        selection=SelectionV1(count=min(20, len(recruits)), unit_kind="recruit"),
                        unit_kind="infantry",
                    ),
                )
            )

        return commands

    # ── Tactical layer (every ~0.25s) ───────────────────────────

    def _tactical_decisions(self, obs: ObservationSnapshotV1) -> list[CommandEnvelopeV1]:
        commands: list[CommandEnvelopeV1] = []
        aggression = float(self.config.get("aggression", 0.7))
        retreat_ratio = float(self.config.get("retreat_hp_ratio", 0.25))

        # Detect commander death → desperation mode
        if not self._commander_dead and not self._has_commander(obs):
            self._commander_dead = True
        if self._commander_dead:
            aggression = min(1.0, aggression + 0.3)

        # Retreat damaged units
        commands.extend(self._retreat_damaged(obs, retreat_ratio))

        # Focus fire on enemy commander if visible
        commands.extend(self._focus_fire_commander(obs))

        # Charge/sprint when advantageous
        if bool(self.config.get("allow_inspire", False)):
            commands.extend(self._use_tactical(obs, aggression))

        # Protect commander: guard assignment (skip if commander is dead)
        if not self._commander_dead:
            commands.extend(self._protect_commander(obs))

        return commands

    def _retreat_damaged(
        self, obs: ObservationSnapshotV1, retreat_ratio: float
    ) -> list[CommandEnvelopeV1]:
        """Pull back units with HP below threshold."""
        commands: list[CommandEnvelopeV1] = []
        damaged = [
            u
            for u in obs.own_units
            if u.hp < retreat_ratio and u.kind not in ("commander", "guard")
        ]
        if not damaged:
            return commands

        # Retreat toward own spawn zone (right side of map)
        retreat_x = self.width * (0.15 if obs.faction == Faction.PLAYER else 0.85)
        retreat_y = self.height * 0.5

        # Only retreat a few at a time to avoid command spam
        retreat_sample = damaged[:8]
        avg_x = sum(u.x for u in retreat_sample) / len(retreat_sample)
        avg_y = sum(u.y for u in retreat_sample) / len(retreat_sample)

        # Retreat away from enemies
        visible = obs.visible_enemies
        if visible:
            enemy_x = sum(e.x for e in visible) / len(visible)
            enemy_y = sum(e.y for e in visible) / len(visible)
            dx = avg_x - enemy_x
            dy = avg_y - enemy_y
            dist = math.hypot(dx, dy) or 1.0
            retreat_x = avg_x + dx / dist * 200
            retreat_y = avg_y + dy / dist * 200

        retreat_x = max(0.0, min(self.width, retreat_x))
        retreat_y = max(0.0, min(self.height, retreat_y))

        # Use unit_ids of the damaged units
        retreat_ids = tuple(u.entity_id for u in retreat_sample)
        commands.append(
            CommandEnvelopeV1(
                command_id=f"ai-{int(obs.faction)}-{obs.tick}-retreat",
                faction=obs.faction,
                issued_tick=obs.tick,
                source=CommandSource.LOCAL_AI,
                payload=MovePayloadV1(
                    kind="move",
                    selection=SelectionV1(unit_ids=retreat_ids),
                    target=PositionV1(x=retreat_x, y=retreat_y),
                ),
            )
        )
        return commands

    def _focus_fire_commander(self, obs: ObservationSnapshotV1) -> list[CommandEnvelopeV1]:
        """Focus fire on enemy commander if visible."""
        if not obs.visible_enemies:
            return []
        cmd = next((u for u in obs.visible_enemies if u.kind == "commander"), None)
        if cmd is None:
            return []
        # Send assassins to focus fire
        assassins = [u for u in obs.own_units if u.kind == "assassin"]
        if len(assassins) < 2:
            return []
        assassin_ids = tuple(u.entity_id for u in assassins[:5])
        return [
            CommandEnvelopeV1(
                command_id=f"ai-{int(obs.faction)}-{obs.tick}-focus-cmd",
                faction=obs.faction,
                issued_tick=obs.tick,
                source=CommandSource.LOCAL_AI,
                payload=FocusFirePayloadV1(
                    selection=SelectionV1(unit_ids=assassin_ids),
                    target_entity_id=cmd.entity_id,
                ),
            )
        ]

    def _use_tactical(
        self, obs: ObservationSnapshotV1, aggression: float
    ) -> list[CommandEnvelopeV1]:
        """Use sprint/charge when units are near enemies."""
        commands: list[CommandEnvelopeV1] = []
        if not obs.visible_enemies:
            return commands

        # Find infantry groups near enemies
        for group_id in (1, 4):
            group_units = [u for u in obs.own_units if u.group_id == group_id and u.stamina > 60]
            if not group_units:
                continue

            avg_x = sum(u.x for u in group_units) / len(group_units)
            avg_y = sum(u.y for u in group_units) / len(group_units)

            # Check if near any enemy
            min_dist = min(math.hypot(avg_x - e.x, avg_y - e.y) for e in obs.visible_enemies)

            if min_dist < 300 and self._rng.random() < aggression:
                # Charge when close
                commands.append(
                    CommandEnvelopeV1(
                        command_id=f"ai-{int(obs.faction)}-{obs.tick}-charge-{group_id}",
                        faction=obs.faction,
                        issued_tick=obs.tick,
                        source=CommandSource.LOCAL_AI,
                        payload=TacticalPayloadV1(
                            kind="charge",
                            selection=SelectionV1(group_id=group_id),
                        ),
                    )
                )
            elif min_dist < 600 and self._rng.random() < aggression * 0.5:
                # Sprint to close distance
                nearest = min(
                    obs.visible_enemies, key=lambda e: math.hypot(avg_x - e.x, avg_y - e.y)
                )
                commands.append(
                    CommandEnvelopeV1(
                        command_id=f"ai-{int(obs.faction)}-{obs.tick}-sprint-{group_id}",
                        faction=obs.faction,
                        issued_tick=obs.tick,
                        source=CommandSource.LOCAL_AI,
                        payload=TacticalPayloadV1(
                            kind="sprint",
                            selection=SelectionV1(group_id=group_id),
                            target=PositionV1(x=nearest.x, y=nearest.y),
                        ),
                    )
                )

        return commands

    def _protect_commander(self, obs: ObservationSnapshotV1) -> list[CommandEnvelopeV1]:
        """Assign guards to protect the commander."""
        commands: list[CommandEnvelopeV1] = []
        cmd_unit = next((u for u in obs.own_units if u.kind == "commander"), None)
        if cmd_unit is None:
            return commands

        # Check if guards exist
        guards = [u for u in obs.own_units if u.kind == "guard"]
        if not guards:
            return commands

        # If enemies are near commander, have guards intercept
        nearby_enemies = [
            e for e in obs.visible_enemies if math.hypot(cmd_unit.x - e.x, cmd_unit.y - e.y) < 400
        ]
        if nearby_enemies:
            threat = nearby_enemies[0]
            guard_ids = tuple(u.entity_id for u in guards[:4])
            commands.append(
                CommandEnvelopeV1(
                    command_id=f"ai-{int(obs.faction)}-{obs.tick}-guard-protect",
                    faction=obs.faction,
                    issued_tick=obs.tick,
                    source=CommandSource.LOCAL_AI,
                    payload=GuardPayloadV1(
                        selection=SelectionV1(unit_ids=guard_ids),
                        target=PositionV1(x=threat.x, y=threat.y),
                    ),
                )
            )

        return commands

    # ── Helpers ──────────────────────────────────────────────────

    def _move_cmd(
        self,
        obs: ObservationSnapshotV1,
        group: int,
        target: PositionV1,
        kind: Literal["move", "attack_move"] = "attack_move",
    ) -> CommandEnvelopeV1:
        return CommandEnvelopeV1(
            command_id=f"ai-{int(obs.faction)}-{obs.tick}-{group}",
            faction=obs.faction,
            issued_tick=obs.tick,
            source=CommandSource.LOCAL_AI,
            payload=MovePayloadV1(
                kind=kind,
                selection=SelectionV1(group_id=group),
                target=target,
            ),
        )
