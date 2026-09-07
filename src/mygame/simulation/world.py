from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from enum import IntEnum, StrEnum
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from mygame.config import BalanceConfig, UnitStats, load_balance
from mygame.maps import BattleMap, Terrain, generate_map
from mygame.protocols import (
    AttackFacilityPayloadV1,
    BuildPayloadV1,
    CommandEnvelopeV1,
    CommandResultV1,
    CommandStatus,
    ConvertPayloadV1,
    FacilityActionPayloadV1,
    Faction,
    FocusFirePayloadV1,
    GameEventV1,
    GroupPayloadV1,
    GuardPayloadV1,
    MovePayloadV1,
    ObservationSnapshotV1,
    QueueMode,
    RecruitPayloadV1,
    SelectionV1,
    TacticalPayloadV1,
)
from mygame.simulation.navigation import FlowFieldCache

if TYPE_CHECKING:
    from mygame.perception.fog import FogOfWar


class UnitKind(IntEnum):
    RECRUIT = 0
    INFANTRY = 1
    SCOUT = 2
    ENGINEER = 3
    ASSASSIN = 4
    COMMANDER = 5
    GUARD = 6

    @property
    def config_name(self) -> str:
        return self.name.lower()


class Order(IntEnum):
    IDLE = 0
    MOVE = 1
    ATTACK_MOVE = 2
    BUILD = 3
    GUARD = 4


class GameOutcome(StrEnum):
    ONGOING = "ongoing"
    PLAYER_WIN = "player_win"
    ENEMY_WIN = "enemy_win"
    DRAW = "draw"


class FacilityKind(StrEnum):
    BRIDGE = "bridge"
    BOAT = "boat"
    ROAD = "road"
    TOWER = "tower"


@dataclass(slots=True)
class Facility:
    facility_id: int
    faction: int
    kind: str
    x: float
    y: float
    progress: float
    required_work: float
    minimum_engineers: int
    hp: float
    max_hp: float
    complete: bool
    builder_ids: list[int]
    target_x: float | None = None
    target_y: float | None = None
    elapsed_ticks: int = 0
    destroyed: bool = False


class UnitStore:
    entity_id: NDArray[Any]
    faction: NDArray[Any]
    kind: NDArray[Any]
    x: NDArray[Any]
    y: NDArray[Any]
    previous_x: NDArray[Any]
    previous_y: NDArray[Any]
    target_x: NDArray[Any]
    target_y: NDArray[Any]
    hp: NDArray[Any]
    max_hp: NDArray[Any]
    damage: NDArray[Any]
    attack_interval: NDArray[Any]
    cooldown: NDArray[Any]
    speed: NDArray[Any]
    vision: NDArray[Any]
    detection: NDArray[Any]
    radius: NDArray[Any]
    attack_range: NDArray[Any]
    stamina: NDArray[Any]
    tactic_kind: NDArray[Any]
    tactic_ticks: NDArray[Any]
    tactic_cooldown: NDArray[Any]
    last_intense_tick: NDArray[Any]
    order: NDArray[Any]
    group_id: NDArray[Any]
    alive: NDArray[Any]
    exposed: NDArray[Any]
    facility_id: NDArray[Any]
    focus_target: NDArray[Any]
    focus_facility: NDArray[Any]

    _fields = (
        "entity_id",
        "faction",
        "kind",
        "x",
        "y",
        "previous_x",
        "previous_y",
        "target_x",
        "target_y",
        "hp",
        "max_hp",
        "damage",
        "attack_interval",
        "cooldown",
        "speed",
        "vision",
        "detection",
        "radius",
        "attack_range",
        "stamina",
        "tactic_kind",
        "tactic_ticks",
        "tactic_cooldown",
        "last_intense_tick",
        "order",
        "group_id",
        "alive",
        "exposed",
        "facility_id",
        "focus_target",
        "focus_facility",
    )

    def __init__(self, capacity: int = 2048) -> None:
        self.capacity = capacity
        self.count = 0
        ints = {
            "entity_id": np.int32,
            "faction": np.int8,
            "kind": np.int8,
            "x": np.int32,
            "y": np.int32,
            "previous_x": np.int32,
            "previous_y": np.int32,
            "target_x": np.int32,
            "target_y": np.int32,
            "attack_interval": np.int16,
            "cooldown": np.int16,
            "tactic_kind": np.int8,
            "tactic_ticks": np.int16,
            "tactic_cooldown": np.int16,
            "last_intense_tick": np.int32,
            "order": np.int8,
            "group_id": np.int16,
            "facility_id": np.int32,
            "focus_target": np.int32,
            "focus_facility": np.int32,
        }
        bools = {"alive", "exposed"}
        for name in self._fields:
            if name in bools:
                array = np.zeros(capacity, dtype=np.bool_)
            elif name in ints:
                array = np.zeros(capacity, dtype=ints[name])
            else:
                array = np.zeros(capacity, dtype=np.float32)
            setattr(self, name, array)

    def _grow(self) -> None:
        new_capacity = self.capacity * 2
        for name in self._fields:
            old = getattr(self, name)
            new = np.zeros(new_capacity, dtype=old.dtype)
            new[: self.count] = old[: self.count]
            setattr(self, name, new)
        self.capacity = new_capacity

    def add(
        self,
        entity_id: int,
        faction: Faction,
        kind: UnitKind,
        x: float,
        y: float,
        stats: UnitStats,
        hz: int,
        subpixels: int,
        group_id: int = 0,
    ) -> int:
        if self.count >= self.capacity:
            self._grow()
        index = self.count
        self.count += 1
        self.entity_id[index] = entity_id
        self.faction[index] = int(faction)
        self.kind[index] = int(kind)
        self.x[index] = round(x * subpixels)
        self.y[index] = round(y * subpixels)
        self.previous_x[index] = self.x[index]
        self.previous_y[index] = self.y[index]
        self.target_x[index] = self.x[index]
        self.target_y[index] = self.y[index]
        self.hp[index] = stats.hp
        self.max_hp[index] = stats.hp
        self.damage[index] = stats.damage
        self.attack_interval[index] = max(1, round(stats.attack_interval * hz))
        self.speed[index] = stats.speed
        self.vision[index] = stats.vision
        self.detection[index] = stats.detection
        self.radius[index] = stats.radius
        self.attack_range[index] = stats.attack_range
        self.stamina[index] = 100.0
        self.order[index] = int(Order.IDLE)
        self.group_id[index] = group_id
        self.alive[index] = True
        self.exposed[index] = kind != UnitKind.ASSASSIN
        self.facility_id[index] = -1
        self.focus_target[index] = -1
        self.focus_facility[index] = -1
        return index

    def active(self, faction: Faction | None = None) -> np.ndarray:
        mask = self.alive[: self.count]
        if faction is not None:
            mask = mask & (self.faction[: self.count] == int(faction))
        return np.flatnonzero(mask)

    def index_of(self, entity_id: int) -> int | None:
        index = entity_id - 1
        if 0 <= index < self.count and int(self.entity_id[index]) == entity_id:
            return index
        matches = np.flatnonzero(self.entity_id[: self.count] == entity_id)
        return int(matches[0]) if len(matches) else None

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name)[: self.count].tolist() for name in self._fields} | {
            "count": self.count,
            "capacity": self.capacity,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UnitStore:
        store = cls(max(int(data["capacity"]), int(data["count"]), 16))
        store.count = int(data["count"])
        for name in cls._fields:
            array = getattr(store, name)
            array[: store.count] = np.asarray(data[name], dtype=array.dtype)
        return store


class World:
    def __init__(
        self,
        balance: BalanceConfig | None = None,
        battle_map: BattleMap | None = None,
        seed: int = 20260907,
        army_size: int | None = None,
        army_composition: dict[str, int] | None = None,
        spawn_armies: bool = True,
    ) -> None:
        self.balance = balance or load_balance()
        wc = self.balance.world
        self.map = battle_map or generate_map(wc.width, wc.height, wc.tile_size, seed)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.units = UnitStore(max(2048, wc.default_army_size * 3))
        self.tick = 0
        self.next_entity_id = 1
        self.next_facility_id = 1
        self.facilities: list[Facility] = []
        self.events: list[GameEventV1] = []
        self.command_log: list[CommandEnvelopeV1] = []
        self.command_results: list[CommandResultV1] = []
        self.statistics: dict[str, list[int]] = {"losses": [0, 0]}
        self.order_queues: dict[int, list[tuple[int, int, int]]] = {}
        self._flow_fields = FlowFieldCache(self.map, self.balance.terrain_speed)
        self.outcome = GameOutcome.ONGOING
        self.victory_enabled = False
        self.groups: dict[int, dict[int, str]] = {0: {}, 1: {}}
        self._perception: FogOfWar | None = None
        if spawn_armies:
            self._spawn_default_armies(army_size or wc.default_army_size, army_composition)

    @property
    def dt(self) -> float:
        return 1.0 / self.balance.world.simulation_hz

    @property
    def subpixels(self) -> int:
        return self.balance.world.subpixels

    def _spawn_default_armies(
        self, army_size: int, composition: dict[str, int] | None = None
    ) -> None:
        army_size = max(10, min(1000, army_size))
        remaining = army_size - 5
        if composition is None:
            infantry = round(remaining * 280 / 495)
            scouts = round(remaining * 60 / 495)
            engineers = round(remaining * 50 / 495)
            assassins = round(remaining * 30 / 495)
            recruits = remaining - infantry - scouts - engineers - assassins
        else:
            allowed = {"infantry", "scout", "engineer", "assassin", "recruit"}
            if set(composition) != allowed or any(value < 0 for value in composition.values()):
                raise ValueError("兵种分配包含未知兵种或负数。")
            if sum(composition.values()) != remaining:
                raise ValueError("兵种分配总数必须等于每方兵力减去将领与护卫。")
            infantry = composition["infantry"]
            scouts = composition["scout"]
            engineers = composition["engineer"]
            assassins = composition["assassin"]
            recruits = composition["recruit"]
        preset = [
            (UnitKind.COMMANDER, 1, 0),
            (UnitKind.GUARD, 4, 0),
            (UnitKind.INFANTRY, infantry, 1),
            (UnitKind.SCOUT, scouts, 2),
            (UnitKind.ENGINEER, engineers, 3),
            (UnitKind.ASSASSIN, assassins, 4),
            (UnitKind.RECRUIT, recruits, 5),
        ]
        for faction in (Faction.PLAYER, Faction.ENEMY):
            kinds: list[tuple[UnitKind, int]] = []
            for kind, amount, group in preset:
                kinds.extend((kind, group) for _ in range(amount))
            base_x = 310.0 if faction == Faction.PLAYER else self.map.width - 310.0
            direction = 1.0 if faction == Faction.PLAYER else -1.0
            for offset, (kind, group) in enumerate(kinds):
                if offset == 0:
                    x, y = base_x - direction * 88.0, self.map.height / 2
                elif offset < 5:
                    guard_offset = offset - 1
                    x = base_x - direction * 60.0
                    y = self.map.height / 2 + (guard_offset - 1.5) * 22.0
                else:
                    row = (offset - 5) // 25
                    col = (offset - 5) % 25
                    x = base_x + direction * (col * 15.0)
                    y = self.map.height / 2 - 150.0 + row * 16.0
                self.spawn_unit(faction, kind, x, y, group)
            self.groups[int(faction)] = {
                1: "第一战团",
                2: "第一侦察队",
                3: "工兵队",
                4: "影刃队",
                5: "预备队",
            }

    def spawn_unit(
        self,
        faction: Faction,
        kind: UnitKind,
        x: float,
        y: float,
        group_id: int = 0,
    ) -> int:
        entity_id = self.next_entity_id
        self.next_entity_id += 1
        stats = self.balance.units[kind.config_name]
        self.units.add(
            entity_id,
            faction,
            kind,
            float(np.clip(x, 4, self.map.width - 4)),
            float(np.clip(y, 4, self.map.height - 4)),
            stats,
            self.balance.world.simulation_hz,
            self.subpixels,
            group_id,
        )
        if kind == UnitKind.COMMANDER:
            self.victory_enabled = all(
                self.commander_index(side) is not None for side in (Faction.PLAYER, Faction.ENEMY)
            )
        return entity_id

    def commander_index(self, faction: Faction) -> int | None:
        mask = (
            self.units.alive[: self.units.count]
            & (self.units.faction[: self.units.count] == int(faction))
            & (self.units.kind[: self.units.count] == int(UnitKind.COMMANDER))
        )
        matches = np.flatnonzero(mask)
        return int(matches[0]) if len(matches) else None

    def resolve_selection(self, selection: SelectionV1, faction: Faction) -> np.ndarray:
        active = self.units.active(faction)
        if selection.unit_ids:
            wanted = set(selection.unit_ids)
            active = np.asarray(
                [idx for idx in active if int(self.units.entity_id[idx]) in wanted], dtype=np.int32
            )
        if selection.group_id is not None:
            active = active[self.units.group_id[active] == selection.group_id]
        if selection.unit_kind:
            try:
                kind = UnitKind[selection.unit_kind.upper()]
            except KeyError:
                return np.empty(0, dtype=np.int32)
            active = active[self.units.kind[active] == int(kind)]
        if selection.count is not None:
            active = active[: selection.count]
        return active

    def execute(self, command: CommandEnvelopeV1) -> CommandResultV1:
        if self.tick - command.issued_tick > self.balance.world.simulation_hz * 10:
            return self._result(
                command, CommandStatus.EXPIRED, "command_expired", "指令已过期，请重新下令。"
            )
        payload = command.payload
        try:
            if isinstance(payload, MovePayloadV1):
                result = self._execute_move(command, payload)
            elif isinstance(payload, AttackFacilityPayloadV1):
                result = self._execute_attack_facility(command, payload)
            elif isinstance(payload, GroupPayloadV1):
                result = self._execute_group(command, payload)
            elif isinstance(payload, ConvertPayloadV1):
                result = self._execute_convert(command, payload)
            elif isinstance(payload, TacticalPayloadV1):
                result = self._execute_tactical(command, payload)
            elif isinstance(payload, GuardPayloadV1):
                result = self._execute_guard(command, payload)
            elif isinstance(payload, FocusFirePayloadV1):
                result = self._execute_focus(command, payload)
            elif isinstance(payload, BuildPayloadV1):
                result = self._execute_build(command, payload)
            elif isinstance(payload, RecruitPayloadV1):
                result = self._execute_recruit(command, payload)
            elif isinstance(payload, FacilityActionPayloadV1):
                result = self._execute_facility_action(command, payload)
            else:
                result = self._result(
                    command, CommandStatus.REJECTED, "unsupported", "不支持该指令。"
                )
        except (ValueError, KeyError) as exc:
            result = self._result(
                command, CommandStatus.REJECTED, "invalid_payload", f"指令参数无效：{exc}"
            )
        if result.status == CommandStatus.ACCEPTED:
            self.command_log.append(command)
        self.command_results.append(result)
        return result

    def _result(
        self,
        command: CommandEnvelopeV1,
        status: CommandStatus,
        reason: str,
        message: str,
    ) -> CommandResultV1:
        event = GameEventV1(
            tick=self.tick,
            kind="command_result",
            visible_to=1 << int(command.faction),
            actor_id=command.issuer_entity_id,
            payload={"command_id": command.command_id, "status": status, "message": message},
        )
        self.events.append(event)
        return CommandResultV1(
            command_id=command.command_id, status=status, reason_code=reason, message_zh=message
        )

    def _execute_move(self, command: CommandEnvelopeV1, payload: MovePayloadV1) -> CommandResultV1:
        indices = self.resolve_selection(payload.selection, command.faction)
        if not len(indices):
            return self._result(
                command, CommandStatus.REJECTED, "empty_selection", "没有可执行指令的单位。"
            )
        x = float(np.clip(payload.target.x, 0, self.map.width))
        y = float(np.clip(payload.target.y, 0, self.map.height))
        crossing_loss = self._crossing_loss_estimate(indices, x, y)
        offsets = self._formation_offsets(len(indices), 18.0)
        order = int(Order.ATTACK_MOVE if payload.kind == "attack_move" else Order.MOVE)
        for position, index in enumerate(indices):
            target_x = round((x + float(offsets[position, 0])) * self.subpixels)
            target_y = round((y + float(offsets[position, 1])) * self.subpixels)
            entity_id = int(self.units.entity_id[index])
            if command.queue_mode == QueueMode.APPEND and self.units.order[index] != int(
                Order.IDLE
            ):
                self.order_queues.setdefault(entity_id, []).append((target_x, target_y, order))
            else:
                if command.queue_mode == QueueMode.REPLACE:
                    self.order_queues.pop(entity_id, None)
                self.units.target_x[index] = target_x
                self.units.target_y[index] = target_y
                self.units.order[index] = order
                self.units.focus_target[index] = -1
                self.units.focus_facility[index] = -1
        message = (
            f"已向 {len(indices)} 名单位下达"
            f"{'攻击移动' if payload.kind == 'attack_move' else '移动'}指令。"
        )
        if crossing_loss > 0:
            message += f" 预计强渡损失约 {crossing_loss:.0f} HP/人；建议架桥或登船。"
        return self._result(
            command,
            CommandStatus.ACCEPTED,
            "ok",
            message,
        )

    def _crossing_loss_estimate(
        self, indices: np.ndarray, target_x: float, target_y: float
    ) -> float:
        if any(
            facility.complete and facility.kind == FacilityKind.BRIDGE
            for facility in self.facilities
        ):
            return 0.0
        start_x = float(np.mean(self.units.x[indices]) / self.subpixels)
        start_y = float(np.mean(self.units.y[indices]) / self.subpixels)
        samples = 48
        ratios = np.linspace(0, 1, samples)
        xs = start_x + (target_x - start_x) * ratios
        ys = start_y + (target_y - start_y) * ratios
        river_fraction = float(np.mean(self.map.terrain_at(xs, ys) == int(Terrain.RIVER)))
        if river_fraction == 0:
            return 0.0
        distance = math.hypot(target_x - start_x, target_y - start_y)
        river_distance = distance * river_fraction
        river_speed = max(
            1.0,
            float(np.mean(self.units.speed[indices])) * self.balance.terrain_speed["river"],
        )
        return min(999.0, river_distance / river_speed * self.balance.river_damage_per_second)

    @staticmethod
    def _formation_offsets(count: int, spacing: float) -> np.ndarray:
        if count <= 1:
            return np.zeros((count, 2), dtype=np.float32)
        width = math.ceil(math.sqrt(count))
        offsets = np.zeros((count, 2), dtype=np.float32)
        for index in range(count):
            offsets[index] = (
                (index % width - (width - 1) / 2) * spacing,
                (index // width) * spacing,
            )
        offsets[:, 1] -= float(np.mean(offsets[:, 1]))
        return offsets

    def _execute_group(
        self, command: CommandEnvelopeV1, payload: GroupPayloadV1
    ) -> CommandResultV1:
        indices = self.resolve_selection(payload.selection, command.faction)
        if not len(indices):
            return self._result(
                command, CommandStatus.REJECTED, "empty_selection", "没有可编组的单位。"
            )
        self.units.group_id[indices] = payload.group_id
        if payload.name:
            self.groups[int(command.faction)][payload.group_id] = payload.name
        return self._result(
            command,
            CommandStatus.ACCEPTED,
            "ok",
            f"已将 {len(indices)} 名单位编入第 {payload.group_id} 组。",
        )

    def _execute_convert(
        self, command: CommandEnvelopeV1, payload: ConvertPayloadV1
    ) -> CommandResultV1:
        indices = self.resolve_selection(payload.selection, command.faction)
        indices = indices[self.units.kind[indices] == int(UnitKind.RECRUIT)]
        if not len(indices):
            return self._result(
                command, CommandStatus.REJECTED, "no_recruits", "选区内没有可分化的初始兵。"
            )
        new_kind = UnitKind[payload.unit_kind.upper()]
        stats = self.balance.units[new_kind.config_name]
        for index in indices:
            hp_ratio = float(self.units.hp[index] / max(self.units.max_hp[index], 1))
            self.units.kind[index] = int(new_kind)
            self._apply_stats(int(index), stats, hp_ratio)
            self.units.exposed[index] = new_kind != UnitKind.ASSASSIN
        return self._result(
            command, CommandStatus.ACCEPTED, "ok", f"已分化 {len(indices)} 名{payload.unit_kind}。"
        )

    def _apply_stats(self, index: int, stats: UnitStats, hp_ratio: float = 1.0) -> None:
        self.units.max_hp[index] = stats.hp
        self.units.hp[index] = stats.hp * hp_ratio
        self.units.damage[index] = stats.damage
        self.units.attack_interval[index] = max(
            1, round(stats.attack_interval * self.balance.world.simulation_hz)
        )
        self.units.speed[index] = stats.speed
        self.units.vision[index] = stats.vision
        self.units.detection[index] = stats.detection
        self.units.radius[index] = stats.radius
        self.units.attack_range[index] = stats.attack_range

    def _execute_tactical(
        self, command: CommandEnvelopeV1, payload: TacticalPayloadV1
    ) -> CommandResultV1:
        indices = self.resolve_selection(payload.selection, command.faction)
        eligible = indices[(self.units.tactic_cooldown[indices] <= 0)]
        cost_key = "sprint_cost" if payload.kind == "sprint" else "charge_cost"
        cost = self.balance.tactics[cost_key]
        eligible = eligible[self.units.stamina[eligible] >= cost]
        if not len(eligible):
            return self._result(
                command,
                CommandStatus.REJECTED,
                "stamina_or_cooldown",
                "没有单位满足耐力与冷却要求。",
            )
        average_strength = float(np.mean(self._tactic_strength(eligible)))
        self.units.stamina[eligible] -= cost
        self.units.tactic_kind[eligible] = 1 if payload.kind == "sprint" else 2
        duration = self.balance.tactics[f"{payload.kind}_duration"]
        self.units.tactic_ticks[eligible] = round(duration * self.balance.world.simulation_hz)
        self.units.tactic_cooldown[eligible] = round(
            self.balance.tactics["cooldown"] * self.balance.world.simulation_hz
        )
        self.units.last_intense_tick[eligible] = self.tick
        if payload.target is not None:
            offsets = self._formation_offsets(len(eligible), 18.0)
            self.units.target_x[eligible] = np.rint(
                (payload.target.x + offsets[:, 0]) * self.subpixels
            ).astype(np.int32)
            self.units.target_y[eligible] = np.rint(
                (payload.target.y + offsets[:, 1]) * self.subpixels
            ).astype(np.int32)
            self.units.order[eligible] = int(
                Order.ATTACK_MOVE if payload.kind == "charge" else Order.MOVE
            )
        configured_speed = self.balance.tactics[f"{payload.kind}_speed_multiplier"] - 1
        effective_speed = configured_speed * average_strength * 100
        effective_damage = (
            (self.balance.tactics["charge_damage_multiplier"] - 1) * average_strength * 100
        )
        extra = (
            f"平均实际速度增益 {effective_speed:.0f}%"
            if payload.kind == "sprint"
            else (f"平均实际速度增益 {effective_speed:.0f}%，接战伤害增益 {effective_damage:.0f}%")
        )
        return self._result(
            command,
            CommandStatus.ACCEPTED,
            "ok",
            (
                f"{len(eligible)} 名单位开始"
                f"{'全速前进' if payload.kind == 'sprint' else '冲锋'}；{extra}。"
            ),
        )

    def _execute_guard(
        self, command: CommandEnvelopeV1, payload: GuardPayloadV1
    ) -> CommandResultV1:
        indices = self.resolve_selection(payload.selection, command.faction)
        if not len(indices):
            return self._result(
                command, CommandStatus.REJECTED, "empty_selection", "没有可执行守卫任务的单位。"
            )
        offsets = self._formation_offsets(len(indices), 20.0)
        self.units.target_x[indices] = np.rint(
            (payload.target.x + offsets[:, 0]) * self.subpixels
        ).astype(np.int32)
        self.units.target_y[indices] = np.rint(
            (payload.target.y + offsets[:, 1]) * self.subpixels
        ).astype(np.int32)
        self.units.order[indices] = int(Order.GUARD)
        self.units.focus_target[indices] = -1
        return self._result(
            command, CommandStatus.ACCEPTED, "ok", f"{len(indices)} 名单位开始守卫目标区域。"
        )

    def _execute_focus(
        self, command: CommandEnvelopeV1, payload: FocusFirePayloadV1
    ) -> CommandResultV1:
        indices = self.resolve_selection(payload.selection, command.faction)
        target = self.units.index_of(payload.target_entity_id)
        visible = {unit.entity_id for unit in self.observation(command.faction).visible_enemies}
        if (
            not len(indices)
            or target is None
            or not self.units.alive[target]
            or payload.target_entity_id not in visible
        ):
            return self._result(
                command,
                CommandStatus.REJECTED,
                "target_not_visible",
                "集火目标当前不可见或已失效。",
            )
        self.units.focus_target[indices] = payload.target_entity_id
        self.units.focus_facility[indices] = -1
        self.units.order[indices] = int(Order.ATTACK_MOVE)
        self.units.target_x[indices] = self.units.x[target]
        self.units.target_y[indices] = self.units.y[target]
        return self._result(
            command, CommandStatus.ACCEPTED, "ok", f"{len(indices)} 名单位开始集火目标。"
        )

    def _execute_attack_facility(
        self, command: CommandEnvelopeV1, payload: AttackFacilityPayloadV1
    ) -> CommandResultV1:
        indices = self.resolve_selection(payload.selection, command.faction)
        facility = next(
            (
                item
                for item in self.facilities
                if item.facility_id == payload.target_facility_id
                and item.faction != int(command.faction)
                and not item.destroyed
            ),
            None,
        )
        known = {
            int(item["facility_id"]) for item in self.observation(command.faction).known_facilities
        }
        if not len(indices) or facility is None or facility.facility_id not in known:
            return self._result(
                command,
                CommandStatus.REJECTED,
                "facility_not_visible",
                "目标设施当前不可见或已被摧毁。",
            )
        self.units.focus_target[indices] = -1
        self.units.focus_facility[indices] = facility.facility_id
        self.units.order[indices] = int(Order.ATTACK_MOVE)
        self.units.target_x[indices] = round(facility.x * self.subpixels)
        self.units.target_y[indices] = round(facility.y * self.subpixels)
        return self._result(
            command, CommandStatus.ACCEPTED, "ok", f"{len(indices)} 名单位开始攻击设施。"
        )

    def _execute_build(
        self, command: CommandEnvelopeV1, payload: BuildPayloadV1
    ) -> CommandResultV1:
        indices = self.resolve_selection(payload.selection, command.faction)
        indices = indices[self.units.kind[indices] == int(UnitKind.ENGINEER)]
        spec = self.balance.facilities[payload.facility_kind]
        minimum = int(spec["minimum_engineers"])
        if len(indices) < minimum:
            return self._result(
                command,
                CommandStatus.REJECTED,
                "not_enough_engineers",
                f"至少需要 {minimum} 名工兵。",
            )
        x, y = float(payload.target.x), float(payload.target.y)
        terrain = Terrain(int(self.map.terrain_at(x, y)))
        if payload.facility_kind in {"bridge", "boat"} and terrain != Terrain.RIVER:
            return self._result(
                command, CommandStatus.REJECTED, "requires_river", "桥梁或船只必须在河流区域建造。"
            )
        if payload.facility_kind == "road" and terrain != Terrain.FOREST:
            return self._result(
                command, CommandStatus.REJECTED, "requires_forest", "道路必须在森林区域开辟。"
            )
        if payload.facility_kind == "tower" and terrain in {Terrain.RIVER, Terrain.SWAMP}:
            return self._result(
                command, CommandStatus.REJECTED, "invalid_terrain", "该地形不能建造防御塔。"
            )
        facility = Facility(
            facility_id=self.next_facility_id,
            faction=int(command.faction),
            kind=payload.facility_kind,
            x=x,
            y=y,
            progress=0.0,
            required_work=float(spec["work"]),
            minimum_engineers=minimum,
            hp=float(spec["hp"]),
            max_hp=float(spec["hp"]),
            complete=False,
            builder_ids=[int(self.units.entity_id[index]) for index in indices],
        )
        self.next_facility_id += 1
        self.facilities.append(facility)
        self.units.order[indices] = int(Order.BUILD)
        self.units.facility_id[indices] = facility.facility_id
        self.units.target_x[indices] = round(x * self.subpixels)
        self.units.target_y[indices] = round(y * self.subpixels)
        return self._result(
            command,
            CommandStatus.ACCEPTED,
            "ok",
            f"{len(indices)} 名工兵开始建造{payload.facility_kind}。",
        )

    def _execute_recruit(
        self, command: CommandEnvelopeV1, payload: RecruitPayloadV1
    ) -> CommandResultV1:
        village = next(
            (item for item in self.map.villages if item.village_id == payload.village_id), None
        )
        if village is None:
            return self._result(
                command, CommandStatus.REJECTED, "unknown_village", "未找到该村庄。"
            )
        friendly = self.units.active(command.faction)
        enemy = self.units.active(
            Faction.ENEMY if command.faction == Faction.PLAYER else Faction.PLAYER
        )
        fx = self.units.x[friendly] / self.subpixels
        fy = self.units.y[friendly] / self.subpixels
        ex = self.units.x[enemy] / self.subpixels
        ey = self.units.y[enemy] / self.subpixels
        has_friendly = bool(
            np.any((fx - village.x) ** 2 + (fy - village.y) ** 2 <= village.radius**2)
        )
        contested = bool(np.any((ex - village.x) ** 2 + (ey - village.y) ** 2 <= village.radius**2))
        if not has_friendly or contested:
            return self._result(
                command,
                CommandStatus.REJECTED,
                "village_unavailable",
                "需要己方接近且村庄范围内不能交战。",
            )
        amount = min(payload.count, village.population)
        if amount <= 0:
            return self._result(
                command, CommandStatus.REJECTED, "village_empty", "村庄已无可征召人口。"
            )
        village.population -= amount
        for number in range(amount):
            angle = number * 2.399963
            radius = 24 + 5 * math.sqrt(number)
            self.spawn_unit(
                command.faction,
                UnitKind.RECRUIT,
                village.x + math.cos(angle) * radius,
                village.y + math.sin(angle) * radius,
                5,
            )
        return self._result(command, CommandStatus.ACCEPTED, "ok", f"已征召 {amount} 名初始兵。")

    def _execute_facility_action(
        self, command: CommandEnvelopeV1, payload: FacilityActionPayloadV1
    ) -> CommandResultV1:
        indices = self.resolve_selection(payload.selection, command.faction)
        expected_kind = "tower" if payload.kind in {"enter_tower", "exit_tower"} else "boat"
        facility = self._find_facility(command.faction, expected_kind, payload.facility_id, indices)
        if facility is None or not facility.complete:
            return self._result(
                command, CommandStatus.REJECTED, "facility_unavailable", "未找到可用的己方设施。"
            )
        if payload.kind == "sail_boat":
            if payload.target is None:
                return self._result(
                    command, CommandStatus.REJECTED, "missing_target", "请指定船只航行目标。"
                )
            facility.target_x = float(np.clip(payload.target.x, 0, self.map.width))
            facility.target_y = float(np.clip(payload.target.y, 0, self.map.height))
            return self._result(command, CommandStatus.ACCEPTED, "ok", "船只开始航行。")
        if payload.kind in {"exit_tower", "disembark"}:
            leaving = indices[self.units.facility_id[indices] == facility.facility_id]
            if not len(leaving):
                return self._result(
                    command, CommandStatus.REJECTED, "no_occupants", "所选单位不在该设施内。"
                )
            for number, index in enumerate(leaving):
                angle = number * 2.399963
                radius = 30 + 4 * math.sqrt(number)
                self.units.x[index] = round(
                    np.clip(facility.x + math.cos(angle) * radius, 0, self.map.width)
                    * self.subpixels
                )
                self.units.y[index] = round(
                    np.clip(facility.y + math.sin(angle) * radius, 0, self.map.height)
                    * self.subpixels
                )
                self.units.target_x[index] = self.units.x[index]
                self.units.target_y[index] = self.units.y[index]
                self.units.facility_id[index] = -1
                self.units.order[index] = int(Order.IDLE)
                if payload.kind == "exit_tower":
                    hp_ratio = float(self.units.hp[index] / max(self.units.max_hp[index], 1))
                    self._apply_stats(int(index), self.balance.units["infantry"], hp_ratio)
            label = "离开防御塔" if payload.kind == "exit_tower" else "完成下船"
            return self._result(
                command, CommandStatus.ACCEPTED, "ok", f"{len(leaving)} 名单位已{label}。"
            )
        nearby = indices[self.units.facility_id[indices] < 0]
        if payload.kind == "enter_tower":
            nearby = nearby[self.units.kind[nearby] == int(UnitKind.INFANTRY)]
        if len(nearby):
            dx = self.units.x[nearby] / self.subpixels - facility.x
            dy = self.units.y[nearby] / self.subpixels - facility.y
            nearby = nearby[dx * dx + dy * dy <= 144**2]
        capacity = int(self.balance.facilities[expected_kind].get("capacity", 20))
        occupied = int(np.sum(self.units.facility_id[: self.units.count] == facility.facility_id))
        entering = nearby[: max(0, capacity - occupied)]
        if not len(entering):
            return self._result(
                command,
                CommandStatus.REJECTED,
                "capacity_or_distance",
                "设施已满，或所选单位距离设施过远。",
            )
        self.units.facility_id[entering] = facility.facility_id
        self.units.x[entering] = round(facility.x * self.subpixels)
        self.units.y[entering] = round(facility.y * self.subpixels)
        self.units.target_x[entering] = self.units.x[entering]
        self.units.target_y[entering] = self.units.y[entering]
        self.units.order[entering] = int(Order.IDLE)
        if payload.kind == "enter_tower":
            self.units.attack_range[entering] = float(
                self.balance.facilities["tower"]["attack_range"]
            )
            self.units.vision[entering] = float(self.balance.facilities["tower"]["vision"])
        label = "进入防御塔" if payload.kind == "enter_tower" else "完成登船"
        return self._result(
            command, CommandStatus.ACCEPTED, "ok", f"{len(entering)} 名单位已{label}。"
        )

    def _find_facility(
        self,
        faction: Faction,
        kind: str,
        facility_id: int | None,
        indices: np.ndarray,
    ) -> Facility | None:
        candidates = [
            facility
            for facility in self.facilities
            if facility.faction == int(faction)
            and facility.kind == kind
            and not facility.destroyed
            and (facility_id is None or facility.facility_id == facility_id)
        ]
        if not candidates:
            return None
        if facility_id is not None or not len(indices):
            return candidates[0]
        x = float(np.mean(self.units.x[indices]) / self.subpixels)
        y = float(np.mean(self.units.y[indices]) / self.subpixels)
        return min(candidates, key=lambda item: (item.x - x) ** 2 + (item.y - y) ** 2)

    def step(self, steps: int = 1) -> None:
        for _ in range(steps):
            if self.outcome != GameOutcome.ONGOING:
                return
            self.units.previous_x[: self.units.count] = self.units.x[: self.units.count]
            self.units.previous_y[: self.units.count] = self.units.y[: self.units.count]
            self._update_timers()
            self._update_guard_following()
            self._move_units()
            collision_due = self.tick % 5 == 1
            if collision_due:
                self._resolve_collisions()
            if self.tick % 4 == 0:
                self._update_assassin_exposure()
            if self.tick % 4 == 0 and not collision_due:
                self._resolve_combat()
            self._update_engineering()
            self._update_boats()
            self._apply_river_damage()
            self._check_victory()
            if self._perception is not None:
                if self.tick % 10 == 0:
                    self._perception.update(self)
                else:
                    self._perception.invalidate_dynamic()
            self.tick += 1

    def _update_timers(self) -> None:
        active = self.units.active()
        self.units.cooldown[active] = np.maximum(0, self.units.cooldown[active] - 1)
        self.units.tactic_ticks[active] = np.maximum(0, self.units.tactic_ticks[active] - 1)
        self.units.tactic_cooldown[active] = np.maximum(0, self.units.tactic_cooldown[active] - 1)
        ended = active[self.units.tactic_ticks[active] == 0]
        self.units.tactic_kind[ended] = 0
        recovery_ticks = round(
            self.balance.tactics["recovery_delay"] * self.balance.world.simulation_hz
        )
        recovering = active[(self.tick - self.units.last_intense_tick[active]) >= recovery_ticks]
        self.units.stamina[recovering] = np.minimum(
            self.balance.tactics["max_stamina"],
            self.units.stamina[recovering] + self.balance.tactics["recovery_per_second"] * self.dt,
        )

    def _update_guard_following(self) -> None:
        offsets = ((-34, -28), (-34, 28), (34, -28), (34, 28))
        for faction in (Faction.PLAYER, Faction.ENEMY):
            commander = self.commander_index(faction)
            if commander is None:
                continue
            guards = np.flatnonzero(
                self.units.alive[: self.units.count]
                & (self.units.faction[: self.units.count] == int(faction))
                & (self.units.kind[: self.units.count] == int(UnitKind.GUARD))
            )
            for position, guard in enumerate(guards):
                dx = (self.units.x[guard] - self.units.x[commander]) / self.subpixels
                dy = (self.units.y[guard] - self.units.y[commander]) / self.subpixels
                if dx * dx + dy * dy <= 80**2:
                    continue
                offset_x, offset_y = offsets[position % len(offsets)]
                self.units.target_x[guard] = self.units.x[commander] + offset_x * self.subpixels
                self.units.target_y[guard] = self.units.y[commander] + offset_y * self.subpixels
                self.units.order[guard] = int(Order.GUARD)

    def _tactic_strength(self, indices: np.ndarray) -> np.ndarray:
        strength = np.zeros(len(indices), dtype=np.float32)
        for faction in (Faction.PLAYER, Faction.ENEMY):
            commander = self.commander_index(faction)
            faction_mask = self.units.faction[indices] == int(faction)
            if commander is None or not np.any(faction_mask):
                continue
            selected = indices[faction_mask]
            dx = (self.units.x[selected] - self.units.x[commander]) / self.subpixels
            dy = (self.units.y[selected] - self.units.y[commander]) / self.subpixels
            distance = np.sqrt(dx * dx + dy * dy)
            near = self.balance.tactics["full_effect_distance"]
            far = self.balance.tactics["max_effect_distance"]
            strength[faction_mask] = np.clip(1 - (distance - near) / max(far - near, 1), 0, 1)
        return strength

    def _move_units(self) -> None:
        active = self.units.active()
        movable = active[self.units.order[active] != int(Order.IDLE)]
        if not len(movable):
            return
        dx = (self.units.target_x[movable] - self.units.x[movable]).astype(
            np.float64
        ) / self.subpixels
        dy = (self.units.target_y[movable] - self.units.y[movable]).astype(
            np.float64
        ) / self.subpixels
        distance = np.sqrt(dx * dx + dy * dy)
        moving = distance > 1.0
        arrived = movable[~moving]
        if len(arrived):
            self._advance_orders(arrived)
        if not np.any(moving):
            return
        indices = movable[moving]
        dx, dy, distance = dx[moving], dy[moving], distance[moving]
        target_cols = self.units.target_x[indices] // self.subpixels // self.map.tile_size // 8
        target_rows = self.units.target_y[indices] // self.subpixels // self.map.tile_size // 8
        for target_col, target_row in np.unique(
            np.column_stack((target_cols, target_rows)), axis=0
        ):
            group_mask = (
                (target_cols == target_col)
                & (target_rows == target_row)
                & (distance > self.map.tile_size * 2)
            )
            if not np.any(group_mask):
                continue
            group = indices[group_mask]
            field = self._flow_fields.get(
                float(self.units.target_x[group[0]] / self.subpixels),
                float(self.units.target_y[group[0]] / self.subpixels),
            )
            cols = np.clip(
                self.units.x[group] // self.subpixels // self.map.tile_size,
                0,
                self.map.cols - 1,
            )
            rows = np.clip(
                self.units.y[group] // self.subpixels // self.map.tile_size,
                0,
                self.map.rows - 1,
            )
            steer_x = field.direction_x[rows, cols].astype(np.float64)
            steer_y = field.direction_y[rows, cols].astype(np.float64)
            usable = (steer_x != 0) | (steer_y != 0)
            positions = np.flatnonzero(group_mask)
            dx[positions[usable]] = steer_x[usable]
            dy[positions[usable]] = steer_y[usable]
        direction_length = np.sqrt(dx * dx + dy * dy)
        terrain = self.map.terrain_at(
            self.units.x[indices] / self.subpixels, self.units.y[indices] / self.subpixels
        )
        speed_table = np.asarray(
            [
                self.balance.terrain_speed["plain"],
                self.balance.terrain_speed["grass"],
                self.balance.terrain_speed["forest"],
                self.balance.terrain_speed["swamp"],
                self.balance.terrain_speed["river"],
                self.balance.terrain_speed["road"],
            ],
            dtype=np.float32,
        )
        terrain_multiplier = speed_table[terrain]
        for facility in self.facilities:
            if facility.complete and facility.kind == FacilityKind.BRIDGE:
                bx = self.units.x[indices] / self.subpixels - facility.x
                by = self.units.y[indices] / self.subpixels - facility.y
                on_bridge = bx * bx + by * by < 52**2
                terrain_multiplier[on_bridge] = 0.95
        tactic = self.units.tactic_kind[indices]
        strength = self._tactic_strength(indices)
        multiplier = np.ones(len(indices), dtype=np.float32)
        sprint = tactic == 1
        charge = tactic == 2
        multiplier[sprint] += (self.balance.tactics["sprint_speed_multiplier"] - 1) * strength[
            sprint
        ]
        multiplier[charge] += (self.balance.tactics["charge_speed_multiplier"] - 1) * strength[
            charge
        ]
        step = np.minimum(
            distance, self.units.speed[indices] * terrain_multiplier * multiplier * self.dt
        )
        self.units.x[indices] += np.rint(dx / direction_length * step * self.subpixels).astype(
            np.int32
        )
        self.units.y[indices] += np.rint(dy / direction_length * step * self.subpixels).astype(
            np.int32
        )
        self.units.x[indices] = np.clip(self.units.x[indices], 0, self.map.width * self.subpixels)
        self.units.y[indices] = np.clip(self.units.y[indices], 0, self.map.height * self.subpixels)

    def _advance_orders(self, arrived: np.ndarray) -> None:
        for index in arrived:
            if self.units.facility_id[index] >= 0 or self.units.order[index] == int(Order.GUARD):
                continue
            entity_id = int(self.units.entity_id[index])
            queue = self.order_queues.get(entity_id)
            if queue:
                target_x, target_y, order = queue.pop(0)
                self.units.target_x[index] = target_x
                self.units.target_y[index] = target_y
                self.units.order[index] = order
                if not queue:
                    self.order_queues.pop(entity_id, None)
            else:
                self.units.order[index] = int(Order.IDLE)

    def _resolve_collisions(self) -> None:
        active = self.units.active()
        cell_size = 20.0
        buckets: dict[tuple[int, int], list[int]] = {}
        for index in active:
            x = int(self.units.x[index] / self.subpixels / cell_size)
            y = int(self.units.y[index] / self.subpixels / cell_size)
            buckets.setdefault((x, y), []).append(int(index))
        for key, members in buckets.items():
            neighbor_keys = ((0, 0), (1, 0), (0, 1), (1, 1), (-1, 1))
            for ox, oy in neighbor_keys:
                candidates = buckets.get((key[0] + ox, key[1] + oy), ())
                for first_position, first in enumerate(members):
                    start = first_position + 1 if ox == 0 and oy == 0 else 0
                    available = len(candidates) - start
                    sample_count = min(6, max(0, available))
                    sample_start = (
                        start + int(self.units.entity_id[first]) % available
                        if available > sample_count
                        else start
                    )
                    for offset in range(sample_count):
                        second = candidates[start + (sample_start - start + offset) % available]
                        dx = float(self.units.x[second] - self.units.x[first]) / self.subpixels
                        dy = float(self.units.y[second] - self.units.y[first]) / self.subpixels
                        minimum = float(self.units.radius[first] + self.units.radius[second]) * 0.78
                        distance_sq = dx * dx + dy * dy
                        if 0 < distance_sq < minimum * minimum:
                            distance = math.sqrt(distance_sq)
                            push = min((minimum - distance) * 0.35, 2.0) * self.subpixels
                            px, py = dx / distance * push, dy / distance * push
                            self.units.x[first] -= round(px)
                            self.units.y[first] -= round(py)
                            self.units.x[second] += round(px)
                            self.units.y[second] += round(py)

    def _update_assassin_exposure(self) -> None:
        hidden = np.flatnonzero(
            self.units.alive[: self.units.count]
            & (self.units.kind[: self.units.count] == int(UnitKind.ASSASSIN))
            & ~self.units.exposed[: self.units.count]
        )
        for assassin in hidden:
            enemies = self.units.active(Faction(1 - int(self.units.faction[assassin])))
            dx = (self.units.x[enemies] - self.units.x[assassin]) / self.subpixels
            dy = (self.units.y[enemies] - self.units.y[assassin]) / self.subpixels
            if np.any(dx * dx + dy * dy <= self.units.detection[enemies] ** 2):
                self.units.exposed[assassin] = True
                self.events.append(
                    GameEventV1(
                        tick=self.tick,
                        kind="assassin_exposed",
                        actor_id=int(self.units.entity_id[assassin]),
                    )
                )

    def _resolve_combat(self) -> None:
        active = self.units.active()
        boat_ids = {
            facility.facility_id
            for facility in self.facilities
            if facility.complete and facility.kind == FacilityKind.BOAT
        }
        if boat_ids:
            active = active[~np.isin(self.units.facility_id[active], list(boat_ids))]
        attackers = active
        terrain = self.map.terrain_at(
            self.units.x[attackers] / self.subpixels,
            self.units.y[attackers] / self.subpixels,
        )
        crossing = terrain == int(Terrain.RIVER)
        if np.any(crossing):
            protected = np.zeros(len(attackers), dtype=np.bool_)
            for facility in self.facilities:
                if not facility.complete or facility.kind != FacilityKind.BRIDGE:
                    continue
                dx = self.units.x[attackers] / self.subpixels - facility.x
                dy = self.units.y[attackers] / self.subpixels - facility.y
                protected |= dx * dx + dy * dy < 52**2
            attackers = attackers[~crossing | protected]
        buckets: list[dict[tuple[int, int], list[int]]] = [{}, {}]
        cell = 64.0
        for index in active:
            if self.units.kind[index] == int(UnitKind.ASSASSIN) and not self.units.exposed[index]:
                continue
            key = (
                int(self.units.x[index] / self.subpixels / cell),
                int(self.units.y[index] / self.subpixels / cell),
            )
            buckets[int(self.units.faction[index])].setdefault(key, []).append(int(index))
        attacks: list[tuple[int, int, float]] = []
        facility_attacks: list[tuple[int, int, float]] = []
        engagement_counts: dict[int, int] = {}
        visible_targets: list[set[int]] = [set(), set()]
        if self._perception is not None and np.any(self.units.focus_target[attackers] >= 0):
            for faction in (Faction.PLAYER, Faction.ENEMY):
                visible_targets[int(faction)] = {
                    unit.entity_id for unit in self.observation(faction).visible_enemies
                }
        visible_facilities: list[set[int]] = [set(), set()]
        if self._perception is not None and np.any(self.units.focus_facility[attackers] >= 0):
            for faction in (Faction.PLAYER, Faction.ENEMY):
                visible_facilities[int(faction)] = {
                    int(item["facility_id"])
                    for item in self.observation(faction).known_facilities
                    if int(item["faction"]) != int(faction)
                }
        for attacker in attackers:
            if self.units.cooldown[attacker] > 0 or self.units.order[attacker] == int(Order.BUILD):
                continue
            focus_facility_id = int(self.units.focus_facility[attacker])
            if focus_facility_id >= 0:
                focused_facility = next(
                    (
                        item
                        for item in self.facilities
                        if item.facility_id == focus_facility_id and not item.destroyed
                    ),
                    None,
                )
                if (
                    focused_facility is None
                    or focus_facility_id
                    not in visible_facilities[int(self.units.faction[attacker])]
                ):
                    self.units.focus_facility[attacker] = -1
                else:
                    dx = focused_facility.x - self.units.x[attacker] / self.subpixels
                    dy = focused_facility.y - self.units.y[attacker] / self.subpixels
                    distance = math.hypot(dx, dy)
                    if distance <= float(self.units.attack_range[attacker]) + 12:
                        facility_attacks.append(
                            (
                                int(attacker),
                                focused_facility.facility_id,
                                float(self.units.damage[attacker]),
                            )
                        )
                        self.units.cooldown[attacker] = self.units.attack_interval[attacker]
                    else:
                        self.units.target_x[attacker] = round(focused_facility.x * self.subpixels)
                        self.units.target_y[attacker] = round(focused_facility.y * self.subpixels)
                    continue
            key = (
                int(self.units.x[attacker] / self.subpixels / cell),
                int(self.units.y[attacker] / self.subpixels / cell),
            )
            best_target: int | None = None
            best_distance = float("inf")
            focus_id = int(self.units.focus_target[attacker])
            if focus_id >= 0:
                focused = self.units.index_of(focus_id)
                if (
                    focused is not None
                    and self.units.alive[focused]
                    and focus_id in visible_targets[int(self.units.faction[attacker])]
                ):
                    best_target = focused
                else:
                    self.units.focus_target[attacker] = -1
            if best_target is None:
                for ox in (-1, 0, 1):
                    for oy in (-1, 0, 1):
                        opposing = buckets[1 - int(self.units.faction[attacker])]
                        candidates = opposing.get((key[0] + ox, key[1] + oy), ())
                        sample_count = min(4, len(candidates))
                        start = (
                            int(self.units.entity_id[attacker]) % len(candidates)
                            if len(candidates) > sample_count
                            else 0
                        )
                        for offset in range(sample_count):
                            target = candidates[(start + offset) % len(candidates)]
                            dx = (
                                float(self.units.x[target] - self.units.x[attacker])
                                / self.subpixels
                            )
                            dy = (
                                float(self.units.y[target] - self.units.y[attacker])
                                / self.subpixels
                            )
                            distance_sq = dx * dx + dy * dy
                            if distance_sq < best_distance:
                                best_target, best_distance = target, distance_sq
            if best_target is None:
                continue
            best_target = self._guard_intercept(best_target)
            dx = float(self.units.x[best_target] - self.units.x[attacker]) / self.subpixels
            dy = float(self.units.y[best_target] - self.units.y[attacker]) / self.subpixels
            distance = math.sqrt(dx * dx + dy * dy)
            attack_range = float(self.units.attack_range[attacker] + self.units.radius[best_target])
            if distance <= attack_range:
                melee = self.units.attack_range[attacker] <= 30
                if melee and engagement_counts.get(best_target, 0) >= 6:
                    continue
                if melee:
                    engagement_counts[best_target] = engagement_counts.get(best_target, 0) + 1
                if self.units.kind[attacker] == int(UnitKind.ASSASSIN):
                    self.units.exposed[attacker] = True
                attack_damage = float(self.units.damage[attacker])
                charge_damage_threshold = round(
                    (
                        self.balance.tactics["charge_duration"]
                        - self.balance.tactics["charge_damage_duration"]
                    )
                    * self.balance.world.simulation_hz
                )
                if (
                    self.units.tactic_kind[attacker] == 2
                    and self.units.tactic_ticks[attacker] > charge_damage_threshold
                ):
                    attack_damage *= self.balance.tactics["charge_damage_multiplier"]
                attacks.append((int(attacker), int(best_target), attack_damage))
                self.units.cooldown[attacker] = self.units.attack_interval[attacker]
                self.units.last_intense_tick[attacker] = self.tick
            elif self.units.order[attacker] == int(Order.ATTACK_MOVE) and distance < 180:
                self.units.target_x[attacker] = self.units.x[best_target]
                self.units.target_y[attacker] = self.units.y[best_target]
        if facility_attacks:
            facility_damage: dict[int, float] = {}
            for _attacker, facility_id, amount in facility_attacks:
                facility_damage[facility_id] = facility_damage.get(facility_id, 0.0) + amount
            for facility_id, amount in facility_damage.items():
                facility = next(item for item in self.facilities if item.facility_id == facility_id)
                facility.hp -= amount
                if facility.hp <= 0:
                    self._destroy_facility(facility)
        if not attacks:
            return
        damage_totals = np.zeros(self.units.count, dtype=np.float32)
        for _attacker, target, amount in attacks:
            damage_totals[target] += amount
        first_attacker, first_target, _ = attacks[0]
        self.events.append(
            GameEventV1(
                tick=self.tick,
                kind="combat_exchange",
                actor_id=int(self.units.entity_id[first_attacker]),
                target_id=int(self.units.entity_id[first_target]),
                payload={
                    "attacks": len(attacks),
                    "total_damage": round(sum(item[2] for item in attacks), 2),
                },
            )
        )
        targets = np.flatnonzero(damage_totals > 0)
        self.units.hp[targets] -= damage_totals[targets]
        dead = targets[self.units.hp[targets] <= 0]
        for index in dead:
            self._kill_unit(int(index))

    def _guard_intercept(self, target: int) -> int:
        if self.units.kind[target] != int(UnitKind.COMMANDER):
            return target
        faction = self.units.faction[target]
        guards = np.flatnonzero(
            self.units.alive[: self.units.count]
            & (self.units.faction[: self.units.count] == faction)
            & (self.units.kind[: self.units.count] == int(UnitKind.GUARD))
        )
        if not len(guards):
            return target
        dx = (self.units.x[guards] - self.units.x[target]) / self.subpixels
        dy = (self.units.y[guards] - self.units.y[target]) / self.subpixels
        nearby = guards[dx * dx + dy * dy <= 160**2]
        return int(nearby[0]) if len(nearby) else target

    def _update_engineering(self) -> None:
        for facility in self.facilities:
            if facility.complete or facility.destroyed:
                continue
            builders: list[int] = []
            for entity_id in facility.builder_ids:
                index = self.units.index_of(entity_id)
                if (
                    index is None
                    or not self.units.alive[index]
                    or self.units.kind[index] != int(UnitKind.ENGINEER)
                ):
                    continue
                dx = self.units.x[index] / self.subpixels - facility.x
                dy = self.units.y[index] / self.subpixels - facility.y
                if dx * dx + dy * dy <= 48**2:
                    builders.append(index)
            if len(builders) < facility.minimum_engineers:
                continue
            facility.elapsed_ticks += 1
            efficiency = float(self.balance.facilities[facility.kind]["engineer_work_per_second"])
            facility.progress += len(builders) * efficiency * self.dt
            minimum_ticks = round(
                float(self.balance.facilities[facility.kind]["minimum_build_seconds"])
                * self.balance.world.simulation_hz
            )
            if (
                facility.progress >= facility.required_work
                and facility.elapsed_ticks >= minimum_ticks
            ):
                facility.progress = facility.required_work
                facility.complete = True
                builder_array = np.asarray(builders, dtype=np.int32)
                self.units.facility_id[builder_array] = -1
                self.units.order[builder_array] = int(Order.IDLE)
                if facility.kind == FacilityKind.ROAD:
                    self.map.clear_forest(facility.x, facility.y)
                self.events.append(
                    GameEventV1(
                        tick=self.tick,
                        kind="facility_completed",
                        visible_to=1 << facility.faction,
                        payload={"facility_id": facility.facility_id, "kind": facility.kind},
                    )
                )

    def _destroy_facility(self, facility: Facility) -> None:
        facility.hp = 0
        facility.complete = False
        facility.destroyed = True
        occupants = np.flatnonzero(
            self.units.facility_id[: self.units.count] == facility.facility_id
        )
        damage = (
            float(self.balance.facilities["boat"].get("destroyed_damage", 40.0))
            if facility.kind == FacilityKind.BOAT
            else 0.0
        )
        for number, index in enumerate(occupants):
            if damage:
                self.units.hp[index] -= damage
            angle = number * 2.399963
            self.units.x[index] = round(
                np.clip(facility.x + math.cos(angle) * 34, 0, self.map.width) * self.subpixels
            )
            self.units.y[index] = round(
                np.clip(facility.y + math.sin(angle) * 34, 0, self.map.height) * self.subpixels
            )
            self.units.facility_id[index] = -1
            self.units.order[index] = int(Order.IDLE)
            if facility.kind == FacilityKind.TOWER:
                hp_ratio = float(self.units.hp[index] / max(self.units.max_hp[index], 1))
                self._apply_stats(int(index), self.balance.units["infantry"], hp_ratio)
            if self.units.hp[index] <= 0:
                self._kill_unit(int(index))
        self.events.append(
            GameEventV1(
                tick=self.tick,
                kind="facility_destroyed",
                payload={"facility_id": facility.facility_id, "kind": facility.kind},
            )
        )

    def _update_boats(self) -> None:
        for facility in self.facilities:
            if (
                not facility.complete
                or facility.kind != FacilityKind.BOAT
                or facility.target_x is None
                or facility.target_y is None
            ):
                continue
            dx, dy = facility.target_x - facility.x, facility.target_y - facility.y
            distance = math.hypot(dx, dy)
            if distance <= 1:
                facility.target_x = None
                facility.target_y = None
                continue
            speed = float(self.balance.facilities["boat"]["speed"])
            step = min(distance, speed * self.dt)
            facility.x += dx / distance * step
            facility.y += dy / distance * step
            occupants = np.flatnonzero(
                self.units.facility_id[: self.units.count] == facility.facility_id
            )
            self.units.x[occupants] = round(facility.x * self.subpixels)
            self.units.y[occupants] = round(facility.y * self.subpixels)

    def _apply_river_damage(self) -> None:
        active = self.units.active()
        terrain = self.map.terrain_at(
            self.units.x[active] / self.subpixels, self.units.y[active] / self.subpixels
        )
        in_river = active[terrain == int(Terrain.RIVER)]
        if not len(in_river):
            return
        protected = np.zeros(len(in_river), dtype=np.bool_)
        for facility in self.facilities:
            if not facility.complete or facility.kind not in {
                FacilityKind.BRIDGE,
                FacilityKind.BOAT,
            }:
                continue
            if facility.kind == FacilityKind.BOAT:
                protected |= self.units.facility_id[in_river] == facility.facility_id
                continue
            dx = self.units.x[in_river] / self.subpixels - facility.x
            dy = self.units.y[in_river] / self.subpixels - facility.y
            protected |= dx * dx + dy * dy < 52**2
        affected = in_river[~protected]
        self.units.hp[affected] -= self.balance.river_damage_per_second * self.dt
        self.units.last_intense_tick[affected] = self.tick
        dead = affected[self.units.hp[affected] <= 0]
        for index in dead:
            self._kill_unit(int(index))

    def _kill_unit(self, index: int) -> None:
        if not self.units.alive[index]:
            return
        self.units.alive[index] = False
        faction = int(self.units.faction[index])
        self.statistics["losses"][faction] += 1
        self.events.append(
            GameEventV1(
                tick=self.tick,
                kind="unit_died",
                target_id=int(self.units.entity_id[index]),
            )
        )

    def _check_victory(self) -> None:
        if not self.victory_enabled:
            return
        player = self.commander_index(Faction.PLAYER)
        enemy = self.commander_index(Faction.ENEMY)
        player_dead = player is None or not self.units.alive[player]
        enemy_dead = enemy is None or not self.units.alive[enemy]
        if player_dead and enemy_dead:
            self.outcome = GameOutcome.DRAW
        elif enemy_dead:
            self.outcome = GameOutcome.PLAYER_WIN
        elif player_dead:
            self.outcome = GameOutcome.ENEMY_WIN
        if self.outcome != GameOutcome.ONGOING:
            self.events.append(
                GameEventV1(tick=self.tick, kind="match_ended", payload={"outcome": self.outcome})
            )

    def observation(self, faction: Faction) -> ObservationSnapshotV1:
        if self._perception is None:
            from mygame.perception.fog import FogOfWar

            self._perception = FogOfWar(self.map.rows, self.map.cols, self.map.tile_size)
            self._perception.update(self)
        return self._perception.observation(self, faction)

    def to_payload(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "tick": self.tick,
            "next_entity_id": self.next_entity_id,
            "next_facility_id": self.next_facility_id,
            "outcome": str(self.outcome),
            "victory_enabled": self.victory_enabled,
            "units": self.units.to_dict(),
            "map": self.map.to_dict(),
            "facilities": [asdict(facility) for facility in self.facilities],
            "groups": self.groups,
            "rng_state": self.rng.bit_generator.state,
            "commands": [command.model_dump(mode="json") for command in self.command_log],
            "order_queues": self.order_queues,
            "statistics": self.statistics,
            "perception": self._perception.to_dict() if self._perception is not None else None,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any], balance: BalanceConfig | None = None) -> World:
        world = cls(
            balance=balance,
            battle_map=BattleMap.from_dict(payload["map"]),
            seed=int(payload["seed"]),
            spawn_armies=False,
        )
        world.tick = int(payload["tick"])
        world.next_entity_id = int(payload["next_entity_id"])
        world.next_facility_id = int(payload["next_facility_id"])
        world.outcome = GameOutcome(payload["outcome"])
        world.victory_enabled = bool(payload.get("victory_enabled", True))
        world.units = UnitStore.from_dict(payload["units"])
        world.facilities = [Facility(**item) for item in payload["facilities"]]
        world.groups = {
            int(side): {int(key): value for key, value in groups.items()}
            for side, groups in payload["groups"].items()
        }
        world.rng.bit_generator.state = payload["rng_state"]
        world.command_log = [
            CommandEnvelopeV1.model_validate(item) for item in payload.get("commands", [])
        ]
        world.order_queues = {
            int(entity_id): [(int(order[0]), int(order[1]), int(order[2])) for order in orders]
            for entity_id, orders in payload.get("order_queues", {}).items()
        }
        world.statistics = {
            "losses": [int(value) for value in payload.get("statistics", {}).get("losses", [0, 0])]
        }
        if payload.get("perception") is not None:
            from mygame.perception.fog import FogOfWar

            world._perception = FogOfWar.from_dict(
                payload["perception"], world.map.rows, world.map.cols, world.map.tile_size
            )
        return world

    def checksum(self) -> str:
        stable = json.dumps(
            self.to_payload(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(stable.encode("utf-8")).hexdigest()

    def visible_events(self, faction: Faction, since: int = 0) -> Iterable[GameEventV1]:
        bit = 1 << int(faction)
        return (event for event in self.events[since:] if event.visible_to & bit)
