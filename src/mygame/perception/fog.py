from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from mygame.protocols import Faction, ObservationSnapshotV1, UnitObservationV1
from mygame.simulation.world import UnitKind

if TYPE_CHECKING:
    from mygame.simulation.world import World


class FogOfWar:
    """Owns faction-specific visibility; never exposes the authority object."""

    def __init__(self, rows: int, cols: int, tile_size: int) -> None:
        self.rows = rows
        self.cols = cols
        self.tile_size = tile_size
        self.visible = np.zeros((2, rows, cols), dtype=np.bool_)
        self.explored = np.zeros((2, rows, cols), dtype=np.bool_)
        self.history: list[dict[int, dict[str, Any]]] = [{}, {}]
        self.village_history: list[dict[int, dict[str, Any]]] = [{}, {}]
        self._offset_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._observation_cache: dict[int, ObservationSnapshotV1] = {}

    def invalidate_dynamic(self) -> None:
        self._observation_cache.clear()

    def update(self, world: World) -> None:
        self.visible[:] = False
        self._observation_cache.clear()
        for faction in (Faction.PLAYER, Faction.ENEMY):
            active = world.units.active(faction)
            if len(active):
                cols = (world.units.x[active] / world.subpixels // self.tile_size).astype(np.int16)
                rows = (world.units.y[active] / world.subpixels // self.tile_size).astype(np.int16)
                terrain = world.map.terrain_at(
                    world.units.x[active] / world.subpixels,
                    world.units.y[active] / world.subpixels,
                )
                vision_table = np.asarray(
                    [
                        world.balance.terrain_vision["plain"],
                        world.balance.terrain_vision["grass"],
                        world.balance.terrain_vision["forest"],
                        world.balance.terrain_vision["swamp"],
                        world.balance.terrain_vision["river"],
                        world.balance.terrain_vision["road"],
                    ],
                    dtype=np.float32,
                )
                radii = np.ceil(
                    world.units.vision[active] * vision_table[terrain] / self.tile_size
                ).astype(np.int16)
                reveals = np.unique(np.column_stack((rows, cols, radii)), axis=0)
                for row, col, radius in reveals:
                    self._reveal_cells(int(faction), int(row), int(col), int(radius))
            self.explored[int(faction)] |= self.visible[int(faction)]
            for village in world.map.villages:
                col = int(np.clip(village.x // self.tile_size, 0, self.cols - 1))
                row = int(np.clip(village.y // self.tile_size, 0, self.rows - 1))
                if self.visible[int(faction), row, col]:
                    self.village_history[int(faction)][village.village_id] = {
                        "village_id": village.village_id,
                        "x": village.x,
                        "y": village.y,
                        "population": village.population,
                        "size": village.size,
                        "last_seen_tick": world.tick,
                    }
            for index in self._visible_enemy_indices(world, faction):
                entity_id = int(world.units.entity_id[index])
                self.history[int(faction)][entity_id] = {
                    "entity_id": entity_id,
                    "kind": UnitKind(int(world.units.kind[index])).name.lower(),
                    "x": float(world.units.x[index] / world.subpixels),
                    "y": float(world.units.y[index] / world.subpixels),
                    "last_seen_tick": world.tick,
                }

    def _reveal_cells(self, faction: int, row: int, col: int, radius: int) -> None:
        if radius not in self._offset_cache:
            yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
            mask = xx * xx + yy * yy <= radius * radius
            self._offset_cache[radius] = (yy[mask], xx[mask])
        dy, dx = self._offset_cache[radius]
        rows = row + dy
        cols = col + dx
        valid = (rows >= 0) & (rows < self.rows) & (cols >= 0) & (cols < self.cols)
        self.visible[faction, rows[valid], cols[valid]] = True

    def _visible_enemy_indices(self, world: World, faction: Faction) -> np.ndarray:
        enemy = world.units.active(Faction(1 - int(faction)))
        if not len(enemy):
            return enemy
        cols = np.clip(
            (world.units.x[enemy] / world.subpixels // self.tile_size).astype(np.int32),
            0,
            self.cols - 1,
        )
        rows = np.clip(
            (world.units.y[enemy] / world.subpixels // self.tile_size).astype(np.int32),
            0,
            self.rows - 1,
        )
        visible = self.visible[int(faction), rows, cols]
        hidden_assassin = (
            world.units.kind[enemy] == int(UnitKind.ASSASSIN)
        ) & ~world.units.exposed[enemy]
        return np.asarray(enemy[visible & ~hidden_assassin], dtype=np.intp)

    @staticmethod
    def _unit(world: World, index: int) -> UnitObservationV1:
        return UnitObservationV1(
            entity_id=int(world.units.entity_id[index]),
            faction=Faction(int(world.units.faction[index])),
            kind=UnitKind(int(world.units.kind[index])).name.lower(),
            x=float(world.units.x[index] / world.subpixels),
            y=float(world.units.y[index] / world.subpixels),
            hp=float(world.units.hp[index]),
            stamina=float(world.units.stamina[index]),
            group_id=int(world.units.group_id[index]),
            exposed=bool(world.units.exposed[index]),
        )

    def observation(self, world: World, faction: Faction) -> ObservationSnapshotV1:
        cached = self._observation_cache.get(int(faction))
        if cached is not None:
            return cached
        own = tuple(self._unit(world, int(index)) for index in world.units.active(faction))
        visible_indices = self._visible_enemy_indices(world, faction)
        enemies = tuple(self._unit(world, int(index)) for index in visible_indices)
        known_villages = list(self.village_history[int(faction)].values())
        known_facilities = tuple(
            {
                "facility_id": item.facility_id,
                "faction": item.faction,
                "kind": item.kind,
                "x": item.x,
                "y": item.y,
                "complete": item.complete,
                "destroyed": item.destroyed,
                "progress": item.progress / max(item.required_work, 1),
            }
            for item in world.facilities
            if item.faction == int(faction)
            or self.visible[
                int(faction), int(item.y // self.tile_size), int(item.x // self.tile_size)
            ]
        )
        snapshot = ObservationSnapshotV1(
            faction=faction,
            tick=world.tick,
            own_units=own,
            visible_enemies=enemies,
            historical_sightings=tuple(self.history[int(faction)].values()),
            known_terrain=np.where(self.explored[int(faction)], world.map.terrain, 255)
            .astype(np.uint8)
            .tobytes(),
            terrain_shape=(self.rows, self.cols),
            known_villages=tuple(known_villages),
            known_facilities=known_facilities,
        )
        self._observation_cache[int(faction)] = snapshot
        return snapshot

    def to_dict(self) -> dict[str, Any]:
        return {
            "visible": self.visible.tolist(),
            "explored": self.explored.tolist(),
            "history": self.history,
            "village_history": self.village_history,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], rows: int, cols: int, tile_size: int) -> FogOfWar:
        fog = cls(rows, cols, tile_size)
        fog.visible = np.asarray(data["visible"], dtype=np.bool_)
        fog.explored = np.asarray(data["explored"], dtype=np.bool_)
        fog.history = [
            {int(entity_id): sighting for entity_id, sighting in side.items()}
            for side in data["history"]
        ]
        fog.village_history = [
            {int(village_id): sighting for village_id, sighting in side.items()}
            for side in data.get("village_history", ({}, {}))
        ]
        return fog
