from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import IntEnum
from typing import Any

import numpy as np


class Terrain(IntEnum):
    PLAIN = 0
    GRASS = 1
    FOREST = 2
    SWAMP = 3
    RIVER = 4
    ROAD = 5


@dataclass(slots=True)
class Village:
    village_id: int
    x: float
    y: float
    radius: float
    population: int
    size: str


@dataclass(slots=True)
class BattleMap:
    width: int
    height: int
    tile_size: int
    terrain: np.ndarray
    villages: list[Village]
    seed: int
    symmetric: bool
    revision: int = 0

    @property
    def cols(self) -> int:
        return self.width // self.tile_size

    @property
    def rows(self) -> int:
        return self.height // self.tile_size

    def terrain_at(self, x: np.ndarray | float, y: np.ndarray | float) -> np.ndarray:
        col = np.clip(np.asarray(x, dtype=np.int32) // self.tile_size, 0, self.cols - 1)
        row = np.clip(np.asarray(y, dtype=np.int32) // self.tile_size, 0, self.rows - 1)
        return self.terrain[row, col]

    def clear_forest(self, x: float, y: float, radius_tiles: int = 2) -> None:
        col, row = int(x // self.tile_size), int(y // self.tile_size)
        y0, y1 = max(0, row - radius_tiles), min(self.rows, row + radius_tiles + 1)
        x0, x1 = max(0, col - radius_tiles), min(self.cols, col + radius_tiles + 1)
        patch = self.terrain[y0:y1, x0:x1]
        patch[patch == Terrain.FOREST] = Terrain.ROAD
        self.revision += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "tile_size": self.tile_size,
            "terrain": self.terrain.tolist(),
            "villages": [asdict(village) for village in self.villages],
            "seed": self.seed,
            "symmetric": self.symmetric,
            "revision": self.revision,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BattleMap:
        return cls(
            width=int(data["width"]),
            height=int(data["height"]),
            tile_size=int(data["tile_size"]),
            terrain=np.asarray(data["terrain"], dtype=np.uint8),
            villages=[Village(**item) for item in data["villages"]],
            seed=int(data["seed"]),
            symmetric=bool(data["symmetric"]),
            revision=int(data.get("revision", 0)),
        )


def _paint_ellipse(grid: np.ndarray, cx: int, cy: int, rx: int, ry: int, value: Terrain) -> None:
    yy, xx = np.ogrid[: grid.shape[0], : grid.shape[1]]
    mask = ((xx - cx) / max(rx, 1)) ** 2 + ((yy - cy) / max(ry, 1)) ** 2 <= 1
    grid[mask] = value


def generate_map(
    width: int = 4096,
    height: int = 2304,
    tile_size: int = 32,
    seed: int = 20260907,
    symmetric: bool = True,
) -> BattleMap:
    rng = np.random.default_rng(seed)
    rows, cols = height // tile_size, width // tile_size
    grid = np.full((rows, cols), Terrain.PLAIN, dtype=np.uint8)
    mid = cols // 2
    river_wobble = np.rint(2.0 * np.sin(np.linspace(0, 3 * np.pi, rows))).astype(int)
    for row, wobble in enumerate(river_wobble):
        center = mid + int(wobble)
        grid[row, center - 2 : center + 3] = Terrain.RIVER

    feature_types = [Terrain.GRASS, Terrain.FOREST, Terrain.SWAMP]
    for _ in range(14):
        cx = int(rng.integers(12, mid - 10))
        cy = int(rng.integers(7, rows - 7))
        rx = int(rng.integers(3, 10))
        ry = int(rng.integers(2, 7))
        value = feature_types[int(rng.integers(0, len(feature_types)))]
        _paint_ellipse(grid, cx, cy, rx, ry, value)
        if symmetric:
            _paint_ellipse(grid, cols - 1 - cx, rows - 1 - cy, rx, ry, value)
        else:
            right_x = int(rng.integers(mid + 10, cols - 12))
            right_y = int(rng.integers(7, rows - 7))
            _paint_ellipse(grid, right_x, right_y, rx, ry, value)
    if symmetric:
        # The left half is authoritative; a 180-degree copy guarantees exact fairness.
        grid[:, mid:] = np.rot90(grid[:, :mid], 2)

    base_villages = [(32, 18, "small"), (46, 52, "medium")]
    villages: list[Village] = []
    next_id = 1
    for cx, cy, size in base_villages:
        population = int(rng.integers(40, 81) if size == "small" else rng.integers(80, 121))
        pairs = (
            [(cx, cy), (cols - 1 - cx, rows - 1 - cy)]
            if symmetric
            else [
                (cx, cy),
                (
                    int(rng.integers(mid + 12, cols - 12)),
                    int(rng.integers(8, rows - 8)),
                ),
            ]
        )
        for px, py in pairs:
            _paint_ellipse(grid, px, py, 3, 3, Terrain.PLAIN)
            villages.append(
                Village(
                    next_id, (px + 0.5) * tile_size, (py + 0.5) * tile_size, 92.0, population, size
                )
            )
            next_id += 1
    spawn_col = max(4, round(310 / tile_size))
    _paint_ellipse(grid, spawn_col, rows // 2, 8, 12, Terrain.PLAIN)
    _paint_ellipse(grid, cols - 1 - spawn_col, rows // 2, 8, 12, Terrain.PLAIN)
    if symmetric:
        grid[:, mid:] = np.rot90(grid[:, :mid], 2)
    return BattleMap(width, height, tile_size, grid, villages, seed, symmetric)


def validate_map(battle_map: BattleMap) -> list[str]:
    problems: list[str] = []
    if battle_map.terrain.shape != (battle_map.rows, battle_map.cols):
        problems.append("terrain_dimensions")
    if len(battle_map.villages) != 4:
        problems.append("village_count")
    left_population = sum(
        village.population for village in battle_map.villages if village.x < battle_map.width / 2
    )
    right_population = sum(
        village.population for village in battle_map.villages if village.x >= battle_map.width / 2
    )
    if left_population != right_population:
        problems.append("population_imbalance")
    if battle_map.symmetric and not np.array_equal(
        battle_map.terrain, np.rot90(battle_map.terrain, 2)
    ):
        problems.append("rotation_asymmetry")
    spawn_y = battle_map.height / 2
    for spawn_x in (310.0, battle_map.width - 310.0):
        if int(battle_map.terrain_at(spawn_x, spawn_y)) == int(Terrain.RIVER):
            problems.append("blocked_spawn")
    return problems
