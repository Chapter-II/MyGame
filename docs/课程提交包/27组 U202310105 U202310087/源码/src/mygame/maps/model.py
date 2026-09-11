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


def _paint_brush(grid: np.ndarray, cx: int, cy: int, radius: int, value: Terrain) -> None:
    """Paint a small local brush without allocating a full-map mask."""
    y0, y1 = max(0, cy - radius), min(grid.shape[0], cy + radius + 1)
    x0, x1 = max(0, cx - radius), min(grid.shape[1], cx + radius + 1)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    patch = grid[y0:y1, x0:x1]
    patch[(xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2] = value


def _clear_land_ellipse(grid: np.ndarray, cx: int, cy: int, rx: int, ry: int) -> None:
    """Clear a settlement footprint without cutting holes into the river."""
    y0, y1 = max(0, cy - ry), min(grid.shape[0], cy + ry + 1)
    x0, x1 = max(0, cx - rx), min(grid.shape[1], cx + rx + 1)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    patch = grid[y0:y1, x0:x1]
    inside = ((xx - cx) / max(rx, 1)) ** 2 + ((yy - cy) / max(ry, 1)) ** 2 <= 1
    patch[inside & (patch != int(Terrain.RIVER))] = Terrain.PLAIN


def _paint_route(
    grid: np.ndarray,
    points: list[tuple[int, int]],
    width: int = 1,
    value: Terrain = Terrain.ROAD,
) -> None:
    """Paint a connected road through several strategic waypoints."""
    for (start_x, start_y), (end_x, end_y) in zip(points, points[1:], strict=False):
        steps = max(abs(end_x - start_x), abs(end_y - start_y), 1)
        xs = np.rint(np.linspace(start_x, end_x, steps + 1)).astype(np.int16)
        ys = np.rint(np.linspace(start_y, end_y, steps + 1)).astype(np.int16)
        for x, y in zip(xs, ys, strict=True):
            _paint_brush(grid, int(x), int(y), width, value)


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

    def cell_x(ratio: float) -> int:
        return int(np.clip(round(cols * ratio), 1, cols - 2))

    def cell_y(ratio: float) -> int:
        return int(np.clip(round(rows * ratio), 1, rows - 2))

    def paint_pair(cx: int, cy: int, rx: int, ry: int, value: Terrain) -> None:
        _paint_ellipse(grid, cx, cy, rx, ry, value)
        if symmetric:
            _paint_ellipse(grid, cols - 1 - cx, rows - 1 - cy, rx, ry, value)
        else:
            mirror_x = int(rng.integers(mid + max(5, rx), cols - max(5, rx)))
            mirror_y = int(rng.integers(max(3, ry), rows - max(3, ry)))
            _paint_ellipse(grid, mirror_x, mirror_y, rx, ry, value)

    # Broad meadows provide readable assembly spaces, while authored forest ridges
    # and floodplain swamps create meaningful fast, concealed, and slow routes.
    strategic_zones = (
        (0.18, 0.50, 14, 15, Terrain.GRASS),
        (0.35, 0.26, 11, 8, Terrain.GRASS),
        (0.39, 0.76, 10, 8, Terrain.GRASS),
        (0.27, 0.13, 13, 5, Terrain.FOREST),
        (0.31, 0.48, 8, 11, Terrain.FOREST),
        (0.23, 0.83, 12, 6, Terrain.FOREST),
        (0.43, 0.63, 5, 8, Terrain.FOREST),
        (0.45, 0.18, 5, 7, Terrain.SWAMP),
        (0.46, 0.84, 5, 6, Terrain.SWAMP),
        (0.40, 0.42, 5, 4, Terrain.SWAMP),
    )
    for x_ratio, y_ratio, rx, ry, value in strategic_zones:
        paint_pair(cell_x(x_ratio), cell_y(y_ratio), rx, ry, value)

    # Seeded secondary patches make individual matches distinct without disturbing
    # the strategic skeleton or faction symmetry.
    feature_types = (Terrain.GRASS, Terrain.GRASS, Terrain.FOREST, Terrain.SWAMP)
    for _ in range(12):
        cx = int(rng.integers(max(5, cell_x(0.08)), mid - 7))
        cy = int(rng.integers(4, rows - 4))
        rx = int(rng.integers(3, 8))
        ry = int(rng.integers(2, 6))
        value = feature_types[int(rng.integers(0, len(feature_types)))]
        paint_pair(cx, cy, rx, ry, value)

    spawn_col = max(4, round(310 / tile_size))
    spawn = (spawn_col, rows // 2)
    home_village = (cell_x(0.20), cell_y(0.22))
    woodland_village = (cell_x(0.34), cell_y(0.73))
    frontier_village = (cell_x(0.43), cell_y(0.40))
    left_routes = (
        [spawn, home_village, frontier_village],
        [spawn, woodland_village, frontier_village],
    )
    for route in left_routes:
        _paint_route(grid, route)
        if symmetric:
            _paint_route(grid, [(cols - 1 - x, rows - 1 - y) for x, y in route])
        else:
            _paint_route(
                grid,
                [
                    (cols - 1 - x, int(np.clip(y + rng.integers(-5, 6), 1, rows - 2)))
                    for x, y in route
                ],
            )

    # The river is continuous but has five materially different crossing costs:
    # a narrow central ford, wide basins, and medium upper/lower reaches. Mirrored
    # row pairs guarantee exact 180-degree competitive fairness.
    width_positions = np.asarray([0.0, 0.18, 0.38, 0.58, 0.78, 1.0])
    width_values = np.asarray([4.0, 8.0, 12.0, 6.0, 10.0, 8.0])
    meander_scale = float(rng.uniform(0.85, 1.15))
    for row in range((rows + 1) // 2):
        centred_y = (row - (rows - 1) / 2) / max(rows - 1, 1)
        distance_from_centre = abs(centred_y) * 2
        raw_width = float(np.interp(distance_from_centre, width_positions, width_values))
        river_width = int(np.clip(2 * round(raw_width / 2), 4, 12))
        offset = round(
            meander_scale
            * (4.0 * np.sin(2 * np.pi * centred_y) + 1.5 * np.sin(6 * np.pi * centred_y))
        )
        start = int(np.clip(mid - river_width // 2 + offset, 1, cols - river_width - 1))
        grid[row, start : start + river_width] = Terrain.RIVER
        mirror_row = rows - 1 - row
        mirror_start = cols - start - river_width
        grid[mirror_row, mirror_start : mirror_start + river_width] = Terrain.RIVER

    base_villages = [
        (*home_village, "small"),
        (*woodland_village, "medium"),
        (*frontier_village, "small"),
    ]
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
            _clear_land_ellipse(grid, px, py, 3, 3)
            villages.append(
                Village(
                    next_id, (px + 0.5) * tile_size, (py + 0.5) * tile_size, 92.0, population, size
                )
            )
            next_id += 1
    _clear_land_ellipse(grid, spawn_col, rows // 2, 8, 12)
    _clear_land_ellipse(grid, cols - 1 - spawn_col, rows // 2, 8, 12)
    if symmetric:
        grid[:, mid:] = np.rot90(grid[:, :mid], 2)
    return BattleMap(width, height, tile_size, grid, villages, seed, symmetric)


def validate_map(battle_map: BattleMap) -> list[str]:
    problems: list[str] = []
    if battle_map.terrain.shape != (battle_map.rows, battle_map.cols):
        problems.append("terrain_dimensions")
    if len(battle_map.villages) != 6:
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
    missing_terrain = [
        terrain.name.lower()
        for terrain in Terrain
        if not np.any(battle_map.terrain == int(terrain))
    ]
    if missing_terrain:
        problems.append("missing_terrain:" + ",".join(missing_terrain))
    river_widths = np.sum(battle_map.terrain == int(Terrain.RIVER), axis=1)
    if np.any(river_widths == 0):
        problems.append("river_gap")
    elif int(np.max(river_widths) - np.min(river_widths)) < 6:
        problems.append("river_lacks_width_variation")
    for row in range(battle_map.rows):
        river_cols = np.flatnonzero(battle_map.terrain[row] == int(Terrain.RIVER))
        if len(river_cols) and not np.all(np.diff(river_cols) == 1):
            problems.append("river_disconnected")
            break
    spawn_y = battle_map.height / 2
    for spawn_x in (310.0, battle_map.width - 310.0):
        if int(battle_map.terrain_at(spawn_x, spawn_y)) == int(Terrain.RIVER):
            problems.append("blocked_spawn")
    return problems
