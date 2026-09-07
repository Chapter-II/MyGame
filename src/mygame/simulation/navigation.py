from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np

from mygame.maps import BattleMap


@dataclass(slots=True, frozen=True)
class FlowField:
    target_cell: tuple[int, int]
    direction_x: np.ndarray
    direction_y: np.ndarray
    integration: np.ndarray


class FlowFieldCache:
    """Shared terrain-aware routes keyed by target region and terrain revision."""

    def __init__(self, battle_map: BattleMap, terrain_speed: dict[str, float]) -> None:
        self.map = battle_map
        self.speed = np.asarray(
            [
                terrain_speed["plain"],
                terrain_speed["grass"],
                terrain_speed["forest"],
                terrain_speed["swamp"],
                terrain_speed["river"],
                terrain_speed["road"],
            ],
            dtype=np.float32,
        )
        self.cache: dict[tuple[int, int, str, int], FlowField] = {}

    def get(self, target_x: float, target_y: float, movement: str = "land") -> FlowField:
        col = int(np.clip(target_x // self.map.tile_size, 0, self.map.cols - 1))
        row = int(np.clip(target_y // self.map.tile_size, 0, self.map.rows - 1))
        # Nearby formation destinations share one field and one cache entry.
        region = 8
        col = min(self.map.cols - 1, col // region * region + region // 2)
        row = min(self.map.rows - 1, row // region * region + region // 2)
        key = (col, row, movement, self.map.revision)
        field = self.cache.get(key)
        if field is None:
            field = self._build(col, row, movement)
            self.cache[key] = field
            if len(self.cache) > 48:
                oldest = next(iter(self.cache))
                self.cache.pop(oldest)
        return field

    def _build(self, target_col: int, target_row: int, movement: str) -> FlowField:
        rows, cols = self.map.rows, self.map.cols
        integration = np.full((rows, cols), np.inf, dtype=np.float32)
        integration[target_row, target_col] = 0.0
        queue: list[tuple[float, int, int]] = [(0.0, target_row, target_col)]
        terrain_cost = 1.0 / np.maximum(self.speed[self.map.terrain], 0.01)
        if movement == "boat":
            terrain_cost = np.where(self.map.terrain == 4, 1.0, 5.0).astype(np.float32)
        neighbors = (
            (-1, 0, 1.0),
            (1, 0, 1.0),
            (0, -1, 1.0),
            (0, 1, 1.0),
            (-1, -1, 1.4142),
            (-1, 1, 1.4142),
            (1, -1, 1.4142),
            (1, 1, 1.4142),
        )
        while queue:
            cost, row, col = heapq.heappop(queue)
            if cost > float(integration[row, col]) + 1e-5:
                continue
            for dy, dx, length in neighbors:
                next_row, next_col = row + dy, col + dx
                if not (0 <= next_row < rows and 0 <= next_col < cols):
                    continue
                candidate = cost + float(terrain_cost[next_row, next_col]) * length
                if candidate < integration[next_row, next_col]:
                    integration[next_row, next_col] = candidate
                    heapq.heappush(queue, (candidate, next_row, next_col))
        direction_x = np.zeros((rows, cols), dtype=np.int8)
        direction_y = np.zeros((rows, cols), dtype=np.int8)
        best = integration.copy()
        for dy, dx, _ in neighbors:
            shifted = np.full_like(integration, np.inf)
            source_rows = slice(max(0, dy), min(rows, rows + dy))
            source_cols = slice(max(0, dx), min(cols, cols + dx))
            target_rows = slice(max(0, -dy), min(rows, rows - dy))
            target_cols = slice(max(0, -dx), min(cols, cols - dx))
            shifted[target_rows, target_cols] = integration[source_rows, source_cols]
            better = shifted < best
            best = np.where(better, shifted, best)
            direction_x[better] = dx
            direction_y[better] = dy
        direction_x[target_row, target_col] = 0
        direction_y[target_row, target_col] = 0
        return FlowField((target_col, target_row), direction_x, direction_y, integration)
