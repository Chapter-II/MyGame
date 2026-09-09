from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pygame

logger = logging.getLogger(__name__)


class ArtBook:
    """Loads optional generated art and exposes small, cached render sprites."""

    def __init__(self) -> None:
        self.terrain_sources: list[pygame.Surface] = []
        self.unit_sources: dict[tuple[int, int], pygame.Surface] = {}
        self.facility_sources: dict[str, pygame.Surface] = {}
        self.village_source: pygame.Surface | None = None
        self.effect_sources: dict[str, pygame.Surface] = {}
        self.terrain_cache: dict[tuple[int, int, int, int], pygame.Surface] = {}
        self.unit_cache: dict[tuple[int, int, int], pygame.Surface] = {}
        self.facility_cache: dict[tuple[str, int, int, bool], pygame.Surface] = {}
        self.village_cache: dict[int, pygame.Surface] = {}
        self.effect_cache: dict[tuple[str, int], pygame.Surface] = {}
        self._load()

    @staticmethod
    def _asset_root() -> Path | None:
        roots = [Path(__file__).resolve().parents[3], Path.cwd()]
        bundle_root = getattr(sys, "_MEIPASS", None)
        if bundle_root:
            roots.insert(0, Path(bundle_root))
        for root in roots:
            generated = root / "assets" / "generated"
            if generated.exists():
                return generated
        return None

    def _load(self) -> None:
        root = self._asset_root()
        if root is None:
            return
        try:
            atlas = pygame.image.load(root / "terrain_atlas.png").convert_alpha()
            cell_w, cell_h = atlas.get_width() // 3, atlas.get_height() // 2
            for row in range(2):
                for col in range(3):
                    self.terrain_sources.append(
                        atlas.subsurface((col * cell_w, row * cell_h, cell_w, cell_h)).copy()
                    )
        except (FileNotFoundError, pygame.error, ValueError):
            logger.info("地区纹理图集不可用，使用程序化颜色回退", exc_info=True)

        try:
            atlas_path = next(
                path
                for path in (root / "units_atlas_v2.png", root / "units_atlas.png")
                if path.exists()
            )
            atlas = pygame.image.load(atlas_path).convert_alpha()
            for faction in range(2):
                top = round(faction * atlas.get_height() / 2)
                bottom = round((faction + 1) * atlas.get_height() / 2)
                for kind in range(7):
                    left = round(kind * atlas.get_width() / 7)
                    right = round((kind + 1) * atlas.get_width() / 7)
                    source = atlas.subsurface((left, top, right - left, bottom - top)).copy()
                    bounds = source.get_bounding_rect(min_alpha=8)
                    cropped = source.subsurface(bounds).copy() if bounds.width else source
                    side = max(cropped.get_width(), cropped.get_height())
                    normalized = pygame.Surface((side, side), pygame.SRCALPHA)
                    normalized.blit(cropped, cropped.get_rect(center=normalized.get_rect().center))
                    self.unit_sources[(faction, kind)] = normalized
        except (FileNotFoundError, pygame.error, ValueError):
            logger.info("单位贴图图集不可用，使用战术标记回退", exc_info=True)

        for facility_kind in ("bridge", "tower"):
            try:
                self.facility_sources[facility_kind] = pygame.image.load(
                    root / f"{facility_kind}.png"
                ).convert_alpha()
            except (FileNotFoundError, pygame.error, ValueError):
                logger.info(
                    "%s 设施贴图不可用，使用战术标记回退",
                    facility_kind,
                    exc_info=True,
                )

        try:
            self.village_source = pygame.image.load(root / "village.png").convert_alpha()
        except (FileNotFoundError, pygame.error, ValueError):
            logger.info("村庄贴图不可用，使用战术标记回退", exc_info=True)

        try:
            atlas = pygame.image.load(root / "effects_atlas.png").convert_alpha()
            cell_w, cell_h = atlas.get_width() // 2, atlas.get_height() // 2
            for index, name in enumerate(("slash", "impact", "sprint", "charge")):
                col, row = index % 2, index // 2
                self.effect_sources[name] = atlas.subsurface(
                    (col * cell_w, row * cell_h, cell_w, cell_h)
                ).copy()
        except (FileNotFoundError, pygame.error, ValueError):
            logger.info("战斗特效图集不可用，使用程序化特效回退", exc_info=True)

    def terrain(
        self, kind: int, size: int, tile_x: int = 0, tile_y: int = 0
    ) -> pygame.Surface | None:
        if not self.terrain_sources:
            return None
        size = max(1, size)
        source = self.terrain_sources[kind % len(self.terrain_sources)]
        patch_w = source.get_width() // 8
        patch_h = source.get_height() // 8
        variant_x = tile_x % 8
        variant_y = tile_y % 8
        key = (kind, size, variant_x, variant_y)
        if key not in self.terrain_cache:
            patch = source.subsurface((variant_x * patch_w, variant_y * patch_h, patch_w, patch_h))
            self.terrain_cache[key] = pygame.transform.smoothscale(patch, (size, size))
        return self.terrain_cache[key]

    def unit(self, faction: int, kind: int, size: int) -> pygame.Surface | None:
        source = self.unit_sources.get((faction, kind))
        if source is None:
            return None
        size = max(4, size)
        key = (faction, kind, size)
        if key not in self.unit_cache:
            # A high-quality downsample becomes the pixel source; the final display
            # layer still scales it with nearest-neighbour filtering.
            self.unit_cache[key] = pygame.transform.smoothscale(source, (size, size))
        return self.unit_cache[key]

    def facility(
        self, kind: str, width: int, height: int, *, vertical: bool = False
    ) -> pygame.Surface | None:
        source = self.facility_sources.get(kind)
        if source is None:
            return None
        width, height = max(4, width), max(4, height)
        key = (kind, width, height, vertical)
        if key not in self.facility_cache:
            scaled = pygame.transform.smoothscale(source, (width, height))
            self.facility_cache[key] = pygame.transform.rotate(scaled, 90) if vertical else scaled
        return self.facility_cache[key]

    def village(self, size: int) -> pygame.Surface | None:
        if self.village_source is None:
            return None
        size = max(8, size)
        if size not in self.village_cache:
            source = self.village_source
            width = max(8, round(size * source.get_width() / source.get_height()))
            self.village_cache[size] = pygame.transform.smoothscale(source, (width, size))
        return self.village_cache[size]

    def effect(self, kind: str, size: int) -> pygame.Surface | None:
        source = self.effect_sources.get(kind)
        if source is None:
            return None
        size = max(6, size)
        key = (kind, size)
        if key not in self.effect_cache:
            self.effect_cache[key] = pygame.transform.smoothscale(source, (size, size))
        return self.effect_cache[key]

    def compose_terrain(self, terrain: np.ndarray, tile_pixels: int = 16) -> pygame.Surface | None:
        """Build one static textured map and soften biome seams with irregular blending."""
        if not self.terrain_sources:
            return None
        rows, cols = terrain.shape
        surface = pygame.Surface((cols * tile_pixels, rows * tile_pixels))
        for row in range(rows):
            for col in range(cols):
                kind = int(terrain[row, col])
                tile = self.terrain(kind, tile_pixels, col, row)
                if tile is not None:
                    surface.blit(tile, (col * tile_pixels, row * tile_pixels))

        pixels = pygame.surfarray.array3d(surface)
        original = pixels.copy()
        # A wider, deterministic feather removes the square tile edge while
        # retaining a deliberate pixel-art shoreline at normal zoom levels.
        max_depth = max(2, tile_pixels // 3)

        for row in range(rows):
            y0 = row * tile_pixels
            for col in range(1, cols):
                if terrain[row, col - 1] == terrain[row, col]:
                    continue
                x = col * tile_pixels
                for offset in range(tile_pixels):
                    y = y0 + offset
                    depth = 1 + ((row * 17 + col * 31 + offset * 13) % max_depth)
                    left = original[x - 1, y].astype(np.float32)
                    right = original[x, y].astype(np.float32)
                    for distance in range(depth):
                        strength = 0.52 * (depth - distance) / depth
                        pixels[x - 1 - distance, y] = left * (1 - strength) + right * strength
                        pixels[x + distance, y] = right * (1 - strength) + left * strength

        original = pixels.copy()
        for row in range(1, rows):
            y = row * tile_pixels
            for col in range(cols):
                if terrain[row - 1, col] == terrain[row, col]:
                    continue
                x0 = col * tile_pixels
                for offset in range(tile_pixels):
                    x = x0 + offset
                    depth = 1 + ((row * 29 + col * 19 + offset * 11) % max_depth)
                    top = original[x, y - 1].astype(np.float32)
                    bottom = original[x, y].astype(np.float32)
                    for distance in range(depth):
                        strength = 0.52 * (depth - distance) / depth
                        pixels[x, y - 1 - distance] = top * (1 - strength) + bottom * strength
                        pixels[x, y + distance] = bottom * (1 - strength) + top * strength

        pygame.surfarray.blit_array(surface, pixels)
        return surface
