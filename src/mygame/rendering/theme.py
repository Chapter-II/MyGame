from __future__ import annotations

from dataclasses import dataclass

Color = tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class Theme:
    # sRGB render approximations of the canonical OKLCH tokens in DESIGN.md.
    background: Color = (3, 4, 7)
    surface: Color = (20, 24, 39)
    surface_high: Color = (31, 38, 58)
    border: Color = (62, 73, 101)
    ink: Color = (235, 239, 249)
    muted: Color = (166, 178, 205)
    primary: Color = (82, 120, 238)
    primary_hover: Color = (105, 143, 255)
    warning: Color = (235, 176, 65)
    ally: Color = (64, 202, 224)
    enemy: Color = (244, 105, 82)
    success: Color = (82, 203, 143)
    fog_explored: Color = (7, 10, 17)
    fog_unknown: Color = (1, 2, 4)
    terrain_plain: Color = (50, 66, 61)
    terrain_grass: Color = (61, 85, 65)
    terrain_forest: Color = (32, 61, 48)
    terrain_swamp: Color = (54, 58, 52)
    terrain_river: Color = (35, 65, 91)
    terrain_road: Color = (91, 85, 69)
