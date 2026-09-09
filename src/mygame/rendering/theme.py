from __future__ import annotations

from dataclasses import dataclass

Color = tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class Theme:
    # Cold iron, dark oak and restrained brass: a medieval field-command table.
    background: Color = (7, 9, 12)
    surface: Color = (20, 24, 29)
    surface_high: Color = (34, 40, 47)
    border: Color = (91, 83, 67)
    ink: Color = (239, 237, 226)
    muted: Color = (179, 181, 176)
    primary: Color = (60, 99, 160)
    primary_hover: Color = (79, 121, 190)
    warning: Color = (222, 176, 82)
    ally: Color = (81, 184, 207)
    enemy: Color = (218, 91, 70)
    success: Color = (86, 180, 126)
    fog_explored: Color = (10, 13, 16)
    fog_unknown: Color = (3, 4, 6)
    terrain_plain: Color = (50, 66, 61)
    terrain_grass: Color = (61, 85, 65)
    terrain_forest: Color = (32, 61, 48)
    terrain_swamp: Color = (54, 58, 52)
    terrain_river: Color = (35, 65, 91)
    terrain_road: Color = (91, 85, 69)
