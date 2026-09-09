from __future__ import annotations

import tomllib
from importlib.resources import files
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class UnitStats(BaseModel):
    model_config = ConfigDict(frozen=True)

    hp: float = Field(gt=0)
    damage: float = Field(ge=0)
    attack_interval: float = Field(gt=0)
    speed: float = Field(gt=0)
    vision: float = Field(gt=0)
    detection: float = Field(ge=0)
    radius: float = Field(gt=0)
    attack_range: float = Field(gt=0)


class WorldConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    width: int = Field(gt=0)
    height: int = Field(gt=0)
    tile_size: int = Field(gt=0)
    simulation_hz: int = Field(gt=0)
    subpixels: int = Field(gt=0)
    default_army_size: int = Field(ge=10, le=1000)


class BalanceConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    world: WorldConfig
    tactics: dict[str, float]
    units: dict[str, UnitStats]
    terrain_speed: dict[str, float]
    terrain_vision: dict[str, float]
    facilities: dict[str, dict[str, float | int]]
    ai: dict[str, dict[str, float | int | bool]]


def load_balance() -> BalanceConfig:
    path = files("mygame.config").joinpath("data/balance_v1.toml")
    with path.open("rb") as handle:
        raw: dict[str, Any] = tomllib.load(handle)
    world = WorldConfig(**raw["world"])
    units = {name: UnitStats(**values) for name, values in raw["units"].items()}
    facility_root = raw["facilities"]
    common = {
        "engineer_work_per_second": facility_root["engineer_work_per_second"],
        "minimum_build_seconds": facility_root["minimum_build_seconds"],
    }
    facilities: dict[str, dict[str, float | int]] = {
        name: {**values, **common}
        for name, values in facility_root.items()
        if isinstance(values, dict)
    }
    return BalanceConfig(
        world=world,
        tactics={key: float(value) for key, value in raw["tactics"].items()},
        units=units,
        terrain_speed={key: float(value) for key, value in raw["terrain"]["speed"].items()},
        terrain_vision={key: float(value) for key, value in raw["terrain"]["vision"].items()},
        facilities=facilities,
        ai={name: dict(values) for name, values in raw["ai"].items()},
    )
