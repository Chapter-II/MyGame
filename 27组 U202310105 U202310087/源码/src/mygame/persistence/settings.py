from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from mygame.persistence.store import SaveManager


class SettingsV1(BaseModel):
    model_config = ConfigDict(extra="ignore")

    schema_version: Literal[1] = 1
    ui_scale: float = 1.0
    reduced_motion: bool = False
    fullscreen: bool = False
    master_volume: float = 0.5
    group_names: dict[int, str] = Field(default_factory=dict)
    voice_allow_download: bool = False


class SettingsManager:
    def __init__(self, root: Path | None = None) -> None:
        self.storage = SaveManager(root)
        self.path = self.storage.root / "settings.json"

    def load(self) -> SettingsV1:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            settings = SettingsV1.model_validate(data)
            if settings.ui_scale not in {1.0, 1.25, 1.5, 2.0}:
                settings.ui_scale = 1.0
            if settings.master_volume not in {0.0, 0.25, 0.5, 0.75, 1.0}:
                settings.master_volume = 0.5
            return settings
        except (OSError, ValueError, TypeError):
            return SettingsV1()

    def save(self, settings: SettingsV1) -> None:
        payload = settings.model_dump_json(indent=2).encode("utf-8")
        self.storage._atomic_write(self.path, payload)
