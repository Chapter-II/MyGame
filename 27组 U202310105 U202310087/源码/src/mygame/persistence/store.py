from __future__ import annotations

import hashlib
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import msgpack  # type: ignore[import-untyped]
import zstandard
from platformdirs import user_data_path

from mygame import __version__
from mygame.config import load_balance
from mygame.protocols import ReplayV1, SaveSnapshotV1
from mygame.simulation import World

MAGIC_SAVE = b"MGSV1\0"
MAGIC_REPLAY = b"MGRP1\0"


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, int) and not -(2**63) <= value < 2**64:
        return {"__bigint__": str(value)}
    return value


def _restore(value: Any) -> Any:
    if isinstance(value, dict):
        if set(value) == {"__bigint__"}:
            return int(value["__bigint__"])
        return {key: _restore(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_restore(item) for item in value]
    return value


def _encode(data: dict[str, Any], magic: bytes) -> bytes:
    packed = msgpack.packb(_safe(data), use_bin_type=True)
    return magic + zstandard.ZstdCompressor(level=7).compress(packed)


def _decode(blob: bytes, magic: bytes) -> dict[str, Any]:
    if not blob.startswith(magic):
        raise ValueError("文件格式或版本不受支持。")
    packed = zstandard.ZstdDecompressor().decompress(blob[len(magic) :])
    value = msgpack.unpackb(packed, raw=False, strict_map_key=False)
    if not isinstance(value, dict):
        raise ValueError("存档内容损坏。")
    return _restore(value)  # type: ignore[no-any-return]


def _config_hash() -> str:
    config = load_balance()
    return hashlib.sha256(config.model_dump_json().encode()).hexdigest()


class SaveManager:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or user_data_path("CommanderTacticalArena", "MyGame")
        self.save_dir = self.root / "saves"
        self.replay_dir = self.root / "replays"

    def snapshot(self, world: World) -> SaveSnapshotV1:
        return SaveSnapshotV1(
            game_version=__version__,
            config_hash=_config_hash(),
            tick=world.tick,
            payload=world.to_payload(),
        )

    def save(self, world: World, name: str = "quicksave") -> Path:
        safe_name = "".join(ch for ch in name if ch.isalnum() or ch in "-_ ").strip() or "quicksave"
        path = self.save_dir / f"{safe_name}.mgsave"
        self._atomic_write(
            path, _encode(self.snapshot(world).model_dump(mode="python"), MAGIC_SAVE)
        )
        return path

    def load(self, path: Path | None = None) -> World:
        target = path or self.save_dir / "quicksave.mgsave"
        data = _decode(target.read_bytes(), MAGIC_SAVE)
        if data.get("schema_version") != 1:
            raise ValueError("该存档来自未知的新版本，当前 V1.0 无法读取。")
        snapshot = SaveSnapshotV1.model_validate(data)
        if snapshot.config_hash != _config_hash():
            raise ValueError("存档使用了不同的平衡配置，无法保证确定性恢复。")
        return World.from_payload(snapshot.payload)

    def save_replay(self, replay: ReplayV1, name: str | None = None) -> Path:
        replay_name = name or datetime.now(UTC).strftime("battle-%Y%m%d-%H%M%S")
        path = self.replay_dir / f"{replay_name}.mgreplay"
        self._atomic_write(path, _encode(replay.model_dump(mode="python"), MAGIC_REPLAY))
        return path

    def load_replay(self, path: Path) -> ReplayV1:
        data = _decode(path.read_bytes(), MAGIC_REPLAY)
        if data.get("schema_version") != 1:
            raise ValueError("该回放来自未知的新版本，当前 V1.0 无法读取。")
        replay = ReplayV1.model_validate(data)
        if replay.initial_snapshot.config_hash != _config_hash():
            raise ValueError("回放使用了不同的平衡配置，无法可靠重演。")
        return replay

    def latest_replay(self) -> Path | None:
        if not self.replay_dir.exists():
            return None
        paths = sorted(self.replay_dir.glob("*.mgreplay"), key=lambda item: item.stat().st_mtime)
        return paths[-1] if paths else None

    @staticmethod
    def _atomic_write(path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        backup = path.with_suffix(path.suffix + ".bak")
        if path.exists():
            backup.write_bytes(path.read_bytes())
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


class ReplayRecorder:
    def __init__(self, world: World) -> None:
        manager = SaveManager(root=Path("."))
        self.initial = manager.snapshot(world)
        self.checkpoints: list[SaveSnapshotV1] = []
        self.recorded_command_count = 0

    def update(self, world: World) -> None:
        if world.tick and world.tick % (world.balance.world.simulation_hz * 5) == 0:
            self.checkpoints.append(SaveManager(root=Path(".")).snapshot(world))

    def finish(self, world: World) -> ReplayV1:
        if not self.checkpoints or self.checkpoints[-1].tick != world.tick:
            self.checkpoints.append(SaveManager(root=Path(".")).snapshot(world))
        return ReplayV1(
            initial_snapshot=self.initial,
            commands=tuple(world.command_log),
            events=tuple(world.events),
            checkpoints=tuple(self.checkpoints),
            final_checksum=world.checksum(),
        )


class ReplayPlayer:
    def __init__(self, replay: ReplayV1) -> None:
        self.replay = replay
        self.world = World.from_payload(replay.initial_snapshot.payload)
        self.commands = sorted(replay.commands, key=lambda command: command.issued_tick)
        self.command_index = 0

    @property
    def final_tick(self) -> int:
        if self.replay.checkpoints:
            return max(item.tick for item in self.replay.checkpoints)
        if self.commands:
            return max(item.issued_tick for item in self.commands) + 1
        return self.world.tick

    @property
    def finished(self) -> bool:
        return self.world.tick >= self.final_tick

    @property
    def checksum_valid(self) -> bool | None:
        if not self.finished or self.replay.final_checksum is None:
            return None
        return self.world.checksum() == self.replay.final_checksum

    def step(self) -> None:
        if self.finished:
            return
        executed = {command.command_id for command in self.world.command_log}
        while self.command_index < len(self.commands):
            command = self.commands[self.command_index]
            if command.issued_tick > self.world.tick:
                break
            if command.command_id not in executed:
                self.world.execute(command)
            self.command_index += 1
        self.world.step()

    def seek(self, target_tick: int) -> World:
        target = max(0, target_tick)
        candidates = [
            snapshot
            for snapshot in (self.replay.initial_snapshot, *self.replay.checkpoints)
            if snapshot.tick <= target
        ]
        snapshot = max(candidates, key=lambda item: item.tick)
        self.world = World.from_payload(snapshot.payload)
        self.command_index = 0
        while (
            self.command_index < len(self.commands)
            and self.commands[self.command_index].issued_tick < self.world.tick
        ):
            self.command_index += 1
        while self.world.tick < target and self.world.outcome == "ongoing":
            self.step()
        return self.world
