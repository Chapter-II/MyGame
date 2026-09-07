from __future__ import annotations

from enum import IntEnum, StrEnum
from typing import Annotated, Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class ProtocolModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Faction(IntEnum):
    PLAYER = 0
    ENEMY = 1


class CommandSource(StrEnum):
    MOUSE = "mouse"
    KEYBOARD = "keyboard"
    TEXT = "text"
    VOICE = "voice"
    LOCAL_AI = "local_ai"
    EXTERNAL_AI = "external_ai"


class QueueMode(StrEnum):
    REPLACE = "replace"
    APPEND = "append"


class CommandStatus(StrEnum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    EXPIRED = "expired"
    CANCELLED = "cancelled"
    COMPLETED = "completed"


class PositionV1(ProtocolModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    x: float
    y: float


class SelectionV1(ProtocolModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    unit_ids: tuple[int, ...] = ()
    group_id: int | None = None
    count: int | None = Field(default=None, ge=1)
    unit_kind: str | None = None


class MovePayloadV1(ProtocolModel):
    kind: Literal["move", "attack_move"]
    selection: SelectionV1
    target: PositionV1


class GroupPayloadV1(ProtocolModel):
    kind: Literal["group"] = "group"
    selection: SelectionV1
    group_id: int = Field(ge=1, le=9)
    name: str | None = Field(default=None, max_length=24)


class ConvertPayloadV1(ProtocolModel):
    kind: Literal["convert"] = "convert"
    selection: SelectionV1
    unit_kind: Literal["infantry", "scout", "engineer", "assassin"]


class TacticalPayloadV1(ProtocolModel):
    kind: Literal["sprint", "charge"]
    selection: SelectionV1
    target: PositionV1 | None = None


class GuardPayloadV1(ProtocolModel):
    kind: Literal["guard"] = "guard"
    selection: SelectionV1
    target: PositionV1


class FocusFirePayloadV1(ProtocolModel):
    kind: Literal["focus_fire"] = "focus_fire"
    selection: SelectionV1
    target_entity_id: int = Field(ge=1)


class AttackFacilityPayloadV1(ProtocolModel):
    kind: Literal["attack_facility"] = "attack_facility"
    selection: SelectionV1
    target_facility_id: int = Field(ge=1)


class BuildPayloadV1(ProtocolModel):
    kind: Literal["build"] = "build"
    selection: SelectionV1
    facility_kind: Literal["bridge", "boat", "road", "tower"]
    target: PositionV1


class RecruitPayloadV1(ProtocolModel):
    kind: Literal["recruit"] = "recruit"
    village_id: int
    count: int = Field(ge=1, le=120)


class FacilityActionPayloadV1(ProtocolModel):
    kind: Literal["enter_tower", "exit_tower", "board_boat", "disembark", "sail_boat"]
    selection: SelectionV1 = Field(default_factory=SelectionV1)
    facility_id: int | None = Field(default=None, ge=1)
    target: PositionV1 | None = None


CommandPayloadV1 = Annotated[
    MovePayloadV1
    | AttackFacilityPayloadV1
    | GroupPayloadV1
    | ConvertPayloadV1
    | TacticalPayloadV1
    | GuardPayloadV1
    | FocusFirePayloadV1
    | BuildPayloadV1
    | RecruitPayloadV1
    | FacilityActionPayloadV1,
    Field(discriminator="kind"),
]


class CommandEnvelopeV1(ProtocolModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    schema_version: Literal[1] = 1
    command_id: str
    faction: Faction
    issuer_entity_id: int | None = None
    issued_tick: int = Field(ge=0)
    source: CommandSource
    queue_mode: QueueMode = QueueMode.REPLACE
    payload: CommandPayloadV1


class CommandResultV1(ProtocolModel):
    schema_version: Literal[1] = 1
    command_id: str
    status: CommandStatus
    reason_code: str = "ok"
    message_zh: str
    candidates: tuple[CommandEnvelopeV1, ...] = ()


class UnitObservationV1(ProtocolModel):
    entity_id: int
    faction: Faction
    kind: str
    x: float
    y: float
    hp: float
    stamina: float
    group_id: int
    exposed: bool


class ObservationSnapshotV1(ProtocolModel):
    schema_version: Literal[1] = 1
    faction: Faction
    tick: int
    own_units: tuple[UnitObservationV1, ...]
    visible_enemies: tuple[UnitObservationV1, ...]
    historical_sightings: tuple[dict[str, Any], ...] = ()
    known_terrain: bytes = b""
    terrain_shape: tuple[int, int] = (0, 0)
    known_villages: tuple[dict[str, Any], ...] = ()
    known_facilities: tuple[dict[str, Any], ...] = ()


class GameEventV1(ProtocolModel):
    schema_version: Literal[1] = 1
    tick: int
    kind: str
    visible_to: int = 3
    actor_id: int | None = None
    target_id: int | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class SaveSnapshotV1(ProtocolModel):
    schema_version: Literal[1] = 1
    game_version: str
    config_hash: str
    tick: int
    payload: dict[str, Any]


class ReplayV1(ProtocolModel):
    schema_version: Literal[1] = 1
    initial_snapshot: SaveSnapshotV1
    commands: tuple[CommandEnvelopeV1, ...]
    events: tuple[GameEventV1, ...] = ()
    checkpoints: tuple[SaveSnapshotV1, ...]
    final_checksum: str | None = None


@runtime_checkable
class WorldView(Protocol):
    @property
    def tick(self) -> int: ...

    def observation(self, faction: Faction) -> ObservationSnapshotV1: ...


@runtime_checkable
class LanguageCommandParser(Protocol):
    def parse(self, text: str, observation: ObservationSnapshotV1) -> CommandResultV1: ...


@runtime_checkable
class SpeechRecognizer(Protocol):
    def transcribe(self, samples: bytes, sample_rate: int = 16_000) -> str: ...


@runtime_checkable
class StrategicController(Protocol):
    def decide(self, observation: ObservationSnapshotV1) -> list[CommandEnvelopeV1]: ...
