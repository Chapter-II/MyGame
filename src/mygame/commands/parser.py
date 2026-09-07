from __future__ import annotations

import re
import uuid
from typing import Literal, cast

from mygame.protocols import (
    AttackFacilityPayloadV1,
    BuildPayloadV1,
    CommandEnvelopeV1,
    CommandResultV1,
    CommandSource,
    CommandStatus,
    ConvertPayloadV1,
    FacilityActionPayloadV1,
    FocusFirePayloadV1,
    GroupPayloadV1,
    GuardPayloadV1,
    MovePayloadV1,
    ObservationSnapshotV1,
    PositionV1,
    RecruitPayloadV1,
    SelectionV1,
    TacticalPayloadV1,
)


class RuleCommandParser:
    def __init__(
        self,
        world_width: int = 4096,
        world_height: int = 2304,
        group_names: dict[int, str] | None = None,
    ) -> None:
        self.world_width = world_width
        self.world_height = world_height
        self.group_names = group_names or {}

    def parse(self, text: str, observation: ObservationSnapshotV1) -> CommandResultV1:
        cleaned = re.sub(r"[，。！？,.!?]", " ", text.strip())
        command_id = uuid.uuid4().hex
        if not cleaned:
            return CommandResultV1(
                command_id=command_id,
                status=CommandStatus.REJECTED,
                reason_code="empty_text",
                message_zh="请输入指令。",
            )
        selection = self._selection(cleaned, observation)
        target = self._target(cleaned, observation)
        source = CommandSource.TEXT
        envelope: CommandEnvelopeV1 | None = None

        if "摧毁" in cleaned or "攻击设施" in cleaned:
            requested_kind = next(
                (
                    kind
                    for word, kind in (
                        ("桥", "bridge"),
                        ("船", "boat"),
                        ("塔", "tower"),
                    )
                    if word in cleaned
                ),
                None,
            )
            targets = [
                item
                for item in observation.known_facilities
                if int(item["faction"]) != int(observation.faction)
                and not bool(item.get("destroyed", False))
                and (requested_kind is None or item["kind"] == requested_kind)
            ]
            if not targets:
                return CommandResultV1(
                    command_id=command_id,
                    status=CommandStatus.REJECTED,
                    reason_code="facility_not_visible",
                    message_zh="没有符合描述的当前可见敌方设施。",
                )
            chosen_facility = min(
                targets,
                key=lambda item: (
                    (float(item["x"]) - target.x) ** 2 + (float(item["y"]) - target.y) ** 2
                ),
            )
            envelope = CommandEnvelopeV1(
                command_id=command_id,
                faction=observation.faction,
                issued_tick=observation.tick,
                source=source,
                payload=AttackFacilityPayloadV1(
                    selection=selection,
                    target_facility_id=int(chosen_facility["facility_id"]),
                ),
            )

        group_match = re.search(r"(?:编入|编为|组成)第?\s*([1-9一二三四五六七八九])\s*组", cleaned)
        if group_match:
            numerals = "一二三四五六七八九"
            token = group_match.group(1)
            group_id = int(token) if token.isdigit() else numerals.index(token) + 1
            name_match = re.search(r"(?:命名为|叫做)\s*([^\s]{1,24})", cleaned)
            envelope = CommandEnvelopeV1(
                command_id=command_id,
                faction=observation.faction,
                issued_tick=observation.tick,
                source=source,
                payload=GroupPayloadV1(
                    selection=selection,
                    group_id=group_id,
                    name=name_match.group(1) if name_match else None,
                ),
            )

        action: (
            Literal["enter_tower", "exit_tower", "board_boat", "disembark", "sail_boat"] | None
        ) = None
        if "入塔" in cleaned or "进入防御塔" in cleaned:
            action = "enter_tower"
        elif "出塔" in cleaned or "离开防御塔" in cleaned:
            action = "exit_tower"
        elif "登船" in cleaned:
            action = "board_boat"
        elif "下船" in cleaned:
            action = "disembark"
        elif "开船" in cleaned or "航行" in cleaned or "船只前往" in cleaned:
            action = "sail_boat"
        if envelope is None and action is not None:
            facility_kind = "tower" if action in {"enter_tower", "exit_tower"} else "boat"
            facilities = [
                item
                for item in observation.known_facilities
                if item["kind"] == facility_kind
                and int(item["faction"]) == int(observation.faction)
                and bool(item["complete"])
            ]
            if not facilities:
                return CommandResultV1(
                    command_id=command_id,
                    status=CommandStatus.REJECTED,
                    reason_code="facility_unavailable",
                    message_zh="尚未发现可用的己方设施。",
                )
            chosen = min(
                facilities,
                key=lambda item: (
                    (float(item["x"]) - target.x) ** 2 + (float(item["y"]) - target.y) ** 2
                ),
            )
            envelope = CommandEnvelopeV1(
                command_id=command_id,
                faction=observation.faction,
                issued_tick=observation.tick,
                source=source,
                payload=FacilityActionPayloadV1(
                    kind=action,
                    selection=selection,
                    facility_id=int(chosen["facility_id"]),
                    target=target if action == "sail_boat" else None,
                ),
            )

        facility: Literal["bridge", "boat", "road", "tower"] | None = None
        for words, value in (
            ("架桥 桥梁", "bridge"),
            ("造船 船只", "boat"),
            ("砍树 开路 道路", "road"),
            ("防御塔 建塔", "tower"),
        ):
            if any(word in cleaned for word in words.split()):
                facility = cast(Literal["bridge", "boat", "road", "tower"], value)
                break
        if envelope is None and facility:
            envelope = CommandEnvelopeV1(
                command_id=command_id,
                faction=observation.faction,
                issued_tick=observation.tick,
                source=source,
                payload=BuildPayloadV1(selection=selection, facility_kind=facility, target=target),
            )
        elif "征召" in cleaned or "征兵" in cleaned:
            if not observation.known_villages:
                return CommandResultV1(
                    command_id=command_id,
                    status=CommandStatus.REJECTED,
                    reason_code="no_known_village",
                    message_zh="尚未发现可征召的村庄。",
                )
            village = min(
                observation.known_villages,
                key=lambda item: (
                    (float(item["x"]) - target.x) ** 2 + (float(item["y"]) - target.y) ** 2
                ),
            )
            amount = self._count(cleaned) or min(20, int(village["population"]))
            envelope = CommandEnvelopeV1(
                command_id=command_id,
                faction=observation.faction,
                issued_tick=observation.tick,
                source=source,
                payload=RecruitPayloadV1(village_id=int(village["village_id"]), count=amount),
            )
        elif "分化" in cleaned or "训练" in cleaned or "转为" in cleaned:
            kind = self._kind(cleaned)
            if kind is None or kind == "recruit":
                return CommandResultV1(
                    command_id=command_id,
                    status=CommandStatus.REJECTED,
                    reason_code="missing_unit_kind",
                    message_zh="请说明要分化为步兵、侦察兵、工兵或刺客。",
                )
            envelope = CommandEnvelopeV1(
                command_id=command_id,
                faction=observation.faction,
                issued_tick=observation.tick,
                source=source,
                payload=ConvertPayloadV1(selection=selection, unit_kind=kind),
            )
        elif "全速" in cleaned:
            envelope = CommandEnvelopeV1(
                command_id=command_id,
                faction=observation.faction,
                issued_tick=observation.tick,
                source=source,
                payload=TacticalPayloadV1(kind="sprint", selection=selection, target=target),
            )
        elif "冲锋" in cleaned or "突击" in cleaned:
            envelope = CommandEnvelopeV1(
                command_id=command_id,
                faction=observation.faction,
                issued_tick=observation.tick,
                source=source,
                payload=TacticalPayloadV1(kind="charge", selection=selection, target=target),
            )
        elif "集火" in cleaned:
            visible = list(observation.visible_enemies)
            if "将领" in cleaned:
                visible = [unit for unit in visible if unit.kind == "commander"]
            if not visible:
                return CommandResultV1(
                    command_id=command_id,
                    status=CommandStatus.REJECTED,
                    reason_code="target_not_visible",
                    message_zh="没有符合描述的当前可见集火目标。",
                )
            chosen_enemy = min(
                visible, key=lambda unit: (unit.x - target.x) ** 2 + (unit.y - target.y) ** 2
            )
            envelope = CommandEnvelopeV1(
                command_id=command_id,
                faction=observation.faction,
                issued_tick=observation.tick,
                source=source,
                payload=FocusFirePayloadV1(
                    selection=selection, target_entity_id=chosen_enemy.entity_id
                ),
            )
        elif "守卫" in cleaned or "守住" in cleaned or "驻守" in cleaned:
            envelope = CommandEnvelopeV1(
                command_id=command_id,
                faction=observation.faction,
                issued_tick=observation.tick,
                source=source,
                payload=GuardPayloadV1(selection=selection, target=target),
            )
        elif any(
            word in cleaned
            for word in ("前往", "移动", "出发", "攻击", "进攻", "搜索", "侦察", "守住")
        ):
            attack = any(word in cleaned for word in ("攻击", "进攻", "搜索", "侦察"))
            envelope = CommandEnvelopeV1(
                command_id=command_id,
                faction=observation.faction,
                issued_tick=observation.tick,
                source=source,
                payload=MovePayloadV1(
                    kind="attack_move" if attack else "move", selection=selection, target=target
                ),
            )
        if envelope is None:
            return CommandResultV1(
                command_id=command_id,
                status=CommandStatus.REJECTED,
                reason_code="unrecognized",
                message_zh="未识别该指令。可尝试“第一侦察队前往中央”或按 F1 查看指令书。",
            )
        candidates = [envelope]
        if isinstance(envelope.payload, MovePayloadV1) and ("还是" in cleaned or "或" in cleaned):
            alternatives: list[SelectionV1] = []
            numerals = "一二三四五六七八九"
            for token in re.findall(r"第\s*([1-9一二三四五六七八九])\s*(?:组|队)", cleaned):
                group_id = int(token) if token.isdigit() else numerals.index(token) + 1
                alternatives.append(SelectionV1(group_id=group_id))
            if len(alternatives) >= 2:
                candidates = [
                    envelope.model_copy(
                        update={
                            "payload": envelope.payload.model_copy(
                                update={"selection": alternative}
                            )
                        }
                    )
                    for alternative in alternatives[:3]
                ]
            elif "东" in cleaned and "西" in cleaned:
                candidates = [
                    envelope.model_copy(
                        update={
                            "payload": envelope.payload.model_copy(
                                update={
                                    "target": PositionV1(
                                        x=self.world_width * ratio,
                                        y=envelope.payload.target.y,
                                    )
                                }
                            )
                        }
                    )
                    for ratio in (0.22, 0.78)
                ]
        return CommandResultV1(
            command_id=command_id,
            status=CommandStatus.PENDING,
            reason_code="ambiguous" if len(candidates) > 1 else "parsed",
            message_zh=(
                "该命令会显著影响执行结果，请从候选解释中选择。"
                if len(candidates) > 1
                else f"已理解：{cleaned}"
            ),
            candidates=tuple(candidates),
        )

    def _selection(self, text: str, observation: ObservationSnapshotV1) -> SelectionV1:
        group_map = {
            "第一战团": 1,
            "第一侦察队": 2,
            "侦察队": 2,
            "工兵队": 3,
            "影刃队": 4,
            "预备队": 5,
        }
        group_map.update({name: group for group, name in self.group_names.items()})
        group = next((value for name, value in group_map.items() if name in text), None)
        match = re.search(r"第\s*([1-9一二三四五六七八九])\s*(?:组|队)", text)
        if match:
            numerals = "一二三四五六七八九"
            group = (
                int(match.group(1))
                if match.group(1).isdigit()
                else numerals.index(match.group(1)) + 1
            )
        kind = self._kind(text)
        count = self._count(text)
        if group is None and kind is None and count is None:
            commander = next(
                (unit.entity_id for unit in observation.own_units if unit.kind == "commander"), None
            )
            return SelectionV1(unit_ids=(commander,) if commander else ())
        return SelectionV1(group_id=group, unit_kind=kind, count=count)

    @staticmethod
    def _kind(
        text: str,
    ) -> Literal["recruit", "infantry", "scout", "engineer", "assassin"] | None:
        for words, kind in (
            ("初始兵 预备兵", "recruit"),
            ("步兵", "infantry"),
            ("侦察兵 侦察队", "scout"),
            ("工兵", "engineer"),
            ("刺客 影刃", "assassin"),
        ):
            if any(word in text for word in words.split()):
                return cast(Literal["recruit", "infantry", "scout", "engineer", "assassin"], kind)
        return None

    @staticmethod
    def _count(text: str) -> int | None:
        match = re.search(r"(\d{1,3})\s*(?:名|人|个)?", text)
        return int(match.group(1)) if match else None

    def _target(self, text: str, observation: ObservationSnapshotV1) -> PositionV1:
        x, y = self.world_width / 2, self.world_height / 2
        if "东" in text:
            x = self.world_width * 0.78
        elif "西" in text:
            x = self.world_width * 0.22
        if "北" in text:
            y = self.world_height * 0.25
        elif "南" in text:
            y = self.world_height * 0.75
        if "最近" in text and observation.known_villages:
            anchor = observation.own_units[0] if observation.own_units else None
            if anchor:
                village = min(
                    observation.known_villages,
                    key=lambda item: (
                        (float(item["x"]) - anchor.x) ** 2 + (float(item["y"]) - anchor.y) ** 2
                    ),
                )
                x, y = float(village["x"]), float(village["y"])
        return PositionV1(x=x, y=y)
