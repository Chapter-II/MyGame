from __future__ import annotations

import json
import os
import uuid

import httpx

from mygame.protocols import (
    CommandEnvelopeV1,
    CommandResultV1,
    CommandSource,
    CommandStatus,
    ObservationSnapshotV1,
)


class DeepSeekCommandParser:
    def __init__(self, timeout: float = 5.0) -> None:
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        self.model = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
        self.base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
        self.timeout = timeout

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def parse(self, text: str, observation: ObservationSnapshotV1) -> CommandResultV1:
        command_id = uuid.uuid4().hex
        if not self.api_key:
            return CommandResultV1(
                command_id=command_id,
                status=CommandStatus.REJECTED,
                reason_code="missing_api_key",
                message_zh="未设置 DEEPSEEK_API_KEY，已保留离线指令解析。",
            )
        schema = CommandEnvelopeV1.model_json_schema()
        tool = {
            "type": "function",
            "function": {
                "name": "issue_command",
                "description": (
                    "把中文战场命令转换为一个合法的结构化命令。不得引用观察快照中不存在的敌军。"
                ),
                "strict": True,
                "parameters": schema,
            },
        }
        visible_context = {
            "tick": observation.tick,
            "faction": int(observation.faction),
            "own_groups": sorted({unit.group_id for unit in observation.own_units}),
            "visible_enemies": [unit.model_dump() for unit in observation.visible_enemies[:40]],
            "known_villages": list(observation.known_villages),
        }
        body = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你是游戏命令解析器。只能调用 issue_command，"
                        "并输出当前阵营在当前帧可验证的 JSON。"
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {"text": text, "observation": visible_context}, ensure_ascii=False
                    ),
                },
            ],
            "tools": [tool],
            "tool_choice": {"type": "function", "function": {"name": "issue_command"}},
            "stream": False,
        }
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=body,
                )
                response.raise_for_status()
                message = response.json()["choices"][0]["message"]
                arguments = message["tool_calls"][0]["function"]["arguments"]
            raw = json.loads(arguments)
            raw.update(
                {
                    "schema_version": 1,
                    "command_id": command_id,
                    "faction": int(observation.faction),
                    "issued_tick": observation.tick,
                    "source": CommandSource.TEXT,
                }
            )
            command = CommandEnvelopeV1.model_validate(raw)
            return CommandResultV1(
                command_id=command_id,
                status=CommandStatus.PENDING,
                reason_code="parsed",
                message_zh="DeepSeek 已生成待校验指令。",
                candidates=(command,),
            )
        except (
            httpx.HTTPError,
            KeyError,
            IndexError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            return CommandResultV1(
                command_id=command_id,
                status=CommandStatus.REJECTED,
                reason_code="provider_error",
                message_zh=f"DeepSeek 解析失败：{type(exc).__name__}。可切换离线规则解析。",
            )
