import json

import httpx

from mygame.commands import DeepSeekCommandParser
from mygame.protocols import CommandStatus, Faction
from mygame.simulation import World


class FakeResponse:
    def __init__(self, arguments: dict) -> None:
        self.arguments = arguments

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return {
            "choices": [
                {
                    "message": {
                        "tool_calls": [{"function": {"arguments": json.dumps(self.arguments)}}]
                    }
                }
            ]
        }


class FakeClient:
    response: FakeResponse | Exception

    def __init__(self, **kwargs) -> None:
        del kwargs

    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:
        pass

    def post(self, *args, **kwargs):
        del args, kwargs
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def parser(monkeypatch, response: FakeResponse | Exception) -> DeepSeekCommandParser:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    FakeClient.response = response
    monkeypatch.setattr(httpx, "Client", FakeClient)
    return DeepSeekCommandParser()


def valid_arguments() -> dict:
    return {
        "payload": {
            "kind": "move",
            "selection": {"group_id": 1},
            "target": {"x": 1000, "y": 1000},
        }
    }


def test_deepseek_valid_tool_result_is_pending(monkeypatch) -> None:
    world = World(seed=31)
    result = parser(monkeypatch, FakeResponse(valid_arguments())).parse(
        "第一战团前进", world.observation(Faction.PLAYER)
    )
    assert result.status == CommandStatus.PENDING
    assert result.candidates[0].payload.kind == "move"


def test_deepseek_hallucinated_field_is_rejected(monkeypatch) -> None:
    world = World(seed=32)
    arguments = valid_arguments()
    arguments["payload"]["teleport"] = True
    result = parser(monkeypatch, FakeResponse(arguments)).parse(
        "传送", world.observation(Faction.PLAYER)
    )
    assert result.status == CommandStatus.REJECTED
    assert result.reason_code == "provider_error"


def test_deepseek_timeout_isolated_as_provider_error(monkeypatch) -> None:
    world = World(seed=33)
    request = httpx.Request("POST", "https://api.deepseek.com/chat/completions")
    result = parser(monkeypatch, httpx.ReadTimeout("late", request=request)).parse(
        "前进", world.observation(Faction.PLAYER)
    )
    assert result.status == CommandStatus.REJECTED
    assert result.reason_code == "provider_error"
