from mygame.commands import DeepSeekCommandParser, RuleCommandParser
from mygame.protocols import CommandStatus, Faction
from mygame.simulation import World


def test_rule_parser_understands_group_move() -> None:
    world = World(seed=9)
    parser = RuleCommandParser()
    result = parser.parse("第一侦察队前往北部中央", world.observation(Faction.PLAYER))
    assert result.status == CommandStatus.PENDING
    assert result.candidates[0].payload.selection.group_id == 2
    assert result.candidates[0].payload.target.y < world.map.height / 2


def test_rule_parser_rejects_unsupported_text() -> None:
    world = World(seed=9)
    result = RuleCommandParser().parse("给我讲个故事", world.observation(Faction.PLAYER))
    assert result.status == CommandStatus.REJECTED
    assert result.reason_code == "unrecognized"


def test_deepseek_is_optional(monkeypatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    world = World(seed=9)
    result = DeepSeekCommandParser().parse("向中央进攻", world.observation(Faction.PLAYER))
    assert result.reason_code == "missing_api_key"


def test_custom_group_name_can_be_created_and_referenced() -> None:
    world = World(seed=10)
    parser = RuleCommandParser(group_names={6: "河西守军"})
    create = parser.parse("将20名步兵编为第6组 命名为河西守军", world.observation(Faction.PLAYER))
    assert create.status == CommandStatus.PENDING
    assert create.candidates[0].payload.kind == "group"
    move = parser.parse("河西守军前往中央", world.observation(Faction.PLAYER))
    assert move.candidates[0].payload.selection.group_id == 6


def test_high_impact_ambiguity_returns_at_most_three_candidates() -> None:
    world = World(seed=11)
    result = RuleCommandParser().parse("第1组还是第2组前往中央", world.observation(Faction.PLAYER))
    assert result.reason_code == "ambiguous"
    assert 2 <= len(result.candidates) <= 3
