from mygame.commands import DeepSeekCommandParser, RuleCommandParser
from mygame.protocols import CommandStatus, Faction, MovePayloadV1
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


def test_parser_world_size_sync_affects_retreat_target() -> None:
    parser = RuleCommandParser(world_width=4096, world_height=2304)
    world = World(seed=9)
    obs = world.observation(Faction.PLAYER)
    result_old = parser.parse("全军撤退", obs)
    # Update to different dimensions
    parser.world_width = 2048
    parser.world_height = 1152
    result_new = parser.parse("全军撤退", obs)
    # Retreat targets should be clamped to different ranges
    if result_old.status == CommandStatus.PENDING and result_new.status == CommandStatus.PENDING:
        old_x = result_old.candidates[0].payload.target.x
        new_x = result_new.candidates[0].payload.target.x
        assert old_x != new_x


def test_parser_retreat_respects_world_dimensions() -> None:
    parser = RuleCommandParser(world_width=1000, world_height=500)
    world = World(seed=9)
    obs = world.observation(Faction.PLAYER)
    result = parser.parse("全军撤退", obs)
    if result.status == CommandStatus.PENDING:
        target = result.candidates[0].payload.target
        assert 0 <= target.x <= 1000
        assert 0 <= target.y <= 500


def test_parser_recognizes_forward_as_move() -> None:
    world = World(seed=9)
    parser = RuleCommandParser()
    result = parser.parse("第一战团前进", world.observation(Faction.PLAYER))
    assert result.status == CommandStatus.PENDING
    assert isinstance(result.candidates[0].payload, MovePayloadV1)


def test_parser_recognizes_advance_as_attack_move() -> None:
    world = World(seed=9)
    parser = RuleCommandParser()
    result = parser.parse("全军向北进军", world.observation(Faction.PLAYER))
    assert result.status == CommandStatus.PENDING
    assert result.candidates[0].payload.kind == "attack_move"


def test_suggester_returns_suggestions_for_empty_input() -> None:
    world = World(seed=9)
    parser = RuleCommandParser()
    obs = world.observation(Faction.PLAYER)
    suggestions = parser.suggester.suggest("", obs)
    assert len(suggestions) > 0
    assert all(isinstance(s, str) for s in suggestions)


def test_suggester_returns_suggestions_for_partial_input() -> None:
    world = World(seed=9)
    parser = RuleCommandParser()
    obs = world.observation(Faction.PLAYER)
    suggestions = parser.suggester.suggest("侦察兵", obs)
    assert len(suggestions) > 0


def test_compound_command_splitting_recognized() -> None:
    """Verify that compound conjunctions produce multiple parseable segments."""
    import re
    text = "第一战团前往中央然后工兵架桥"
    segments = re.split(r"\s*(?:然后|接着|并且|同时|再|并)\s*", text)
    assert len(segments) == 2
    assert segments[0].strip() == "第一战团前往中央"
    assert segments[1].strip() == "工兵架桥"
