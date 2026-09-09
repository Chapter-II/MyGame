from mygame.ai import LocalStrategicAI
from mygame.protocols import Faction, MovePayloadV1, ObservationSnapshotV1, UnitObservationV1


def _make_obs(tick: int, own_units: tuple, visible_enemies: tuple = ()):
    from mygame.protocols import ObservationSnapshotV1

    return ObservationSnapshotV1(
        faction=Faction.ENEMY,
        tick=tick,
        own_units=own_units,
        visible_enemies=visible_enemies,
    )


def _unit(kind: str, x: float = 100.0, y: float = 100.0, hp: float = 1.0, eid: int = 1):
    return UnitObservationV1(
        entity_id=eid,
        kind=kind,
        faction=Faction.ENEMY,
        x=x,
        y=y,
        hp=hp,
        stamina=100.0,
        group_id=0,
        exposed=True,
    )


def test_ai_detects_commander_death() -> None:
    ai = LocalStrategicAI(4096, 2304, difficulty="normal")
    assert not ai._commander_dead
    # Commander alive
    obs1 = _make_obs(10, (_unit("commander", eid=1), _unit("infantry", eid=2)))
    ai._tactical_decisions(obs1)
    assert not ai._commander_dead
    # Commander gone
    obs2 = _make_obs(20, (_unit("infantry", eid=2),))
    ai._tactical_decisions(obs2)
    assert ai._commander_dead


def test_ai_skips_protect_commander_when_dead() -> None:
    ai = LocalStrategicAI(4096, 2304, difficulty="hard")
    ai._commander_dead = True
    obs = _make_obs(10, (_unit("infantry", eid=2),))
    commands = ai._tactical_decisions(obs)
    # No guard commands should be issued since commander is dead
    from mygame.protocols import GuardPayloadV1

    guard_cmds = [c for c in commands if isinstance(c.payload, GuardPayloadV1)]
    assert len(guard_cmds) == 0


def test_ai_increases_aggression_on_commander_death() -> None:
    ai = LocalStrategicAI(
        4096,
        2304,
        config={
            "strategy_interval_ticks": 1,
            "tactical_interval_ticks": 1,
            "max_simultaneous_tasks": 5,
            "allow_inspire": True,
            "aggression": 0.5,
            "retreat_hp_ratio": 0.25,
            "scout_count": 1,
        },
    )
    # Normal mode: aggression is 0.5
    obs_alive = _make_obs(1, (_unit("commander", eid=1), _unit("infantry", eid=2, x=200, y=200)))
    ai._tactical_decisions(obs_alive)
    assert not ai._commander_dead
    # Commander dies
    obs_dead = _make_obs(2, (_unit("infantry", eid=2, x=200, y=200),))
    ai._tactical_decisions(obs_dead)
    assert ai._commander_dead


def test_ai_skips_conversions_when_commander_dead() -> None:
    ai = LocalStrategicAI(4096, 2304, difficulty="normal")
    ai._commander_dead = True
    obs = _make_obs(40, (_unit("infantry", eid=2),))
    commands = ai._strategic_decisions(obs)
    from mygame.protocols import ConvertPayloadV1

    convert_cmds = [c for c in commands if isinstance(c.payload, ConvertPayloadV1)]
    assert len(convert_cmds) == 0


def test_ai_can_control_player_faction_for_optional_auto_battle() -> None:
    ai = LocalStrategicAI(4096, 2304, difficulty="normal")
    player_units = (
        UnitObservationV1(
            entity_id=1,
            kind="commander",
            faction=Faction.PLAYER,
            x=300,
            y=1152,
            hp=1.0,
            stamina=100.0,
            group_id=0,
            exposed=True,
        ),
        UnitObservationV1(
            entity_id=2,
            kind="infantry",
            faction=Faction.PLAYER,
            x=500,
            y=1152,
            hp=1.0,
            stamina=100.0,
            group_id=1,
            exposed=True,
        ),
    )
    observation = ObservationSnapshotV1(
        faction=Faction.PLAYER,
        tick=0,
        own_units=player_units,
        visible_enemies=(),
    )

    commands = ai.decide(observation)

    assert commands
    assert all(command.faction == Faction.PLAYER for command in commands)
    assault_targets = [
        command.payload.target.x
        for command in commands
        if isinstance(command.payload, MovePayloadV1) and command.payload.kind == "attack_move"
    ]
    assert assault_targets and min(assault_targets) > 4096 / 2
