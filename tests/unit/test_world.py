import uuid

from mygame.protocols import (
    AttackFacilityPayloadV1,
    BuildPayloadV1,
    CommandEnvelopeV1,
    CommandSource,
    CommandStatus,
    ConvertPayloadV1,
    FacilityActionPayloadV1,
    Faction,
    MovePayloadV1,
    PositionV1,
    QueueMode,
    SelectionV1,
    TacticalPayloadV1,
)
from mygame.simulation import Facility, GameOutcome, UnitKind, World


def command(world: World, payload: object) -> CommandEnvelopeV1:
    return CommandEnvelopeV1(
        command_id=uuid.uuid4().hex,
        faction=Faction.PLAYER,
        issued_tick=world.tick,
        source=CommandSource.KEYBOARD,
        payload=payload,
    )


def test_default_armies_have_real_entities_and_required_roles() -> None:
    world = World(seed=5)
    assert world.units.count == 1000
    for faction in (Faction.PLAYER, Faction.ENEMY):
        active = world.units.active(faction)
        assert len(active) == 500
        assert sum(world.units.kind[active] == int(UnitKind.COMMANDER)) == 1
        assert sum(world.units.kind[active] == int(UnitKind.GUARD)) == 4


def test_move_command_changes_position() -> None:
    world = World(seed=7, spawn_armies=False)
    entity_id = world.spawn_unit(Faction.PLAYER, UnitKind.INFANTRY, 100, 100)
    payload = MovePayloadV1(
        kind="move",
        selection=SelectionV1(unit_ids=(entity_id,)),
        target=PositionV1(x=220, y=100),
    )
    result = world.execute(command(world, payload))
    before = int(world.units.x[0])
    world.step(10)
    assert result.status == CommandStatus.ACCEPTED
    assert world.units.x[0] > before


def test_recruits_convert_once_and_use_configured_stats() -> None:
    world = World(seed=7, spawn_armies=False)
    entity_id = world.spawn_unit(Faction.PLAYER, UnitKind.RECRUIT, 100, 100)
    payload = ConvertPayloadV1(selection=SelectionV1(unit_ids=(entity_id,)), unit_kind="scout")
    assert world.execute(command(world, payload)).status == CommandStatus.ACCEPTED
    assert world.units.kind[0] == int(UnitKind.SCOUT)
    assert world.execute(command(world, payload)).reason_code == "no_recruits"


def test_tactical_command_spends_stamina_and_sets_cooldown() -> None:
    world = World(seed=7, spawn_armies=False)
    commander = world.spawn_unit(Faction.PLAYER, UnitKind.COMMANDER, 100, 100)
    infantry = world.spawn_unit(Faction.PLAYER, UnitKind.INFANTRY, 130, 100)
    payload = TacticalPayloadV1(
        kind="sprint",
        selection=SelectionV1(unit_ids=(infantry,)),
        target=PositionV1(x=300, y=100),
    )
    result = world.execute(command(world, payload))
    assert commander == 1
    assert result.status == CommandStatus.ACCEPTED
    assert world.units.stamina[1] == 70
    assert world.units.tactic_cooldown[1] > 0


def test_guard_intercepts_commander_target() -> None:
    world = World(seed=7, spawn_armies=False)
    commander = world.spawn_unit(Faction.ENEMY, UnitKind.COMMANDER, 120, 100)
    guard = world.spawn_unit(Faction.ENEMY, UnitKind.GUARD, 112, 100)
    commander_index = world.units.index_of(commander)
    guard_index = world.units.index_of(guard)
    assert commander_index is not None and guard_index is not None
    assert world._guard_intercept(commander_index) == guard_index


def test_guard_follows_commander_when_separated() -> None:
    world = World(seed=15, spawn_armies=False)
    commander = world.spawn_unit(Faction.PLAYER, UnitKind.COMMANDER, 300, 100)
    guard = world.spawn_unit(Faction.PLAYER, UnitKind.GUARD, 100, 100)
    guard_index = world.units.index_of(guard)
    assert commander and guard_index is not None
    before = int(world.units.x[guard_index])
    world.step(20)
    assert world.units.x[guard_index] > before


def test_simultaneous_commander_deaths_are_draw() -> None:
    world = World(seed=7, spawn_armies=False)
    first = world.spawn_unit(Faction.PLAYER, UnitKind.COMMANDER, 100, 100)
    second = world.spawn_unit(Faction.ENEMY, UnitKind.COMMANDER, 116, 100)
    first_index = world.units.index_of(first)
    second_index = world.units.index_of(second)
    assert first_index is not None and second_index is not None
    world.units.hp[[first_index, second_index]] = 1
    world.step()
    assert world.outcome == GameOutcome.DRAW


def test_engineers_complete_bridge() -> None:
    world = World(seed=7, spawn_armies=False)
    river_cells = (world.map.terrain == 4).nonzero()
    row, col = int(river_cells[0][0]), int(river_cells[1][0])
    x, y = (col + 0.5) * world.map.tile_size, (row + 0.5) * world.map.tile_size
    ids = tuple(world.spawn_unit(Faction.PLAYER, UnitKind.ENGINEER, x, y) for _ in range(8))
    payload = BuildPayloadV1(
        selection=SelectionV1(unit_ids=ids),
        facility_kind="bridge",
        target=PositionV1(x=x, y=y),
    )
    assert world.execute(command(world, payload)).status == CommandStatus.ACCEPTED
    world.step(160)
    assert world.facilities[0].complete


def test_custom_army_composition_is_exact() -> None:
    composition = {
        "infantry": 50,
        "scout": 15,
        "engineer": 10,
        "assassin": 5,
        "recruit": 15,
    }
    world = World(seed=8, army_size=100, army_composition=composition)
    player = world.units.active(Faction.PLAYER)
    assert len(player) == 100
    assert sum(world.units.kind[player] == int(UnitKind.INFANTRY)) == 50
    assert sum(world.units.kind[player] == int(UnitKind.RECRUIT)) == 15


def test_tower_capacity_and_exit_restore_infantry() -> None:
    world = World(seed=9, spawn_armies=False)
    engineers = tuple(
        world.spawn_unit(Faction.PLAYER, UnitKind.ENGINEER, 300, 300) for _ in range(10)
    )
    build = BuildPayloadV1(
        selection=SelectionV1(unit_ids=engineers),
        facility_kind="tower",
        target=PositionV1(x=300, y=300),
    )
    assert world.execute(command(world, build)).status == CommandStatus.ACCEPTED
    world.step(170)
    tower = world.facilities[0]
    infantry = tuple(
        world.spawn_unit(Faction.PLAYER, UnitKind.INFANTRY, 320, 300) for _ in range(25)
    )
    enter = FacilityActionPayloadV1(
        kind="enter_tower",
        selection=SelectionV1(unit_ids=infantry),
        facility_id=tower.facility_id,
    )
    result = world.execute(command(world, enter))
    assert result.status == CommandStatus.ACCEPTED
    occupants = world.resolve_selection(SelectionV1(unit_ids=infantry), Faction.PLAYER)
    inside = occupants[world.units.facility_id[occupants] == tower.facility_id]
    assert len(inside) == 20
    assert min(world.units.attack_range[inside]) == 260
    leave = FacilityActionPayloadV1(
        kind="exit_tower",
        selection=SelectionV1(unit_ids=tuple(world.units.entity_id[inside])),
        facility_id=tower.facility_id,
    )
    assert world.execute(command(world, leave)).status == CommandStatus.ACCEPTED
    assert all(world.units.facility_id[inside] == -1)
    assert max(world.units.attack_range[inside]) == 20


def test_boat_capacity_movement_and_disembark() -> None:
    world = World(seed=10, spawn_armies=False)
    river_cells = (world.map.terrain == 4).nonzero()
    row, col = int(river_cells[0][20]), int(river_cells[1][20])
    x, y = (col + 0.5) * world.map.tile_size, (row + 0.5) * world.map.tile_size
    engineers = tuple(world.spawn_unit(Faction.PLAYER, UnitKind.ENGINEER, x, y) for _ in range(4))
    build = BuildPayloadV1(
        selection=SelectionV1(unit_ids=engineers),
        facility_kind="boat",
        target=PositionV1(x=x, y=y),
    )
    assert world.execute(command(world, build)).status == CommandStatus.ACCEPTED
    world.step(160)
    boat = world.facilities[0]
    passengers = tuple(
        world.spawn_unit(Faction.PLAYER, UnitKind.INFANTRY, x + 10, y) for _ in range(25)
    )
    board = FacilityActionPayloadV1(
        kind="board_boat",
        selection=SelectionV1(unit_ids=passengers),
        facility_id=boat.facility_id,
    )
    assert world.execute(command(world, board)).status == CommandStatus.ACCEPTED
    assert sum(world.units.facility_id[: world.units.count] == boat.facility_id) == 20
    sail = FacilityActionPayloadV1(
        kind="sail_boat",
        facility_id=boat.facility_id,
        target=PositionV1(x=x + 100, y=y),
    )
    assert world.execute(command(world, sail)).status == CommandStatus.ACCEPTED
    before = boat.x
    world.step(20)
    assert boat.x > before


def test_shift_append_runs_orders_in_sequence() -> None:
    world = World(seed=11, spawn_armies=False)
    entity_id = world.spawn_unit(Faction.PLAYER, UnitKind.INFANTRY, 100, 100)
    first = command(
        world,
        MovePayloadV1(
            kind="move",
            selection=SelectionV1(unit_ids=(entity_id,)),
            target=PositionV1(x=120, y=100),
        ),
    )
    second = command(
        world,
        MovePayloadV1(
            kind="move",
            selection=SelectionV1(unit_ids=(entity_id,)),
            target=PositionV1(x=180, y=100),
        ),
    ).model_copy(update={"queue_mode": QueueMode.APPEND})
    world.execute(first)
    world.execute(second)
    world.step(50)
    assert world.units.x[0] / world.subpixels > 170


def test_melee_target_has_six_engagement_slots() -> None:
    world = World(seed=12, spawn_armies=False)
    for _ in range(10):
        world.spawn_unit(Faction.PLAYER, UnitKind.INFANTRY, 100, 100)
    target = world.spawn_unit(Faction.ENEMY, UnitKind.COMMANDER, 112, 100)
    target_index = world.units.index_of(target)
    assert target_index is not None
    world.units.hp[target_index] = 1000
    world.units.max_hp[target_index] = 1000
    world._resolve_combat()
    assert world.units.hp[target_index] == 1000 - 6 * 16


def test_destroyed_tower_ejects_and_restores_occupants() -> None:
    world = World(seed=13, spawn_armies=False)
    attackers = tuple(
        world.spawn_unit(Faction.PLAYER, UnitKind.INFANTRY, 100, 100) for _ in range(6)
    )
    occupant = world.spawn_unit(Faction.ENEMY, UnitKind.INFANTRY, 112, 100)
    facility = Facility(
        facility_id=1,
        faction=int(Faction.ENEMY),
        kind="tower",
        x=112,
        y=100,
        progress=800,
        required_work=800,
        minimum_engineers=10,
        hp=50,
        max_hp=1000,
        complete=True,
        builder_ids=[],
    )
    world.facilities.append(facility)
    occupant_index = world.units.index_of(occupant)
    assert occupant_index is not None
    world.units.facility_id[occupant_index] = facility.facility_id
    world.units.attack_range[occupant_index] = 260
    payload = AttackFacilityPayloadV1(
        selection=SelectionV1(unit_ids=attackers), target_facility_id=facility.facility_id
    )
    assert world.execute(command(world, payload)).status == CommandStatus.ACCEPTED
    world._resolve_combat()
    assert facility.destroyed
    assert world.units.facility_id[occupant_index] == -1
    assert world.units.attack_range[occupant_index] == 20


def test_destroyed_boat_damages_passengers() -> None:
    world = World(seed=14, spawn_armies=False)
    passenger = world.spawn_unit(Faction.PLAYER, UnitKind.INFANTRY, 100, 100)
    facility = Facility(
        facility_id=1,
        faction=int(Faction.PLAYER),
        kind="boat",
        x=100,
        y=100,
        progress=300,
        required_work=300,
        minimum_engineers=4,
        hp=1,
        max_hp=300,
        complete=True,
        builder_ids=[],
    )
    world.facilities.append(facility)
    passenger_index = world.units.index_of(passenger)
    assert passenger_index is not None
    world.units.facility_id[passenger_index] = facility.facility_id
    world._destroy_facility(facility)
    assert world.units.hp[passenger_index] == 80
