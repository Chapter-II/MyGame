import uuid

import numpy as np

from mygame.constants import BRIDGE_DECK_HALF_WIDTH
from mygame.maps import Terrain
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


def test_default_army_roles_spawn_in_separate_square_formations() -> None:
    world = World(seed=5)
    active = world.units.active(Faction.PLAYER)
    centroids: list[tuple[float, float]] = []
    for kind in (
        UnitKind.INFANTRY,
        UnitKind.SCOUT,
        UnitKind.ENGINEER,
        UnitKind.ASSASSIN,
        UnitKind.RECRUIT,
    ):
        indices = active[world.units.kind[active] == int(kind)]
        assert len(indices)
        xs = world.units.x[indices] / world.subpixels
        ys = world.units.y[indices] / world.subpixels
        centroids.append((float(np.mean(xs)), float(np.mean(ys))))
        # A square formation should not collapse into the former long mixed strip.
        assert max(float(np.ptp(xs)), float(np.ptp(ys))) <= 300
    for first, second in zip(centroids, centroids[1:], strict=False):
        assert (first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2 > 300**2


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


def test_short_commander_move_keeps_the_requested_cardinal_direction() -> None:
    world = World(seed=7, army_size=100)
    commander = world.commander_index(Faction.PLAYER)
    assert commander is not None
    entity_id = int(world.units.entity_id[commander])
    start_x = float(world.units.x[commander] / world.subpixels)
    start_y = float(world.units.y[commander] / world.subpixels)
    payload = MovePayloadV1(
        kind="move",
        selection=SelectionV1(unit_ids=(entity_id,)),
        target=PositionV1(x=start_x, y=start_y - 64),
    )
    assert world.execute(command(world, payload)).status == CommandStatus.ACCEPTED
    world.step(80)
    assert abs(float(world.units.x[commander] / world.subpixels) - start_x) < 2
    assert float(world.units.y[commander] / world.subpixels) < start_y - 50


def test_friendly_collision_makes_units_yield_to_the_commander() -> None:
    world = World(seed=7, spawn_armies=False)
    commander_id = world.spawn_unit(Faction.PLAYER, UnitKind.COMMANDER, 100, 100)
    guard_id = world.spawn_unit(Faction.PLAYER, UnitKind.GUARD, 105, 100)
    commander = world.units.index_of(commander_id)
    guard = world.units.index_of(guard_id)
    assert commander is not None and guard is not None
    commander_before = (int(world.units.x[commander]), int(world.units.y[commander]))
    guard_before = int(world.units.x[guard])
    world._resolve_collisions()
    assert (int(world.units.x[commander]), int(world.units.y[commander])) == commander_before
    assert int(world.units.x[guard]) > guard_before


def test_unknown_river_does_not_leak_through_move_rejection() -> None:
    world = World(seed=7, spawn_armies=False)
    entity_id = world.spawn_unit(Faction.PLAYER, UnitKind.INFANTRY, 1000, 1152)
    payload = MovePayloadV1(
        kind="move",
        selection=SelectionV1(unit_ids=(entity_id,)),
        target=PositionV1(x=3000, y=1152),
    )

    explored_before = world.observation(Faction.PLAYER).known_terrain
    before = int(world.units.x[0])
    accepted = world.execute(command(world, payload))
    explored_after_command = world.observation(Faction.PLAYER).known_terrain
    world.step(20)

    assert accepted.status == CommandStatus.ACCEPTED
    assert accepted.reason_code == "ok"
    assert "河" not in accepted.message_zh
    assert explored_after_command == explored_before
    assert world.units.x[0] > before


def test_known_river_requires_a_bridge_then_uses_the_bridge_deck() -> None:
    world = World(seed=7, spawn_armies=False)
    row = world.map.rows // 2
    river_cols = np.flatnonzero(world.map.terrain[row] == 4)
    start_x = (int(river_cols.min()) - 1 + 0.5) * world.map.tile_size
    target_x = (int(river_cols.max()) + 8 + 0.5) * world.map.tile_size
    y = (row + 0.5) * world.map.tile_size
    entity_id = world.spawn_unit(Faction.PLAYER, UnitKind.INFANTRY, start_x, y)
    payload = MovePayloadV1(
        kind="move",
        selection=SelectionV1(unit_ids=(entity_id,)),
        target=PositionV1(x=target_x, y=y),
    )

    blocked = world.execute(command(world, payload))
    assert blocked.status == CommandStatus.REJECTED
    assert blocked.reason_code == "river_requires_bridge"

    bridge_x, bridge_y, bridge_length, bridge_vertical = world.bridge_geometry_at(
        float(np.mean(river_cols) * world.map.tile_size), y
    )

    bridge = Facility(
        facility_id=1,
        faction=int(Faction.PLAYER),
        kind="bridge",
        x=bridge_x,
        y=bridge_y,
        progress=500,
        required_work=500,
        minimum_engineers=8,
        hp=600,
        max_hp=600,
        complete=True,
        builder_ids=[],
        bridge_length=bridge_length,
        bridge_vertical=bridge_vertical,
    )
    world.facilities.append(bridge)
    world._sync_bridge_navigation()
    result = world.execute(command(world, payload))
    assert result.status == CommandStatus.ACCEPTED
    world.step(500)
    unit_index = world.units.index_of(entity_id)
    assert unit_index is not None
    assert world.units.x[unit_index] / world.subpixels > river_cols.max() * world.map.tile_size


def test_bridge_planning_does_not_measure_an_unexplored_opposite_bank() -> None:
    world = World(seed=7, spawn_armies=False)
    row = 0
    river_cols = np.flatnonzero(world.map.terrain[row] == int(Terrain.RIVER))
    near_bank_x = (int(river_cols.min()) - 1 + 0.5) * world.map.tile_size
    y = (row + 0.5) * world.map.tile_size
    world.spawn_unit(Faction.PLAYER, UnitKind.INFANTRY, near_bank_x, y)
    river_x = (int(river_cols.min()) + 0.5) * world.map.tile_size

    assert world.bridge_geometry_at(river_x, y)[2] > 0
    assert world.known_bridge_geometry_at(Faction.PLAYER, river_x, y) is None


def test_enemy_bridge_stays_out_of_faction_pathfinding_until_discovered() -> None:
    world = World(seed=7, spawn_armies=False)
    row = 0
    river_cols = np.flatnonzero(world.map.terrain[row] == int(Terrain.RIVER))
    river_x = (float(np.mean(river_cols)) + 0.5) * world.map.tile_size
    river_y = (row + 0.5) * world.map.tile_size
    x, y, length, vertical = world.bridge_geometry_at(river_x, river_y)
    world.facilities.append(
        Facility(
            facility_id=1,
            faction=int(Faction.ENEMY),
            kind="bridge",
            x=x,
            y=y,
            progress=500,
            required_work=500,
            minimum_engineers=8,
            hp=600,
            max_hp=600,
            complete=True,
            builder_ids=[],
            bridge_length=length,
            bridge_vertical=vertical,
        )
    )

    world._sync_bridge_navigation()

    assert not world._flow_fields[Faction.PLAYER].bridge_mask.any()
    assert world._flow_fields[Faction.ENEMY].bridge_mask.any()


def test_river_no_longer_deals_forced_crossing_damage() -> None:
    world = World(seed=7, spawn_armies=False)
    river_rows, river_cols = (world.map.terrain == 4).nonzero()
    x = (int(river_cols[0]) + 0.5) * world.map.tile_size
    y = (int(river_rows[0]) + 0.5) * world.map.tile_size
    entity_id = world.spawn_unit(Faction.PLAYER, UnitKind.INFANTRY, x, y)
    unit_index = world.units.index_of(entity_id)
    assert unit_index is not None
    hp_before = float(world.units.hp[unit_index])
    world.step(100)
    assert float(world.units.hp[unit_index]) == hp_before


def test_bridge_edge_collision_push_is_restored_out_of_open_water() -> None:
    world = World(seed=7, spawn_armies=False)
    river_rows, river_cols = (world.map.terrain == 4).nonzero()
    bridge_x = (int(river_cols[0]) + 0.5) * world.map.tile_size
    bridge_y = (int(river_rows[0]) + 0.5) * world.map.tile_size
    far_x = (int(river_cols[-1]) + 0.5) * world.map.tile_size
    far_y = (int(river_rows[-1]) + 0.5) * world.map.tile_size
    world.facilities.append(
        Facility(
            facility_id=1,
            faction=int(Faction.PLAYER),
            kind="bridge",
            x=bridge_x,
            y=bridge_y,
            progress=500,
            required_work=500,
            minimum_engineers=8,
            hp=600,
            max_hp=600,
            complete=True,
            builder_ids=[],
        )
    )
    entity_id = world.spawn_unit(Faction.PLAYER, UnitKind.INFANTRY, bridge_x, bridge_y)
    unit = world.units.index_of(entity_id)
    assert unit is not None
    world.units.previous_x[unit] = round(bridge_x * world.subpixels)
    world.units.previous_y[unit] = round(bridge_y * world.subpixels)
    world.units.x[unit] = round(far_x * world.subpixels)
    world.units.y[unit] = round(far_y * world.subpixels)

    world._restore_illegal_river_entries()

    assert world.units.x[unit] == world.units.previous_x[unit]
    assert world.units.y[unit] == world.units.previous_y[unit]


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
    tactic_event = next(event for event in world.events if event.kind == "tactic_started")
    assert tactic_event.payload["kind"] == "sprint"
    assert tactic_event.payload["unit_ids"] == [infantry]


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
    commander_deaths = [
        event for event in world.events if event.kind == "unit_died" and event.payload["commander"]
    ]
    assert len(commander_deaths) == 2
    assert all("x" in event.payload and "y" in event.payload for event in commander_deaths)


def test_engineers_complete_bridge() -> None:
    world = World(seed=7, spawn_armies=False)
    river_cells = (world.map.terrain == 4).nonzero()
    row, col = int(river_cells[0][0]), int(river_cells[1][0])
    x, y = (col + 0.5) * world.map.tile_size, (row + 0.5) * world.map.tile_size
    river_cols = np.flatnonzero(world.map.terrain[row] == int(Terrain.RIVER))
    near_bank_x = (int(river_cols.min()) - 1 + 0.5) * world.map.tile_size
    far_bank_x = (int(river_cols.max()) + 1 + 0.5) * world.map.tile_size
    ids = tuple(
        world.spawn_unit(Faction.PLAYER, UnitKind.ENGINEER, near_bank_x, y) for _ in range(8)
    )
    world.spawn_unit(Faction.PLAYER, UnitKind.SCOUT, far_bank_x, y)
    payload = BuildPayloadV1(
        selection=SelectionV1(unit_ids=ids),
        facility_kind="bridge",
        target=PositionV1(x=x, y=y),
    )
    assert world.execute(command(world, payload)).status == CommandStatus.ACCEPTED
    world.step(200)
    assert world.facilities[0].complete


def test_engineers_can_build_a_bridge_from_the_river_bank() -> None:
    world = World(seed=7, spawn_armies=False)
    row = world.map.rows // 2
    river_cols = (world.map.terrain[row] == 4).nonzero()[0]
    bridge_x = (float(river_cols.mean()) + 0.5) * world.map.tile_size
    bridge_y = (row + 0.5) * world.map.tile_size
    bank_x = (int(river_cols.min()) - 1 + 0.5) * world.map.tile_size
    ids = tuple(
        world.spawn_unit(Faction.PLAYER, UnitKind.ENGINEER, bank_x, bridge_y) for _ in range(8)
    )
    payload = BuildPayloadV1(
        selection=SelectionV1(unit_ids=ids),
        facility_kind="bridge",
        target=PositionV1(x=bridge_x, y=bridge_y),
    )
    assert world.execute(command(world, payload)).status == CommandStatus.ACCEPTED
    world.step(260)
    assert world.facilities[0].complete


def test_bridge_length_and_work_scale_with_local_river_width() -> None:
    reference = World(seed=7, spawn_armies=False)
    widths = np.sum(reference.map.terrain == 4, axis=1)
    narrow_row = int(np.flatnonzero(widths == widths.min())[0])
    wide_row = int(np.flatnonzero(widths == widths.max())[0])

    def start_bridge(row: int) -> tuple[float, float]:
        world = World(seed=7, spawn_armies=False)
        river_cols = np.flatnonzero(world.map.terrain[row] == 4)
        x = (float(np.mean(river_cols)) + 0.5) * world.map.tile_size
        y = (row + 0.5) * world.map.tile_size
        ids = tuple(world.spawn_unit(Faction.PLAYER, UnitKind.ENGINEER, x, y) for _ in range(8))
        payload = BuildPayloadV1(
            selection=SelectionV1(unit_ids=ids),
            facility_kind="bridge",
            target=PositionV1(x=x, y=y),
        )
        assert world.execute(command(world, payload)).status == CommandStatus.ACCEPTED
        facility = world.facilities[-1]
        half_length = facility.bridge_length / 2
        endpoint_x = 0.0 if facility.bridge_vertical else half_length
        endpoint_y = half_length if facility.bridge_vertical else 0.0
        endpoints = world.map.terrain_at(
            np.asarray([facility.x - endpoint_x, facility.x + endpoint_x]),
            np.asarray([facility.y - endpoint_y, facility.y + endpoint_y]),
        )
        assert np.all(endpoints != int(Terrain.RIVER))
        return facility.bridge_length, facility.required_work

    narrow_length, narrow_work = start_bridge(narrow_row)
    wide_length, wide_work = start_bridge(wide_row)

    assert wide_length > narrow_length
    assert wide_work > narrow_work


def test_bridge_passability_matches_the_visible_rectangular_deck() -> None:
    world = World(seed=7, spawn_armies=False)
    row = world.map.rows // 2
    river_cols = np.flatnonzero(world.map.terrain[row] == int(Terrain.RIVER))
    x = (float(np.mean(river_cols)) + 0.5) * world.map.tile_size
    y = (row + 0.5) * world.map.tile_size
    bridge_x, bridge_y, length, vertical = world.bridge_geometry_at(x, y)
    facility = Facility(
        facility_id=1,
        faction=int(Faction.PLAYER),
        kind="bridge",
        x=bridge_x,
        y=bridge_y,
        progress=500,
        required_work=500,
        minimum_engineers=8,
        hp=600,
        max_hp=600,
        complete=True,
        builder_ids=[],
        bridge_length=length,
        bridge_vertical=vertical,
    )
    world.facilities.append(facility)
    across_x = BRIDGE_DECK_HALF_WIDTH + 1 if vertical else 0
    across_y = 0 if vertical else BRIDGE_DECK_HALF_WIDTH + 1

    assert world._positions_on_bridges(np.asarray([bridge_x]), np.asarray([bridge_y])).item()
    assert not world._positions_on_bridges(
        np.asarray([bridge_x + across_x]), np.asarray([bridge_y + across_y])
    ).item()


def test_collision_push_is_clamped_inside_bridge_rails() -> None:
    world = World(seed=7, spawn_armies=False)
    row = world.map.rows // 2
    river_cols = np.flatnonzero(world.map.terrain[row] == int(Terrain.RIVER))
    x = (float(np.mean(river_cols)) + 0.5) * world.map.tile_size
    y = (row + 0.5) * world.map.tile_size
    bridge_x, bridge_y, length, vertical = world.bridge_geometry_at(x, y)
    facility = Facility(
        facility_id=1,
        faction=int(Faction.PLAYER),
        kind="bridge",
        x=bridge_x,
        y=bridge_y,
        progress=500,
        required_work=500,
        minimum_engineers=8,
        hp=600,
        max_hp=600,
        complete=True,
        builder_ids=[],
        bridge_length=length,
        bridge_vertical=vertical,
    )
    world.facilities.append(facility)
    entity_id = world.spawn_unit(Faction.PLAYER, UnitKind.INFANTRY, bridge_x, bridge_y)
    unit = world.units.index_of(entity_id)
    assert unit is not None
    world.units.previous_x[unit] = round(bridge_x * world.subpixels)
    world.units.previous_y[unit] = round(bridge_y * world.subpixels)
    pushed = BRIDGE_DECK_HALF_WIDTH + 12
    if vertical:
        world.units.x[unit] = round((bridge_x + pushed) * world.subpixels)
    else:
        world.units.y[unit] = round((bridge_y + pushed) * world.subpixels)

    world._constrain_units_to_bridge_decks()
    world._restore_illegal_river_entries()

    unit_x = float(world.units.x[unit] / world.subpixels)
    unit_y = float(world.units.y[unit] / world.subpixels)
    across = abs(unit_x - bridge_x) if vertical else abs(unit_y - bridge_y)
    assert across <= BRIDGE_DECK_HALF_WIDTH - float(world.units.radius[unit]) + 0.01
    assert world._positions_on_bridges(np.asarray([unit_x]), np.asarray([unit_y])).item()


def test_building_rejects_overlapping_facilities() -> None:
    world = World(seed=7, spawn_armies=False)
    plain_rows, plain_cols = (world.map.terrain == 0).nonzero()
    x = (int(plain_cols[0]) + 0.5) * world.map.tile_size
    y = (int(plain_rows[0]) + 0.5) * world.map.tile_size
    ids = tuple(world.spawn_unit(Faction.PLAYER, UnitKind.ENGINEER, x, y) for _ in range(10))
    first = BuildPayloadV1(
        selection=SelectionV1(unit_ids=ids),
        facility_kind="tower",
        target=PositionV1(x=x, y=y),
    )
    second = first.model_copy(update={"target": PositionV1(x=x + 10, y=y)})
    assert world.execute(command(world, first)).status == CommandStatus.ACCEPTED
    rejected = world.execute(command(world, second))
    assert rejected.status == CommandStatus.REJECTED
    assert rejected.reason_code == "facility_overlap"


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


def test_tower_auto_garrisons_builders_then_respects_capacity_and_exit() -> None:
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
    builder_indices = world.resolve_selection(SelectionV1(unit_ids=engineers), Faction.PLAYER)
    assert all(world.units.facility_id[builder_indices] == tower.facility_id)
    assert all(world.units.kind[builder_indices] == int(UnitKind.INFANTRY))
    assert min(world.units.attack_range[builder_indices]) == 260
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
    assert len(inside) == 10
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
    world.step(180)
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
    world.step(100)
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
