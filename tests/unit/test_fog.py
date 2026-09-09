from mygame.protocols import Faction
from mygame.simulation import UnitKind, World


def test_hidden_enemy_is_not_in_observation() -> None:
    world = World(seed=4, spawn_armies=False)
    world.spawn_unit(Faction.PLAYER, UnitKind.SCOUT, 100, 100)
    enemy = world.spawn_unit(Faction.ENEMY, UnitKind.INFANTRY, 3000, 100)
    assert not world.observation(Faction.PLAYER).visible_enemies
    enemy_index = world.units.index_of(enemy)
    assert enemy_index is not None
    world.units.x[enemy_index] = 180 * world.subpixels
    world._perception.update(world)
    visible = world.observation(Faction.PLAYER).visible_enemies
    assert [unit.entity_id for unit in visible] == [enemy]


def test_enemy_entering_a_previously_explored_area_is_visible() -> None:
    world = World(seed=4, spawn_armies=False)
    scout = world.spawn_unit(Faction.PLAYER, UnitKind.SCOUT, 100, 100)
    enemy = world.spawn_unit(Faction.ENEMY, UnitKind.INFANTRY, 3000, 100)
    world.observation(Faction.PLAYER)

    scout_index = world.units.index_of(scout)
    enemy_index = world.units.index_of(enemy)
    assert scout_index is not None and enemy_index is not None and world._perception is not None
    world.units.x[scout_index] = 1000 * world.subpixels
    world.units.x[enemy_index] = 150 * world.subpixels
    world._perception.update(world)

    visible = world.observation(Faction.PLAYER).visible_enemies
    assert [unit.entity_id for unit in visible] == [enemy]


def test_hidden_assassin_requires_exposure() -> None:
    world = World(seed=4, spawn_armies=False)
    world.spawn_unit(Faction.PLAYER, UnitKind.INFANTRY, 100, 100)
    assassin = world.spawn_unit(Faction.ENEMY, UnitKind.ASSASSIN, 150, 100)
    world.observation(Faction.PLAYER)
    assert not world.observation(Faction.PLAYER).visible_enemies
    index = world.units.index_of(assassin)
    assert index is not None
    world.units.exposed[index] = True
    world._perception.update(world)
    assert world.observation(Faction.PLAYER).visible_enemies[0].entity_id == assassin


def test_explored_village_population_stays_live() -> None:
    world = World(seed=6, spawn_armies=False)
    village = world.map.villages[0]
    scout = world.spawn_unit(Faction.PLAYER, UnitKind.SCOUT, village.x, village.y)
    first = world.observation(Faction.PLAYER)
    known_population = int(first.known_villages[0]["population"])
    scout_index = world.units.index_of(scout)
    assert scout_index is not None and world._perception is not None
    world.units.x[scout_index] = 10 * world.subpixels
    world.units.y[scout_index] = 10 * world.subpixels
    world._perception.update(world)
    village.population = max(0, village.population - 10)
    explored_snapshot = world.observation(Faction.PLAYER)
    assert int(explored_snapshot.known_villages[0]["population"]) == known_population - 10
