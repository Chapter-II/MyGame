import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from mygame.maps import BattleMap, generate_map, validate_map
from mygame.simulation import FlowFieldCache


def test_symmetric_map_is_exactly_rotation_symmetric() -> None:
    battle_map = generate_map(seed=17, symmetric=True)
    assert np.array_equal(battle_map.terrain, np.rot90(battle_map.terrain, 2))
    by_position = {(round(v.x), round(v.y)): v for v in battle_map.villages}
    for village in battle_map.villages:
        mirror = (round(battle_map.width - village.x), round(battle_map.height - village.y))
        assert mirror in by_position
        assert by_position[mirror].population == village.population
        assert by_position[mirror].size == village.size


def test_map_seed_is_reproducible() -> None:
    first = generate_map(seed=2026)
    second = generate_map(seed=2026)
    assert np.array_equal(first.terrain, second.terrain)
    assert [v.population for v in first.villages] == [v.population for v in second.villages]


def test_generated_river_has_no_walkable_gaps() -> None:
    battle_map = generate_map(seed=29)
    widths = np.sum(battle_map.terrain == 4, axis=1)
    assert np.all(widths >= 4)
    assert set(widths) == {4, 6, 8, 10, 12}
    assert np.max(np.abs(np.diff(widths))) <= 2
    for row in range(battle_map.rows):
        river_cols = np.flatnonzero(battle_map.terrain[row] == 4)
        assert np.all(np.diff(river_cols) == 1)


def test_generated_map_has_all_terrain_roles_and_six_objectives() -> None:
    battle_map = generate_map(seed=29)
    terrain_counts = np.bincount(battle_map.terrain.ravel(), minlength=6)

    assert np.all(terrain_counts > 0)
    assert terrain_counts[5] >= 300  # A real road network, not isolated decoration.
    assert len(battle_map.villages) == 6
    assert sum(village.x < battle_map.width / 2 for village in battle_map.villages) == 3
    assert sum(village.x > battle_map.width / 2 for village in battle_map.villages) == 3


@settings(max_examples=12, deadline=None)
@given(seed=st.integers(min_value=0, max_value=2**31 - 1), symmetric=st.booleans())
def test_generated_maps_pass_fairness_and_serialization(seed: int, symmetric: bool) -> None:
    battle_map = generate_map(seed=seed, symmetric=symmetric)
    assert validate_map(battle_map) == []
    restored = BattleMap.from_dict(battle_map.to_dict())
    assert np.array_equal(restored.terrain, battle_map.terrain)
    assert [v.population for v in restored.villages] == [v.population for v in battle_map.villages]


def test_flow_field_is_shared_and_invalidated_by_terrain_revision() -> None:
    battle_map = generate_map(seed=29)
    speed = {
        "plain": 1.0,
        "grass": 0.95,
        "forest": 0.6,
        "swamp": 0.4,
        "river": 0.2,
        "road": 1.1,
    }
    cache = FlowFieldCache(battle_map, speed)
    first = cache.get(3500, 1152)
    assert cache.get(3501, 1153) is first
    assert not np.isfinite(first.integration).all()
    assert first.direction_x[36, 10] == 0 and first.direction_y[36, 10] == 0
    cache.set_bridges([(battle_map.width / 2, battle_map.height / 2)], 160)
    bridged = cache.get(3500, 1152)
    assert bridged is not first
    assert np.isfinite(bridged.integration[36, 10])
    assert bridged.direction_x[36, 10] != 0 or bridged.direction_y[36, 10] != 0
    battle_map.revision += 1
    assert cache.get(3500, 1152) is not bridged
