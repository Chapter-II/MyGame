from __future__ import annotations

from types import SimpleNamespace

import pytest

from mygame.app.game import BATTLE_RECT, MINIMAP_RECT, BattleEffect, Camera, GameApp
from mygame.protocols import Faction, SelectionV1
from mygame.rendering import Theme
from mygame.simulation import UnitKind, World


def app_with_world() -> GameApp:
    app = GameApp.__new__(GameApp)
    app.world = World(seed=31, spawn_armies=False)
    app.camera = Camera(app.world)
    app.scene = "battle"
    app.theme = Theme()
    app.reduced_motion = False
    app.selected_ids = set()
    app.messages = []
    app.sounds = SimpleNamespace(play=lambda _name: None)
    app.context_menu_point = None
    app.context_menu_world = None
    return app


def test_zoom_eases_toward_target_around_pointer_anchor() -> None:
    app = app_with_world()
    anchor = (BATTLE_RECT.centerx + 120, BATTLE_RECT.centery - 60)
    world_before = app.camera.screen_to_world(*anchor)
    zoom_before = app.camera.zoom

    app._request_zoom(1.5, anchor)
    app._update_camera_zoom(1 / 60)

    assert zoom_before < app.camera.zoom < app.camera.target_zoom
    assert app.camera.screen_to_world(*anchor) == pytest.approx(world_before, abs=0.01)


def test_minimap_drag_maps_pointer_to_world_position() -> None:
    app = app_with_world()
    app.camera.zoom = 1.0
    app.camera.target_zoom = 1.0

    app._camera_from_minimap(MINIMAP_RECT.center)

    assert app.camera.x == pytest.approx(app.world.map.width / 2)
    assert app.camera.y == pytest.approx(app.world.map.height / 2)


def test_context_build_selection_finds_nearest_engineers() -> None:
    app = app_with_world()
    far = app.world.spawn_unit(Faction.PLAYER, UnitKind.ENGINEER, 100, 100)
    near = app.world.spawn_unit(Faction.PLAYER, UnitKind.ENGINEER, 500, 500)

    selected = app._context_selection(520, 500, 1)

    assert selected == (near,)
    assert far not in selected


def test_context_terrain_lookup_keeps_unexplored_cells_unknown() -> None:
    app = app_with_world()
    app.world.spawn_unit(Faction.PLAYER, UnitKind.SCOUT, 200, 200)
    app.world.observation(Faction.PLAYER)
    explored_before = app.world._perception.explored.copy()

    terrain = app._known_terrain_at(app.world.map.width / 2, app.world.map.height / 2)

    assert terrain is None
    assert (app.world._perception.explored == explored_before).all()


def test_battle_effects_expire_and_stay_bounded() -> None:
    app = app_with_world()
    app.battle_effects = [BattleEffect("impact", 10, 10, 10, 10, 0.2) for _ in range(110)]

    app._update_battle_effects(0.05)
    assert len(app.battle_effects) == 96

    app._update_battle_effects(0.2)
    assert app.battle_effects == []


def test_mouse_quick_select_can_select_all_engineers() -> None:
    app = app_with_world()
    engineer = app.world.spawn_unit(Faction.PLAYER, UnitKind.ENGINEER, 100, 100)
    app.world.spawn_unit(Faction.PLAYER, UnitKind.SCOUT, 120, 100)
    app.messages = []

    app._select_kind(UnitKind.ENGINEER)

    assert app.selected_ids == {engineer}


def test_village_recruits_are_selected_and_can_be_trained_with_mouse_ui() -> None:
    app = app_with_world()
    village = app.world.map.villages[0]
    app.world.spawn_unit(Faction.PLAYER, UnitKind.INFANTRY, village.x, village.y)
    app.context_menu_point = MINIMAP_RECT.center
    app.context_menu_world = (village.x, village.y)
    count_before = app.world.units.count

    app._issue_context_recruit(village.village_id)

    assert app.world.units.count > count_before
    assert len(app.selected_ids) == app.world.units.count - count_before
    app._convert_selected("engineer")
    selected = app.world.resolve_selection(
        SelectionV1(unit_ids=tuple(app.selected_ids)), Faction.PLAYER
    )
    assert all(app.world.units.kind[selected] == int(UnitKind.ENGINEER))
