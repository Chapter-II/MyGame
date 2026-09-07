from mygame.ai import LocalStrategicAI
from mygame.persistence import (
    ReplayPlayer,
    ReplayRecorder,
    SaveManager,
    SettingsManager,
    SettingsV1,
)
from mygame.protocols import (
    CommandEnvelopeV1,
    CommandSource,
    Faction,
    MovePayloadV1,
    PositionV1,
    SelectionV1,
)
from mygame.simulation import World


def test_save_round_trip_preserves_checksum(tmp_path) -> None:
    world = World(seed=42)
    world.observation(Faction.PLAYER)
    world.step(12)
    manager = SaveManager(tmp_path)
    path = manager.save(world, "roundtrip")
    restored = manager.load(path)
    assert restored.checksum() == world.checksum()


def test_replay_container_round_trip(tmp_path) -> None:
    world = World(seed=43)
    recorder = ReplayRecorder(world)
    world.step(100)
    recorder.update(world)
    replay = recorder.finish(world)
    manager = SaveManager(tmp_path)
    path = manager.save_replay(replay, "test")
    restored = manager.load_replay(path)
    assert restored.final_checksum == replay.final_checksum
    assert restored.checkpoints[0].tick == 100


def test_replay_final_checkpoint_and_checksum() -> None:
    world = World(seed=44)
    recorder = ReplayRecorder(world)
    world.step(37)
    replay = recorder.finish(world)
    assert replay.checkpoints[-1].tick == 37
    player = ReplayPlayer(replay)
    player.seek(player.final_tick)
    assert player.checksum_valid is True


def test_settings_round_trip(tmp_path) -> None:
    manager = SettingsManager(tmp_path)
    manager.save(SettingsV1(ui_scale=1.5, reduced_motion=True))
    restored = manager.load()
    assert restored.ui_scale == 1.5
    assert restored.reduced_motion is True


def test_identical_seed_and_command_stream_is_deterministic(tmp_path) -> None:
    command = CommandEnvelopeV1(
        command_id="deterministic-move",
        faction=Faction.PLAYER,
        issued_tick=0,
        source=CommandSource.KEYBOARD,
        payload=MovePayloadV1(
            kind="attack_move",
            selection=SelectionV1(group_id=1),
            target=PositionV1(x=1800, y=1152),
        ),
    )
    uninterrupted = World(seed=45)
    resumed = World(seed=45)
    uninterrupted.execute(command)
    resumed.execute(command)
    uninterrupted.step(180)
    resumed.step(80)
    manager = SaveManager(tmp_path)
    restored = manager.load(manager.save(resumed, "resume"))
    restored.step(100)
    assert restored.checksum() == uninterrupted.checksum()


def test_save_resume_with_local_ai_matches_uninterrupted(tmp_path) -> None:
    def advance(world: World, ai: LocalStrategicAI, ticks: int) -> None:
        for _ in range(ticks):
            if ai.ready(world.tick):
                for issued in ai.decide(world.observation(Faction.ENEMY)):
                    world.execute(issued)
            world.step()

    uninterrupted = World(seed=46)
    uninterrupted_ai = LocalStrategicAI(
        uninterrupted.map.width,
        uninterrupted.map.height,
        config=uninterrupted.balance.ai["normal"],
    )
    advance(uninterrupted, uninterrupted_ai, 200)

    split = World(seed=46)
    split_ai = LocalStrategicAI(
        split.map.width, split.map.height, config=split.balance.ai["normal"]
    )
    advance(split, split_ai, 100)
    manager = SaveManager(tmp_path)
    restored = manager.load(manager.save(split, "ai-resume"))
    restored_ai = LocalStrategicAI(
        restored.map.width, restored.map.height, config=restored.balance.ai["normal"]
    )
    advance(restored, restored_ai, 100)
    assert restored.checksum() == uninterrupted.checksum()
