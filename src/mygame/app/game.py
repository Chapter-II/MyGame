from __future__ import annotations

import math
import os
import sys
import time
import uuid
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import pygame

from mygame.ai import LocalStrategicAI
from mygame.commands import DeepSeekCommandParser, RuleCommandParser
from mygame.input import MicrophoneRecorder, VoiceUnavailable, WhisperRecognizer
from mygame.maps import generate_map
from mygame.persistence import (
    ReplayPlayer,
    ReplayRecorder,
    SaveManager,
    SettingsManager,
    SettingsV1,
)
from mygame.protocols import (
    CommandEnvelopeV1,
    CommandResultV1,
    CommandSource,
    CommandStatus,
    Faction,
    FocusFirePayloadV1,
    GroupPayloadV1,
    MovePayloadV1,
    PositionV1,
    QueueMode,
    SelectionV1,
)
from mygame.rendering import Theme
from mygame.simulation import GameOutcome, UnitKind, World

LOGICAL_SIZE = (1280, 720)
BATTLE_RECT = pygame.Rect(0, 40, 960, 548)


@dataclass(slots=True)
class Button:
    rect: pygame.Rect
    label: str
    action: Callable[[], None]
    enabled: bool = True


class FontBook:
    def __init__(self, scale: float = 1.0) -> None:
        self.scale = scale
        self.path = self._find_font()
        self.cache: dict[tuple[int, bool], pygame.font.Font] = {}

    @staticmethod
    def _find_font() -> str | None:
        roots = [Path(__file__).resolve().parents[3], Path.cwd()]
        bundle_root = getattr(sys, "_MEIPASS", None)
        if bundle_root:
            roots.insert(0, Path(bundle_root))
        for root in roots:
            bundled = root / "assets" / "fonts" / "NotoSansSC.ttf"
            if bundled.exists():
                return str(bundled)
        candidates = (
            "Noto Sans CJK SC",
            "Microsoft YaHei",
            "Droid Sans Fallback",
            "WenQuanYi Zen Hei",
            "SimHei",
            "Arial Unicode MS",
        )
        for candidate in candidates:
            path = pygame.font.match_font(candidate)
            if path:
                return path
        return None

    def get(self, size: int, bold: bool = False) -> pygame.font.Font:
        key = (max(10, round(size * self.scale)), bold)
        if key not in self.cache:
            font = pygame.font.Font(self.path, key[0])
            font.set_bold(bold)
            self.cache[key] = font
        return self.cache[key]


class SoundBook:
    def __init__(self, volume: float) -> None:
        self.volume = volume
        self.sounds: dict[str, pygame.mixer.Sound] = {}
        if pygame.mixer.get_init() is None:
            return
        roots = [Path(__file__).resolve().parents[3], Path.cwd()]
        bundle_root = getattr(sys, "_MEIPASS", None)
        if bundle_root:
            roots.insert(0, Path(bundle_root))
        for root in roots:
            audio_root = root / "assets" / "audio"
            if not audio_root.exists():
                continue
            for name in ("command", "impact", "complete", "victory"):
                try:
                    self.sounds[name] = pygame.mixer.Sound(audio_root / f"{name}.wav")
                except pygame.error:
                    pass
            break
        self.set_volume(volume)

    def set_volume(self, volume: float) -> None:
        self.volume = volume
        for sound in self.sounds.values():
            sound.set_volume(volume)

    def play(self, name: str) -> None:
        sound = self.sounds.get(name)
        if sound is not None and self.volume > 0:
            sound.play()


class Camera:
    def __init__(self, world: World) -> None:
        self.x = world.map.width / 2
        self.y = world.map.height / 2
        self.zoom = 0.235

    def world_to_screen(self, x: float, y: float) -> tuple[int, int]:
        sx = BATTLE_RECT.centerx + (x - self.x) * self.zoom
        sy = BATTLE_RECT.centery + (y - self.y) * self.zoom
        return round(sx), round(sy)

    def screen_to_world(self, x: float, y: float) -> tuple[float, float]:
        return self.x + (x - BATTLE_RECT.centerx) / self.zoom, self.y + (
            y - BATTLE_RECT.centery
        ) / self.zoom

    def clamp(self, world: World) -> None:
        half_w = BATTLE_RECT.width / max(self.zoom, 0.01) / 2
        half_h = BATTLE_RECT.height / max(self.zoom, 0.01) / 2
        if half_w * 2 >= world.map.width:
            self.x = world.map.width / 2
        else:
            self.x = float(np.clip(self.x, half_w, world.map.width - half_w))
        if half_h * 2 >= world.map.height:
            self.y = world.map.height / 2
        else:
            self.y = float(np.clip(self.y, half_h, world.map.height - half_h))


class GameApp:
    def __init__(self) -> None:
        pygame.init()
        pygame.font.init()
        self.theme = Theme()
        self.settings_manager = SettingsManager()
        self.settings = self.settings_manager.load()
        flags = pygame.FULLSCREEN if self.settings.fullscreen else pygame.RESIZABLE
        self.screen = pygame.display.set_mode(LOGICAL_SIZE, flags)
        pygame.display.set_caption("指挥官战术对抗")
        self.canvas = pygame.Surface(LOGICAL_SIZE)
        self.clock = pygame.time.Clock()
        self.fonts = FontBook(self.settings.ui_scale)
        self.sounds = SoundBook(self.settings.master_volume)
        self.running = True
        self.scene = "menu"
        self.buttons: list[Button] = []
        self.world: World | None = None
        self.camera: Camera | None = None
        self.ai: LocalStrategicAI | None = None
        self.save_manager = SaveManager()
        self.recorder: ReplayRecorder | None = None
        self.replay_player: ReplayPlayer | None = None
        self.rule_parser = RuleCommandParser(group_names=self.settings.group_names)
        self.deepseek = DeepSeekCommandParser()
        self.online_enabled = True
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="mygame-io")
        self.pending_text: Future[CommandResultV1] | None = None
        self.pending_voice: Future[str] | None = None
        self.microphone = MicrophoneRecorder()
        self.whisper = WhisperRecognizer(
            os.getenv("MYGAME_WHISPER_MODEL", "small"),
            allow_download=os.getenv("MYGAME_ALLOW_MODEL_DOWNLOAD") == "1",
        )
        self.pending_voice_samples: bytes | None = None
        self.recording = False
        self.selected_ids: set[int] = set()
        self.drag_start: tuple[int, int] | None = None
        self.command_input = ""
        self.composition = ""
        self.typing = False
        self.suggestions: list[str] = []
        self.selected_suggestion = 0
        self.messages: list[tuple[str, tuple[int, int, int]]] = []
        self.paused = False
        self.speed = 1.0
        self.accumulator = 0.0
        self.help_open = False
        self.reduced_motion = self.settings.reduced_motion
        self.fullscreen = self.settings.fullscreen
        self.last_event_index = 0
        self.stats = {"player_losses": 0, "enemy_losses": 0}
        self.result_sound_played = False
        self.tick_timings: deque[float] = deque(maxlen=240)
        self.setup_seed = 20260907
        self.setup_symmetric = True
        self.setup_army_size = 500
        self.setup_difficulty = "normal"
        self.setup_composition: dict[str, int] = {}
        self._reset_composition()
        self.view_faction: Faction | None = Faction.PLAYER
        self.tutorial_page = 0
        self.pending_retry_text: str | None = None
        self.provider_retry_count = 0
        self.ambiguity_candidates: list[CommandEnvelopeV1] = []

    def run(self, max_frames: int | None = None) -> int:
        frames = 0
        while self.running:
            dt = min(self.clock.tick(60) / 1000.0, 0.1)
            self._events()
            self._update(dt)
            self._draw()
            self._present()
            frames += 1
            if max_frames is not None and frames >= max_frames:
                self.running = False
        self.executor.shutdown(wait=False, cancel_futures=True)
        pygame.quit()
        return 0

    def open_setup(self) -> None:
        self.scene = "setup"

    def new_battle(self) -> None:
        battle_map = generate_map(seed=self.setup_seed, symmetric=self.setup_symmetric)
        self.world = World(
            battle_map=battle_map,
            seed=self.setup_seed,
            army_size=self.setup_army_size,
            army_composition=self.setup_composition,
        )
        self.world.groups[int(Faction.PLAYER)].update(self.settings.group_names)
        self.camera = Camera(self.world)
        self.ai = LocalStrategicAI(
            self.world.map.width,
            self.world.map.height,
            difficulty=self.setup_difficulty,
            config=self.world.balance.ai.get(self.setup_difficulty, self.world.balance.ai["normal"]),
        )
        self.recorder = ReplayRecorder(self.world)
        self.replay_player = None
        self.view_faction = Faction.PLAYER
        self.scene = "battle"
        self.selected_ids.clear()
        self.messages = [("侦察地图并寻找敌方将领。F1 打开指令书。", self.theme.ink)]
        self.paused = False
        self.accumulator = 0.0
        self.last_event_index = 0
        self.stats = {"player_losses": 0, "enemy_losses": 0}
        self.result_sound_played = False

    def load_battle(self) -> None:
        try:
            self.world = self.save_manager.load()
            self.camera = Camera(self.world)
            self.ai = LocalStrategicAI(
                self.world.map.width,
                self.world.map.height,
                config=self.world.balance.ai.get("normal", {}),
            )
            self.recorder = ReplayRecorder(self.world)
            self.replay_player = None
            self.view_faction = Faction.PLAYER
            self.scene = "battle"
            self.stats = {
                "player_losses": self.world.statistics["losses"][int(Faction.PLAYER)],
                "enemy_losses": self.world.statistics["losses"][int(Faction.ENEMY)],
            }
            self._message("已载入快速存档。", self.theme.success)
        except (OSError, ValueError) as exc:
            self._message(f"载入失败：{exc}", self.theme.enemy)

    def load_latest_replay(self) -> None:
        path = self.save_manager.latest_replay()
        if path is None:
            self._message("还没有可用回放。", self.theme.warning)
            return
        try:
            self.replay_player = ReplayPlayer(self.save_manager.load_replay(path))
            self.world = self.replay_player.world
            self.camera = Camera(self.world)
            self.ai = None
            self.recorder = None
            self.scene = "replay"
            self.paused = False
            self.speed = 1.0
            self.view_faction = Faction.PLAYER
            self.messages = [(f"正在回放：{path.name}", self.theme.ink)]
        except (OSError, ValueError) as exc:
            self._message(f"回放载入失败：{exc}", self.theme.enemy)

    def _events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.VIDEORESIZE and not self.fullscreen:
                self.screen = pygame.display.set_mode(event.size, pygame.RESIZABLE)
            elif self.scene in {"menu", "setup", "settings", "tutorial"}:
                self._menu_event(event)
            elif self.scene in {"battle", "replay", "result"}:
                self._battle_event(event)

    def _menu_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self.scene = "menu"
            return
        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            point = self._logical_mouse(event.pos)
            for button in self.buttons:
                if button.enabled and button.rect.collidepoint(point):
                    button.action()
                    break

    def _battle_event(self, event: pygame.event.Event) -> None:
        if self.world is None or self.camera is None:
            return
        if event.type == pygame.TEXTINPUT and self.typing:
            if len(self.command_input) + len(event.text) <= 120:
                self.command_input += event.text
            self.composition = ""
            self._update_ime_rect()
            return
        if event.type == pygame.TEXTEDITING and self.typing:
            self.composition = event.text
            return
        if self.scene == "result" and event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            point = self._logical_mouse(event.pos)
            for button in self.buttons:
                if button.enabled and button.rect.collidepoint(point):
                    button.action()
                    return
        if event.type == pygame.KEYDOWN:
            if self.scene == "replay":
                if event.key == pygame.K_F2:
                    self.view_faction = Faction.PLAYER
                    self._message("回放视角：玩家", self.theme.ally)
                    return
                if event.key == pygame.K_F3:
                    self.view_faction = Faction.ENEMY
                    self._message("回放视角：敌方", self.theme.enemy)
                    return
                if event.key == pygame.K_F4:
                    self.view_faction = None
                    self._message("回放视角：全局", self.theme.warning)
                    return
                if event.key in (pygame.K_LEFT, pygame.K_RIGHT) and self.replay_player:
                    delta = -100 if event.key == pygame.K_LEFT else 100
                    self.world = self.replay_player.seek(self.world.tick + delta)
                    self._message(f"已跳转至 {self.world.tick / 20:.1f} 秒", self.theme.ink)
                    return
                if event.key not in {
                    pygame.K_ESCAPE,
                    pygame.K_F1,
                    pygame.K_F11,
                    pygame.K_SPACE,
                    pygame.K_PLUS,
                    pygame.K_EQUALS,
                    pygame.K_KP_PLUS,
                    pygame.K_MINUS,
                    pygame.K_KP_MINUS,
                }:
                    return
            if self.typing:
                if event.key == pygame.K_ESCAPE:
                    self.typing = False
                    self.composition = ""
                    if hasattr(pygame.key, "stop_text_input"):
                        pygame.key.stop_text_input()
                elif event.key == pygame.K_RETURN:
                    self._submit_text()
                elif event.key == pygame.K_BACKSPACE:
                    self.command_input = self.command_input[:-1]
                return
            if (
                self.scene == "battle"
                and self.ambiguity_candidates
                and pygame.K_1 <= event.key <= pygame.K_3
            ):
                choice = event.key - pygame.K_1
                if choice < len(self.ambiguity_candidates):
                    command = self.ambiguity_candidates[choice]
                    self.ambiguity_candidates = []
                    result = self.world.execute(command)
                    self._remember_group_name(command, result)
                    self._message(result.message_zh, self.theme.success)
                    self.sounds.play("command")
                return
            if event.key == pygame.K_RETURN:
                self.typing = True
                self.command_input = ""
                self.composition = ""
                if hasattr(pygame.key, "start_text_input"):
                    self._update_ime_rect()
                    pygame.key.start_text_input()
            elif event.key == pygame.K_ESCAPE:
                if self.help_open:
                    self.help_open = False
                else:
                    self.paused = not self.paused
            elif event.key == pygame.K_F1:
                self.help_open = not self.help_open
            elif event.key == pygame.K_F5 and self.scene == "battle":
                try:
                    path = self.save_manager.save(self.world)
                    self._message(f"已保存：{path.name}", self.theme.success)
                except OSError as exc:
                    self._message(f"保存失败：{exc}", self.theme.enemy)
            elif event.key == pygame.K_F9 and self.scene == "battle":
                self.load_battle()
            elif event.key == pygame.K_F11:
                self._toggle_fullscreen()
            elif event.key == pygame.K_SPACE:
                self.paused = not self.paused
            elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                self.speed = min(4.0, self.speed * 2)
            elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                self.speed = max(0.5, self.speed / 2)
            elif event.key == pygame.K_v and not self.recording and self.scene == "battle":
                try:
                    self.microphone.start()
                    self.recording = True
                    self._message("正在聆听……松开 V 结束。", self.theme.warning)
                except VoiceUnavailable as exc:
                    self._message(str(exc), self.theme.enemy)
            elif pygame.K_1 <= event.key <= pygame.K_9:
                group = event.key - pygame.K_0
                if event.mod & pygame.KMOD_CTRL:
                    self._assign_group(group)
                else:
                    self._select_group(group)
            elif event.key in (pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT):
                self._move_commander(event.key)
            elif event.key == pygame.K_y and self.pending_voice_samples:
                self.whisper.allow_download = True
                self.pending_voice = self.executor.submit(
                    self.whisper.transcribe, self.pending_voice_samples
                )
                self.pending_voice_samples = None
                self._message("正在下载/加载语音模型并转写……", self.theme.warning)
            elif event.key == pygame.K_n and self.pending_voice_samples:
                self.pending_voice_samples = None
                self._message("已取消语音模型下载。", self.theme.muted)
            elif event.key == pygame.K_r and self.pending_retry_text and self.scene == "battle":
                self._retry_provider()
            elif event.key == pygame.K_o and self.pending_retry_text and self.scene == "battle":
                if self.pending_text is not None:
                    self.pending_text.cancel()
                    self.pending_text = None
                self.pending_retry_text = None
                self.online_enabled = False
                self.paused = False
                self._message("已切换离线游玩；鼠标、键盘和标准文字命令可用。", self.theme.success)
            elif event.key == pygame.K_q and self.pending_retry_text and self.scene == "battle":
                try:
                    if self.pending_text is not None:
                        self.pending_text.cancel()
                        self.pending_text = None
                    self.save_manager.save(self.world, "network-emergency")
                finally:
                    self.running = False
        elif event.type == pygame.KEYUP and event.key == pygame.K_v and self.recording:
            samples = self.microphone.stop()
            self.recording = False
            self.pending_voice_samples = samples
            self.pending_voice = self.executor.submit(self.whisper.transcribe, samples)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            point = self._logical_mouse(event.pos)
            if event.button == 1 and BATTLE_RECT.collidepoint(point):
                self.drag_start = point
            elif event.button == 2 and BATTLE_RECT.collidepoint(point):
                self.drag_start = point
            elif event.button == 3 and BATTLE_RECT.collidepoint(point) and self.scene == "battle":
                self._mouse_order(
                    point,
                    pygame.key.get_pressed()[pygame.K_a],
                    pygame.key.get_mods() & pygame.KMOD_SHIFT,
                )
            elif event.button in (4, 5) and BATTLE_RECT.collidepoint(point):
                factor = 1.16 if event.button == 4 else 1 / 1.16
                self.camera.zoom = float(np.clip(self.camera.zoom * factor, 0.18, 1.5))
                self.camera.clamp(self.world)
        elif event.type == pygame.MOUSEBUTTONUP:
            point = self._logical_mouse(event.pos)
            if (
                event.button == 1
                and self.scene == "replay"
                and self.replay_player is not None
                and pygame.Rect(40, 636, 880, 28).collidepoint(point)
            ):
                ratio = float(np.clip((point[0] - 40) / 880, 0, 1))
                self.world = self.replay_player.seek(round(self.replay_player.final_tick * ratio))
                self.accumulator = 0.0
            elif event.button == 1 and self.drag_start:
                self._finish_selection(self.drag_start, point)
                self.drag_start = None
            elif event.button == 2:
                self.drag_start = None
        elif event.type == pygame.MOUSEMOTION and event.buttons[1] and self.drag_start:
            dx = event.rel[0] / max(self.screen.get_width() / LOGICAL_SIZE[0], 0.01)
            dy = event.rel[1] / max(self.screen.get_height() / LOGICAL_SIZE[1], 0.01)
            self.camera.x -= dx / self.camera.zoom
            self.camera.y -= dy / self.camera.zoom
            self.camera.clamp(self.world)

    def _logical_mouse(self, point: tuple[int, int]) -> tuple[int, int]:
        viewport = self._viewport()
        x = (point[0] - viewport.x) * LOGICAL_SIZE[0] / max(viewport.width, 1)
        y = (point[1] - viewport.y) * LOGICAL_SIZE[1] / max(viewport.height, 1)
        return round(x), round(y)

    def _submit_text(self) -> None:
        if self.world is None or self.scene != "battle":
            return
        text = self.command_input.strip()
        self.typing = False
        self.command_input = ""
        self.composition = ""
        if hasattr(pygame.key, "stop_text_input"):
            pygame.key.stop_text_input()
        parsed = self.rule_parser.parse(text, self.world.observation(Faction.PLAYER))
        if parsed.status == CommandStatus.PENDING and parsed.candidates:
            if len(parsed.candidates) > 1:
                self.ambiguity_candidates = list(parsed.candidates[:3])
                self._message(parsed.message_zh, self.theme.warning)
                return
            command = parsed.candidates[0]
            result = self.world.execute(command)
            self._remember_group_name(command, result)
            if result.status == CommandStatus.ACCEPTED:
                self.sounds.play("command")
            self._message(
                result.message_zh,
                self.theme.success if result.status == CommandStatus.ACCEPTED else self.theme.enemy,
            )
        elif self.online_enabled and self.deepseek.available:
            self.pending_retry_text = text
            self.provider_retry_count = 0
            self.pending_text = self.executor.submit(
                self.deepseek.parse, text, self.world.observation(Faction.PLAYER)
            )
            self._message("离线规则未识别，正在请求 DeepSeek……", self.theme.warning)
        else:
            self._message(parsed.message_zh, self.theme.enemy)

    def _update_ime_rect(self) -> None:
        if not hasattr(pygame.key, "set_text_input_rect"):
            return
        viewport = self._viewport()
        scale = viewport.width / LOGICAL_SIZE[0]
        box = pygame.Rect(340, 618, 590, 42)
        screen_rect = pygame.Rect(
            viewport.x + int(box.x * scale),
            viewport.y + int(box.y * scale),
            int(box.width * scale),
            int(box.height * scale),
        )
        pygame.key.set_text_input_rect(screen_rect)

    def _retry_provider(self) -> None:
        if self.world is None or not self.pending_retry_text or self.pending_text is not None:
            return
        if self.provider_retry_count >= 1:
            self._message("在线解析已重试一次，请按 O 切换离线模式。", self.theme.enemy)
            return
        self.provider_retry_count += 1
        self.paused = True
        self.pending_text = self.executor.submit(
            self.deepseek.parse,
            self.pending_retry_text,
            self.world.observation(Faction.PLAYER),
        )
        self._message("正在重试在线命令解析……", self.theme.warning)

    def _assign_group(self, group: int) -> None:
        if self.world is None:
            return
        command = CommandEnvelopeV1(
            command_id=uuid.uuid4().hex,
            faction=Faction.PLAYER,
            issued_tick=self.world.tick,
            source=CommandSource.KEYBOARD,
            payload=GroupPayloadV1(
                selection=SelectionV1(unit_ids=tuple(sorted(self.selected_ids))), group_id=group
            ),
        )
        result = self.world.execute(command)
        if result.status == CommandStatus.ACCEPTED:
            self.sounds.play("command")

    def _select_group(self, group: int) -> None:
        if self.world is None:
            return
        indices = self.world.resolve_selection(SelectionV1(group_id=group), Faction.PLAYER)
        self.selected_ids = {int(self.world.units.entity_id[index]) for index in indices}
        self._message(f"已选择第 {group} 组：{len(indices)} 人。", self.theme.ink)

    def _move_commander(self, key: int) -> None:
        if self.world is None:
            return
        index = self.world.commander_index(Faction.PLAYER)
        if index is None:
            return
        dx = -90 if key == pygame.K_LEFT else 90 if key == pygame.K_RIGHT else 0
        dy = -90 if key == pygame.K_UP else 90 if key == pygame.K_DOWN else 0
        x = self.world.units.x[index] / self.world.subpixels + dx
        y = self.world.units.y[index] / self.world.subpixels + dy
        self._issue_move(
            (int(self.world.units.entity_id[index]),), x, y, False, False, CommandSource.KEYBOARD
        )

    def _mouse_order(self, point: tuple[int, int], attack: bool, append: int) -> None:
        if self.camera is None or self.world is None:
            return
        x, y = self.camera.screen_to_world(*point)
        ids = tuple(sorted(self.selected_ids))
        if not ids:
            commander = self.world.commander_index(Faction.PLAYER)
            ids = (int(self.world.units.entity_id[commander]),) if commander is not None else ()
        observation = self.world.observation(Faction.PLAYER)
        clicked_enemy = next(
            (
                unit
                for unit in observation.visible_enemies
                if sum(
                    (first - second) ** 2
                    for first, second in zip(
                        self.camera.world_to_screen(unit.x, unit.y), point, strict=True
                    )
                )
                <= 14**2
            ),
            None,
        )
        if clicked_enemy is not None:
            self._issue_focus(ids, clicked_enemy.entity_id)
            return
        self._issue_move(ids, x, y, attack, bool(append), CommandSource.MOUSE)

    def _issue_focus(self, ids: tuple[int, ...], target_entity_id: int) -> None:
        if self.world is None:
            return
        command = CommandEnvelopeV1(
            command_id=uuid.uuid4().hex,
            faction=Faction.PLAYER,
            issued_tick=self.world.tick,
            source=CommandSource.MOUSE,
            payload=FocusFirePayloadV1(
                selection=SelectionV1(unit_ids=ids), target_entity_id=target_entity_id
            ),
        )
        result = self.world.execute(command)
        if result.status == CommandStatus.ACCEPTED:
            self.sounds.play("command")
        self._message(
            result.message_zh,
            self.theme.success if result.status == CommandStatus.ACCEPTED else self.theme.enemy,
        )

    def _issue_move(
        self,
        ids: tuple[int, ...],
        x: float,
        y: float,
        attack: bool,
        append: bool,
        source: CommandSource,
    ) -> None:
        if self.world is None:
            return
        command = CommandEnvelopeV1(
            command_id=uuid.uuid4().hex,
            faction=Faction.PLAYER,
            issued_tick=self.world.tick,
            source=source,
            queue_mode=QueueMode.APPEND if append else QueueMode.REPLACE,
            payload=MovePayloadV1(
                kind="attack_move" if attack else "move",
                selection=SelectionV1(unit_ids=ids),
                target=PositionV1(x=x, y=y),
            ),
        )
        result = self.world.execute(command)
        if result.status == CommandStatus.ACCEPTED:
            self.sounds.play("command")
        self._message(
            result.message_zh,
            self.theme.success if result.status == CommandStatus.ACCEPTED else self.theme.enemy,
        )

    def _finish_selection(self, start: tuple[int, int], end: tuple[int, int]) -> None:
        if self.world is None or self.camera is None or self.scene != "battle":
            return
        rect = pygame.Rect(
            min(start[0], end[0]),
            min(start[1], end[1]),
            abs(end[0] - start[0]),
            abs(end[1] - start[1]),
        )
        click = rect.width < 5 and rect.height < 5
        chosen: list[tuple[float, int]] = []
        for index in self.world.units.active(Faction.PLAYER):
            sx, sy = self.camera.world_to_screen(
                self.world.units.x[index] / self.world.subpixels,
                self.world.units.y[index] / self.world.subpixels,
            )
            if (click and (sx - end[0]) ** 2 + (sy - end[1]) ** 2 <= 12**2) or (
                not click and rect.collidepoint(sx, sy)
            ):
                chosen.append(
                    (
                        (sx - end[0]) ** 2 + (sy - end[1]) ** 2,
                        int(self.world.units.entity_id[index]),
                    )
                )
        if click and chosen:
            self.selected_ids = {min(chosen)[1]}
        else:
            self.selected_ids = {entity_id for _, entity_id in chosen}

    def _update(self, dt: float) -> None:
        if self.scene not in {"battle", "replay"} or self.world is None or self.camera is None:
            return
        keys = pygame.key.get_pressed()
        camera_speed = 900 * dt / max(self.camera.zoom, 0.1)
        self.camera.x += (keys[pygame.K_d] - keys[pygame.K_a]) * camera_speed
        self.camera.y += (keys[pygame.K_s] - keys[pygame.K_w]) * camera_speed
        self.camera.clamp(self.world)
        self._poll_workers()
        if self.paused or self.help_open:
            return
        self.accumulator += dt * self.speed
        fixed = self.world.dt
        max_steps = 8
        steps = 0
        while self.accumulator >= fixed and steps < max_steps:
            tick_started = time.perf_counter()
            if self.scene == "replay" and self.replay_player is not None:
                self.replay_player.step()
                self.world = self.replay_player.world
            else:
                if self.ai is not None and self.ai.ready(self.world.tick):
                    for command in self.ai.decide(self.world.observation(Faction.ENEMY)):
                        self.world.execute(command)
                self.world.step()
                if self.recorder is not None:
                    self.recorder.update(self.world)
            self.tick_timings.append((time.perf_counter() - tick_started) * 1000)
            self.accumulator -= fixed
            steps += 1
        if steps == max_steps:
            self.accumulator = 0.0
        self._consume_events()
        if self.scene == "battle" and self.world.outcome != GameOutcome.ONGOING:
            self.scene = "result"
            self.paused = True
            if not self.result_sound_played:
                self.sounds.play(
                    "victory" if self.world.outcome == GameOutcome.PLAYER_WIN else "impact"
                )
                self.result_sound_played = True
            if self.recorder is not None:
                try:
                    self.save_manager.save_replay(self.recorder.finish(self.world))
                except OSError:
                    pass

    def _poll_workers(self) -> None:
        if self.world is None:
            return
        if self.pending_text is not None and self.pending_text.done():
            try:
                result = self.pending_text.result()
            except Exception as exc:  # provider isolation boundary
                result = CommandResultV1(
                    command_id="provider",
                    status=CommandStatus.REJECTED,
                    reason_code="provider_error",
                    message_zh=f"在线解析失败：{type(exc).__name__}",
                )
            self.pending_text = None
            if result.status == CommandStatus.PENDING and result.candidates:
                self.pending_retry_text = None
                command = result.candidates[0]
                if self.world.tick - command.issued_tick <= 200:
                    executed = self.world.execute(command)
                    self._remember_group_name(command, executed)
                    if executed.status == CommandStatus.ACCEPTED:
                        self.sounds.play("command")
                    self._message(
                        executed.message_zh,
                        self.theme.success
                        if executed.status == CommandStatus.ACCEPTED
                        else self.theme.enemy,
                    )
                else:
                    self._message("在线指令返回过晚，已取消。", self.theme.enemy)
            else:
                self.paused = True
                self._message(
                    f"服务失败，战局暂停。{result.message_zh} R 重试 / O 离线 / Q 存档退出",
                    self.theme.enemy,
                )
        if self.pending_voice is not None and self.pending_voice.done():
            try:
                transcript = self.pending_voice.result()
                self.pending_voice = None
                if transcript:
                    self.pending_voice_samples = None
                    self.command_input = transcript
                    self._message(f"识别：{transcript}", self.theme.ink)
                    self._submit_text()
                else:
                    self._message("没有识别到语音。", self.theme.enemy)
            except VoiceUnavailable as exc:
                self.pending_voice = None
                self._message(f"{exc} 按 Y 允许下载，按 N 取消。", self.theme.warning)
            except Exception as exc:
                self.pending_voice = None
                self._message(f"语音识别失败：{type(exc).__name__}", self.theme.enemy)

    def _consume_events(self) -> None:
        if self.world is None:
            return
        impact = False
        completed = False
        for event in self.world.visible_events(Faction.PLAYER, self.last_event_index):
            impact |= event.kind == "unit_died"
            completed |= event.kind == "facility_completed"
        self.stats = {
            "player_losses": self.world.statistics["losses"][int(Faction.PLAYER)],
            "enemy_losses": self.world.statistics["losses"][int(Faction.ENEMY)],
        }
        if completed:
            self.sounds.play("complete")
        elif impact:
            self.sounds.play("impact")
        self.last_event_index = len(self.world.events)

    def _message(self, text: str, color: tuple[int, int, int]) -> None:
        self.messages.append((text, color))
        self.messages = self.messages[-40:]

    def _remember_group_name(self, command: CommandEnvelopeV1, result: CommandResultV1) -> None:
        if (
            result.status == CommandStatus.ACCEPTED
            and isinstance(command.payload, GroupPayloadV1)
            and command.payload.name
        ):
            self.settings.group_names[command.payload.group_id] = command.payload.name
            self.rule_parser.group_names = dict(self.settings.group_names)
            self._save_settings()

    def _draw(self) -> None:
        self.canvas.fill(self.theme.background)
        self.buttons = []
        if self.scene == "menu":
            self._draw_menu()
        elif self.scene == "setup":
            self._draw_setup()
        elif self.scene == "settings":
            self._draw_settings()
        elif self.scene == "tutorial":
            self._draw_tutorial()
        else:
            self._draw_battle()
            if self.scene == "result":
                self._draw_result()

    def _draw_menu(self) -> None:
        title = self.fonts.get(34, True).render("指挥官战术对抗", True, self.theme.ink)
        subtitle = self.fonts.get(16).render(
            "在战争迷雾中，以军令统筹一千名真实士兵", True, self.theme.muted
        )
        self.canvas.blit(title, (96, 92))
        self.canvas.blit(subtitle, (98, 142))
        pygame.draw.line(self.canvas, self.theme.primary, (98, 182), (470, 182), 3)
        labels: list[tuple[str, Callable[[], None], bool]] = [
            ("开始对局", self.open_setup, True),
            (
                "载入快速存档",
                self.load_battle,
                (self.save_manager.save_dir / "quicksave.mgsave").exists(),
            ),
            (
                "观看最近回放",
                self.load_latest_replay,
                self.save_manager.latest_replay() is not None,
            ),
            ("设置与辅助", lambda: setattr(self, "scene", "settings"), True),
            ("新手教程", self._open_tutorial, True),
            ("退出", lambda: setattr(self, "running", False), True),
        ]
        mouse = self._logical_mouse(pygame.mouse.get_pos())
        for index, (label, action, enabled) in enumerate(labels):
            rect = pygame.Rect(98, 220 + index * 52, 300, 40)
            hovered = rect.collidepoint(mouse) and enabled
            color = (
                self.theme.primary_hover
                if hovered
                else self.theme.primary
                if index == 0 and enabled
                else self.theme.surface_high
            )
            pygame.draw.rect(self.canvas, color, rect, border_radius=6)
            if index > 0:
                pygame.draw.rect(self.canvas, self.theme.border, rect, width=1, border_radius=6)
            text_color = self.theme.ink if enabled else self.theme.muted
            self._blit_text(label, rect.x + 16, rect.centery, 16, text_color, True, center_y=True)
            self.buttons.append(Button(rect, label, action, enabled))
        self._blit_text(
            "默认：500 对 500 · 本地公平 AI · 约 10 分钟", 98, 554, 14, self.theme.muted
        )
        self._blit_text(
            "离线可完整游玩 · DeepSeek 与本地语音均为可选能力", 98, 580, 14, self.theme.muted
        )
        panel = pygame.Rect(760, 80, 420, 550)
        pygame.draw.rect(self.canvas, self.theme.surface, panel, border_radius=10)
        self._blit_text("战前简报", 792, 116, 22, self.theme.ink, True)
        briefing = [
            "胜利条件：击杀敌方将领",
            "将领由四名护卫优先承伤",
            "侦察兵开图并发现未暴露刺客",
            "工兵可以架桥、开路、造船与建塔",
            "敌我双方拥有完全独立的战争迷雾",
            "同一模拟帧双方将领阵亡时判为平局",
        ]
        for index, line in enumerate(briefing):
            pygame.draw.circle(self.canvas, self.theme.ally, (796, 174 + index * 52), 4)
            self._blit_text(line, 812, 174 + index * 52, 15, self.theme.ink, center_y=True)

    def _draw_setup(self) -> None:
        self._blit_text("对局配置", 100, 86, 30, self.theme.ink, True)
        self._blit_text(
            "地图和兵力在开局后写入存档与回放。随机地图可由种子精确复现。",
            102,
            132,
            15,
            self.theme.muted,
        )
        panel = pygame.Rect(100, 170, 1080, 410)
        pygame.draw.rect(self.canvas, self.theme.surface, panel, border_radius=10)
        rows = [
            ("每方兵力", str(self.setup_army_size), self._cycle_army_size),
            (
                "地图模式",
                "严格旋转对称" if self.setup_symmetric else "非镜像随机",
                lambda: setattr(self, "setup_symmetric", not self.setup_symmetric),
            ),
            ("地图种子", str(self.setup_seed), self._next_seed),
            (
                "敌方难度",
                {"easy": "简单", "normal": "普通", "hard": "困难"}[self.setup_difficulty],
                self._cycle_difficulty,
            ),
        ]
        mouse = self._logical_mouse(pygame.mouse.get_pos())
        for index, (label, value, action) in enumerate(rows):
            y = 220 + index * 72
            self._blit_text(label, 138, y + 18, 15, self.theme.muted, center_y=True)
            rect = pygame.Rect(270, y, 300, 38)
            pygame.draw.rect(
                self.canvas,
                self.theme.surface_high if not rect.collidepoint(mouse) else self.theme.primary,
                rect,
                border_radius=6,
            )
            pygame.draw.rect(self.canvas, self.theme.border, rect, 1, border_radius=6)
            self._blit_text(
                value, rect.x + 14, rect.centery, 15, self.theme.ink, True, center_y=True
            )
            self.buttons.append(Button(rect, label, action))
        self._blit_text("兵种分配", 640, 198, 18, self.theme.ink, True)
        names = {
            "infantry": "步兵",
            "scout": "侦察兵",
            "engineer": "工兵",
            "assassin": "刺客",
            "recruit": "保留初始兵",
        }
        for index, (kind, label) in enumerate(names.items()):
            y = 236 + index * 52
            self._blit_text(label, 640, y + 16, 14, self.theme.muted, center_y=True)
            self._blit_text(
                str(self.setup_composition[kind]),
                932,
                y + 16,
                15,
                self.theme.ink,
                True,
                center=True,
            )
            if kind != "recruit":
                minus = pygame.Rect(850, y, 34, 32)
                plus = pygame.Rect(980, y, 34, 32)
                for rect, delta, glyph in ((minus, -5, "−"), (plus, 5, "+")):
                    pygame.draw.rect(self.canvas, self.theme.surface_high, rect, border_radius=5)
                    pygame.draw.rect(self.canvas, self.theme.border, rect, 1, border_radius=5)
                    self._blit_text(
                        glyph, rect.centerx, rect.centery, 16, self.theme.ink, center=True
                    )
                    self.buttons.append(
                        Button(
                            rect,
                            f"{label}{glyph}",
                            partial(self._adjust_composition, kind, delta),
                        )
                    )
        diff_label = {"easy": "简单", "normal": "普通", "hard": "困难"}[self.setup_difficulty]
        inspire = "可使用鼓舞" if self.setup_difficulty == "hard" else "不使用鼓舞"
        self._blit_text(f"敌方 AI：{diff_label} · 相同兵力 · {inspire}", 138, 520, 14, self.theme.muted)
        self._draw_action_button(
            pygame.Rect(780, 610, 190, 42), "返回", lambda: setattr(self, "scene", "menu")
        )
        self._draw_action_button(pygame.Rect(990, 610, 190, 42), "部署部队", self.new_battle, True)

    def _cycle_army_size(self) -> None:
        values = (100, 250, 500, 750)
        self.setup_army_size = values[(values.index(self.setup_army_size) + 1) % len(values)]
        self._reset_composition()

    def _cycle_difficulty(self) -> None:
        values = ["easy", "normal", "hard"]
        idx = values.index(self.setup_difficulty)
        self.setup_difficulty = values[(idx + 1) % len(values)]

    def _reset_composition(self) -> None:
        remaining = self.setup_army_size - 5
        infantry = round(remaining * 280 / 495)
        scouts = round(remaining * 60 / 495)
        engineers = round(remaining * 50 / 495)
        assassins = round(remaining * 30 / 495)
        self.setup_composition = {
            "infantry": infantry,
            "scout": scouts,
            "engineer": engineers,
            "assassin": assassins,
            "recruit": remaining - infantry - scouts - engineers - assassins,
        }

    def _adjust_composition(self, kind: str, delta: int) -> None:
        if kind == "recruit":
            return
        amount = (
            min(5, self.setup_composition[kind])
            if delta < 0
            else min(5, self.setup_composition["recruit"])
        )
        if not amount:
            return
        self.setup_composition[kind] += amount if delta > 0 else -amount
        self.setup_composition["recruit"] += -amount if delta > 0 else amount

    def _next_seed(self) -> None:
        self.setup_seed = (self.setup_seed * 1664525 + 1013904223) & 0x7FFFFFFF

    def _draw_settings(self) -> None:
        self._blit_text("设置与辅助", 100, 86, 30, self.theme.ink, True)
        self._blit_text(
            "高对比与非仅颜色编码始终启用；以下偏好可在运行中调整。",
            102,
            132,
            15,
            self.theme.muted,
        )
        panel = pygame.Rect(100, 190, 1080, 390)
        pygame.draw.rect(self.canvas, self.theme.surface, panel, border_radius=10)
        rows = [
            (
                "界面缩放",
                f"{round(self.fonts.scale * 100)}%",
                self._cycle_ui_scale,
            ),
            (
                "减弱动态",
                "开启" if self.reduced_motion else "关闭",
                self._toggle_reduced_motion,
            ),
            (
                "主音量",
                f"{round(self.settings.master_volume * 100)}%",
                self._cycle_volume,
            ),
            (
                "窗口模式",
                "全屏" if self.fullscreen else "窗口化",
                self._toggle_fullscreen,
            ),
            (
                "在线命令",
                (
                    "环境变量密钥已就绪"
                    if self.online_enabled and self.deepseek.available
                    else "已禁用或未设置密钥 · 离线可玩"
                ),
                self._toggle_online,
            ),
        ]
        mouse = self._logical_mouse(pygame.mouse.get_pos())
        for index, (label, value, action) in enumerate(rows):
            y = 226 + index * 66
            self._blit_text(label, 138, y + 18, 15, self.theme.muted, center_y=True)
            rect = pygame.Rect(540, y, 540, 38)
            pygame.draw.rect(
                self.canvas,
                self.theme.surface_high if not rect.collidepoint(mouse) else self.theme.primary,
                rect,
                border_radius=6,
            )
            pygame.draw.rect(self.canvas, self.theme.border, rect, 1, border_radius=6)
            self._blit_text(
                value, rect.x + 14, rect.centery, 15, self.theme.ink, True, center_y=True
            )
            enabled = index < 4 or (index == 4 and self.deepseek.available)
            self.buttons.append(Button(rect, label, action, enabled))
        self._draw_action_button(
            pygame.Rect(990, 610, 190, 42), "返回主菜单", lambda: setattr(self, "scene", "menu")
        )

    def _open_tutorial(self) -> None:
        self.tutorial_page = 0
        self.scene = "tutorial"

    def _draw_tutorial(self) -> None:
        pages = [
            (
                "01 · 观察战场",
                [
                    "WASD 或按住中键拖动摄像机，滚轮缩放。",
                    "黑色区域尚未探索；灰暗区域只保留历史地形。",
                    "敌军一旦离开当前视野就会消失，小地图遵守同一情报边界。",
                ],
            ),
            (
                "02 · 选择与编队",
                [
                    "左键点选或拖动框选部队，右键下达移动命令。",
                    "按住 A 再右键为攻击移动；Shift 追加任务。",
                    "Ctrl+数字保存编队，数字键重新选择；方向键直接移动将领。",
                ],
            ),
            (
                "03 · 侦察、征兵与工程",
                [
                    "侦察兵速度快、视野广，并能发现未暴露刺客。",
                    "接近无敌军争夺的村庄后，可用文字命令征召人口。",
                    "选择足量工兵后，可架桥、造船、开路或建塔；F1 查看句式。",
                ],
            ),
            (
                "04 · 战术与胜利",
                [
                    "全速前进消耗 30 耐力；冲锋消耗 40 耐力并强化首次接战。",
                    "部队离将领越远，战术增益越弱；个体冷却为 20 秒。",
                    "四名护卫会优先承受攻击。击杀敌方将领即可获胜。",
                ],
            ),
            (
                "05 · 存档与离线保障",
                [
                    "F5 快速保存，F9 载入；Space 暂停，- / + 调整速度。",
                    "DeepSeek、语音和网络均为可选能力，离线输入始终保留。",
                    "服务失败时按 R 重试、O 离线继续、Q 创建临时存档并退出。",
                ],
            ),
        ]
        title, lines = pages[self.tutorial_page]
        self._blit_text("新手教程", 100, 82, 30, self.theme.ink, True)
        self._blit_text(f"{self.tutorial_page + 1} / {len(pages)}", 1080, 96, 14, self.theme.muted)
        panel = pygame.Rect(100, 160, 1080, 410)
        pygame.draw.rect(self.canvas, self.theme.surface, panel, border_radius=10)
        pygame.draw.rect(self.canvas, self.theme.border, panel, 1, border_radius=10)
        self._blit_text(title, 146, 208, 22, self.theme.ally, True)
        for index, line in enumerate(lines):
            y = 282 + index * 76
            pygame.draw.circle(self.canvas, self.theme.warning, (152, y + 8), 5)
            self._blit_text(line, 176, y, 16, self.theme.ink)
        if self.tutorial_page > 0:
            self._draw_action_button(
                pygame.Rect(760, 610, 130, 42),
                "上一步",
                lambda: setattr(self, "tutorial_page", self.tutorial_page - 1),
            )
        next_label = "完成" if self.tutorial_page == len(pages) - 1 else "下一步"
        next_action = (
            (lambda: setattr(self, "scene", "menu"))
            if self.tutorial_page == len(pages) - 1
            else (lambda: setattr(self, "tutorial_page", self.tutorial_page + 1))
        )
        self._draw_action_button(pygame.Rect(910, 610, 130, 42), next_label, next_action, True)
        self._draw_action_button(
            pygame.Rect(1050, 610, 130, 42), "退出教程", lambda: setattr(self, "scene", "menu")
        )

    def _cycle_ui_scale(self) -> None:
        values = (1.0, 1.25, 1.5, 2.0)
        current = min(range(len(values)), key=lambda index: abs(values[index] - self.fonts.scale))
        self.fonts.scale = values[(current + 1) % len(values)]
        self.fonts.cache.clear()
        self._save_settings()

    def _toggle_reduced_motion(self) -> None:
        self.reduced_motion = not self.reduced_motion
        self._save_settings()

    def _cycle_volume(self) -> None:
        values = (0.0, 0.25, 0.5, 0.75, 1.0)
        current = min(
            range(len(values)),
            key=lambda index: abs(values[index] - self.settings.master_volume),
        )
        self.settings.master_volume = values[(current + 1) % len(values)]
        self.sounds.set_volume(self.settings.master_volume)
        self.sounds.play("command")
        self._save_settings()

    def _toggle_online(self) -> None:
        if self.deepseek.available:
            self.online_enabled = not self.online_enabled

    def _toggle_fullscreen(self) -> None:
        self.fullscreen = not self.fullscreen
        flags = pygame.FULLSCREEN if self.fullscreen else pygame.RESIZABLE
        self.screen = pygame.display.set_mode(LOGICAL_SIZE, flags)
        self._save_settings()

    def _save_settings(self) -> None:
        self.settings = SettingsV1(
            ui_scale=self.fonts.scale,
            reduced_motion=self.reduced_motion,
            fullscreen=self.fullscreen,
            master_volume=self.settings.master_volume,
            group_names=self.settings.group_names,
        )
        try:
            self.settings_manager.save(self.settings)
        except OSError:
            pass

    def _draw_action_button(
        self, rect: pygame.Rect, label: str, action: Callable[[], None], primary: bool = False
    ) -> None:
        hovered = rect.collidepoint(self._logical_mouse(pygame.mouse.get_pos()))
        color = (
            self.theme.primary_hover
            if hovered and primary
            else self.theme.primary
            if primary
            else self.theme.surface_high
        )
        pygame.draw.rect(self.canvas, color, rect, border_radius=6)
        if not primary:
            pygame.draw.rect(self.canvas, self.theme.border, rect, 1, border_radius=6)
        self._blit_text(label, rect.centerx, rect.centery, 15, self.theme.ink, True, center=True)
        self.buttons.append(Button(rect, label, action))

    def _draw_battle(self) -> None:
        if self.world is None or self.camera is None:
            return
        pygame.draw.rect(self.canvas, self.theme.fog_unknown, BATTLE_RECT)
        self._draw_terrain()
        self._draw_facilities_and_villages()
        self._draw_units()
        self._draw_hud()
        if self.drag_start and pygame.mouse.get_pressed()[0]:
            end = self._logical_mouse(pygame.mouse.get_pos())
            rect = pygame.Rect(
                min(self.drag_start[0], end[0]),
                min(self.drag_start[1], end[1]),
                abs(end[0] - self.drag_start[0]),
                abs(end[1] - self.drag_start[1]),
            )
            pygame.draw.rect(self.canvas, self.theme.ally, rect, 1)
        if self.help_open:
            self._draw_help()
        elif self.paused and self.scene == "battle":
            self._draw_pause()

    def _draw_terrain(self) -> None:
        assert self.world is not None and self.camera is not None
        fog = self.world._perception
        view = self.view_faction
        if fog is None and view is not None:
            self.world.observation(view)
            fog = self.world._perception
        if view is not None:
            assert fog is not None
        colors = [
            self.theme.terrain_plain,
            self.theme.terrain_grass,
            self.theme.terrain_forest,
            self.theme.terrain_swamp,
            self.theme.terrain_river,
            self.theme.terrain_road,
        ]
        tile = self.world.map.tile_size
        left, top = self.camera.screen_to_world(BATTLE_RECT.left, BATTLE_RECT.top)
        right, bottom = self.camera.screen_to_world(BATTLE_RECT.right, BATTLE_RECT.bottom)
        c0, c1 = max(0, int(left // tile)), min(self.world.map.cols, int(right // tile) + 2)
        r0, r1 = max(0, int(top // tile)), min(self.world.map.rows, int(bottom // tile) + 2)
        size = max(1, math.ceil(tile * self.camera.zoom) + 1)
        for row in range(r0, r1):
            for col in range(c0, c1):
                sx, sy = self.camera.world_to_screen(col * tile, row * tile)
                if view is None:
                    color = colors[int(self.world.map.terrain[row, col])]
                else:
                    assert fog is not None
                    if not fog.explored[int(view), row, col]:
                        color = self.theme.fog_unknown
                    else:
                        color = colors[int(self.world.map.terrain[row, col])]
                        if not fog.visible[int(view), row, col]:
                            color = (
                                max(2, color[0] // 3),
                                max(2, color[1] // 3),
                                max(2, color[2] // 3),
                            )
                pygame.draw.rect(self.canvas, color, (sx, sy, size, size))

    def _draw_facilities_and_villages(self) -> None:
        assert self.world is not None and self.camera is not None
        fog = self.world._perception
        view = self.view_faction
        observation = self.world.observation(view) if view is not None else None
        known_villages = (
            {int(item["village_id"]): item for item in observation.known_villages}
            if observation
            else {}
        )
        known_facility_ids = (
            {int(item["facility_id"]) for item in observation.known_facilities}
            if observation
            else set()
        )
        tile = self.world.map.tile_size
        for village in self.world.map.villages:
            col, row = int(village.x // tile), int(village.y // tile)
            if view is not None and (fog is None or not fog.explored[int(view), row, col]):
                continue
            sx, sy = self.camera.world_to_screen(village.x, village.y)
            radius = max(3, round(village.radius * self.camera.zoom))
            pygame.draw.circle(self.canvas, self.theme.warning, (sx, sy), radius, 1)
            if self.camera.zoom > 0.35:
                population = (
                    village.population
                    if view is None
                    else int(known_villages.get(village.village_id, {}).get("population", 0))
                )
                self._blit_text(str(population), sx, sy - 4, 11, self.theme.warning, center=True)
        for item in self.world.facilities:
            if view is not None and item.facility_id not in known_facility_ids:
                continue
            sx, sy = self.camera.world_to_screen(item.x, item.y)
            if not BATTLE_RECT.collidepoint(sx, sy):
                continue
            color = self.theme.ally if item.faction == 0 else self.theme.enemy
            if item.destroyed:
                pygame.draw.line(self.canvas, color, (sx - 6, sy - 6), (sx + 6, sy + 6), 2)
                pygame.draw.line(self.canvas, color, (sx + 6, sy - 6), (sx - 6, sy + 6), 2)
            else:
                pygame.draw.rect(
                    self.canvas, color, (sx - 5, sy - 5, 10, 10), 0 if item.complete else 1
                )
            if not item.complete and not item.destroyed:
                width = round(20 * item.progress / max(item.required_work, 1))
                pygame.draw.rect(self.canvas, self.theme.background, (sx - 10, sy - 11, 20, 3))
                pygame.draw.rect(self.canvas, color, (sx - 10, sy - 11, width, 3))
            else:
                occupants = int(
                    np.sum(
                        self.world.units.facility_id[: self.world.units.count] == item.facility_id
                    )
                )
                if occupants:
                    self._blit_text(str(occupants), sx, sy - 14, 10, color, True, center=True)

    def _draw_units(self) -> None:
        assert self.world is not None and self.camera is not None
        view = self.view_faction
        observation = self.world.observation(view) if view is not None else None
        visible_enemy_ids = (
            {unit.entity_id for unit in observation.visible_enemies} if observation else set()
        )
        concealed_facilities = {
            item.facility_id
            for item in self.world.facilities
            if item.complete and item.kind in {"tower", "boat"}
        }
        # Per-unit-kind colors
        kind_colors = {
            UnitKind.INFANTRY: (220, 60, 60),     # red
            UnitKind.SCOUT: (70, 130, 230),       # blue
            UnitKind.ENGINEER: (230, 200, 50),    # yellow
            UnitKind.ASSASSIN: (170, 80, 220),    # purple
        }
        for index in self.world.units.active():
            if int(self.world.units.facility_id[index]) in concealed_facilities:
                continue
            entity_id = int(self.world.units.entity_id[index])
            faction = int(self.world.units.faction[index])
            if view is not None and faction != int(view) and entity_id not in visible_enemy_ids:
                continue
            alpha = float(np.clip(self.accumulator / self.world.dt, 0, 1))
            x = (
                self.world.units.previous_x[index] * (1 - alpha) + self.world.units.x[index] * alpha
            ) / self.world.subpixels
            y = (
                self.world.units.previous_y[index] * (1 - alpha) + self.world.units.y[index] * alpha
            ) / self.world.subpixels
            sx, sy = self.camera.world_to_screen(x, y)
            if not BATTLE_RECT.collidepoint(sx, sy):
                continue
            kind = UnitKind(int(self.world.units.kind[index]))
            radius = max(2, min(7, round(self.world.units.radius[index] * self.camera.zoom + 1)))
            color = kind_colors.get(kind, self.theme.ally if faction == 0 else self.theme.enemy)
            if kind == UnitKind.COMMANDER:
                color = self.theme.warning
            # Enemy black halo
            if faction == 1:
                pygame.draw.circle(self.canvas, (0, 0, 0), (sx, sy), radius + 3)
            if kind == UnitKind.COMMANDER:
                pygame.draw.circle(self.canvas, self.theme.warning, (sx, sy), radius + 4, 2)
            if faction == 0:
                pygame.draw.circle(self.canvas, color, (sx, sy), radius)
                pygame.draw.line(
                    self.canvas,
                    self.theme.background,
                    (sx, sy - radius),
                    (sx + radius, sy - radius),
                    1,
                )
            else:
                pygame.draw.polygon(
                    self.canvas,
                    color,
                    [
                        (sx, sy - radius - 1),
                        (sx + radius + 1, sy),
                        (sx, sy + radius + 1),
                        (sx - radius - 1, sy),
                    ],
                )
            if entity_id in self.selected_ids:
                pygame.draw.circle(self.canvas, self.theme.ink, (sx, sy), radius + 4, 1)
            if (
                self.world.units.hp[index] < self.world.units.max_hp[index]
                and self.camera.zoom > 0.4
            ):
                ratio = float(self.world.units.hp[index] / self.world.units.max_hp[index])
                pygame.draw.rect(
                    self.canvas, self.theme.background, (sx - 7, sy - radius - 5, 14, 2)
                )
                pygame.draw.rect(
                    self.canvas,
                    self.theme.success if ratio > 0.4 else self.theme.enemy,
                    (sx - 7, sy - radius - 5, round(14 * ratio), 2),
                )

    def _draw_hud(self) -> None:
        assert self.world is not None
        pygame.draw.rect(self.canvas, self.theme.surface, (0, 0, 1280, 40))
        pygame.draw.rect(self.canvas, self.theme.surface, (960, 40, 320, 680))
        pygame.draw.rect(self.canvas, self.theme.surface, (0, 588, 960, 132))
        pygame.draw.line(self.canvas, self.theme.border, (960, 40), (960, 720))
        pygame.draw.line(self.canvas, self.theme.border, (0, 588), (960, 588))
        minutes, seconds = divmod(self.world.tick // self.world.balance.world.simulation_hz, 60)
        self._blit_text(
            f"{minutes:02d}:{seconds:02d}", 18, 20, 14, self.theme.ink, True, center_y=True
        )
        status = "暂停" if self.paused else f"{self.speed:g}×"
        self._blit_text(
            status,
            92,
            20,
            14,
            self.theme.warning if self.paused else self.theme.ink,
            True,
            center_y=True,
        )
        online = "DeepSeek 就绪" if self.online_enabled and self.deepseek.available else "离线指令"
        self._blit_text(
            online,
            154,
            20,
            13,
            (
                self.theme.success
                if self.online_enabled and self.deepseek.available
                else self.theme.muted
            ),
            center_y=True,
        )
        p95 = float(np.percentile(self.tick_timings, 95)) if self.tick_timings else 0.0
        self._blit_text(
            f"FPS {self.clock.get_fps():.0f}  SIM P95 {p95:.1f}ms",
            310,
            20,
            12,
            self.theme.muted if p95 <= 10 else self.theme.warning,
            center_y=True,
        )
        shortcuts = (
            "F2/F3/F4 视角   ←/→ 跳转   F11 全屏"
            if self.scene == "replay"
            else "F1 指令书   F5 保存   F9 载入   F11 全屏"
        )
        self._blit_text(shortcuts, 640, 20, 12, self.theme.muted, center_y=True)

        self._blit_text("战场态势", 980, 64, 18, self.theme.ink, True)
        view = self.view_faction
        side = view if view is not None else Faction.PLAYER
        player_alive = len(self.world.units.active(side))
        known_enemy = (
            len(self.world.observation(side).visible_enemies)
            if view is not None
            else len(self.world.units.active(Faction.ENEMY))
        )
        side_label = "我军" if side == Faction.PLAYER else "敌军"
        self._blit_text(f"{side_label}存活  {player_alive}", 980, 100, 14, self.theme.ally)
        self._blit_text(f"当前可见敌军  {known_enemy}", 980, 126, 14, self.theme.enemy)
        commander = self.world.commander_index(side)
        guards = np.sum(
            self.world.units.alive[: self.world.units.count]
            & (self.world.units.faction[: self.world.units.count] == int(side))
            & (self.world.units.kind[: self.world.units.count] == int(UnitKind.GUARD))
        )
        hp = 0 if commander is None else round(float(self.world.units.hp[commander]))
        self._blit_text(f"将领生命  {hp}   护卫  {guards}/4", 980, 152, 14, self.theme.warning)
        if self.scene == "replay":
            view_label = (
                "玩家" if view == Faction.PLAYER else "敌方" if view == Faction.ENEMY else "全局"
            )
            self._blit_text(f"回放视角：{view_label}", 980, 178, 12, self.theme.muted)

        self._blit_text("指挥记录", 980, 212, 17, self.theme.ink, True)
        history_lines = 7 if self.ambiguity_candidates else 12
        for line, (message, color) in enumerate(self.messages[-history_lines:]):
            clipped = message if len(message) <= 20 else message[:19] + "…"
            self._blit_text(clipped, 980, 242 + line * 25, 13, color)
        if self.ambiguity_candidates:
            self._blit_text("选择解释", 980, 430, 14, self.theme.warning, True)
            for index, command in enumerate(self.ambiguity_candidates):
                payload = command.payload
                detail: str = payload.kind
                if isinstance(payload, MovePayloadV1):
                    detail = (
                        f"{payload.kind} · 组{payload.selection.group_id}"
                        if payload.selection.group_id
                        else f"{payload.kind} · ({payload.target.x:.0f},{payload.target.y:.0f})"
                    )
                self._blit_text(f"{index + 1}  {detail}", 980, 458 + index * 24, 12, self.theme.ink)

        self._draw_minimap()

        if self.scene == "replay":
            self._draw_replay_controls()
            return

        selected_indices = [self.world.units.index_of(entity_id) for entity_id in self.selected_ids]
        selected = [
            index
            for index in selected_indices
            if index is not None and self.world.units.alive[index]
        ]
        self._blit_text(f"当前选择  {len(selected)}", 20, 612, 17, self.theme.ink, True)
        if selected:
            kinds: dict[str, int] = {}
            for index in selected:
                name = UnitKind(int(self.world.units.kind[index])).name.lower()
                kinds[name] = kinds.get(name, 0) + 1
            average_hp = (
                np.mean(
                    [
                        self.world.units.hp[index] / self.world.units.max_hp[index]
                        for index in selected
                    ]
                )
                * 100
            )
            average_stamina = np.mean([self.world.units.stamina[index] for index in selected])
            summary = "  ".join(f"{name}:{count}" for name, count in kinds.items())
            self._blit_text(summary, 20, 646, 14, self.theme.muted)
            self._blit_text(
                f"平均生命 {average_hp:.0f}%   平均耐力 {average_stamina:.0f}",
                20,
                676,
                14,
                self.theme.ink,
            )
        else:
            self._blit_text("左键点选或拖动框选；数字键选择编队。", 20, 650, 14, self.theme.muted)
        if self.typing:
            box = pygame.Rect(340, 618, 590, 42)
            pygame.draw.rect(self.canvas, self.theme.background, box, border_radius=6)
            pygame.draw.rect(self.canvas, self.theme.primary, box, 2, border_radius=6)
            self._blit_text(
                self.command_input + self.composition + "│", box.x + 12, box.centery, 15, self.theme.ink, center_y=True
            )
            self._blit_text("Enter 执行 · Esc 取消", 340, 674, 12, self.theme.muted)
        else:
            self._blit_text("Enter 输入文字军令 · 按住 V 语音下令", 340, 638, 15, self.theme.ink)
            self._blit_text(
                "A+右键攻击移动 · Ctrl+数字编组 · Shift 追加", 340, 674, 12, self.theme.muted
            )

    def _draw_minimap(self) -> None:
        assert self.world is not None and self.camera is not None
        rect = pygame.Rect(980, 548, 280, 152)
        terrain_colors = np.asarray(
            [
                self.theme.terrain_plain,
                self.theme.terrain_grass,
                self.theme.terrain_forest,
                self.theme.terrain_swamp,
                self.theme.terrain_river,
                self.theme.terrain_road,
            ],
            dtype=np.uint8,
        )
        pixels = terrain_colors[self.world.map.terrain]
        view = self.view_faction
        fog = self.world._perception
        if view is not None and fog is not None:
            explored = fog.explored[int(view)]
            visible = fog.visible[int(view)]
            pixels = pixels.copy()
            pixels[~explored] = self.theme.fog_unknown
            pixels[explored & ~visible] //= 3
        surface = pygame.surfarray.make_surface(np.transpose(pixels, (1, 0, 2)))
        self.canvas.blit(pygame.transform.scale(surface, rect.size), rect)
        observation = self.world.observation(view) if view is not None else None
        visible_ids = (
            {unit.entity_id for unit in observation.visible_enemies} if observation else set()
        )
        for index in self.world.units.active():
            faction = int(self.world.units.faction[index])
            entity_id = int(self.world.units.entity_id[index])
            if view is not None and faction != int(view) and entity_id not in visible_ids:
                continue
            x = self.world.units.x[index] / self.world.subpixels
            y = self.world.units.y[index] / self.world.subpixels
            sx = rect.x + round(x / self.world.map.width * rect.width)
            sy = rect.y + round(y / self.world.map.height * rect.height)
            position = (
                int(np.clip(sx, rect.left, rect.right - 1)),
                int(np.clip(sy, rect.top, rect.bottom - 1)),
            )
            self.canvas.set_at(position, self.theme.ally if faction == 0 else self.theme.enemy)
        world_left, world_top = self.camera.screen_to_world(BATTLE_RECT.left, BATTLE_RECT.top)
        world_right, world_bottom = self.camera.screen_to_world(
            BATTLE_RECT.right, BATTLE_RECT.bottom
        )
        camera_rect = pygame.Rect(
            rect.x + round(world_left / self.world.map.width * rect.width),
            rect.y + round(world_top / self.world.map.height * rect.height),
            max(2, round((world_right - world_left) / self.world.map.width * rect.width)),
            max(2, round((world_bottom - world_top) / self.world.map.height * rect.height)),
        )
        pygame.draw.rect(self.canvas, self.theme.ink, camera_rect.clip(rect), 1)
        pygame.draw.rect(self.canvas, self.theme.border, rect, 1)

    def _draw_replay_controls(self) -> None:
        assert self.world is not None and self.replay_player is not None
        self._blit_text("回放时间轴", 22, 612, 15, self.theme.ink, True)
        track = pygame.Rect(40, 642, 880, 8)
        pygame.draw.rect(self.canvas, self.theme.surface_high, track, border_radius=4)
        final_tick = max(1, self.replay_player.final_tick)
        progress = float(np.clip(self.world.tick / final_tick, 0, 1))
        pygame.draw.rect(
            self.canvas,
            self.theme.primary,
            (track.x, track.y, max(2, round(track.width * progress)), track.height),
            border_radius=4,
        )
        for checkpoint in self.replay_player.replay.checkpoints:
            x = track.x + round(checkpoint.tick / final_tick * track.width)
            pygame.draw.line(self.canvas, self.theme.muted, (x, 637), (x, 655), 1)
        cursor_x = track.x + round(track.width * progress)
        pygame.draw.circle(self.canvas, self.theme.ink, (cursor_x, track.centery), 7)
        current = self.world.tick / self.world.balance.world.simulation_hz
        duration = final_tick / self.world.balance.world.simulation_hz
        self._blit_text(
            f"{current:06.1f}s / {duration:06.1f}s   {self.speed:g}×",
            40,
            674,
            13,
            self.theme.muted,
        )
        commands = [
            command
            for command in self.replay_player.replay.commands
            if command.issued_tick <= self.world.tick
        ]
        latest_command = commands[-1] if commands else None
        if latest_command is not None:
            self._blit_text(
                f"最近命令：{latest_command.source} · {latest_command.payload.kind}",
                330,
                674,
                13,
                self.theme.ink,
            )
        events = [
            event for event in self.replay_player.replay.events if event.tick <= self.world.tick
        ]
        if events:
            self._blit_text(f"关键事件：{events[-1].kind}", 650, 674, 13, self.theme.warning)
        validity = self.replay_player.checksum_valid
        if validity is not None:
            self._blit_text(
                "校验通过" if validity else "校验失败",
                850,
                612,
                12,
                self.theme.success if validity else self.theme.enemy,
            )

    def _draw_help(self) -> None:
        overlay = pygame.Surface(LOGICAL_SIZE, pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 190))
        self.canvas.blit(overlay, (0, 0))
        panel = pygame.Rect(80, 50, 1120, 620)
        pygame.draw.rect(self.canvas, self.theme.surface, panel, border_radius=10)
        pygame.draw.rect(self.canvas, self.theme.border, panel, 1, border_radius=10)
        self._blit_text("指令书", panel.centerx, 78, 24, self.theme.ink, True, center=True)

        left = [
            "选择与移动",
            "左键点选或框选，右键移动。A+右键攻击移动。",
            "Ctrl+1–9 保存编组，1–9 选中编组，Shift 追加选择。",
            "方向键直接移动将领。Space 暂停，-/+ 调整速度。",
            "",
            "文字命令（Enter 输入）",
            "移动：第一侦察队前往北部中央",
            "攻击：第1组攻击敌方将领",
            "搜索：侦察兵搜索东部",
            "编组：20名初始兵编为第6组",
            "命名：第6组命名为河西守军",
            "分化：20名初始兵分化为步兵",
            "征兵：在最近村庄征召20人",
            "集火：第1组集火敌方将领",
            "",
            "设施命令",
            "架桥：工兵队在中央架桥",
            "造船：工兵在西岸造船",
            "开路：工兵砍伐东侧森林",
            "建塔：工兵在村庄建塔",
        ]
        right = [
            "战术命令",
            "全速：第1组全速前进",
            "冲锋：第1组冲锋",
            "突击：第1组突击敌方将领",
            "守卫：第1组守住最近村庄",
            "保护：保护将领（全员围成一圈）",
            "",
            "载具命令",
            "登船：第1组登船",
            "开船：船只前往西岸",
            "下船：第1组下船",
            "入塔：第6组进入防御塔",
            "出塔：第6组离开防御塔",
            "",
            "兵种职责",
            "步兵：正面作战，可入塔获得远程攻击",
            "侦察兵：速度快、视野大，反隐刺客",
            "工兵：建造桥梁/船只/道路/防御塔",
            "刺客：高伤害，暴露后永久可见",
            "将领：阵亡即败，4名护卫拦截攻击",
            "",
            "其他操作",
            "Esc 返回 / 暂停  F1 指令书",
            "F5 保存  F9 载入  V 按住说话",
            "Enter 提交命令  Esc 取消输入",
        ]
        self._draw_lines(left, 108, 112, 490)
        self._draw_lines(right, 620, 112, 490)

    def _draw_lines(self, lines: list[str], x: int, y: int, width: int) -> None:
        cursor = y
        for line in lines:
            if not line:
                cursor += 14
                continue
            heading = len(line) <= 6
            words = self._wrap(line, self.fonts.get(14, heading), width)
            for wrapped in words:
                self._blit_text(
                    wrapped,
                    x,
                    cursor,
                    16 if heading else 14,
                    self.theme.ink if heading else self.theme.muted,
                    heading,
                )
                cursor += 28 if heading else 22
            if heading:
                cursor += 4

    def _draw_pause(self) -> None:
        overlay = pygame.Surface(LOGICAL_SIZE, pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        self.canvas.blit(overlay, (0, 0))
        panel = pygame.Rect(440, 220, 400, 250)
        pygame.draw.rect(self.canvas, self.theme.surface, panel, border_radius=10)
        self._blit_text("战局已暂停", panel.centerx, 264, 24, self.theme.ink, True, center=True)
        self._blit_text("Esc / Space 继续", panel.centerx, 320, 15, self.theme.muted, center=True)
        self._blit_text(
            "F5 保存 · F9 载入 · F11 全屏", panel.centerx, 352, 14, self.theme.muted, center=True
        )
        self._blit_text(
            "断网时可继续使用鼠标、键盘和离线文字命令。",
            panel.centerx,
            408,
            13,
            self.theme.ink,
            center=True,
        )

    def _draw_result(self) -> None:
        assert self.world is not None
        overlay = pygame.Surface(LOGICAL_SIZE, pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 190))
        self.canvas.blit(overlay, (0, 0))
        panel = pygame.Rect(250, 80, 780, 560)
        pygame.draw.rect(self.canvas, self.theme.surface, panel, border_radius=10)
        titles = {
            GameOutcome.PLAYER_WIN: "胜利",
            GameOutcome.ENEMY_WIN: "战败",
            GameOutcome.DRAW: "平局",
        }
        title = titles[self.world.outcome]
        color = (
            self.theme.success
            if self.world.outcome == GameOutcome.PLAYER_WIN
            else self.theme.warning
            if self.world.outcome == GameOutcome.DRAW
            else self.theme.enemy
        )
        self._blit_text(title, panel.centerx, 118, 32, color, True, center=True)

        # Game duration
        duration_s = self.world.tick / 20.0
        minutes = int(duration_s) // 60
        seconds = int(duration_s) % 60
        self._blit_text(
            f"对局时长 {minutes}分{seconds:02d}秒 · {self.world.tick} 帧",
            panel.centerx, 164, 15, self.theme.muted, center=True,
        )

        # Statistics table
        stats = self.world.statistics
        y = 200
        self._blit_text("战      统", panel.centerx, y, 20, self.theme.ink, True, center=True)
        y += 40

        # Header
        self._blit_text("", 310, y, 14, self.theme.muted)
        self._blit_text("我方", 480, y, 14, self.theme.ally, True, center=True)
        self._blit_text("敌方", 620, y, 14, self.theme.enemy, True, center=True)
        y += 30

        # Losses
        self._blit_text("损失", 310, y, 14, self.theme.muted)
        self._blit_text(str(stats["losses"][0]), 480, y, 15, self.theme.ink, True, center=True)
        self._blit_text(str(stats["losses"][1]), 620, y, 15, self.theme.ink, True, center=True)
        y += 28

        # Kills
        self._blit_text("击杀", 310, y, 14, self.theme.muted)
        self._blit_text(str(stats.get("kills", [0, 0])[0]), 480, y, 15, self.theme.ink, True, center=True)
        self._blit_text(str(stats.get("kills", [0, 0])[1]), 620, y, 15, self.theme.ink, True, center=True)
        y += 28

        # Damage dealt
        dmg = stats.get("damage_dealt", [0.0, 0.0])
        self._blit_text("输出伤害", 310, y, 14, self.theme.muted)
        self._blit_text(f"{dmg[0]:.0f}", 480, y, 15, self.theme.ink, True, center=True)
        self._blit_text(f"{dmg[1]:.0f}", 620, y, 15, self.theme.ink, True, center=True)
        y += 28

        # Per-unit-type losses breakdown
        kind_names = {
            "recruit": "初始兵", "infantry": "步兵", "scout": "侦察兵",
            "engineer": "工兵", "assassin": "刺客", "commander": "将领", "guard": "护卫",
        }
        losses_by_kind = stats.get("losses_by_kind", [{}, {}])
        has_breakdown = any(losses_by_kind[0]) or any(losses_by_kind[1])
        if has_breakdown:
            y += 10
            self._blit_text("兵种损失明细", panel.centerx, y, 15, self.theme.ink, True, center=True)
            y += 26
            all_kinds = sorted(set(list(losses_by_kind[0].keys()) + list(losses_by_kind[1].keys())))
            for kind in all_kinds:
                label = kind_names.get(kind, kind)
                p_loss = losses_by_kind[0].get(kind, 0)
                e_loss = losses_by_kind[1].get(kind, 0)
                if p_loss == 0 and e_loss == 0:
                    continue
                self._blit_text(label, 310, y, 13, self.theme.muted)
                self._blit_text(str(p_loss), 480, y, 14, self.theme.ink, center=True)
                self._blit_text(str(e_loss), 620, y, 14, self.theme.ink, center=True)
                y += 22

        # Buttons
        y = max(y + 20, 470)
        mouse = self._logical_mouse(pygame.mouse.get_pos())
        options = [
            ("再来一局", self.new_battle),
            ("返回主菜单", lambda: setattr(self, "scene", "menu")),
        ]
        for index, (label, action) in enumerate(options):
            rect = pygame.Rect(420, y + index * 54, 340, 42)
            pygame.draw.rect(
                self.canvas,
                self.theme.primary_hover
                if rect.collidepoint(mouse)
                else self.theme.primary
                if index == 0
                else self.theme.surface_high,
                rect,
                border_radius=6,
            )
            if index:
                pygame.draw.rect(self.canvas, self.theme.border, rect, 1, border_radius=6)
            self._blit_text(
                label, rect.centerx, rect.centery, 16, self.theme.ink, True, center=True
            )
            self.buttons.append(Button(rect, label, action))

    def _blit_text(
        self,
        text: str,
        x: int,
        y: int,
        size: int,
        color: tuple[int, int, int],
        bold: bool = False,
        center: bool = False,
        center_y: bool = False,
    ) -> None:
        surface = self.fonts.get(size, bold).render(str(text), True, color)
        rect = surface.get_rect()
        if center:
            rect.center = (x, y)
        elif center_y:
            rect.midleft = (x, y)
        else:
            rect.topleft = (x, y)
        self.canvas.blit(surface, rect)

    @staticmethod
    def _wrap(text: str, font: pygame.font.Font, width: int) -> list[str]:
        result: list[str] = []
        current = ""
        for char in text:
            if current and font.size(current + char)[0] > width:
                result.append(current)
                current = char
            else:
                current += char
        if current:
            result.append(current)
        return result

    def _viewport(self) -> pygame.Rect:
        sw, sh = self.screen.get_size()
        scale = min(sw / LOGICAL_SIZE[0], sh / LOGICAL_SIZE[1])
        width, height = round(LOGICAL_SIZE[0] * scale), round(LOGICAL_SIZE[1] * scale)
        return pygame.Rect((sw - width) // 2, (sh - height) // 2, width, height)

    def _present(self) -> None:
        viewport = self._viewport()
        self.screen.fill(self.theme.background)
        scaled = pygame.transform.smoothscale(self.canvas, viewport.size)
        self.screen.blit(scaled, viewport)
        pygame.display.flip()
