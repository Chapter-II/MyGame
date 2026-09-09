from __future__ import annotations

import logging
import math
import os
import sys
import time
import uuid
from collections import OrderedDict, deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Literal

import numpy as np
import pygame

from mygame.ai import LocalStrategicAI
from mygame.commands import DeepSeekCommandParser, RuleCommandParser
from mygame.constants import BRIDGE_DECK_HALF_WIDTH, ENEMY_CLICK_RADIUS_SQ, UNIT_CLICK_RADIUS_SQ
from mygame.input import MicrophoneRecorder, VoiceUnavailable, WhisperRecognizer
from mygame.maps import Terrain, generate_map
from mygame.persistence import (
    ReplayPlayer,
    ReplayRecorder,
    SaveManager,
    SettingsManager,
    SettingsV1,
)
from mygame.protocols import (
    BuildPayloadV1,
    CommandEnvelopeV1,
    CommandResultV1,
    CommandSource,
    CommandStatus,
    ConvertPayloadV1,
    Faction,
    FocusFirePayloadV1,
    GroupPayloadV1,
    MovePayloadV1,
    PositionV1,
    QueueMode,
    RecruitPayloadV1,
    SelectionV1,
)
from mygame.rendering import ArtBook, Theme
from mygame.simulation import GameOutcome, UnitKind, World

logger = logging.getLogger(__name__)
LOGICAL_SIZE = (1280, 720)
BATTLE_RECT = pygame.Rect(0, 40, 960, 548)
MINIMAP_RECT = pygame.Rect(980, 548, 280, 152)
MIN_WINDOW_SIZE = (480, 270)


@dataclass(slots=True)
class Button:
    rect: pygame.Rect
    label: str
    action: Callable[[], None]
    enabled: bool = True


@dataclass(slots=True)
class TextCommand:
    text: str
    x: int
    y: int
    size: int
    color: tuple[int, int, int]
    bold: bool
    center: bool
    center_y: bool


@dataclass(slots=True)
class BattleEffect:
    kind: str
    x: float
    y: float
    target_x: float
    target_y: float
    duration: float
    age: float = 0.0
    size: int = 28
    faction: int = 0


class FontBook:
    def __init__(self, scale: float = 1.0) -> None:
        self.scale = scale
        self.path = self._find_font()
        self.cache: dict[tuple[int, bool], pygame.font.Font] = {}
        self.render_cache: OrderedDict[
            tuple[str, int, bool, tuple[int, int, int], int], pygame.Surface
        ] = OrderedDict()

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

    def get(self, size: int, bold: bool = False, resolution_scale: float = 1.0) -> pygame.font.Font:
        key = (max(8, round(size * self.scale * resolution_scale)), bold)
        if key not in self.cache:
            font = pygame.font.Font(self.path, key[0])
            font.set_bold(bold)
            self.cache[key] = font
        return self.cache[key]

    def render(
        self,
        text: str,
        size: int,
        color: tuple[int, int, int],
        bold: bool = False,
        resolution_scale: float = 1.0,
    ) -> pygame.Surface:
        scale_key = round(resolution_scale * 1000)
        key = (text, size, bold, color, scale_key)
        cached = self.render_cache.get(key)
        if cached is not None:
            self.render_cache.move_to_end(key)
            return cached
        surface = self.get(size, bold, resolution_scale).render(text, True, color)
        self.render_cache[key] = surface
        if len(self.render_cache) > 768:
            self.render_cache.popitem(last=False)
        return surface


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
                    logger.debug("音效加载失败: %s", name)
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
        self.target_zoom = self.zoom
        self.zoom_anchor_screen: tuple[int, int] = BATTLE_RECT.center
        self.zoom_anchor_world: tuple[float, float] | None = None

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
        if sys.platform == "win32":
            try:
                import ctypes

                ctypes.windll.shcore.SetProcessDpiAwareness(1)
            except Exception:
                logger.debug("DPI 感知设置失败", exc_info=True)
        pygame.init()
        pygame.font.init()
        self.theme = Theme()
        self.settings_manager = SettingsManager()
        self.settings = self.settings_manager.load()
        self.fullscreen = self.settings.fullscreen
        self._windowed_size = self._initial_window_size()
        self.screen = self._set_display_mode(self.fullscreen)
        pygame.display.set_caption("指挥官战术对抗")
        self.canvas = pygame.Surface(LOGICAL_SIZE)
        self.clock = pygame.time.Clock()
        self.fonts = FontBook(float(np.clip(self.settings.ui_scale, 0.9, 1.35)))
        self.art = ArtBook()
        self._text_commands: list[TextCommand] = []
        self._native_text = False
        self.sounds = SoundBook(self.settings.master_volume)
        self.running = True
        self.scene = "menu"
        self.buttons: list[Button] = []
        self.menu_focus = 0
        self.world: World | None = None
        self.camera: Camera | None = None
        self.ai: LocalStrategicAI | None = None
        self.player_ai: LocalStrategicAI | None = None
        self.auto_command = False
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
        self.minimap_dragging = False
        self.context_menu_point: tuple[int, int] | None = None
        self.context_menu_world: tuple[float, float] | None = None
        self.command_input = ""
        self.composition = ""
        self.typing = False
        self.suggestions: list[str] = []
        self.selected_suggestion = 0
        self.messages: list[tuple[str, tuple[int, int, int]]] = []
        self.battle_effects: list[BattleEffect] = []
        self.result_cinematic_time = 0.0
        self.result_cinematic_duration = 1.35
        self.cinematic_target: tuple[float, float] | None = None
        self.paused = False
        self.speed = 1.0
        self.accumulator = 0.0
        self.help_open = False
        self.reduced_motion = self.settings.reduced_motion
        self.last_event_index = 0
        self.stats = {"player_losses": 0, "enemy_losses": 0}
        self.result_sound_played = False
        self.tick_timings: deque[float] = deque(maxlen=240)
        self._terrain_cache: pygame.Surface | None = None
        self._terrain_cache_key: tuple[object, ...] = ()
        self._terrain_world_surface: pygame.Surface | None = None
        self._terrain_world_key: tuple[object, ...] = ()
        self._fogged_world_surface: pygame.Surface | None = None
        self._fogged_world_key: tuple[object, ...] = ()
        self._minimap_cache: pygame.Surface | None = None
        self._minimap_cache_key: tuple[object, ...] = ()
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

    @staticmethod
    def _desktop_size() -> tuple[int, int]:
        try:
            sizes = pygame.display.get_desktop_sizes()
            if sizes and sizes[0][0] > 0 and sizes[0][1] > 0:
                return sizes[0]
        except (AttributeError, pygame.error):
            logger.debug("无法读取桌面分辨率", exc_info=True)
        info = pygame.display.Info()
        if info.current_w > 0 and info.current_h > 0:
            return info.current_w, info.current_h
        return LOGICAL_SIZE

    def _initial_window_size(self) -> tuple[int, int]:
        desktop_w, desktop_h = self._desktop_size()
        scale = min(
            1.0,
            (desktop_w - 64) / LOGICAL_SIZE[0],
            (desktop_h - 96) / LOGICAL_SIZE[1],
        )
        scale = max(0.25, scale)
        return round(LOGICAL_SIZE[0] * scale), round(LOGICAL_SIZE[1] * scale)

    def _set_display_mode(self, fullscreen: bool) -> pygame.Surface:
        if fullscreen:
            try:
                return pygame.display.set_mode(self._desktop_size(), pygame.FULLSCREEN)
            except pygame.error:
                logger.warning("全屏模式不可用，已回退到自适应窗口", exc_info=True)
                self.fullscreen = False
        return pygame.display.set_mode(self._windowed_size, pygame.RESIZABLE)

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
        self._terrain_cache = None
        self._terrain_cache_key = ()
        self._terrain_world_surface = None
        self._terrain_world_key = ()
        self._fogged_world_surface = None
        self._fogged_world_key = ()
        self._minimap_cache = None
        self._minimap_cache_key = ()
        battle_map = generate_map(seed=self.setup_seed, symmetric=self.setup_symmetric)
        self.world = World(
            battle_map=battle_map,
            seed=self.setup_seed,
            army_size=self.setup_army_size,
            army_composition=self.setup_composition,
        )
        self.world.groups[int(Faction.PLAYER)].update(self.settings.group_names)
        self.rule_parser.world_width = self.world.map.width
        self.rule_parser.world_height = self.world.map.height
        self.rule_parser.suggester.world_width = self.world.map.width
        self.rule_parser.suggester.world_height = self.world.map.height
        self.camera = Camera(self.world)
        self.ai = LocalStrategicAI(
            self.world.map.width,
            self.world.map.height,
            difficulty=self.setup_difficulty,
            config=self.world.balance.ai.get(
                self.setup_difficulty, self.world.balance.ai["normal"]
            ),
        )
        self.player_ai = LocalStrategicAI(
            self.world.map.width,
            self.world.map.height,
            difficulty=self.setup_difficulty,
            config=self.world.balance.ai.get(
                self.setup_difficulty, self.world.balance.ai["normal"]
            ),
        )
        self.auto_command = False
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
        self.context_menu_point = None
        self.context_menu_world = None
        self.battle_effects.clear()
        self.result_cinematic_time = 0.0
        self.cinematic_target = None

    def load_battle(self) -> None:
        try:
            self.world = self.save_manager.load()
            self.battle_effects.clear()
            self.result_cinematic_time = 0.0
            self.cinematic_target = None
            self._terrain_world_surface = None
            self._terrain_world_key = ()
            self._fogged_world_surface = None
            self._fogged_world_key = ()
            self._minimap_cache = None
            self.rule_parser.world_width = self.world.map.width
            self.rule_parser.world_height = self.world.map.height
            self.rule_parser.suggester.world_width = self.world.map.width
            self.rule_parser.suggester.world_height = self.world.map.height
            self.camera = Camera(self.world)
            self.ai = LocalStrategicAI(
                self.world.map.width,
                self.world.map.height,
                config=self.world.balance.ai.get("normal", {}),
            )
            self.player_ai = LocalStrategicAI(
                self.world.map.width,
                self.world.map.height,
                config=self.world.balance.ai.get("normal", {}),
            )
            self.auto_command = False
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
            self.battle_effects.clear()
            self.result_cinematic_time = 0.0
            self.cinematic_target = None
            self._terrain_world_surface = None
            self._terrain_world_key = ()
            self._fogged_world_surface = None
            self._fogged_world_key = ()
            self._minimap_cache = None
            self.camera = Camera(self.world)
            self.ai = None
            self.player_ai = None
            self.auto_command = False
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
                self._windowed_size = (
                    max(MIN_WINDOW_SIZE[0], event.w),
                    max(MIN_WINDOW_SIZE[1], event.h),
                )
                self.screen = pygame.display.set_mode(self._windowed_size, pygame.RESIZABLE)
            elif event.type == pygame.WINDOWFOCUSLOST and self.recording:
                self.recording = False
                self._message("窗口失焦，语音录制已取消。", self.theme.warning)
            elif self.scene in {"menu", "setup", "settings", "tutorial"}:
                self._menu_event(event)
            elif self.scene in {"battle", "replay", "result"}:
                self._battle_event(event)

    def _menu_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self.scene = "menu"
            return
        if event.type == pygame.KEYDOWN:
            enabled_buttons = [b for b in self.buttons if b.enabled]
            if not enabled_buttons:
                return
            if event.key in (pygame.K_TAB, pygame.K_DOWN):
                self.menu_focus = (self.menu_focus + 1) % len(enabled_buttons)
                return
            if event.key == pygame.K_UP:
                self.menu_focus = (self.menu_focus - 1) % len(enabled_buttons)
                return
            if event.key in (pygame.K_RETURN, pygame.K_SPACE):
                if 0 <= self.menu_focus < len(enabled_buttons):
                    enabled_buttons[self.menu_focus].action()
                return
        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            point = self._logical_mouse(event.pos)
            for button in self.buttons:
                if button.enabled and button.rect.collidepoint(point):
                    button.action()
                    # Update focus to clicked button
                    enabled_buttons = [b for b in self.buttons if b.enabled]
                    try:
                        self.menu_focus = enabled_buttons.index(button)
                    except ValueError:
                        logger.debug("按钮焦点同步失败")
                    break

    def _battle_event(self, event: pygame.event.Event) -> None:
        if self.world is None or self.camera is None:
            return
        if event.type == pygame.TEXTINPUT and self.typing:
            if len(self.command_input) + len(event.text) <= 120:
                self.command_input += event.text
            self.composition = ""
            self._update_ime_rect()
            self._refresh_suggestions()
            return
        if event.type == pygame.TEXTEDITING and self.typing:
            self.composition = event.text
            return
        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            point = self._logical_mouse(event.pos)
            candidates = self.buttons
            if self.help_open:
                candidates = [button for button in candidates if button.label == "关闭"]
            elif self.paused and self.scene == "battle":
                pause_panel = pygame.Rect(400, 190, 480, 330)
                candidates = [button for button in candidates if pause_panel.contains(button.rect)]
            for button in reversed(candidates):
                if button.enabled and button.rect.collidepoint(point):
                    self.drag_start = None
                    button.action()
                    return
            if self.context_menu_point is not None:
                self.context_menu_point = None
                self.context_menu_world = None
                self.drag_start = None
                return
        if event.type in {
            pygame.MOUSEBUTTONDOWN,
            pygame.MOUSEBUTTONUP,
            pygame.MOUSEMOTION,
            pygame.MOUSEWHEEL,
        } and (self.help_open or (self.paused and self.scene == "battle")):
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
                    self.suggestions = []
                    if hasattr(pygame.key, "stop_text_input"):
                        pygame.key.stop_text_input()
                elif event.key == pygame.K_RETURN:
                    if self.suggestions and self.selected_suggestion < len(self.suggestions):
                        self.command_input = self.suggestions[self.selected_suggestion]
                        self.suggestions = []
                        self.selected_suggestion = 0
                    else:
                        self._submit_text()
                elif event.key == pygame.K_TAB:
                    if self.suggestions and self.selected_suggestion < len(self.suggestions):
                        self.command_input = self.suggestions[self.selected_suggestion]
                        self.suggestions = []
                        self.selected_suggestion = 0
                elif event.key == pygame.K_UP:
                    if self.suggestions:
                        self.selected_suggestion = max(0, self.selected_suggestion - 1)
                elif event.key == pygame.K_DOWN:
                    if self.suggestions:
                        self.selected_suggestion = min(
                            len(self.suggestions) - 1, self.selected_suggestion + 1
                        )
                elif event.key == pygame.K_BACKSPACE:
                    self.command_input = self.command_input[:-1]
                    self._refresh_suggestions()
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
                self._begin_text_input()
            elif event.key == pygame.K_ESCAPE:
                if self.help_open:
                    self.help_open = False
                else:
                    self.paused = not self.paused
            elif event.key == pygame.K_F1:
                self.help_open = not self.help_open
            elif event.key == pygame.K_F5 and self.scene == "battle":
                self._quick_save()
            elif event.key == pygame.K_F9 and self.scene == "battle":
                self.load_battle()
            elif event.key == pygame.K_F11:
                self._toggle_fullscreen()
            elif event.key == pygame.K_SPACE:
                self._toggle_pause()
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
            if event.button == 1 and MINIMAP_RECT.collidepoint(point):
                self.minimap_dragging = True
                self._camera_from_minimap(point)
            elif event.button == 1 and self.context_menu_point is not None:
                return
            elif event.button == 1 and BATTLE_RECT.collidepoint(point):
                self.drag_start = point
            elif event.button == 2 and BATTLE_RECT.collidepoint(point):
                self.drag_start = point
            elif event.button == 3 and BATTLE_RECT.collidepoint(point) and self.scene == "battle":
                self._open_context_menu(point)
        elif event.type == pygame.MOUSEBUTTONUP:
            point = self._logical_mouse(event.pos)
            if event.button == 1 and self.minimap_dragging:
                self._camera_from_minimap(point)
                self.minimap_dragging = False
                self.drag_start = None
            elif (
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
        elif event.type == pygame.MOUSEMOTION:
            point = self._logical_mouse(event.pos)
            if self.minimap_dragging and event.buttons[0]:
                self._camera_from_minimap(point)
            elif event.buttons[1] and self.drag_start:
                viewport = self._viewport()
                dx = event.rel[0] / max(viewport.width / LOGICAL_SIZE[0], 0.01)
                dy = event.rel[1] / max(viewport.height / LOGICAL_SIZE[1], 0.01)
                self.camera.zoom_anchor_world = None
                self.camera.x -= dx / self.camera.zoom
                self.camera.y -= dy / self.camera.zoom
                self.camera.clamp(self.world)
        elif event.type == pygame.MOUSEWHEEL:
            point = self._logical_mouse(pygame.mouse.get_pos())
            amount = float(getattr(event, "precise_y", event.y))
            if MINIMAP_RECT.collidepoint(point):
                self._camera_from_minimap(point)
                self._request_zoom(1.16**amount, BATTLE_RECT.center)
            elif BATTLE_RECT.collidepoint(point):
                self._request_zoom(1.16**amount, point)

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
        self.suggestions = []
        if hasattr(pygame.key, "stop_text_input"):
            pygame.key.stop_text_input()
        # Split compound commands on conjunctions
        import re

        segments = re.split(r"\s*(?:然后|接着|并且|同时|再|并)\s*", text)
        segments = [s.strip() for s in segments if s.strip()]
        if not segments:
            return
        executed = 0
        for segment in segments:
            if not self._execute_single_command(segment):
                break
            executed += 1
        if executed > 1:
            self._message(f"已执行 {executed} 条指令。", self.theme.success)

    def _pick_suggestion(self, index: int) -> None:
        if index >= len(self.suggestions):
            return
        self.command_input = self.suggestions[index]
        self.selected_suggestion = index
        self.suggestions = []
        self._update_ime_rect()

    def _execute_single_command(self, text: str) -> bool:
        """Parse and execute one command segment. Returns False if rejected."""
        if self.world is None:
            return False
        parsed = self.rule_parser.parse(text, self.world.observation(Faction.PLAYER))
        if parsed.status == CommandStatus.PENDING and parsed.candidates:
            if len(parsed.candidates) > 1:
                self.ambiguity_candidates = list(parsed.candidates[:3])
                self._message(parsed.message_zh, self.theme.warning)
                return False
            command = parsed.candidates[0]
            result = self.world.execute(command)
            self._remember_group_name(command, result)
            if result.status == CommandStatus.ACCEPTED:
                self.sounds.play("command")
            self._message(
                result.message_zh,
                self.theme.success if result.status == CommandStatus.ACCEPTED else self.theme.enemy,
            )
            return result.status == CommandStatus.ACCEPTED
        # Try online fallback only for single-segment input
        if self.online_enabled and self.deepseek.available:
            self.pending_retry_text = text
            self.provider_retry_count = 0
            self.pending_text = self.executor.submit(
                self.deepseek.parse, text, self.world.observation(Faction.PLAYER)
            )
            self._message("离线规则未识别，正在请求 DeepSeek……", self.theme.warning)
            return False
        self._message(parsed.message_zh, self.theme.enemy)
        return False

    def _refresh_suggestions(self) -> None:
        if self.world is None:
            self.suggestions = []
            return
        obs = self.world.observation(Faction.PLAYER)
        self.suggestions = self.rule_parser.suggester.suggest(self.command_input, obs)
        self.selected_suggestion = 0

    def _update_ime_rect(self) -> None:
        if not hasattr(pygame.key, "set_text_input_rect"):
            return
        viewport = self._viewport()
        scale = viewport.width / LOGICAL_SIZE[0]
        box = pygame.Rect(342, 636, 484, 38)
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
        self.context_menu_point = None
        self.context_menu_world = None
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

    def _select_kind(self, kind: UnitKind) -> None:
        if self.world is None or self.scene != "battle":
            return
        indices = self.world.resolve_selection(
            SelectionV1(unit_kind=kind.config_name), Faction.PLAYER
        )
        self.selected_ids = {int(self.world.units.entity_id[index]) for index in indices}
        labels = {
            UnitKind.COMMANDER: "将领",
            UnitKind.ENGINEER: "工兵",
            UnitKind.SCOUT: "侦察兵",
            UnitKind.RECRUIT: "预备兵",
        }
        self._message(
            f"已选择全部{labels.get(kind, kind.config_name)}：{len(indices)} 人。", self.theme.ink
        )

    def _select_all_player(self) -> None:
        if self.world is None or self.scene != "battle":
            return
        indices = self.world.units.active(Faction.PLAYER)
        self.selected_ids = {int(self.world.units.entity_id[index]) for index in indices}
        self._message(f"已选择全军：{len(indices)} 人。", self.theme.ink)

    def _move_commander(self, key: int) -> None:
        if self.world is None:
            return
        index = self.world.commander_index(Faction.PLAYER)
        if index is None:
            return
        dx = -64 if key == pygame.K_LEFT else 64 if key == pygame.K_RIGHT else 0
        dy = -64 if key == pygame.K_UP else 64 if key == pygame.K_DOWN else 0
        x = self.world.units.x[index] / self.world.subpixels + dx
        y = self.world.units.y[index] / self.world.subpixels + dy
        self._issue_move(
            (int(self.world.units.entity_id[index]),), x, y, False, False, CommandSource.KEYBOARD
        )

    def _request_zoom(self, factor: float, anchor: tuple[int, int] | None = None) -> None:
        if self.camera is None:
            return
        anchor = anchor or BATTLE_RECT.center
        self.camera.zoom_anchor_screen = anchor
        self.camera.zoom_anchor_world = self.camera.screen_to_world(*anchor)
        base = self.camera.target_zoom
        self.camera.target_zoom = float(np.clip(base * factor, 0.18, 1.5))

    def _update_camera_zoom(self, dt: float) -> None:
        if self.camera is None or self.world is None:
            return
        difference = self.camera.target_zoom - self.camera.zoom
        if abs(difference) < 0.0005:
            self.camera.zoom = self.camera.target_zoom
            self.camera.zoom_anchor_world = None
            return
        amount = 1.0 if self.reduced_motion else 1.0 - math.exp(-14.0 * dt)
        self.camera.zoom += difference * amount
        if self.camera.zoom_anchor_world is not None:
            after_x, after_y = self.camera.screen_to_world(*self.camera.zoom_anchor_screen)
            self.camera.x += self.camera.zoom_anchor_world[0] - after_x
            self.camera.y += self.camera.zoom_anchor_world[1] - after_y
        self.camera.clamp(self.world)

    def _camera_from_minimap(self, point: tuple[int, int]) -> None:
        if self.camera is None or self.world is None:
            return
        ratio_x = float(np.clip((point[0] - MINIMAP_RECT.left) / MINIMAP_RECT.width, 0, 1))
        ratio_y = float(np.clip((point[1] - MINIMAP_RECT.top) / MINIMAP_RECT.height, 0, 1))
        self.camera.x = ratio_x * self.world.map.width
        self.camera.y = ratio_y * self.world.map.height
        self.camera.zoom_anchor_world = None
        self.camera.clamp(self.world)

    def _open_context_menu(self, point: tuple[int, int]) -> None:
        if self.camera is None:
            return
        self.context_menu_point = point
        self.context_menu_world = self.camera.screen_to_world(*point)

    def _known_terrain_at(self, x: float, y: float) -> Terrain | None:
        """Read terrain through the player's observation without exposing hidden cells."""
        if self.world is None:
            return None
        observation = self.world.observation(Faction.PLAYER)
        known = np.frombuffer(observation.known_terrain, dtype=np.uint8).reshape(
            observation.terrain_shape
        )
        col = int(np.clip(x // self.world.map.tile_size, 0, self.world.map.cols - 1))
        row = int(np.clip(y // self.world.map.tile_size, 0, self.world.map.rows - 1))
        value = int(known[row, col])
        return None if value == 255 else Terrain(value)

    def _context_selection(self, x: float, y: float, minimum: int = 0) -> tuple[int, ...]:
        if self.world is None:
            return ()
        world = self.world
        selected: list[int] = []
        for entity_id in sorted(self.selected_ids):
            index = world.units.index_of(entity_id)
            if (
                index is not None
                and world.units.alive[index]
                and world.units.faction[index] == int(Faction.PLAYER)
                and world.units.kind[index] == int(UnitKind.ENGINEER)
            ):
                selected.append(entity_id)
        if len(selected) >= minimum:
            return tuple(selected)
        engineers = world.resolve_selection(SelectionV1(unit_kind="engineer"), Faction.PLAYER)
        ordered = sorted(
            engineers,
            key=lambda index: (
                (world.units.x[index] / world.subpixels - x) ** 2
                + (world.units.y[index] / world.subpixels - y) ** 2
            ),
        )
        return tuple(int(world.units.entity_id[index]) for index in ordered[:minimum])

    def _issue_context_move(self, attack: bool = False) -> None:
        if self.context_menu_world is None or self.world is None:
            return
        x, y = self.context_menu_world
        ids = tuple(sorted(self.selected_ids))
        if not ids:
            commander = self.world.commander_index(Faction.PLAYER)
            ids = (int(self.world.units.entity_id[commander]),) if commander is not None else ()
        self._issue_move(ids, x, y, attack, False, CommandSource.MOUSE)
        self.context_menu_point = None
        self.context_menu_world = None

    def _issue_context_build(
        self, facility_kind: Literal["bridge", "boat", "road", "tower"]
    ) -> None:
        if self.context_menu_world is None or self.world is None:
            return
        x, y = self.context_menu_world
        minimum = int(self.world.balance.facilities[facility_kind]["minimum_engineers"])
        ids = self._context_selection(x, y, minimum)
        command = CommandEnvelopeV1(
            command_id=uuid.uuid4().hex,
            faction=Faction.PLAYER,
            issued_tick=self.world.tick,
            source=CommandSource.MOUSE,
            payload=BuildPayloadV1(
                selection=SelectionV1(unit_ids=ids),
                facility_kind=facility_kind,
                target=PositionV1(x=x, y=y),
            ),
        )
        result = self.world.execute(command)
        if result.status == CommandStatus.ACCEPTED:
            self.selected_ids = set(ids)
            self.sounds.play("command")
        self._message(
            result.message_zh,
            self.theme.success if result.status == CommandStatus.ACCEPTED else self.theme.enemy,
        )
        self.context_menu_point = None
        self.context_menu_world = None

    def _issue_context_recruit(self, village_id: int) -> None:
        if self.world is None:
            return
        first_id = self.world.next_entity_id
        command = CommandEnvelopeV1(
            command_id=uuid.uuid4().hex,
            faction=Faction.PLAYER,
            issued_tick=self.world.tick,
            source=CommandSource.MOUSE,
            payload=RecruitPayloadV1(village_id=village_id, count=20),
        )
        result = self.world.execute(command)
        if result.status == CommandStatus.ACCEPTED:
            self.selected_ids = set(range(first_id, self.world.next_entity_id))
            self.sounds.play("command")
        self._message(
            result.message_zh
            + (
                " 已选中新兵，请在底栏训练兵种。" if result.status == CommandStatus.ACCEPTED else ""
            ),
            self.theme.success if result.status == CommandStatus.ACCEPTED else self.theme.enemy,
        )
        self.context_menu_point = None
        self.context_menu_world = None

    def _convert_selected(
        self, unit_kind: Literal["infantry", "scout", "engineer", "assassin"]
    ) -> None:
        if self.world is None or self.scene != "battle":
            return
        recruit_ids: list[int] = []
        for entity_id in sorted(self.selected_ids):
            index = self.world.units.index_of(entity_id)
            if (
                index is not None
                and self.world.units.alive[index]
                and self.world.units.faction[index] == int(Faction.PLAYER)
                and self.world.units.kind[index] == int(UnitKind.RECRUIT)
            ):
                recruit_ids.append(entity_id)
        command = CommandEnvelopeV1(
            command_id=uuid.uuid4().hex,
            faction=Faction.PLAYER,
            issued_tick=self.world.tick,
            source=CommandSource.MOUSE,
            payload=ConvertPayloadV1(
                selection=SelectionV1(unit_ids=tuple(recruit_ids)), unit_kind=unit_kind
            ),
        )
        result = self.world.execute(command)
        if result.status == CommandStatus.ACCEPTED:
            self.sounds.play("command")
        self._message(
            result.message_zh,
            self.theme.success if result.status == CommandStatus.ACCEPTED else self.theme.enemy,
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
                <= ENEMY_CLICK_RADIUS_SQ
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
            if (click and (sx - end[0]) ** 2 + (sy - end[1]) ** 2 <= UNIT_CLICK_RADIUS_SQ) or (
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
        self._update_camera_zoom(dt)
        self._update_battle_effects(dt)
        if self.result_cinematic_time > 0:
            self._update_result_cinematic(dt)
            return
        keys = pygame.key.get_pressed()
        camera_speed = 900 * dt / max(self.camera.zoom, 0.1)
        horizontal = keys[pygame.K_d] - keys[pygame.K_a]
        vertical = keys[pygame.K_s] - keys[pygame.K_w]
        if horizontal or vertical:
            self.camera.zoom_anchor_world = None
        self.camera.x += horizontal * camera_speed
        self.camera.y += vertical * camera_speed
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
                if (
                    self.auto_command
                    and self.player_ai is not None
                    and self.player_ai.ready(self.world.tick)
                ):
                    for command in self.player_ai.decide(self.world.observation(Faction.PLAYER)):
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
            self._begin_result_cinematic()

    def _update_battle_effects(self, dt: float) -> None:
        if not self.battle_effects:
            return
        for effect in self.battle_effects:
            effect.age += dt
        self.battle_effects = [
            effect for effect in self.battle_effects if effect.age < effect.duration
        ][-96:]

    def _begin_result_cinematic(self) -> None:
        if self.world is None or self.result_cinematic_time > 0 or self.scene != "battle":
            return
        self.paused = True
        self.context_menu_point = None
        self.context_menu_world = None
        commander_deaths = [
            event
            for event in reversed(self.world.events)
            if event.kind == "unit_died" and bool(event.payload.get("commander"))
        ]
        if commander_deaths:
            payload = commander_deaths[0].payload
            self.cinematic_target = (float(payload["x"]), float(payload["y"]))
        self.result_cinematic_time = 0.08 if self.reduced_motion else self.result_cinematic_duration
        if (
            self.camera is not None
            and self.cinematic_target is not None
            and not self.reduced_motion
        ):
            self.camera.zoom_anchor_world = None
            self.camera.target_zoom = max(self.camera.target_zoom, 0.68)

    def _update_result_cinematic(self, dt: float) -> None:
        if self.world is None or self.camera is None:
            return
        if self.cinematic_target is not None and not self.reduced_motion:
            amount = 1.0 - math.exp(-5.5 * dt)
            self.camera.x += (self.cinematic_target[0] - self.camera.x) * amount
            self.camera.y += (self.cinematic_target[1] - self.camera.y) * amount
            self.camera.clamp(self.world)
        self.result_cinematic_time = max(0.0, self.result_cinematic_time - dt)
        if self.result_cinematic_time > 0:
            return
        self.scene = "result"
        if not self.result_sound_played:
            self.sounds.play(
                "victory" if self.world.outcome == GameOutcome.PLAYER_WIN else "impact"
            )
            self.result_sound_played = True
        if self.recorder is not None:
            try:
                self.save_manager.save_replay(self.recorder.finish(self.world))
            except OSError:
                logger.warning("回放保存失败", exc_info=True)

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
            if event.kind == "combat_exchange":
                target_x = float(event.payload.get("target_x", 0.0))
                target_y = float(event.payload.get("target_y", 0.0))
                if self._world_point_visible(target_x, target_y):
                    attacker_x = float(event.payload.get("attacker_x", target_x))
                    attacker_y = float(event.payload.get("attacker_y", target_y))
                    ranged = bool(event.payload.get("ranged"))
                    tactic = int(event.payload.get("tactic", 0))
                    self._append_effect(
                        BattleEffect(
                            "projectile" if ranged else "charge" if tactic == 2 else "slash",
                            attacker_x,
                            attacker_y,
                            target_x,
                            target_y,
                            0.20 if ranged else 0.28,
                            size=28 if tactic != 2 else 36,
                        )
                    )
                    self._append_effect(
                        BattleEffect(
                            "impact", target_x, target_y, target_x, target_y, 0.24, size=24
                        )
                    )
            elif event.kind == "unit_died":
                x = float(event.payload.get("x", 0.0))
                y = float(event.payload.get("y", 0.0))
                if self._world_point_visible(x, y):
                    commander = bool(event.payload.get("commander"))
                    self._append_effect(
                        BattleEffect(
                            "death",
                            x,
                            y,
                            x,
                            y,
                            1.05 if commander else 0.52,
                            size=72 if commander else 34,
                            faction=int(event.payload.get("faction", 0)),
                        )
                    )
            elif event.kind == "tactic_started":
                effect_kind = "sprint" if event.payload.get("kind") == "sprint" else "charge"
                for entity_id in list(event.payload.get("unit_ids", []))[::3]:
                    index = self.world.units.index_of(int(entity_id))
                    if index is None:
                        continue
                    x = float(self.world.units.x[index]) / self.world.subpixels
                    y = float(self.world.units.y[index]) / self.world.subpixels
                    if self._world_point_visible(x, y):
                        self._append_effect(BattleEffect(effect_kind, x, y, x, y, 0.42, size=30))
            elif event.kind == "facility_completed":
                facility_id = int(event.payload.get("facility_id", -1))
                facility = next(
                    (item for item in self.world.facilities if item.facility_id == facility_id),
                    None,
                )
                if facility is not None and self._world_point_visible(facility.x, facility.y):
                    self._append_effect(
                        BattleEffect(
                            "impact",
                            facility.x,
                            facility.y,
                            facility.x,
                            facility.y,
                            0.48,
                            size=40,
                        )
                    )
            elif event.kind == "tower_garrisoned":
                count = int(event.payload.get("count", 0))
                self._message(f"{count} 名工兵已登塔并转为守塔弓箭手。", self.theme.success)
        self.stats = {
            "player_losses": self.world.statistics["losses"][int(Faction.PLAYER)],
            "enemy_losses": self.world.statistics["losses"][int(Faction.ENEMY)],
        }
        if completed:
            self.sounds.play("complete")
        elif impact:
            self.sounds.play("impact")
        self.last_event_index = len(self.world.events)

    def _append_effect(self, effect: BattleEffect) -> None:
        if self.reduced_motion and effect.kind not in {"death", "impact"}:
            return
        self.battle_effects.append(effect)
        if len(self.battle_effects) > 96:
            self.battle_effects = self.battle_effects[-96:]

    def _world_point_visible(self, x: float, y: float) -> bool:
        if self.world is None or self.view_faction is None:
            return True
        fog = self.world._perception
        if fog is None:
            return False
        col = int(np.clip(x // self.world.map.tile_size, 0, self.world.map.cols - 1))
        row = int(np.clip(y // self.world.map.tile_size, 0, self.world.map.rows - 1))
        return bool(fog.explored[int(self.view_faction), row, col])

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
        self._mouse_pos = self._logical_mouse(pygame.mouse.get_pos())
        self._native_text = self._viewport().size != LOGICAL_SIZE
        self._text_commands = []
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
        self._blit_text("指挥官战术对抗", 96, 92, 34, self.theme.ink, True)
        self._blit_text("在战争迷雾中，以军令统筹一千名真实士兵", 98, 142, 16, self.theme.muted)
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
        mouse = self._mouse_pos
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
            "侦察兵永久擦除迷雾并发现未暴露刺客",
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
        mouse = self._mouse_pos
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
        self._blit_text(
            f"敌方 AI：{diff_label} · 相同兵力 · {inspire}", 138, 520, 14, self.theme.muted
        )
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
        self._blit_text(
            "像素战场最近邻缩放 · 文字按显示器分辨率原生重绘",
            102,
            158,
            13,
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
                "全屏 · 自动适配" if self.fullscreen else "窗口化 · 自动适配",
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
        mouse = self._mouse_pos
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
                    "WASD 或按住中键拖动摄像机；滚轮和 + / − 按钮平滑缩放。",
                    "黑色区域尚未探索；走过的区域会永久擦除战争迷雾。",
                    "敌军进入任意已探索区域就会显示，小地图遵守同一情报边界。",
                ],
            ),
            (
                "02 · 选择与编队",
                [
                    "左键点选或拖动框选部队，右键打开地形操作菜单。",
                    "菜单可移动、攻击移动；河面可架桥，陆地可建塔。",
                    "Ctrl+数字保存编队，数字键重新选择；方向键直接移动将领。",
                ],
            ),
            (
                "03 · 侦察、征兵与工程",
                [
                    "侦察兵速度快、视野广，并能发现未暴露刺客。",
                    "接近无敌军争夺的村庄后，可用文字命令征召人口。",
                    "陆军不能强行渡河；选择足量工兵架桥后才能通过。",
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
        values = (0.9, 1.0, 1.1, 1.2, 1.35)
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
        if not self.fullscreen:
            self._windowed_size = self.screen.get_size()
        self.fullscreen = not self.fullscreen
        self.screen = self._set_display_mode(self.fullscreen)
        self._save_settings()

    def _begin_text_input(self) -> None:
        if self.scene != "battle":
            return
        self.typing = True
        self.command_input = ""
        self.composition = ""
        self._refresh_suggestions()
        if hasattr(pygame.key, "start_text_input"):
            self._update_ime_rect()
            pygame.key.start_text_input()

    def _toggle_pause(self) -> None:
        self.help_open = False
        self.paused = not self.paused

    def _cycle_speed(self) -> None:
        values = (0.5, 1.0, 2.0, 4.0)
        current = min(range(len(values)), key=lambda i: abs(values[i] - self.speed))
        self.speed = values[(current + 1) % len(values)]

    def _toggle_auto_command(self) -> None:
        if self.scene != "battle" or self.player_ai is None:
            return
        self.auto_command = not self.auto_command
        state = "开启" if self.auto_command else "关闭"
        self._message(f"自动作战已{state}；手动指令仍可随时接管。", self.theme.warning)

    def _cycle_view(self) -> None:
        if self.scene != "replay":
            return
        values: tuple[Faction | None, ...] = (Faction.PLAYER, Faction.ENEMY, None)
        current = values.index(self.view_faction) if self.view_faction in values else 0
        self.view_faction = values[(current + 1) % len(values)]

    def _quick_save(self) -> None:
        if self.world is None or self.scene != "battle":
            return
        try:
            path = self.save_manager.save(self.world)
            self._message(f"已保存：{path.name}", self.theme.success)
        except OSError as exc:
            self._message(f"保存失败：{exc}", self.theme.enemy)

    def _choose_ambiguity(self, index: int) -> None:
        if self.world is None or index >= len(self.ambiguity_candidates):
            return
        command = self.ambiguity_candidates[index]
        self.ambiguity_candidates = []
        result = self.world.execute(command)
        self._remember_group_name(command, result)
        self._message(result.message_zh, self.theme.success)
        self.sounds.play("command")

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
            logger.warning("设置保存失败", exc_info=True)

    def _draw_action_button(
        self, rect: pygame.Rect, label: str, action: Callable[[], None], primary: bool = False
    ) -> None:
        enabled_buttons = [b for b in self.buttons if b.enabled]
        is_focused = 0 <= self.menu_focus < len(enabled_buttons) and len(enabled_buttons) > 0
        focused_rect = enabled_buttons[self.menu_focus].rect if is_focused else None
        keyboard_focused = focused_rect == rect if focused_rect else False
        hovered = rect.collidepoint(self._mouse_pos)
        color = (
            self.theme.primary_hover
            if (hovered or keyboard_focused) and primary
            else self.theme.primary
            if primary
            else self.theme.surface_high
            if not keyboard_focused
            else self.theme.primary
        )
        pygame.draw.rect(self.canvas, color, rect, border_radius=6)
        if not primary:
            pygame.draw.rect(self.canvas, self.theme.border, rect, 1, border_radius=6)
        if keyboard_focused:
            pygame.draw.rect(self.canvas, self.theme.ink, rect, 2, border_radius=6)
        self._blit_text(label, rect.centerx, rect.centery, 15, self.theme.ink, True, center=True)
        self.buttons.append(Button(rect, label, action, enabled=True))

    def _draw_compact_button(
        self,
        rect: pygame.Rect,
        label: str,
        action: Callable[[], None],
        accent: bool = False,
        enabled: bool = True,
    ) -> None:
        hovered = rect.collidepoint(self._mouse_pos) and enabled
        fill = (
            self.theme.primary_hover
            if hovered
            else self.theme.primary
            if accent and enabled
            else self.theme.surface_high
        )
        pygame.draw.rect(self.canvas, fill, rect, border_radius=5)
        pygame.draw.rect(self.canvas, self.theme.border, rect, 1, border_radius=5)
        self._blit_text(
            label,
            rect.centerx,
            rect.centery,
            12,
            self.theme.ink if enabled else self.theme.muted,
            accent,
            center=True,
            max_width=rect.width - 8,
        )
        self.buttons.append(Button(rect, label, action, enabled))

    def _draw_battle(self) -> None:
        if self.world is None or self.camera is None:
            return
        pygame.draw.rect(self.canvas, self.theme.fog_unknown, BATTLE_RECT)
        self._draw_terrain()
        self._draw_facilities_and_villages()
        self._draw_movement_effects()
        self._draw_units()
        self._draw_battle_effects()
        if self.result_cinematic_time > 0:
            self._draw_result_cinematic()
        self._draw_hud()
        self._draw_context_menu()
        if self.drag_start and pygame.mouse.get_pressed()[0]:
            end = self._mouse_pos
            rect = pygame.Rect(
                min(self.drag_start[0], end[0]),
                min(self.drag_start[1], end[1]),
                abs(end[0] - self.drag_start[0]),
                abs(end[1] - self.drag_start[1]),
            )
            pygame.draw.rect(self.canvas, self.theme.ally, rect, 1)
        if self.help_open:
            self._draw_help()
        elif self.paused and self.scene == "battle" and self.result_cinematic_time <= 0:
            self._draw_pause()

    def _draw_context_menu(self) -> None:
        if (
            self.context_menu_point is None
            or self.context_menu_world is None
            or self.world is None
            or self.scene != "battle"
        ):
            return
        world_x, world_y = self.context_menu_world
        terrain = self._known_terrain_at(world_x, world_y)
        terrain_names = {
            Terrain.PLAIN: "平原",
            Terrain.GRASS: "草地",
            Terrain.FOREST: "森林",
            Terrain.SWAMP: "沼泽",
            Terrain.RIVER: "河流",
            Terrain.ROAD: "道路",
        }
        engineers = len(
            self.world.resolve_selection(SelectionV1(unit_kind="engineer"), Faction.PLAYER)
        )
        observation = self.world.observation(Faction.PLAYER)
        village = next(
            (
                item
                for item in observation.known_villages
                if (float(item["x"]) - world_x) ** 2 + (float(item["y"]) - world_y) ** 2 <= 92**2
            ),
            None,
        )
        options: list[tuple[str, Callable[[], None], bool, bool]] = [
            ("移动到这里", self._issue_context_move, True, True),
            ("攻击移动", partial(self._issue_context_move, True), True, False),
        ]
        if village is not None:
            options.append(
                (
                    "征召预备兵 · 20人",
                    partial(self._issue_context_recruit, int(village["village_id"])),
                    int(village["population"]) > 0,
                    False,
                )
            )
        if terrain == Terrain.RIVER:
            minimum = int(self.world.balance.facilities["bridge"]["minimum_engineers"])
            geometry = self.world.known_bridge_geometry_at(Faction.PLAYER, world_x, world_y)
            if geometry is None:
                options.append(
                    (
                        "架桥 · 需先探明两岸",
                        partial(self._issue_context_build, "bridge"),
                        False,
                        False,
                    )
                )
            else:
                bridge_length = geometry[2]
                baseline = self.world.map.tile_size * 7
                bridge_work = round(
                    float(self.world.balance.facilities["bridge"]["work"])
                    * float(np.clip(bridge_length / baseline, 0.75, 3.0))
                )
                options.append(
                    (
                        f"架桥 · {minimum}人 · 跨{bridge_length:.0f} · 工程{bridge_work}",
                        partial(self._issue_context_build, "bridge"),
                        engineers >= minimum,
                        False,
                    )
                )
        elif terrain is not None and terrain != Terrain.SWAMP:
            minimum = int(self.world.balance.facilities["tower"]["minimum_engineers"])
            options.append(
                (
                    f"建造防御塔 · {minimum}人",
                    partial(self._issue_context_build, "tower"),
                    engineers >= minimum,
                    False,
                )
            )
            if terrain == Terrain.FOREST:
                road_minimum = int(self.world.balance.facilities["road"]["minimum_engineers"])
                options.append(
                    (
                        f"开辟道路 · {road_minimum}人",
                        partial(self._issue_context_build, "road"),
                        engineers >= road_minimum,
                        False,
                    )
                )

        width = 246
        height = 38 + len(options) * 36 + 8
        x = int(
            np.clip(
                self.context_menu_point[0] + 8, BATTLE_RECT.left + 6, BATTLE_RECT.right - width - 6
            )
        )
        y = int(
            np.clip(
                self.context_menu_point[1] + 8,
                BATTLE_RECT.top + 6,
                BATTLE_RECT.bottom - height - 6,
            )
        )
        panel = pygame.Rect(x, y, width, height)
        pygame.draw.rect(self.canvas, self.theme.surface, panel, border_radius=8)
        pygame.draw.rect(self.canvas, self.theme.border, panel, 1, border_radius=8)
        heading = (
            f"村庄 · 人口 {int(village['population'])}"
            if village is not None
            else f"{terrain_names[terrain]} · {world_x:.0f}, {world_y:.0f}"
            if terrain is not None
            else f"未探索区域 · {world_x:.0f}, {world_y:.0f}"
        )
        self._blit_text(
            heading,
            panel.x + 12,
            panel.y + 10,
            12,
            self.theme.muted,
            max_width=panel.width - 24,
        )
        for index, (label, action, enabled, accent) in enumerate(options):
            rect = pygame.Rect(panel.x + 8, panel.y + 36 + index * 36, panel.width - 16, 30)
            self._draw_compact_button(rect, label, action, accent, enabled)

    def _draw_terrain(self) -> None:
        assert self.world is not None and self.camera is not None
        fog = self.world._perception
        view = self.view_faction
        if fog is None and view is not None:
            self.world.observation(view)
            fog = self.world._perception
        if view is not None:
            assert fog is not None

        world_key = (id(self.world.map), self.world.map.revision)
        if self._terrain_world_key != world_key or self._terrain_world_surface is None:
            composed = self.art.compose_terrain(self.world.map.terrain)
            if composed is None:
                colors = np.asarray(
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
                pixels = colors[self.world.map.terrain]
                tiny = pygame.surfarray.make_surface(np.transpose(pixels, (1, 0, 2)))
                composed = pygame.transform.scale(
                    tiny, (self.world.map.cols * 16, self.world.map.rows * 16)
                )
            self._terrain_world_surface = composed
            self._terrain_world_key = world_key
            self._fogged_world_surface = None
            self._terrain_cache = None

        fog_key = (
            world_key,
            int(view) if view is not None else -1,
            self.world.tick // 10 if view is not None else -1,
        )
        if self._fogged_world_key != fog_key:
            if view is None:
                self._fogged_world_surface = None
            else:
                assert fog is not None
                fog_mask = pygame.Surface(
                    (self.world.map.cols, self.world.map.rows), pygame.SRCALPHA
                )
                alpha = pygame.surfarray.pixels_alpha(fog_mask)
                alpha[:, :] = np.where(fog.explored[int(view)].T, 0, 255)
                del alpha
                self._fogged_world_surface = fog_mask
            self._fogged_world_key = fog_key
            self._terrain_cache = None

        cam_key = (
            round(self.camera.x, 2),
            round(self.camera.y, 2),
            round(self.camera.zoom, 4),
            fog_key,
        )
        if self._terrain_cache_key == cam_key and self._terrain_cache is not None:
            self.canvas.blit(self._terrain_cache, BATTLE_RECT.topleft)
            return

        view_width = BATTLE_RECT.width / self.camera.zoom
        view_height = BATTLE_RECT.height / self.camera.zoom
        left = self.camera.x - view_width / 2
        top = self.camera.y - view_height / 2
        right = left + view_width
        bottom = top + view_height
        clipped_left = max(0.0, left)
        clipped_top = max(0.0, top)
        clipped_right = min(float(self.world.map.width), right)
        clipped_bottom = min(float(self.world.map.height), bottom)

        result = pygame.Surface(BATTLE_RECT.size)
        result.fill(self.theme.fog_unknown)
        if clipped_right > clipped_left and clipped_bottom > clipped_top:
            assert self._terrain_world_surface is not None
            scale_x = self._terrain_world_surface.get_width() / self.world.map.width
            scale_y = self._terrain_world_surface.get_height() / self.world.map.height
            source = pygame.Rect(
                round(clipped_left * scale_x),
                round(clipped_top * scale_y),
                max(1, round((clipped_right - clipped_left) * scale_x)),
                max(1, round((clipped_bottom - clipped_top) * scale_y)),
            ).clip(self._terrain_world_surface.get_rect())
            target = pygame.Rect(
                round((clipped_left - left) * self.camera.zoom),
                round((clipped_top - top) * self.camera.zoom),
                max(1, round((clipped_right - clipped_left) * self.camera.zoom)),
                max(1, round((clipped_bottom - clipped_top) * self.camera.zoom)),
            )
            crop = self._terrain_world_surface.subsurface(source)
            result.blit(pygame.transform.scale(crop, target.size), target)
            if self._fogged_world_surface is not None:
                fog_scale_x = self._fogged_world_surface.get_width() / self.world.map.width
                fog_scale_y = self._fogged_world_surface.get_height() / self.world.map.height
                fog_source = pygame.Rect(
                    int(clipped_left * fog_scale_x),
                    int(clipped_top * fog_scale_y),
                    max(1, math.ceil((clipped_right - clipped_left) * fog_scale_x)),
                    max(1, math.ceil((clipped_bottom - clipped_top) * fog_scale_y)),
                ).clip(self._fogged_world_surface.get_rect())
                fog_crop = self._fogged_world_surface.subsurface(fog_source)
                result.blit(pygame.transform.scale(fog_crop, target.size), target)

        self._terrain_cache = result
        self._terrain_cache_key = cam_key
        self.canvas.blit(result, BATTLE_RECT.topleft)

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
            village_size = max(24, min(92, round(village.radius * 2.4 * self.camera.zoom)))
            village_sprite = self.art.village(village_size)
            if village_sprite is not None:
                self.canvas.blit(village_sprite, village_sprite.get_rect(center=(sx, sy - 2)))
                pygame.draw.ellipse(
                    self.canvas,
                    self.theme.warning,
                    pygame.Rect(sx - radius, sy - max(3, radius // 2), radius * 2, radius),
                    1,
                )
            else:
                pygame.draw.circle(self.canvas, self.theme.warning, (sx, sy), radius, 1)
            if self.camera.zoom > 0.35:
                population = (
                    village.population
                    if view is None
                    else int(known_villages.get(village.village_id, {}).get("population", 0))
                )
                label = pygame.Rect(sx - 18, sy + radius + 2, 36, 16)
                pygame.draw.rect(self.canvas, self.theme.background, label, border_radius=3)
                self._blit_text(
                    str(population),
                    label.centerx,
                    label.centery,
                    10,
                    self.theme.warning,
                    True,
                    center=True,
                )
        for item in self.world.facilities:
            if view is not None and item.facility_id not in known_facility_ids:
                continue
            sx, sy = self.camera.world_to_screen(item.x, item.y)
            if not BATTLE_RECT.collidepoint(sx, sy):
                continue
            color = self.theme.ally if item.faction == 0 else self.theme.enemy
            facility_sprite: pygame.Surface | None = None
            if item.kind == "bridge":
                bridge_length = item.bridge_length or self.world.map.tile_size * 8
                facility_sprite = self.art.facility(
                    "bridge",
                    max(30, round(bridge_length * self.camera.zoom)),
                    max(14, round(BRIDGE_DECK_HALF_WIDTH * 2 * self.camera.zoom)),
                    vertical=item.bridge_vertical,
                )
            elif item.kind == "tower":
                tower_size = max(18, round(62 * self.camera.zoom))
                facility_sprite = self.art.facility("tower", tower_size, tower_size)
            if item.destroyed:
                pygame.draw.line(self.canvas, color, (sx - 6, sy - 6), (sx + 6, sy + 6), 2)
                pygame.draw.line(self.canvas, color, (sx + 6, sy - 6), (sx - 6, sy + 6), 2)
            elif facility_sprite is not None:
                if not item.complete:
                    facility_sprite = facility_sprite.copy()
                    facility_sprite.set_alpha(120)
                self.canvas.blit(facility_sprite, facility_sprite.get_rect(center=(sx, sy)))
                outline = facility_sprite.get_rect(center=(sx, sy)).inflate(4, 4)
                pygame.draw.rect(self.canvas, color, outline, 1, border_radius=3)
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

    def _draw_unit_tactical_marker(
        self,
        sx: int,
        sy: int,
        radius: int,
        kind: UnitKind,
        faction_color: tuple[int, int, int],
    ) -> None:
        """Draw a role-specific marker used while detail sprites fade in."""
        outline = self.theme.background
        if self.camera is not None and self.camera.zoom < 0.34:
            if kind == UnitKind.INFANTRY:
                pygame.draw.rect(
                    self.canvas,
                    faction_color,
                    (sx - radius, sy - radius, radius * 2, radius * 2),
                )
            elif kind == UnitKind.SCOUT:
                pygame.draw.polygon(
                    self.canvas,
                    faction_color,
                    (
                        (sx, sy - radius),
                        (sx + radius, sy + radius),
                        (sx - radius, sy + radius),
                    ),
                )
            elif kind == UnitKind.ENGINEER:
                pygame.draw.polygon(
                    self.canvas,
                    faction_color,
                    (
                        (sx, sy - radius),
                        (sx + radius, sy),
                        (sx, sy + radius),
                        (sx - radius, sy),
                    ),
                )
            elif kind == UnitKind.ASSASSIN:
                pygame.draw.circle(self.canvas, faction_color, (sx, sy), radius, 1)
            elif kind == UnitKind.COMMANDER:
                pygame.draw.circle(self.canvas, self.theme.warning, (sx, sy), radius + 2)
                pygame.draw.circle(self.canvas, faction_color, (sx, sy), radius - 1)
            else:
                pygame.draw.circle(self.canvas, faction_color, (sx, sy), radius)
            return
        if kind == UnitKind.INFANTRY:
            marker = pygame.Rect(sx - radius, sy - radius, radius * 2, radius * 2)
            pygame.draw.rect(self.canvas, outline, marker.inflate(2, 2), border_radius=2)
            pygame.draw.rect(self.canvas, faction_color, marker, border_radius=2)
        elif kind == UnitKind.SCOUT:
            triangle_points = (
                (sx, sy - radius - 1),
                (sx + radius + 1, sy + radius),
                (sx - radius - 1, sy + radius),
            )
            pygame.draw.polygon(self.canvas, outline, triangle_points)
            inner = tuple(
                (round(sx + (px - sx) * 0.72), round(sy + (py - sy) * 0.72))
                for px, py in triangle_points
            )
            pygame.draw.polygon(self.canvas, faction_color, inner)
        elif kind == UnitKind.ENGINEER:
            diamond_points = (
                (sx, sy - radius - 1),
                (sx + radius + 1, sy),
                (sx, sy + radius + 1),
                (sx - radius - 1, sy),
            )
            pygame.draw.polygon(self.canvas, outline, diamond_points)
            pygame.draw.polygon(self.canvas, faction_color, diamond_points, 2)
            pygame.draw.circle(self.canvas, faction_color, (sx, sy), 2)
        elif kind == UnitKind.ASSASSIN:
            pygame.draw.circle(self.canvas, outline, (sx, sy), radius + 1)
            pygame.draw.circle(self.canvas, faction_color, (sx, sy), radius, 2)
            pygame.draw.circle(self.canvas, faction_color, (sx, sy), 2)
        elif kind == UnitKind.COMMANDER:
            pygame.draw.circle(self.canvas, outline, (sx, sy), radius + 3)
            pygame.draw.circle(self.canvas, self.theme.warning, (sx, sy), radius + 2)
            pygame.draw.circle(self.canvas, faction_color, (sx, sy), radius - 1)
        elif kind == UnitKind.GUARD:
            pygame.draw.circle(self.canvas, outline, (sx, sy), radius + 1)
            pygame.draw.circle(self.canvas, faction_color, (sx, sy), radius)
            pygame.draw.line(
                self.canvas, self.theme.ink, (sx, sy - radius + 1), (sx, sy + radius - 1), 1
            )
        else:
            pygame.draw.circle(self.canvas, outline, (sx, sy), radius + 1)
            pygame.draw.circle(self.canvas, faction_color, (sx, sy), radius)

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
        for index in self.world.units.active():
            if int(self.world.units.facility_id[index]) in concealed_facilities:
                continue
            entity_id = int(self.world.units.entity_id[index])
            faction = int(self.world.units.faction[index])
            if view is not None and faction != int(view) and entity_id not in visible_enemy_ids:
                continue
            alpha = (
                1.0
                if self.reduced_motion
                else float(np.clip(self.accumulator / self.world.dt, 0, 1))
            )
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
            faction_color = self.theme.ally if faction == int(Faction.PLAYER) else self.theme.enemy
            radius = max(3, min(9, round(self.world.units.radius[index] * self.camera.zoom + 2)))
            detail_mix = float(np.clip((self.camera.zoom - 0.42) / 0.16, 0.0, 1.0))
            sprite_size = min(
                50,
                round(
                    14
                    + max(0.0, self.camera.zoom - 0.42) * 48
                    + (5 if kind in {UnitKind.COMMANDER, UnitKind.GUARD} else 0)
                ),
            )
            sprite = self.art.unit(faction, int(kind), sprite_size) if detail_mix > 0.0 else None
            if detail_mix < 1.0:
                self._draw_unit_tactical_marker(sx, sy, radius, kind, faction_color)
            if sprite is not None:
                shadow_radius = max(4, round(sprite_size * 0.22))
                shadow = pygame.Surface((shadow_radius * 2 + 2, shadow_radius + 4), pygame.SRCALPHA)
                pygame.draw.ellipse(shadow, (20, 22, 20, round(92 * detail_mix)), shadow.get_rect())
                self.canvas.blit(shadow, shadow.get_rect(center=(sx, sy + sprite_size // 3)))
                if detail_mix < 1.0:
                    sprite = sprite.copy()
                    sprite.set_alpha(round(255 * detail_mix))
                sprite_rect = sprite.get_rect(center=(sx, sy - sprite_size // 10))
                self.canvas.blit(sprite, sprite_rect)
            if kind == UnitKind.COMMANDER:
                pygame.draw.circle(self.canvas, self.theme.warning, (sx, sy), radius + 5, 2)
            if entity_id in self.selected_ids:
                selection_radius = max(radius + 4, sprite_size // 2 if sprite is not None else 0)
                pygame.draw.circle(self.canvas, self.theme.ink, (sx, sy), selection_radius, 1)
            if (
                self.world.units.hp[index] < self.world.units.max_hp[index]
                and self.camera.zoom > 0.4
            ):
                ratio = float(self.world.units.hp[index] / self.world.units.max_hp[index])
                health_y = sy - (sprite_size // 2 + 5 if sprite is not None else radius + 5)
                pygame.draw.rect(self.canvas, self.theme.background, (sx - 7, health_y, 14, 2))
                pygame.draw.rect(
                    self.canvas,
                    self.theme.success if ratio > 0.4 else self.theme.enemy,
                    (sx - 7, health_y, round(14 * ratio), 2),
                )

    def _draw_movement_effects(self) -> None:
        assert self.world is not None and self.camera is not None
        if self.reduced_motion or self.camera.zoom < 0.26:
            return
        layer = pygame.Surface(BATTLE_RECT.size, pygame.SRCALPHA)
        view = self.view_faction
        observation = self.world.observation(view) if view is not None else None
        visible_enemy_ids = (
            {unit.entity_id for unit in observation.visible_enemies} if observation else set()
        )
        for index in self.world.units.active():
            tactic = int(self.world.units.tactic_kind[index])
            if tactic == 0:
                continue
            entity_id = int(self.world.units.entity_id[index])
            faction = int(self.world.units.faction[index])
            if view is not None and faction != int(view) and entity_id not in visible_enemy_ids:
                continue
            if entity_id % 3:
                continue
            x = float(self.world.units.x[index]) / self.world.subpixels
            y = float(self.world.units.y[index]) / self.world.subpixels
            previous_x = float(self.world.units.previous_x[index]) / self.world.subpixels
            previous_y = float(self.world.units.previous_y[index]) / self.world.subpixels
            dx, dy = x - previous_x, y - previous_y
            length = math.hypot(dx, dy)
            if length < 0.02:
                continue
            sx, sy = self.camera.world_to_screen(x, y)
            if not BATTLE_RECT.collidepoint(sx, sy):
                continue
            nx, ny = dx / length, dy / length
            trail = 9 if tactic == 1 else 13
            color = (*self.theme.ally, 105) if tactic == 1 else (*self.theme.warning, 125)
            local = (sx - BATTLE_RECT.x, sy - BATTLE_RECT.y)
            for offset in (-2, 2):
                perpendicular_x, perpendicular_y = -ny * offset, nx * offset
                start = (
                    round(local[0] - nx * trail + perpendicular_x),
                    round(local[1] - ny * trail + perpendicular_y),
                )
                end = (
                    round(local[0] - nx * 2 + perpendicular_x),
                    round(local[1] - ny * 2 + perpendicular_y),
                )
                pygame.draw.line(layer, color, start, end, 1)
            if entity_id % 12 == 0:
                sprite = self.art.effect("sprint" if tactic == 1 else "charge", 22)
                if sprite is not None:
                    sprite = sprite.copy()
                    sprite.set_alpha(72)
                    layer.blit(sprite, sprite.get_rect(center=local))
        self.canvas.blit(layer, BATTLE_RECT.topleft)

    def _draw_battle_effects(self) -> None:
        if self.camera is None or not self.battle_effects:
            return
        layer = pygame.Surface(BATTLE_RECT.size, pygame.SRCALPHA)
        for effect in self.battle_effects:
            progress = float(np.clip(effect.age / max(effect.duration, 0.001), 0, 1))
            alpha = round(255 * (1.0 - progress))
            if effect.kind == "projectile":
                eased = 1.0 - (1.0 - progress) ** 3
                x = effect.x + (effect.target_x - effect.x) * eased
                y = effect.y + (effect.target_y - effect.y) * eased
                sx, sy = self.camera.world_to_screen(x, y)
                tx, ty = self.camera.world_to_screen(effect.x, effect.y)
                local = (sx - BATTLE_RECT.x, sy - BATTLE_RECT.y)
                tail = (
                    round(local[0] + (tx - sx) * 0.12),
                    round(local[1] + (ty - sy) * 0.12),
                )
                pygame.draw.line(layer, (244, 214, 137, alpha), tail, local, 2)
                pygame.draw.circle(layer, (255, 249, 220, alpha), local, 2)
                continue
            x = effect.x + (effect.target_x - effect.x) * 0.72
            y = effect.y + (effect.target_y - effect.y) * 0.72
            sx, sy = self.camera.world_to_screen(x, y)
            if not BATTLE_RECT.inflate(80, 80).collidepoint(sx, sy):
                continue
            local = (sx - BATTLE_RECT.x, sy - BATTLE_RECT.y)
            art_kind = "impact" if effect.kind == "death" else effect.kind
            size = max(8, round(effect.size * (0.76 + progress * 0.42)))
            sprite = self.art.effect(art_kind, size)
            if sprite is not None:
                sprite = sprite.copy()
                sprite.set_alpha(alpha)
                layer.blit(sprite, sprite.get_rect(center=local))
            else:
                fallback = self.theme.enemy if effect.kind == "death" else self.theme.warning
                pygame.draw.circle(layer, (*fallback, alpha), local, max(2, size // 4), 2)
            if effect.kind == "death":
                ring_color = self.theme.enemy if effect.faction == 0 else self.theme.ally
                pygame.draw.circle(
                    layer,
                    (*ring_color, alpha),
                    local,
                    max(5, round(size * 0.46)),
                    2,
                )
        self.canvas.blit(layer, BATTLE_RECT.topleft)

    def _draw_result_cinematic(self) -> None:
        assert self.world is not None and self.camera is not None
        overlay = pygame.Surface(BATTLE_RECT.size, pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 58))
        self.canvas.blit(overlay, BATTLE_RECT.topleft)
        if self.cinematic_target is not None:
            sx, sy = self.camera.world_to_screen(*self.cinematic_target)
            pulse = 24 + round(8 * math.sin(self.result_cinematic_time * 11))
            pygame.draw.circle(self.canvas, self.theme.warning, (sx, sy), pulse, 2)
            pygame.draw.circle(self.canvas, self.theme.ink, (sx, sy), pulse + 5, 1)
        labels = {
            GameOutcome.PLAYER_WIN: ("敌方将领阵亡", self.theme.success),
            GameOutcome.ENEMY_WIN: ("我方将领阵亡", self.theme.enemy),
            GameOutcome.DRAW: ("双方将领同时阵亡", self.theme.warning),
        }
        label, color = labels[self.world.outcome]
        banner = pygame.Rect(BATTLE_RECT.centerx - 150, BATTLE_RECT.top + 22, 300, 42)
        pygame.draw.rect(self.canvas, self.theme.surface, banner, border_radius=6)
        pygame.draw.rect(self.canvas, color, banner, 2, border_radius=6)
        self._blit_text(label, banner.centerx, banner.centery, 18, color, True, center=True)

    def _draw_hud(self) -> None:
        assert self.world is not None
        topbar = pygame.Rect(0, 0, 1280, 40)
        right_rail = pygame.Rect(960, 40, 320, 680)
        order_deck = pygame.Rect(0, 588, 960, 132)
        pygame.draw.rect(self.canvas, self.theme.surface, topbar)
        pygame.draw.rect(self.canvas, self.theme.surface, right_rail)
        pygame.draw.rect(self.canvas, self.theme.surface, order_deck)
        pygame.draw.line(self.canvas, self.theme.border, (0, 39), (1280, 39), 1)
        pygame.draw.line(self.canvas, self.theme.border, (959, 40), (959, 720), 1)
        pygame.draw.line(self.canvas, self.theme.border, (0, 588), (960, 588), 1)

        # Top status strip: one scan line for time, simulation and global controls.
        pygame.draw.rect(self.canvas, self.theme.warning, (12, 9, 3, 22))
        self._blit_text("前线指挥部", 24, 20, 15, self.theme.ink, True, center_y=True)
        minutes, seconds = divmod(self.world.tick // self.world.balance.world.simulation_hz, 60)
        self._blit_text(
            f"{minutes:02d}:{seconds:02d}", 136, 20, 14, self.theme.warning, True, center_y=True
        )
        self._blit_text(
            "已暂停" if self.paused else f"战局 {self.speed:g}×",
            202,
            20,
            13,
            self.theme.warning if self.paused else self.theme.ink,
            True,
            center_y=True,
        )
        online_ready = self.online_enabled and self.deepseek.available
        pygame.draw.circle(
            self.canvas, self.theme.success if online_ready else self.theme.muted, (292, 20), 4
        )
        self._blit_text(
            "在线军令" if online_ready else "离线军令",
            304,
            20,
            12,
            self.theme.muted,
            center_y=True,
        )
        p95 = float(np.percentile(self.tick_timings, 95)) if self.tick_timings else 0.0
        self._blit_text(
            f"FPS {self.clock.get_fps():.0f} · SIM {p95:.1f}ms",
            402,
            20,
            11,
            self.theme.muted if p95 <= 10 else self.theme.warning,
            center_y=True,
            max_width=118,
        )
        actions: list[tuple[str, Callable[[], None], int]]
        if self.scene == "replay":
            actions = [
                ("视角", self._cycle_view, 52),
                (f"{self.speed:g}×", self._cycle_speed, 48),
                ("暂停" if not self.paused else "继续", self._toggle_pause, 54),
                ("帮助", lambda: setattr(self, "help_open", True), 50),
                ("全屏", self._toggle_fullscreen, 50),
            ]
        else:
            actions = [
                ("暂停" if not self.paused else "继续", self._toggle_pause, 54),
                (f"{self.speed:g}×", self._cycle_speed, 46),
                ("自动开" if self.auto_command else "自动关", self._toggle_auto_command, 56),
                ("军令", self._begin_text_input, 50),
                ("保存", self._quick_save, 46),
                ("载入", self.load_battle, 46),
                ("帮助", lambda: setattr(self, "help_open", True), 46),
                ("全屏", self._toggle_fullscreen, 46),
            ]
        button_x = 530 if self.scene != "replay" else 566
        for label, action, width in actions:
            self._draw_compact_button(
                pygame.Rect(button_x, 5, width, 30),
                label,
                action,
                accent=label in {"继续", "自动开"},
            )
            button_x += width + 5

        # Right command rail.
        commander_panel = pygame.Rect(972, 52, 296, 124)
        log_panel = pygame.Rect(972, 188, 296, 330)
        for panel in (commander_panel, log_panel):
            pygame.draw.rect(self.canvas, self.theme.background, panel, border_radius=8)
            pygame.draw.rect(self.canvas, self.theme.border, panel, 1, border_radius=8)
        view = self.view_faction
        side = view if view is not None else Faction.PLAYER
        side_label = "我军" if side == Faction.PLAYER else "敌军"
        self._blit_text(f"{side_label}将领", 988, 68, 15, self.theme.ink, True)
        commander = self.world.commander_index(side)
        if commander is None:
            hp, max_hp = 0.0, 1.0
        else:
            hp = float(self.world.units.hp[commander])
            max_hp = float(self.world.units.max_hp[commander])
        self._blit_text(
            "阵亡" if hp <= 0 else "指挥中",
            1250,
            69,
            12,
            self.theme.enemy if hp <= 0 else self.theme.success,
            True,
            center=True,
        )
        hp_track = pygame.Rect(988, 94, 264, 8)
        pygame.draw.rect(self.canvas, self.theme.surface_high, hp_track, border_radius=4)
        pygame.draw.rect(
            self.canvas,
            self.theme.success if hp / max_hp > 0.35 else self.theme.enemy,
            (
                hp_track.x,
                hp_track.y,
                round(hp_track.width * max(0.0, hp / max_hp)),
                hp_track.height,
            ),
            border_radius=4,
        )
        self._blit_text(f"生命 {max(0, round(hp))}/{round(max_hp)}", 988, 111, 11, self.theme.muted)
        alive = len(self.world.units.active(side))
        known_enemy = (
            len(self.world.observation(side).visible_enemies)
            if view is not None
            else len(self.world.units.active(Faction.ENEMY))
        )
        guards = int(
            np.sum(
                self.world.units.alive[: self.world.units.count]
                & (self.world.units.faction[: self.world.units.count] == int(side))
                & (self.world.units.kind[: self.world.units.count] == int(UnitKind.GUARD))
            )
        )
        self._blit_text(f"兵力 {alive}", 988, 141, 13, self.theme.ally, True)
        self._blit_text(f"敌情 {known_enemy}", 1078, 141, 13, self.theme.enemy, True)
        self._blit_text(f"护卫 {guards}/4", 1170, 141, 13, self.theme.warning, True)

        self._blit_text("指挥记录", 988, 204, 15, self.theme.ink, True)
        self._blit_text("最新战场反馈", 1162, 205, 10, self.theme.muted, max_width=90)
        pygame.draw.line(self.canvas, self.theme.border, (988, 228), (1252, 228), 1)
        history_lines = 6 if self.ambiguity_candidates else 10
        visible_messages = self.messages[-history_lines:]
        for line, (message, color) in enumerate(visible_messages):
            row_y = 238 + line * 27
            pygame.draw.circle(self.canvas, color, (990, row_y + 7), 3)
            self._blit_text(message, 1001, row_y, 11, self.theme.ink, max_width=251)
        if self.ambiguity_candidates:
            self._blit_text("请选择命令解释", 988, 414, 12, self.theme.warning, True)
            for index, command in enumerate(self.ambiguity_candidates):
                payload = command.payload
                detail: str = payload.kind
                if isinstance(payload, MovePayloadV1):
                    detail = (
                        f"{payload.kind} · 组{payload.selection.group_id}"
                        if payload.selection.group_id
                        else f"{payload.kind} · ({payload.target.x:.0f},{payload.target.y:.0f})"
                    )
                self._draw_compact_button(
                    pygame.Rect(986, 438 + index * 25, 268, 22),
                    f"{index + 1}  {detail}",
                    partial(self._choose_ambiguity, index),
                    index == 0,
                )

        self._draw_minimap()
        if self.camera is not None:
            zoom_panel = pygame.Rect(900, 52, 48, 118)
            pygame.draw.rect(self.canvas, self.theme.surface, zoom_panel, border_radius=6)
            pygame.draw.rect(self.canvas, self.theme.border, zoom_panel, 1, border_radius=6)
            self._draw_compact_button(
                pygame.Rect(907, 59, 34, 32),
                "+",
                partial(self._request_zoom, 1.28, BATTLE_RECT.center),
                True,
            )
            self._blit_text(f"{self.camera.zoom:.2f}", 924, 108, 10, self.theme.ink, center=True)
            self._draw_compact_button(
                pygame.Rect(907, 131, 34, 32),
                "−",
                partial(self._request_zoom, 1 / 1.28, BATTLE_RECT.center),
            )

        if self.scene == "replay":
            self._draw_replay_controls()
            return

        # Bottom order deck: selection at left, command entry at right.
        selection_panel = pygame.Rect(12, 600, 304, 108)
        command_panel = pygame.Rect(328, 600, 620, 108)
        for panel in (selection_panel, command_panel):
            pygame.draw.rect(self.canvas, self.theme.background, panel, border_radius=8)
            pygame.draw.rect(self.canvas, self.theme.border, panel, 1, border_radius=8)
        selected_indices = [self.world.units.index_of(entity_id) for entity_id in self.selected_ids]
        selected = [
            index
            for index in selected_indices
            if index is not None and self.world.units.alive[index]
        ]
        self._blit_text(f"部队选择 · {len(selected)}", 26, 614, 14, self.theme.ink, True)
        if selected:
            average_hp = float(
                np.mean(
                    [
                        self.world.units.hp[index] / self.world.units.max_hp[index]
                        for index in selected
                    ]
                )
            )
            average_stamina = float(
                np.mean([self.world.units.stamina[index] for index in selected])
            )
            self._blit_text(
                f"生命 {average_hp * 100:.0f}%  ·  耐力 {average_stamina:.0f}",
                26,
                639,
                11,
                self.theme.muted,
            )
        else:
            self._blit_text("点选、框选或使用下方快捷选择", 26, 639, 11, self.theme.muted)
        quick_select: list[tuple[str, Callable[[], None]]] = [
            ("全军", self._select_all_player),
            ("将领", partial(self._select_kind, UnitKind.COMMANDER)),
            ("工兵", partial(self._select_kind, UnitKind.ENGINEER)),
            ("侦察", partial(self._select_kind, UnitKind.SCOUT)),
            ("预备", partial(self._select_kind, UnitKind.RECRUIT)),
        ]
        for index, (label, action) in enumerate(quick_select):
            self._draw_compact_button(
                pygame.Rect(24 + index * 56, 665, 51, 31), label, action, index == 0
            )

        selected_recruits = [
            index for index in selected if self.world.units.kind[index] == int(UnitKind.RECRUIT)
        ]
        if selected_recruits:
            self._blit_text(
                f"训练预备兵 · 已选 {len(selected_recruits)} 人",
                342,
                614,
                13,
                self.theme.ink,
                True,
            )
            conversion_options: list[
                tuple[str, Literal["infantry", "scout", "engineer", "assassin"]]
            ] = [
                ("训练步兵", "infantry"),
                ("训练侦察", "scout"),
                ("训练工兵", "engineer"),
                ("训练刺客", "assassin"),
            ]
            for index, (label, kind) in enumerate(conversion_options):
                self._draw_compact_button(
                    pygame.Rect(342 + index * 147, 636, 138, 38),
                    label,
                    partial(self._convert_selected, kind),
                    index == 0,
                )
            self._blit_text(
                "选择预备兵后点击训练；初始预备队和村庄新兵都可转换",
                342,
                686,
                11,
                self.theme.muted,
                max_width=590,
            )
            return

        self._blit_text("文字军令", 342, 614, 13, self.theme.ink, True)
        box = pygame.Rect(342, 636, 484, 38)
        send = pygame.Rect(836, 636, 98, 38)
        pygame.draw.rect(self.canvas, self.theme.surface, box, border_radius=6)
        pygame.draw.rect(
            self.canvas,
            self.theme.primary if self.typing else self.theme.border,
            box,
            2 if self.typing else 1,
            border_radius=6,
        )
        if self.typing:
            self._blit_text(
                self.command_input + self.composition + "│",
                box.x + 12,
                box.centery,
                14,
                self.theme.ink,
                center_y=True,
                max_width=box.width - 24,
            )
            self._draw_compact_button(send, "执行军令", self._submit_text, True)
            if self.suggestions:
                suggestion_height = len(self.suggestions) * 27 + 6
                suggestion_box = pygame.Rect(
                    box.x, box.y - suggestion_height - 6, box.width, suggestion_height
                )
                pygame.draw.rect(
                    self.canvas, self.theme.background, suggestion_box, border_radius=6
                )
                pygame.draw.rect(self.canvas, self.theme.border, suggestion_box, 1, border_radius=6)
                for index, text in enumerate(self.suggestions):
                    option = pygame.Rect(
                        suggestion_box.x + 3,
                        suggestion_box.y + 3 + index * 27,
                        suggestion_box.width - 6,
                        25,
                    )
                    if index == self.selected_suggestion:
                        pygame.draw.rect(self.canvas, self.theme.primary, option, border_radius=4)
                    self._blit_text(
                        text,
                        option.x + 9,
                        option.centery,
                        12,
                        self.theme.ink,
                        center_y=True,
                        max_width=option.width - 18,
                    )
                    self.buttons.append(Button(option, text, partial(self._pick_suggestion, index)))
        else:
            self._blit_text(
                "点击输入命令，例如：工兵队在中央架桥",
                box.x + 12,
                box.centery,
                13,
                self.theme.muted,
                center_y=True,
                max_width=box.width - 24,
            )
            self.buttons.append(Button(box, "输入文字军令", self._begin_text_input))
            self._draw_compact_button(send, "输入军令", self._begin_text_input, True)
        self._blit_text(
            "左键选择 · 右键行动/建造 · 滚轮缩放 · 中键拖动地图",
            342,
            686,
            11,
            self.theme.muted,
            max_width=590,
        )

    def _draw_hud_legacy(self) -> None:
        assert self.world is not None
        pygame.draw.rect(self.canvas, self.theme.surface, (0, 0, 1280, 40))
        pygame.draw.rect(self.canvas, self.theme.surface, (960, 40, 320, 680))
        pygame.draw.rect(self.canvas, self.theme.surface, (0, 588, 960, 132))
        pygame.draw.rect(self.canvas, self.theme.surface_high, (972, 52, 296, 134), border_radius=8)
        pygame.draw.rect(self.canvas, self.theme.background, (972, 200, 296, 326), border_radius=8)
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
            max_width=245,
        )

        actions: list[tuple[str, Callable[[], None], int]]
        if self.scene == "replay":
            actions = [
                ("视角", self._cycle_view, 54),
                (f"{self.speed:g}×", self._cycle_speed, 50),
                ("暂停" if not self.paused else "继续", self._toggle_pause, 54),
                ("帮助", lambda: setattr(self, "help_open", True), 50),
                ("全屏", self._toggle_fullscreen, 50),
            ]
        else:
            actions = [
                ("暂停" if not self.paused else "继续", self._toggle_pause, 54),
                (f"{self.speed:g}×", self._cycle_speed, 48),
                ("军令", self._begin_text_input, 54),
                ("保存", self._quick_save, 48),
                ("载入", self.load_battle, 48),
                ("帮助", lambda: setattr(self, "help_open", True), 46),
                ("全屏", self._toggle_fullscreen, 46),
            ]
        button_x = 572
        for index, (label, action, width) in enumerate(actions):
            self._draw_compact_button(
                pygame.Rect(button_x, 6, width, 28),
                label,
                action,
                accent=index == 0 and self.paused,
            )
            button_x += width + 5

        if self.camera is not None:
            zoom_panel = pygame.Rect(906, 52, 46, 112)
            pygame.draw.rect(self.canvas, self.theme.surface, zoom_panel, border_radius=7)
            pygame.draw.rect(self.canvas, self.theme.border, zoom_panel, 1, border_radius=7)
            self._draw_compact_button(
                pygame.Rect(913, 59, 32, 30),
                "+",
                partial(self._request_zoom, 1.28, BATTLE_RECT.center),
                True,
            )
            self._draw_compact_button(
                pygame.Rect(913, 124, 32, 30),
                "−",
                partial(self._request_zoom, 1 / 1.28, BATTLE_RECT.center),
            )
            self._blit_text(
                f"{self.camera.zoom:.2f}×",
                zoom_panel.centerx,
                106,
                10,
                self.theme.ink,
                center=True,
                max_width=40,
            )

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
            self._blit_text(message, 980, 242 + line * 25, 13, color, max_width=272)
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
                option_rect = pygame.Rect(976, 452 + index * 24, 282, 23)
                self.buttons.append(
                    Button(option_rect, f"解释 {index + 1}", partial(self._choose_ambiguity, index))
                )

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
        box = pygame.Rect(330, 612, 510, 42)
        send = pygame.Rect(850, 612, 80, 42)
        pygame.draw.rect(self.canvas, self.theme.background, box, border_radius=6)
        pygame.draw.rect(
            self.canvas,
            self.theme.primary if self.typing else self.theme.border,
            box,
            2 if self.typing else 1,
            border_radius=6,
        )
        if self.typing:
            self._blit_text(
                self.command_input + self.composition + "│",
                box.x + 12,
                box.centery,
                15,
                self.theme.ink,
                center_y=True,
                max_width=box.width - 24,
            )
            self._draw_compact_button(send, "执行", self._submit_text, True)
            hint = "可点击候选或执行；Esc 取消输入"
            self._blit_text(hint, 330, 672, 12, self.theme.muted, max_width=600)
            # Render suggestions dropdown (above input box)
            if self.suggestions:
                sug_h = len(self.suggestions) * 26 + 4
                sug_box = pygame.Rect(box.x, box.y - sug_h - 2, box.width, sug_h)
                pygame.draw.rect(self.canvas, self.theme.background, sug_box, border_radius=4)
                pygame.draw.rect(self.canvas, self.theme.muted, sug_box, 1, border_radius=4)
                for i, text in enumerate(self.suggestions):
                    is_sel = i == self.selected_suggestion
                    if is_sel:
                        sel_rect = pygame.Rect(
                            sug_box.x + 2, sug_box.y + 2 + i * 26, sug_box.width - 4, 24
                        )
                        pygame.draw.rect(self.canvas, self.theme.primary, sel_rect, border_radius=3)
                    color = self.theme.background if is_sel else self.theme.ink
                    option_rect = pygame.Rect(
                        sug_box.x + 2, sug_box.y + 2 + i * 26, sug_box.width - 4, 24
                    )
                    self._blit_text(
                        text,
                        sug_box.x + 10,
                        sug_box.y + 4 + i * 26,
                        14,
                        color,
                        max_width=sug_box.width - 20,
                    )
                    self.buttons.append(
                        Button(option_rect, text, partial(self._pick_suggestion, i))
                    )
        else:
            self._blit_text(
                "点击输入文字军令……", box.x + 12, box.centery, 15, self.theme.muted, center_y=True
            )
            self.buttons.append(Button(box, "输入文字军令", self._begin_text_input))
            self._draw_compact_button(send, "输入", self._begin_text_input, True)
            self._blit_text(
                "左键选择 · 右键菜单 · 滚轮缩放 · 拖动缩略图",
                330,
                672,
                12,
                self.theme.muted,
                max_width=600,
            )

    def _draw_minimap(self) -> None:
        assert self.world is not None and self.camera is not None
        rect = MINIMAP_RECT
        view = self.view_faction
        fog = self.world._perception

        # Cache terrain surface - only rebuild when fog changes
        fog_tick = self.world.tick // 10
        cache_key = (int(view) if view is not None else -1, fog_tick)
        if self._minimap_cache_key != cache_key or self._minimap_cache is None:
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
            if view is not None and fog is not None:
                explored = fog.explored[int(view)]
                pixels = pixels.copy()
                pixels[~explored] = self.theme.fog_unknown
            surface = pygame.surfarray.make_surface(np.transpose(pixels, (1, 0, 2)))
            self._minimap_cache = pygame.transform.scale(surface, rect.size)
            self._minimap_cache_key = cache_key

        self._blit_text("缩略图 · 拖动 / 滚轮缩放", rect.x, rect.y - 19, 11, self.theme.muted)
        self._draw_compact_button(
            pygame.Rect(1204, rect.y - 27, 24, 23),
            "+",
            partial(self._request_zoom, 1.28, BATTLE_RECT.center),
            True,
        )
        self._draw_compact_button(
            pygame.Rect(1234, rect.y - 27, 24, 23),
            "−",
            partial(self._request_zoom, 1 / 1.28, BATTLE_RECT.center),
        )
        self.canvas.blit(self._minimap_cache, rect)
        observation = self.world.observation(view) if view is not None else None
        visible_ids = (
            {unit.entity_id for unit in observation.visible_enemies} if observation else set()
        )
        active = self.world.units.active()
        if view is not None and len(active):
            factions = self.world.units.faction[active]
            entity_ids = self.world.units.entity_id[active]
            visible_mask = factions == int(view)
            if visible_ids:
                visible_mask |= np.isin(entity_ids, np.fromiter(visible_ids, dtype=np.int32))
            active = active[visible_mask]
        if len(active):
            unit_layer = pygame.Surface(rect.size, pygame.SRCALPHA)
            xs = np.rint(
                self.world.units.x[active]
                / self.world.subpixels
                / self.world.map.width
                * (rect.width - 1)
            ).astype(np.int32)
            ys = np.rint(
                self.world.units.y[active]
                / self.world.subpixels
                / self.world.map.height
                * (rect.height - 1)
            ).astype(np.int32)
            xs = np.clip(xs, 0, rect.width - 1)
            ys = np.clip(ys, 0, rect.height - 1)
            factions = self.world.units.faction[active]
            colors = np.where(
                (factions == int(Faction.PLAYER))[:, None],
                np.asarray(self.theme.ally, dtype=np.uint8),
                np.asarray(self.theme.enemy, dtype=np.uint8),
            )
            rgb = pygame.surfarray.pixels3d(unit_layer)
            alpha = pygame.surfarray.pixels_alpha(unit_layer)
            rgb[xs, ys] = colors
            alpha[xs, ys] = 255
            del rgb, alpha
            self.canvas.blit(unit_layer, rect.topleft)
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
        clipped_camera = camera_rect.clip(rect)
        viewport_overlay = pygame.Surface(clipped_camera.size, pygame.SRCALPHA)
        viewport_overlay.fill((*self.theme.ally, 30))
        self.canvas.blit(viewport_overlay, clipped_camera)
        pygame.draw.rect(self.canvas, self.theme.ink, clipped_camera, 2)
        pygame.draw.rect(
            self.canvas,
            self.theme.primary_hover if self.minimap_dragging else self.theme.border,
            rect,
            2 if self.minimap_dragging else 1,
        )

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
        self._draw_compact_button(
            pygame.Rect(panel.right - 88, panel.top + 18, 64, 32),
            "关闭",
            lambda: setattr(self, "help_open", False),
            True,
        )

        left = [
            "选择与移动",
            "左键点选或框选；右键打开移动、攻击移动与建造菜单。",
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
            "顶栏按钮可暂停、调速、保存、载入与全屏。",
            "所有基础界面均可用鼠标操作，键盘只是快捷方式。",
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
        panel = pygame.Rect(400, 190, 480, 330)
        pygame.draw.rect(self.canvas, self.theme.surface, panel, border_radius=10)
        pygame.draw.rect(self.canvas, self.theme.border, panel, 1, border_radius=10)
        self._blit_text("战局已暂停", panel.centerx, 232, 24, self.theme.ink, True, center=True)
        options = [
            ("继续战斗", self._toggle_pause, True),
            ("保存战局", self._quick_save, False),
            ("载入存档", self.load_battle, False),
            ("切换全屏", self._toggle_fullscreen, False),
            ("返回主菜单", lambda: setattr(self, "scene", "menu"), False),
        ]
        for index, (label, action, accent) in enumerate(options):
            col, row = index % 2, index // 2
            rect = pygame.Rect(panel.x + 38 + col * 202, 274 + row * 54, 190, 40)
            self._draw_compact_button(rect, label, action, accent)
        self._blit_text(
            "断网时可继续使用鼠标、键盘和离线文字命令。",
            panel.centerx,
            478,
            13,
            self.theme.ink,
            center=True,
            max_width=420,
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
            panel.centerx,
            164,
            15,
            self.theme.muted,
            center=True,
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
        self._blit_text(
            str(stats.get("kills", [0, 0])[0]), 480, y, 15, self.theme.ink, True, center=True
        )
        self._blit_text(
            str(stats.get("kills", [0, 0])[1]), 620, y, 15, self.theme.ink, True, center=True
        )
        y += 28

        # Damage dealt
        dmg = stats.get("damage_dealt", [0.0, 0.0])
        self._blit_text("输出伤害", 310, y, 14, self.theme.muted)
        self._blit_text(f"{dmg[0]:.0f}", 480, y, 15, self.theme.ink, True, center=True)
        self._blit_text(f"{dmg[1]:.0f}", 620, y, 15, self.theme.ink, True, center=True)
        y += 28

        # Per-unit-type losses breakdown
        kind_names = {
            "recruit": "初始兵",
            "infantry": "步兵",
            "scout": "侦察兵",
            "engineer": "工兵",
            "assassin": "刺客",
            "commander": "将领",
            "guard": "护卫",
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
        mouse = self._mouse_pos
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
        max_width: int | None = None,
    ) -> None:
        value = str(text)
        font = self.fonts.get(size, bold)
        if max_width is None and not center and x >= 960:
            max_width = LOGICAL_SIZE[0] - x - 12
        if max_width is not None and font.size(value)[0] > max_width:
            suffix = "…"
            low, high = 0, len(value)
            while low < high:
                middle = (low + high + 1) // 2
                if font.size(value[:middle] + suffix)[0] <= max_width:
                    low = middle
                else:
                    high = middle - 1
            value = value[:low] + suffix if low else suffix
        if self._native_text:
            self._text_commands.append(
                TextCommand(value, x, y, size, color, bold, center, center_y)
            )
            return
        surface = self.fonts.render(value, size, color, bold)
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
        if viewport.size == LOGICAL_SIZE:
            self.screen.blit(self.canvas, viewport.topleft)
        else:
            # Nearest-neighbour preserves crisp pixel edges without increasing
            # the pixel-art layer's internal rendering resolution.
            scaled = pygame.transform.scale(self.canvas, viewport.size)
            self.screen.blit(scaled, viewport)
        if self._native_text:
            scale = viewport.width / LOGICAL_SIZE[0]
            for command in self._text_commands:
                surface = self.fonts.render(
                    command.text,
                    command.size,
                    command.color,
                    command.bold,
                    resolution_scale=scale,
                )
                rect = surface.get_rect()
                px = viewport.x + round(command.x * scale)
                py = viewport.y + round(command.y * scale)
                if command.center:
                    rect.center = (px, py)
                elif command.center_y:
                    rect.midleft = (px, py)
                else:
                    rect.topleft = (px, py)
                self.screen.blit(surface, rect)
        pygame.display.flip()
