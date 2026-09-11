from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def _load_local_env() -> None:
    """Load local.env (KEY=value) if present; never override existing env."""
    candidates = [
        Path.cwd() / "local.env",
        Path(__file__).resolve().parents[2] / "local.env",
        Path(__file__).resolve().parents[3] / "local.env",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, _, value = stripped.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
        return


def _headless(ticks: int) -> int:
    from mygame.ai import LocalStrategicAI
    from mygame.protocols import Faction
    from mygame.simulation import World

    world = World()
    ai = LocalStrategicAI(world.map.width, world.map.height, config=world.balance.ai["normal"])
    started = time.perf_counter()
    timings: list[float] = []
    for _ in range(ticks):
        if ai.ready(world.tick):
            for command in ai.decide(world.observation(Faction.ENEMY)):
                world.execute(command)
        before = time.perf_counter()
        world.step()
        timings.append((time.perf_counter() - before) * 1000)
    elapsed = time.perf_counter() - started
    timings.sort()
    result = {
        "ticks": world.tick,
        "entities": world.units.count,
        "alive": int(world.units.alive[: world.units.count].sum()),
        "elapsed_seconds": round(elapsed, 3),
        "ticks_per_second": round(ticks / max(elapsed, 1e-9), 1),
        "p95_tick_ms": round(timings[min(len(timings) - 1, int(len(timings) * 0.95))], 3),
        "checksum": world.checksum(),
    }
    print(json.dumps(result, ensure_ascii=False))
    return 0


def main() -> int:
    _load_local_env()
    from mygame.logging_setup import configure_logging

    configure_logging()
    parser = argparse.ArgumentParser(description="指挥官战术对抗")
    parser.add_argument("--headless", action="store_true", help="运行无界面模拟基准")
    parser.add_argument("--ticks", type=int, default=400, help="无界面模拟帧数")
    parser.add_argument("--smoke", action="store_true", help="使用虚拟显示启动三帧")
    parser.add_argument("--screenshot", help="将主菜单渲染到 PNG 后退出")
    args = parser.parse_args()
    if args.headless:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        return _headless(max(1, args.ticks))
    if args.smoke or args.screenshot:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    from mygame.app import GameApp

    app = GameApp()
    if args.screenshot:
        import pygame

        app._draw()
        pygame.image.save(app.canvas, args.screenshot)
        pygame.quit()
        return 0
    return app.run(max_frames=3 if args.smoke else None)


if __name__ == "__main__":
    raise SystemExit(main())
