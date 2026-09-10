from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from platformdirs import user_log_path


def configure_logging() -> None:
    root = logging.getLogger("mygame")
    if root.handlers:
        return
    root.setLevel(logging.INFO)
    try:
        directory = user_log_path("CommanderTacticalArena", "MyGame", ensure_exists=True)
        handler: logging.Handler = RotatingFileHandler(
            directory / "game.log",
            maxBytes=1_000_000,
            backupCount=2,
            encoding="utf-8",
        )
    except OSError:
        handler = logging.NullHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    root.addHandler(handler)
