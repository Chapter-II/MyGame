# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files


datas = collect_data_files("mygame")
datas += [("assets", "assets")]

analysis = Analysis(
    ["src/mygame/__main__.py"],
    pathex=["src"],
    binaries=[],
    datas=datas,
    hiddenimports=["pygame", "numpy", "pydantic", "httpx"],
    hookspath=[],
    runtime_hooks=[],
    excludes=["faster_whisper", "sounddevice"],
    noarchive=False,
)
archive = PYZ(analysis.pure)
executable = EXE(
    archive,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="CommanderTacticalArena",
    console=False,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
)
distribution = COLLECT(
    executable,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=True,
    name="CommanderTacticalArena",
)
