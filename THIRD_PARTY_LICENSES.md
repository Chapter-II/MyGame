# Third-party software and content

The game code renders its unit, terrain, facility and UI graphics procedurally. It does not redistribute third-party game art or audio.

## Noto Sans SC

- Source: Google Fonts, `ofl/notosanssc`
- Copyright: Google and the Noto project authors
- License: SIL Open Font License 1.1
- Files: `assets/fonts/NotoSansSC.ttf`, `assets/fonts/OFL.txt`
- Use: Simplified Chinese UI text in packaged Windows and Linux builds

## Runtime libraries

The Python packages declared in `pyproject.toml` retain their respective licenses. Release automation preserves package metadata and includes the application source notice. Notable direct dependencies are pygame-ce (LGPL-2.1-or-later), NumPy (BSD-3-Clause), Pydantic (MIT), HTTPX (BSD-3-Clause), MessagePack for Python (Apache-2.0), python-zstandard (BSD-3-Clause), platformdirs (MIT), faster-whisper (MIT) and sounddevice (MIT).
