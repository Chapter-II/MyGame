from __future__ import annotations

import math
import os
import struct
import wave
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")


ROOT = Path(__file__).resolve().parents[1]
PIXEL = ROOT / "assets" / "generated"
AUDIO = ROOT / "assets" / "audio"


def tone(path: Path, notes: list[tuple[float, float]], volume: float = 0.24) -> None:
    sample_rate = 22_050
    frames: list[bytes] = []
    for frequency, duration in notes:
        count = round(sample_rate * duration)
        for index in range(count):
            envelope = min(1.0, index / 80) * max(0.0, 1 - index / count)
            sample = math.sin(2 * math.pi * frequency * index / sample_rate)
            frames.append(struct.pack("<h", round(sample * envelope * volume * 32767)))
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(b"".join(frames))


def sprites() -> None:
    import pygame

    pygame.display.init()
    sheet = pygame.Surface((7 * 16, 2 * 16), pygame.SRCALPHA)
    colors = ((44, 211, 225), (255, 112, 102))
    for faction, color in enumerate(colors):
        for kind in range(7):
            x, y = kind * 16, faction * 16
            if faction == 0:
                pygame.draw.circle(sheet, color, (x + 8, y + 8), 5)
                pygame.draw.rect(sheet, (7, 10, 16, 0), (x + 6, y + 3, 4, 2))
            else:
                pygame.draw.polygon(
                    sheet, color, ((x + 8, y + 2), (x + 14, y + 8), (x + 8, y + 14), (x + 2, y + 8))
                )
            if kind == 5:
                pygame.draw.circle(sheet, (245, 181, 57), (x + 8, y + 8), 7, 2)
            elif kind == 2:
                pygame.draw.line(sheet, (235, 242, 255), (x + 4, y + 12), (x + 12, y + 4), 1)
            elif kind == 3:
                pygame.draw.rect(sheet, (235, 242, 255), (x + 6, y + 5, 4, 7), 1)
            elif kind == 4:
                pygame.draw.line(sheet, (235, 242, 255), (x + 5, y + 5), (x + 11, y + 11), 1)
    pygame.image.save(sheet, PIXEL / "units.png")
    terrain = pygame.Surface((6 * 16, 16))
    palette = ((50, 67, 65), (59, 79, 61), (37, 62, 52), (63, 68, 62), (31, 65, 82), (80, 78, 67))
    for index, color in enumerate(palette):
        pygame.draw.rect(terrain, color, (index * 16, 0, 16, 16))
        for offset in range(2, 15, 4):
            terrain.set_at(
                (index * 16 + offset, (offset * 3 + index) % 16),
                tuple(min(255, c + 10) for c in color),
            )
    pygame.image.save(terrain, PIXEL / "terrain.png")
    pygame.display.quit()


def main() -> None:
    PIXEL.mkdir(parents=True, exist_ok=True)
    AUDIO.mkdir(parents=True, exist_ok=True)
    sprites()
    tone(AUDIO / "command.wav", [(660, 0.05), (880, 0.07)])
    tone(AUDIO / "impact.wav", [(130, 0.05)], 0.18)
    tone(AUDIO / "complete.wav", [(440, 0.06), (660, 0.06), (880, 0.09)])
    tone(AUDIO / "victory.wav", [(392, 0.09), (523, 0.09), (659, 0.16)])


if __name__ == "__main__":
    main()
