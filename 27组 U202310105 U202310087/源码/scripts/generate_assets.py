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


def _paint_commander_sprites() -> None:
    """Replace commander cells (col 5) in units_atlas_v2.png with naiwa & doubao."""
    import pygame

    if pygame.display.get_surface() is None:
        pygame.display.init()
        pygame.display.set_mode((1, 1))
    atlas_path = PIXEL / "units_atlas_v2.png"
    if not atlas_path.exists():
        return
    atlas = pygame.image.load(str(atlas_path)).convert_alpha()
    cell_w = atlas.get_width() // 7
    cell_h = atlas.get_height() // 2

    # Clear both commander cells
    for row in range(2):
        region = pygame.Rect(5 * cell_w, row * cell_h, cell_w, cell_h)
        atlas.fill((0, 0, 0, 0), region)

    # ── 奶娃 (row 0): yellow chubby baby dragon ──
    ox0 = 5 * cell_w  # origin x for column 5
    oy0 = 0  # origin y for row 0
    cx = ox0 + cell_w // 2  # center x within cell

    YELLOW = (255, 210, 50)
    YELLOW_DARK = (200, 160, 30)
    YELLOW_DEEP = (160, 120, 20)
    BELLY = (255, 230, 120)
    BLUSH = (255, 150, 120)
    EYE_WHITE = (255, 255, 255)
    EYE_BLACK = (30, 30, 30)
    MOUTH = (200, 80, 60)
    HORN = (255, 200, 80)
    HORN_DARK = (200, 150, 40)

    # Head (large, ~55% of visual)
    head_cy = oy0 + 130
    head_rx, head_ry = 88, 82
    pygame.draw.ellipse(atlas, YELLOW_DARK, (cx - head_rx - 3, head_cy - head_ry - 3, (head_rx + 3) * 2, (head_ry + 3) * 2))
    pygame.draw.ellipse(atlas, YELLOW, (cx - head_rx, head_cy - head_ry, head_rx * 2, head_ry * 2))

    # Dragon horns (two small triangles on top)
    for hx_sign in (-1, 1):
        hx = cx + hx_sign * 38
        horn_base = head_cy - head_ry + 8
        pygame.draw.polygon(atlas, HORN_DARK, [(hx - 8, horn_base + 4), (hx + 8, horn_base + 4), (hx + hx_sign * 2, horn_base - 28)])
        pygame.draw.polygon(atlas, HORN, [(hx - 5, horn_base + 2), (hx + 5, horn_base + 2), (hx + hx_sign * 1, horn_base - 24)])

    # Body (chubby oval below head)
    body_cy = oy0 + 275
    body_rx, body_ry = 70, 62
    pygame.draw.ellipse(atlas, YELLOW_DARK, (cx - body_rx - 2, body_cy - body_ry - 2, (body_rx + 2) * 2, (body_ry + 2) * 2))
    pygame.draw.ellipse(atlas, YELLOW, (cx - body_rx, body_cy - body_ry, body_rx * 2, body_ry * 2))

    # Belly patch (lighter oval)
    belly_cy = body_cy + 5
    pygame.draw.ellipse(atlas, BELLY, (cx - 38, belly_cy - 35, 76, 70))

    # Neck (connect head to body)
    pygame.draw.ellipse(atlas, YELLOW, (cx - 32, head_cy + head_ry - 30, 64, 60))

    # Arms (short stubby)
    for arm_sign in (-1, 1):
        ax = cx + arm_sign * (body_rx + 8)
        ay = body_cy - 15
        pygame.draw.ellipse(atlas, YELLOW_DARK, (ax - 17, ay - 14, 34, 28))
        pygame.draw.ellipse(atlas, YELLOW, (ax - 14, ay - 12, 28, 24))

    # Legs (short stubby)
    for leg_sign in (-1, 1):
        lx = cx + leg_sign * 32
        ly = oy0 + 330
        pygame.draw.ellipse(atlas, YELLOW_DARK, (lx - 18, ly - 10, 36, 28))
        pygame.draw.ellipse(atlas, YELLOW, (lx - 15, ly - 8, 30, 24))

    # Tail (small curve from bottom-right)
    tail_points = [(cx + 55, body_cy + body_ry - 10), (cx + 85, body_cy + body_ry + 15), (cx + 75, body_cy + body_ry + 35)]
    pygame.draw.lines(atlas, YELLOW_DARK, False, tail_points, 12)
    pygame.draw.lines(atlas, YELLOW, False, tail_points, 8)

    # Eyes (big cute round eyes)
    for eye_sign in (-1, 1):
        ex = cx + eye_sign * 30
        ey = head_cy - 12
        # White sclera
        pygame.draw.circle(atlas, EYE_WHITE, (ex, ey), 22)
        pygame.draw.circle(atlas, EYE_BLACK, (ex, ey), 22, 2)
        # Pupil (slightly off-center for cuteness)
        pygame.draw.circle(atlas, EYE_BLACK, (ex + eye_sign * 3, ey + 2), 12)
        # Eye shine
        pygame.draw.circle(atlas, EYE_WHITE, (ex + eye_sign * 1, ey - 5), 5)

    # Eyebrows (small arcs above eyes)
    for eye_sign in (-1, 1):
        bx = cx + eye_sign * 30
        by = head_cy - 36
        pygame.draw.arc(atlas, EYE_BLACK, (bx - 16, by - 6, 32, 14), 0.2, 2.94, 2)

    # Nostrils (two small dots)
    pygame.draw.circle(atlas, YELLOW_DEEP, (cx - 10, head_cy + 18), 4)
    pygame.draw.circle(atlas, YELLOW_DEEP, (cx + 10, head_cy + 18), 4)

    # Mouth (small happy curve)
    mouth_rect = (cx - 18, head_cy + 22, 36, 20)
    pygame.draw.arc(atlas, MOUTH, mouth_rect, 3.4, 6.0, 3)

    # Blush (rosy cheeks)
    for cheek_sign in (-1, 1):
        chx = cx + cheek_sign * 52
        chy = head_cy + 10
        blush_surf = pygame.Surface((30, 18), pygame.SRCALPHA)
        pygame.draw.ellipse(blush_surf, (*BLUSH, 90), (0, 0, 30, 18))
        atlas.blit(blush_surf, (chx - 15, chy - 9))

    # Small ear-like fins (dragon feature)
    for ear_sign in (-1, 1):
        ear_x = cx + ear_sign * (head_rx - 5)
        ear_y = head_cy - 30
        pygame.draw.polygon(atlas, YELLOW_DARK, [
            (ear_x, ear_y + 15),
            (ear_x + ear_sign * 20, ear_y - 12),
            (ear_x + ear_sign * 8, ear_y + 20),
        ])
        pygame.draw.polygon(atlas, YELLOW, [
            (ear_x + ear_sign * 2, ear_y + 13),
            (ear_x + ear_sign * 17, ear_y - 9),
            (ear_x + ear_sign * 7, ear_y + 18),
        ])

    # ── 豆包AI (row 1): warm-white cute girl ──
    ox1 = 5 * cell_w
    oy1 = cell_h

    SKIN = (255, 235, 220)
    SKIN_DARK = (220, 195, 175)
    HAIR = (80, 60, 50)
    HAIR_HIGHLIGHT = (110, 85, 70)
    DRESS = (160, 200, 240)
    DRESS_DARK = (120, 165, 210)
    DRESS_LIGHT = (195, 225, 255)
    BLUSH2 = (255, 190, 190)
    EYE_BROWN = (70, 50, 40)
    EYE_LIGHT = (120, 90, 70)
    SPROUT_GREEN = (100, 200, 100)
    SPROUT_DARK = (60, 150, 60)
    SPROUT_STEM = (80, 170, 80)

    # Hair (covers top of head, short style)
    hair_cy = oy1 + 120
    hair_rx, hair_ry = 82, 78
    # Hair base (larger than head)
    pygame.draw.ellipse(atlas, HAIR, (cx - hair_rx - 5, hair_cy - hair_ry - 8, (hair_rx + 5) * 2, (hair_ry + 8) * 2))

    # Face (round, sits on top of hair)
    face_cy = oy1 + 128
    face_rx, face_ry = 72, 68
    pygame.draw.ellipse(atlas, SKIN_DARK, (cx - face_rx - 2, face_cy - face_ry - 2, (face_rx + 2) * 2, (face_ry + 2) * 2))
    pygame.draw.ellipse(atlas, SKIN, (cx - face_rx, face_cy - face_ry, face_rx * 2, face_ry * 2))

    # Hair bangs (fringe over forehead)
    bangs_y = face_cy - face_ry + 5
    pygame.draw.ellipse(atlas, HAIR, (cx - 68, bangs_y - 10, 136, 55))
    # Hair highlight
    pygame.draw.ellipse(atlas, HAIR_HIGHLIGHT, (cx - 40, bangs_y - 2, 30, 18))
    pygame.draw.ellipse(atlas, HAIR_HIGHLIGHT, (cx + 15, bangs_y + 2, 25, 14))

    # Side hair (covers ears area)
    for side_sign in (-1, 1):
        sx = cx + side_sign * (face_rx - 8)
        pygame.draw.ellipse(atlas, HAIR, (sx - 22, face_cy - 40, 44, 90))

    # Body / dress
    body_top = oy1 + 210
    dress_w = 75
    # Dress trapezoid shape
    pygame.draw.polygon(atlas, DRESS_DARK, [
        (cx - 40, body_top),
        (cx + 40, body_top),
        (cx + dress_w, oy1 + 370),
        (cx - dress_w, oy1 + 370),
    ])
    pygame.draw.polygon(atlas, DRESS, [
        (cx - 37, body_top + 2),
        (cx + 37, body_top + 2),
        (cx + dress_w - 5, oy1 + 368),
        (cx - dress_w + 5, oy1 + 368),
    ])
    # Dress light stripe
    pygame.draw.polygon(atlas, DRESS_LIGHT, [
        (cx - 12, body_top + 5),
        (cx + 12, body_top + 5),
        (cx + 18, oy1 + 365),
        (cx - 18, oy1 + 365),
    ])

    # Neck
    pygame.draw.rect(atlas, SKIN, (cx - 15, face_cy + face_ry - 15, 30, 25))

    # Arms (small, coming from dress)
    for arm_sign in (-1, 1):
        ax = cx + arm_sign * 45
        ay = body_top + 25
        pygame.draw.ellipse(atlas, SKIN_DARK, (ax - 14, ay - 10, 28, 50))
        pygame.draw.ellipse(atlas, SKIN, (ax - 12, ay - 8, 24, 46))
        # Hand
        pygame.draw.circle(atlas, SKIN, (ax, ay + 38), 10)

    # Legs (barely visible under dress)
    for leg_sign in (-1, 1):
        lx = cx + leg_sign * 25
        ly = oy1 + 365
        pygame.draw.ellipse(atlas, SKIN, (lx - 10, ly, 20, 18))

    # Eyes (big cute anime-style)
    for eye_sign in (-1, 1):
        ex = cx + eye_sign * 26
        ey = face_cy - 5
        # Eye white
        pygame.draw.ellipse(atlas, EYE_WHITE, (ex - 18, ey - 20, 36, 38))
        pygame.draw.ellipse(atlas, EYE_BROWN, (ex - 18, ey - 20, 36, 38), 2)
        # Iris (large)
        pygame.draw.ellipse(atlas, EYE_BROWN, (ex - 12, ey - 14, 24, 30))
        # Inner lighter iris
        pygame.draw.ellipse(atlas, EYE_LIGHT, (ex - 7, ey - 8, 14, 18))
        # Pupil
        pygame.draw.ellipse(atlas, (30, 20, 15), (ex - 6, ey - 5, 12, 14))
        # Eye shine (two highlights)
        pygame.draw.circle(atlas, EYE_WHITE, (ex + 4, ey - 8), 5)
        pygame.draw.circle(atlas, EYE_WHITE, (ex - 3, ey + 3), 3)

    # Eyebrows (thin, gentle arcs)
    for eye_sign in (-1, 1):
        bx = cx + eye_sign * 26
        by = face_cy - 30
        pygame.draw.arc(atlas, HAIR, (bx - 14, by - 4, 28, 12), 0.3, 2.84, 2)

    # Nose (tiny dot)
    pygame.draw.circle(atlas, SKIN_DARK, (cx, face_cy + 10), 3)

    # Mouth (small sweet smile)
    pygame.draw.arc(atlas, (200, 120, 120), (cx - 12, face_cy + 14, 24, 14), 3.4, 6.0, 2)

    # Blush
    for cheek_sign in (-1, 1):
        chx = cx + cheek_sign * 42
        chy = face_cy + 8
        blush_surf = pygame.Surface((24, 14), pygame.SRCALPHA)
        pygame.draw.ellipse(blush_surf, (*BLUSH2, 100), (0, 0, 24, 14))
        atlas.blit(blush_surf, (chx - 12, chy - 7))

    # Bean sprout hair accessory (on top of head)
    sprout_base_x = cx + 15
    sprout_base_y = hair_cy - hair_ry - 5
    # Stem
    pygame.draw.line(atlas, SPROUT_STEM, (sprout_base_x, sprout_base_y), (sprout_base_x + 5, sprout_base_y - 22), 4)
    # Leaves (two small oval leaves)
    leaf_cx = sprout_base_x + 5
    leaf_cy = sprout_base_y - 25
    pygame.draw.ellipse(atlas, SPROUT_DARK, (leaf_cx - 16, leaf_cy - 10, 22, 20))
    pygame.draw.ellipse(atlas, SPROUT_GREEN, (leaf_cx - 14, leaf_cy - 8, 18, 16))
    pygame.draw.ellipse(atlas, SPROUT_DARK, (leaf_cx + 2, leaf_cy - 12, 20, 18))
    pygame.draw.ellipse(atlas, SPROUT_GREEN, (leaf_cx + 4, leaf_cy - 10, 16, 14))

    pygame.image.save(atlas, str(atlas_path))


def main() -> None:
    PIXEL.mkdir(parents=True, exist_ok=True)
    AUDIO.mkdir(parents=True, exist_ok=True)
    sprites()
    _paint_commander_sprites()
    tone(AUDIO / "command.wav", [(660, 0.05), (880, 0.07)])
    tone(AUDIO / "impact.wav", [(130, 0.05)], 0.18)
    tone(AUDIO / "complete.wav", [(440, 0.06), (660, 0.06), (880, 0.09)])
    tone(AUDIO / "victory.wav", [(392, 0.09), (523, 0.09), (659, 0.16)])


if __name__ == "__main__":
    main()
