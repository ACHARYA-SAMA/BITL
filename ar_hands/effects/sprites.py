"""Procedural sprites (zero asset files) with optional PNG overrides.

Everything is generated with NumPy from fixed seeds, so the look is identical
on every run and machine. Drop ``fireball.png`` or ``block_<kind>.png`` into
``assets/sprites/`` to replace a procedural sprite.
"""
from __future__ import annotations
from math import ceil
from pathlib import Path
import logging
from zlib import crc32
import cv2
import numpy as np
from ..config import EffectsConfig
from .compositing import Sprite, ensure_bgra

LOGGER = logging.getLogger(__name__)
TEXEL = 16  # Minecraft-style texture resolution.

# 5-row pixel glyphs for the TNT label (1 = ink). N needs 4 columns to read as N.
_GLYPHS = {
    "T": ("111", "010", "010", "010", "010"),
    "N": ("1001", "1101", "1101", "1011", "1001"),
}


def _fractal_noise(rng: np.random.Generator, size: int, octaves: int = 4) -> np.ndarray:
    """Smooth 0..1 noise: random grids upsampled with bicubic interpolation and summed."""
    total = np.zeros((size, size), dtype=np.float32)
    amplitude, weight = 1.0, 0.0
    for octave in range(octaves):
        cells = 4 * 2 ** octave
        grid = rng.random((cells, cells)).astype(np.float32)
        total += amplitude * cv2.resize(grid, (size, size), interpolation=cv2.INTER_CUBIC)
        weight += amplitude
        amplitude *= 0.5
    total /= weight
    return np.clip((total - total.min()) / max(1e-6, float(total.max() - total.min())), 0, 1)


def _ramp(value: np.ndarray, stops: list[tuple[float, tuple[int, int, int]]]) -> np.ndarray:
    """Map 0..1 values through color stops to float32 BGR."""
    positions = [stop for stop, _ in stops]
    return np.stack([np.interp(value, positions, [color[channel] for _, color in stops])
                     for channel in range(3)], axis=-1).astype(np.float32)


def make_fireball(size: int = 192, seed: int = 7) -> np.ndarray:
    """A turbulent fireball: hot white core, noisy flame ring, tongues licking upward."""
    rng = np.random.default_rng(seed)
    coords = (np.arange(size, dtype=np.float32) + 0.5) / size * 2 - 1
    u, v = np.meshgrid(coords, coords)
    # Flames rise: compress the upper half so the boundary reaches farther up.
    stretched = np.where(v < 0, v * 0.72, v)
    radius = np.hypot(u, stretched)
    noise = _fractal_noise(rng, size)
    edge = 0.58 + 0.30 * (noise - 0.5)
    body = np.clip(1.0 - radius / edge, 0, 1) ** 0.75
    core = np.exp(-(np.hypot(u, v) / 0.30) ** 2)
    heat = np.clip(0.62 * body + 0.85 * core + 0.25 * (noise - 0.5) * body, 0, 1)
    color = _ramp(heat, [(0.0, (10, 20, 140)), (0.30, (20, 40, 215)), (0.55, (25, 130, 255)),
                         (0.80, (110, 225, 255)), (1.0, (255, 255, 255))])
    alpha = np.clip(heat * 1.7, 0, 1)
    bgra = np.dstack((color, alpha * 255)).astype(np.float32)
    bgra = cv2.GaussianBlur(bgra, (0, 0), size / 96)
    return np.clip(bgra, 0, 255).astype(np.uint8)


def make_glow(size: int, color: tuple[int, int, int], white_core: float = 0.6) -> np.ndarray:
    """Soft radial glow, white at the center, for charge-ups and lightning cores."""
    coords = (np.arange(size, dtype=np.float32) + 0.5) / size * 2 - 1
    u, v = np.meshgrid(coords, coords)
    radius = np.hypot(u, v)
    alpha = np.exp(-(radius / 0.45) ** 2) * (radius < 1)
    mix = np.clip(np.exp(-(radius / 0.18) ** 2) * white_core, 0, 1)[..., None]
    rgb = np.asarray(color, np.float32) * (1 - mix) + 255 * mix
    return np.dstack((rgb, alpha * 255)).clip(0, 255).astype(np.uint8)


def _tile(rng: np.random.Generator, bgr: tuple[int, int, int], spread: int) -> np.ndarray:
    """16x16 texture of one color with per-texel brightness jitter."""
    shade = rng.integers(-spread, spread + 1, (TEXEL, TEXEL, 1))
    return np.clip(np.asarray(bgr, np.int32) + shade, 0, 255).astype(np.uint8)


def _speckle(tile: np.ndarray, rng: np.random.Generator, bgr: tuple[int, int, int],
             count: int, cluster: int = 1) -> None:
    for _ in range(count):
        y, x = rng.integers(0, TEXEL - cluster + 1, 2)
        tile[y:y + cluster, x:x + cluster] = bgr


def block_faces(kind: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (top, side) 16x16 BGR textures, deterministic per kind."""
    rng = np.random.default_rng(crc32(kind.encode()))
    if kind == "grass":
        top = _tile(rng, (52, 160, 92), 18)
        _speckle(top, rng, (40, 128, 70), 14)
        side = _tile(rng, (66, 99, 134), 14)
        _speckle(side, rng, (50, 76, 105), 12)
        for x in range(TEXEL):
            depth = 2 + int(rng.integers(0, 3))
            side[:depth, x] = top[:depth, x]
        return top, side
    if kind == "stone":
        stone = _tile(rng, (128, 128, 128), 14)
        _speckle(stone, rng, (98, 98, 100), 10, 2)
        _speckle(stone, rng, (150, 150, 150), 8)
        return stone, stone
    if kind == "planks":
        planks = _tile(rng, (78, 130, 162), 8)
        for row in range(0, TEXEL, 4):
            planks[row + 3] = (52, 92, 118)  # Seam under each board.
            seam = 7 if (row // 4) % 2 else 15
            planks[row:row + 3, seam] = (58, 100, 128)
            streak = row + int(rng.integers(0, 3))
            start = int(rng.integers(0, 10))
            planks[streak, start:start + 5] = (68, 117, 147)
        return planks, planks
    if kind == "tnt":
        side = _tile(rng, (40, 52, 212), 10)
        side[:, ::3] = np.clip(side[:, ::3].astype(np.int32) - 30, 0, 255).astype(np.uint8)
        side[5:11] = (232, 232, 232)
        for letter, x0 in (("T", 2), ("N", 6), ("T", 11)):
            for dy, row in enumerate(_GLYPHS[letter]):
                for dx, ink in enumerate(row):
                    if ink == "1":
                        side[6 + dy, x0 + dx] = (20, 20, 20)
        top = _tile(rng, (105, 160, 190), 10)
        top[5:11, 5:11] = (36, 48, 150)
        top[7:9, 7:9] = (20, 20, 20)
        return top, side
    if kind == "diamond_ore":
        stone = _tile(rng, (124, 124, 124), 12)
        for _ in range(6):
            y, x = rng.integers(1, TEXEL - 3, 2)
            stone[y:y + 2, x:x + 2] = (229, 236, 93)
            stone[y, x] = (255, 255, 230)
        return stone, stone
    raise ValueError(f"Unknown block kind: {kind}")


def crack_mask(stage: int, seed: int = 99) -> np.ndarray:
    """0/1 16x16 crack pattern; each stage adds to the previous one."""
    mask = np.zeros((TEXEL, TEXEL), dtype=np.float32)
    rng = np.random.default_rng(seed)
    for _ in range(2 * max(0, stage)):
        y, x = 8 + rng.integers(-3, 4), 8 + rng.integers(-3, 4)
        dy, dx = rng.choice((-1, 0, 1)), rng.choice((-1, 1))
        for _ in range(3 + 2 * stage):
            mask[int(np.clip(y, 0, 15)), int(np.clip(x, 0, 15))] = 1
            x += dx
            y += dy if rng.random() < 0.6 else 0
    return mask


def iso_cube(top: np.ndarray, side: np.ndarray, width: int, crack_stage: int = 0) -> np.ndarray:
    """Pixel-art isometric cube (BGRA, width x width): three shaded faces.

    Each output pixel is mapped back to texture coordinates of the face it lies
    on (solving a 2x2 system per face), which gives seamless edges and crisp
    nearest-neighbor texels, unlike warping three images and hoping they meet.
    """
    if crack_stage:
        darken = 1.0 - 0.7 * crack_mask(crack_stage)[..., None]
        top = (top * darken).astype(np.uint8)
        side = (side * darken).astype(np.uint8)
    w = float(width)
    ys, xs = np.mgrid[0:width, 0:width].astype(np.float32) + 0.5
    out = np.zeros((width, width, 4), dtype=np.uint8)
    faces = (  # (texture, shade, origin, edge s, edge t); top is lightest, right darkest.
        (top, 1.0, (w / 2, 0.0), (w / 2, w / 4), (-w / 2, w / 4)),
        (side, 0.80, (0.0, w / 4), (w / 2, w / 4), (0.0, w / 2)),
        (side, 0.60, (w / 2, w / 2), (w / 2, -w / 4), (0.0, w / 2)),
    )
    taken = np.zeros((width, width), dtype=bool)
    for texture, shade, origin, edge_s, edge_t in faces:
        inverse = np.linalg.inv(np.array([[edge_s[0], edge_t[0]], [edge_s[1], edge_t[1]]]))
        dx, dy = xs - origin[0], ys - origin[1]
        s = inverse[0, 0] * dx + inverse[0, 1] * dy
        t = inverse[1, 0] * dx + inverse[1, 1] * dy
        inside = (s >= 0) & (s < 1) & (t >= 0) & (t < 1) & ~taken
        texel_x = np.clip((s[inside] * TEXEL).astype(np.int32), 0, TEXEL - 1)
        texel_y = np.clip((t[inside] * TEXEL).astype(np.int32), 0, TEXEL - 1)
        out[inside, :3] = (texture[texel_y, texel_x] * shade).astype(np.uint8)
        out[inside, 3] = 255
        taken |= inside
    return out


def load_user_sprite(path: Path) -> np.ndarray | None:
    """Load a PNG (any channel count) as BGRA, or warn and return None."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)  # Unicode-safe on Windows.
        image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError("not a readable image")
        if image.dtype != np.uint8:
            image = cv2.convertScaleAbs(image, alpha=255.0 / max(1, int(image.max())))
        return ensure_bgra(image)
    except (OSError, ValueError, cv2.error) as exc:
        LOGGER.warning("Ignoring sprite %s (%s); using the procedural one.", path, exc)
        return None


class SpriteLibrary:
    """All sprites the effects need, procedural unless a user PNG overrides them.

    Everything is handed out as a prepared ``Sprite`` (premultiplied once), so
    drawing never repeats that work per frame.
    """

    def __init__(self, config: EffectsConfig, folder: Path | None = None) -> None:
        """Generate (or load) sprites once; block images are cached per size and crack stage."""
        self.config = config
        self.folder = folder
        user_fireball = self._user("fireball.png")
        frames = ([user_fireball] if user_fireball is not None
                  else [make_fireball(seed=seed) for seed in (3, 7, 11, 19)])
        self.fireball_frames = [Sprite.from_image(frame) for frame in frames]
        self.fire_glow = Sprite.from_image(make_glow(128, (40, 150, 255)))
        self.blue_glow = Sprite.from_image(make_glow(128, (255, 170, 60)))
        self.white_glow = Sprite.from_image(make_glow(64, (255, 255, 255), white_core=1.0))
        self.purple_glow = Sprite.from_image(make_glow(128, (255, 70, 200)))
        self.red_glow = Sprite.from_image(make_glow(128, (40, 40, 255)))
        # Imported here: rasenshuriken.py imports this module for its own sprites.
        from .rasenshuriken import make_rasengan_core, make_shuriken_blades
        self.rasen_blades = Sprite.from_image(make_shuriken_blades())
        self.rasen_core = Sprite.from_image(make_rasengan_core())
        from .jjk import make_purple_core
        self.purple_core = Sprite.from_image(make_purple_core())
        self._faces = {kind: block_faces(kind) for kind in config.block_kinds}
        self._user_blocks = {kind: self._user(f"block_{kind}.png") for kind in config.block_kinds}
        self._cache: dict[tuple[str, int, int], Sprite] = {}

    def _user(self, name: str) -> np.ndarray | None:
        if self.folder is None:
            return None
        path = self.folder / name
        return load_user_sprite(path) if path.is_file() else None

    def fireball(self, index: int) -> Sprite:
        """Animation frame ``index`` (wraps); cycling frames makes the fire flicker."""
        return self.fireball_frames[index % len(self.fireball_frames)]

    def block(self, kind: str, width: int, crack_stage: int = 0) -> Sprite:
        """A width x width cube sprite, cached (widths are quantized by the caller)."""
        key = (kind, int(width), int(crack_stage))
        sprite = self._cache.get(key)
        if sprite is None:
            if kind not in self._faces:
                raise ValueError(f"Unknown block kind: {kind}")
            user = self._user_blocks[kind]
            if user is not None:
                sprite = cv2.resize(user, (width, width), interpolation=cv2.INTER_NEAREST)
                if crack_stage:
                    darken = 1 - 0.7 * cv2.resize(crack_mask(crack_stage), (width, width),
                                                  interpolation=cv2.INTER_NEAREST)
                    sprite[..., :3] = (sprite[..., :3] * darken[..., None]).astype(np.uint8)
            else:
                top, side = self._faces[kind]
                sprite = iso_cube(top, side, width, crack_stage)
            sprite = self._cache[key] = Sprite.from_image(sprite)
        return sprite

    def icon(self, kind: str, width: int = 40) -> Sprite:
        """HUD icon of a block kind."""
        return self.block(kind, int(ceil(width / 4) * 4))

    def palette(self, kind: str) -> np.ndarray:
        """Representative colors of a block (for break chunks), as an (n, 3) uint8 array."""
        top, side = self._faces[kind]
        colors = np.concatenate((top.reshape(-1, 3), side.reshape(-1, 3)))
        return colors[:: max(1, len(colors) // 24)]
