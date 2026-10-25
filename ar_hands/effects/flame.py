"""Procedural fire: turbulent flames colored by temperature, plus heat haze and shockwaves.

A flame is a heat field: hot at the core, with a turbulent edge that streams
along the flow (upward for a held fireball, backward for a thrown one). The
turbulence is fractal noise sampled with a domain warp and scrolled along the
flow every frame, so tongues of fire rise, curl and break off. Heat maps to
color through a blackbody ramp (dark red -> orange -> yellow -> white), the
way real flames glow with temperature, and the result is added to the light
buffer, where lighting.py makes it bloom and light up the scene.

The field is computed on a small grid (at most 96x96) with cv2.remap
lookups into a precomputed noise texture and upscaled: flames are soft, so a
fireball costs well under 1 ms however large it is on screen.

Heat haze and shockwaves bend the camera picture itself (cv2.remap of a
region with a smooth displacement), which is what makes the air above a
flame shimmer and a blast ripple outward.
"""
from __future__ import annotations
import cv2
import numpy as np

Point = tuple[float, float]
TEXTURE = 256
EXTENT = 2.6  # The drawn square spans +-2.6 ball radii...
OFFSET = 0.9  # ...centered this far along the flow: tongues reach about 3.5 radii, the base 1.
# Heat (0..1) -> BGR. Blackbody-like: what glows at each temperature.
BLACKBODY = ((0.00, (0, 0, 0)), (0.10, (2, 4, 38)), (0.25, (6, 22, 150)), (0.40, (12, 72, 235)),
             (0.55, (32, 140, 255)), (0.70, (82, 200, 255)), (0.85, (168, 238, 255)), (1.00, (240, 250, 255)))


def _fractal_noise(size: int, rng: np.random.Generator, octaves: tuple[int, ...] = (4, 8, 16, 32)) -> np.ndarray:
    """Seamlessly tiling value noise in 0..1: random grids, cubic-upsampled, summed by octave."""
    total = np.zeros((size, size), np.float32)
    weight_sum = 0.0
    for index, cells in enumerate(octaves):
        grid = rng.random((cells, cells)).astype(np.float32)
        tiled = np.tile(grid, (3, 3))  # Neighbours on every side make the middle tile wrap seamlessly.
        upsampled = cv2.resize(tiled, (size * 3, size * 3), interpolation=cv2.INTER_CUBIC)
        weight = 0.55 ** index
        total += upsampled[size:2 * size, size:2 * size] * weight
        weight_sum += weight
    total /= weight_sum
    return cv2.normalize(total, None, 0.0, 1.0, cv2.NORM_MINMAX)


def _ramp_lut(stops: tuple) -> np.ndarray:
    """A 256-entry BGR lookup table interpolated between color stops."""
    heats = np.linspace(0.0, 1.0, 256)
    positions = [stop for stop, _ in stops]
    lut = np.stack([np.interp(heats, positions, [color[c] for _, color in stops]) for c in range(3)], axis=1)
    return np.clip(lut, 0, 255).astype(np.uint8).reshape(256, 1, 3)


class FlameField:
    """Draws fireballs and fire bursts into a light buffer."""

    def __init__(self, seed: int = 7) -> None:
        """Precompute the noise texture and the color ramp."""
        rng = np.random.default_rng(seed)
        self.noise = _fractal_noise(TEXTURE, rng)
        self.lut = _ramp_lut(BLACKBODY)
        self._grids: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def _grid(self, resolution: int) -> tuple[np.ndarray, np.ndarray]:
        """(u, v) in flame units: 1.0 = the ball's radius, centered."""
        if resolution not in self._grids:
            axis = np.linspace(-EXTENT, EXTENT, resolution, dtype=np.float32)
            self._grids[resolution] = np.meshgrid(axis, axis)
        return self._grids[resolution]

    def _sample(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Noise at texture coordinates (1.0 = one texture width), wrapping."""
        return cv2.remap(self.noise, (x * TEXTURE).astype(np.float32), (y * TEXTURE).astype(np.float32),
                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

    def heat(self, resolution: int, t: float, flow: Point, stretch: float, seed: float,
             turbulence: float = 1.0, billow: float = 0.0) -> np.ndarray:
        """The heat field (0..1+) on a resolution x resolution grid.

        ``billow`` 0 is a flame (tongues streaming along the flow); 1 is a
        lumpy, roughly round fireball, as in an explosion."""
        u, v = self._grid(resolution)
        fx, fy = flow
        # The patch sits shifted along the flow (see draw), so "along" starts below the ball.
        along = u * fx + v * fy + OFFSET * (1.0 - billow)  # Positive in the direction the flames stream.
        across = u * -fy + v * fx
        # Streaky noise, stretched along the flow and scrolled with it so features
        # rise; a warped lookup makes the streaks curl instead of sliding.
        nx = across * 0.58 + seed
        ny = -along * 0.17 + t * 0.62 + seed * 0.37
        warp = self._sample(nx * 1.9 + 0.5, ny * 1.9) - 0.5
        n = self._sample(nx + warp * 0.3 * turbulence, ny + warp * 0.3 * turbulence)
        fine = self._sample(nx * 2.6 + 0.25, ny * 2.6 + t * 0.35)
        tongues = 1.0 - np.abs(2.0 * n - 1.0)  # Ridges: where the noise crosses its middle.
        turb = 0.55 * tongues + 0.45 * fine
        ahead = np.maximum(along, 0.0)
        behind = np.minimum(along, 0.0)
        # A round base and a plume that narrows as it rises; how far each tongue
        # climbs and how wide the body bulges both flicker with the turbulence.
        width = (1.02 + 0.6 * (turb - 0.5) * turbulence) / (1.0 + 0.4 * ahead)
        reach = 0.6 + stretch * (0.6 + 2.0 * turb * turbulence)
        side = np.abs(across) / np.maximum(width, 0.05)
        heat = np.clip(1.0 - np.sqrt(side * side + (ahead / reach) ** 2 + (behind / 0.95) ** 2), 0.0, 1.0)
        heat *= 0.5 + 0.85 * turb  # Gaps and brighter streaks inside the body.
        r = np.sqrt(across * across + along * along)
        if billow > 0:
            # Cauliflower lumps: isotropic noise slowly churning outward.
            lumps = self._sample(across * 0.4 + seed, along * 0.4 + t * 0.25)
            lumps = 0.6 * lumps + 0.4 * self._sample(across * 0.9 + 0.3, along * 0.9 - t * 0.35)
            radius = 0.85 + 0.75 * (lumps - 0.5) * turbulence
            ball = np.clip(1.15 - r / np.maximum(radius, 0.05), 0.0, 1.0) * (0.45 + 0.9 * lumps)
            heat = (1.0 - billow) * heat + billow * ball
        heat = np.maximum(heat, 0.92 * np.exp(-(r / 0.34) ** 2))  # The white-hot heart.
        # Fade out toward the patch border, so no flame is ever cut off square.
        border = np.clip((EXTENT - np.maximum(np.abs(u), np.abs(v))) / 0.7, 0.0, 1.0)
        return heat * border

    def draw(self, light: np.ndarray, center: Point, radius: float, t: float, flow: Point = (0.0, -1.0),
             intensity: float = 1.0, stretch: float = 1.0, seed: float = 0.0, cool: float = 0.0,
             turbulence: float = 1.0, billow: float = 0.0, detail: float = 1.0) -> None:
        """Add a flame of ``radius`` px at ``center`` to ``light`` (``detail`` < 1: a coarser grid).

        ``cool`` 0..1 shifts the colors toward dull red (a dying blast); ``billow``
        see heat()."""
        if radius < 2 or intensity <= 0:
            return
        half = radius * EXTENT
        size = max(8, int(half * 2))
        resolution = int(np.clip(size / 3.2, 32, 96) * detail)
        heat = self.heat(resolution, t, flow, stretch, seed, turbulence, billow)
        heat = np.clip(heat * intensity * (1.0 - 0.45 * cool), 0.0, 1.0) ** 1.35
        index = (heat * 255).astype(np.uint8)
        color = cv2.LUT(cv2.merge((index, index, index)), self.lut)
        # The patch is centered OFFSET radii along the flow: tongues need room above the ball.
        shift = radius * OFFSET * (1.0 - billow)
        mid_x, mid_y = center[0] + flow[0] * shift, center[1] + flow[1] * shift
        x0, y0 = int(round(mid_x - size / 2)), int(round(mid_y - size / 2))
        add_clipped(light, cv2.resize(color, (size, size), interpolation=cv2.INTER_LINEAR), x0, y0)


def add_clipped(target: np.ndarray, patch: np.ndarray, x0: int, y0: int) -> None:
    """Saturating add of ``patch`` at (x0, y0), clipped to ``target``."""
    height, width = target.shape[:2]
    ph, pw = patch.shape[:2]
    cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(width, x0 + pw), min(height, y0 + ph)
    if cx0 >= cx1 or cy0 >= cy1:
        return
    roi = target[cy0:cy1, cx0:cx1]
    cv2.add(roi, patch[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0], dst=roi)


class Distortion:
    """Bends regions of the camera picture: heat haze above flames, blast shockwaves."""

    def __init__(self, noise: np.ndarray) -> None:
        """Shares the flame's noise texture."""
        self.noise = noise

    def _region(self, frame: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> tuple[int, int, int, int] | None:
        height, width = frame.shape[:2]
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(width, x1), min(height, y1)
        return (x0, y0, x1, y1) if x1 - x0 > 8 and y1 - y0 > 8 else None

    def haze(self, frame: np.ndarray, center: Point, radius: float, t: float, strength: float = 1.0) -> None:
        """Shimmering air above a flame (hot air refracts light)."""
        cx, cy = center
        region = self._region(frame, int(cx - radius * 1.6), int(cy - radius * 4.2),
                              int(cx + radius * 1.6), int(cy + radius * 0.6))
        if region is None or strength <= 0:
            return
        x0, y0, x1, y1 = region
        w, h = x1 - x0, y1 - y0
        sw, sh = max(4, w // 6), max(4, h // 6)
        gx, gy = np.meshgrid(np.linspace(x0, x1, sw, dtype=np.float32), np.linspace(y0, y1, sh, dtype=np.float32))
        # Ripples rise: sample noise that scrolls downward in texture space.
        nx = gx / (radius * 2.2)
        ny = gy / (radius * 2.2) + t * 1.4
        wrap = cv2.BORDER_WRAP
        dx = cv2.remap(self.noise, (nx * TEXTURE) % TEXTURE, (ny * TEXTURE) % TEXTURE, cv2.INTER_LINEAR,
                       borderMode=wrap) - 0.5
        dy = cv2.remap(self.noise, (nx * TEXTURE + 97) % TEXTURE, (ny * TEXTURE + 31) % TEXTURE,
                       cv2.INTER_LINEAR, borderMode=wrap) - 0.5
        # Strongest just above the flame, fading to nothing at the region's edges.
        rel_x = (gx - cx) / (radius * 1.6)
        rel_y = (gy - (cy - radius * 1.6)) / (radius * 2.6)
        falloff = np.clip(1.0 - rel_x * rel_x - rel_y * rel_y, 0.0, 1.0) ** 1.5
        amplitude = radius * 0.09 * strength
        dx = cv2.resize(dx * falloff * amplitude, (w, h), interpolation=cv2.INTER_LINEAR)
        dy = cv2.resize(dy * falloff * amplitude, (w, h), interpolation=cv2.INTER_LINEAR)
        self._warp(frame, region, dx, dy)

    def shockwave(self, frame: np.ndarray, center: Point, ring: float, width: float, strength: float) -> None:
        """A ring of refracted air expanding from a blast."""
        cx, cy = center
        reach = ring + width * 3
        region = self._region(frame, int(cx - reach), int(cy - reach), int(cx + reach), int(cy + reach))
        if region is None or strength <= 0 or ring < 2:
            return
        x0, y0, x1, y1 = region
        w, h = x1 - x0, y1 - y0
        sw, sh = max(4, w // 6), max(4, h // 6)
        gx, gy = np.meshgrid(np.linspace(x0, x1, sw, dtype=np.float32), np.linspace(y0, y1, sh, dtype=np.float32))
        ox, oy = gx - cx, gy - cy
        distance = np.sqrt(ox * ox + oy * oy) + 1e-3
        push = np.exp(-((distance - ring) / width) ** 2) * strength * width * 0.6
        dx = cv2.resize(-ox / distance * push, (w, h), interpolation=cv2.INTER_LINEAR)
        dy = cv2.resize(-oy / distance * push, (w, h), interpolation=cv2.INTER_LINEAR)
        self._warp(frame, region, dx, dy)

    @staticmethod
    def _warp(frame: np.ndarray, region: tuple[int, int, int, int], dx: np.ndarray, dy: np.ndarray) -> None:
        x0, y0, x1, y1 = region
        w, h = x1 - x0, y1 - y0
        dx += np.arange(w, dtype=np.float32)[None, :]  # Offsets -> absolute source positions, in place.
        dy += np.arange(h, dtype=np.float32)[:, None]
        roi = frame[y0:y1, x0:x1]
        roi[:] = cv2.remap(roi.copy(), dx, dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
