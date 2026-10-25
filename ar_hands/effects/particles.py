"""Vectorized particles: struct-of-arrays state, frame-rate independent motion.

Rendering method ("splat, then blur once"). Stamping a pre-blurred Gaussian
sprite at every particle costs O(particles x sprite area) of Python work.
Instead, each particle adds its premultiplied color as a single weighted point
into a low-resolution buffer (one ``np.bincount`` per channel), and the
buffer is blurred once with a Gaussian. Convolution is linear, so blurring a
field of points is exactly the same as stamping a Gaussian sprite at every
point. Sizes vary, so particles are grouped into a few size bins, one blur
each. The cost then depends on screen area, not particle count: about 1-2 ms
for 1000 particles. The blur is the "pre-blurred radial gradient", applied to
all particles of a bin at once.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from math import ceil, floor, pi, sqrt
import cv2
import numpy as np
from ..config import EffectsConfig

Point = tuple[float, float]


class Blend(Enum):
    """How a particle system composites onto the frame."""

    ADD = "add"  # Light: fire, sparks.
    ALPHA = "alpha"  # Matter: smoke.
    SQUARES = "squares"  # Crisp pixel chunks (block debris), alpha.
    DUST = "dust"  # Tiny motes with their own colors (the snap); vectorized, alpha.


@dataclass(frozen=True)
class ColorRamp:
    """Color and alpha over normalized age (0 = born, 1 = dead)."""

    stops: tuple[float, ...]
    colors: tuple[tuple[int, int, int], ...]
    alphas: tuple[float, ...]

    def sample(self, age: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (n, 3) float32 BGR and (n,) float32 alpha."""
        rgb = np.stack([np.interp(age, self.stops, [c[i] for c in self.colors])
                        for i in range(3)], axis=1).astype(np.float32)
        return rgb, np.interp(age, self.stops, self.alphas).astype(np.float32)


# Fire: white-yellow, orange, red, then dark smoke with alpha falling to zero.
FIRE_RAMP = ColorRamp((0.0, 0.12, 0.35, 0.65, 1.0),
                      ((255, 255, 255), (110, 235, 255), (30, 150, 255), (20, 50, 225), (30, 25, 40)),
                      (1.0, 0.95, 0.85, 0.5, 0.0))
SMOKE_RAMP = ColorRamp((0.0, 0.2, 1.0), ((80, 80, 86), (92, 92, 98), (115, 115, 120)), (0.0, 0.45, 0.0))
# Susanoo and Hollow Purple: violet spirit flames.
SPIRIT_RAMP = ColorRamp((0.0, 0.25, 0.6, 1.0), ((255, 230, 255), (255, 120, 220), (200, 40, 150), (60, 10, 40)),
                        (1.0, 0.9, 0.55, 0.0))
# Reversal Red and the Malevolent Shrine: blood-red embers.
EMBER_RAMP = ColorRamp((0.0, 0.3, 1.0), ((200, 220, 255), (40, 60, 255), (20, 10, 120)), (1.0, 0.8, 0.0))
SPARK_RAMP = ColorRamp((0.0, 0.3, 1.0), ((255, 255, 255), (255, 230, 150), (255, 140, 50)), (1.0, 0.9, 0.0))


class ParticleSystem:
    """Fixed-capacity particles; when full, new particles recycle the oldest slots."""

    def __init__(self, capacity: int, blend: Blend, ramp: ColorRamp | None = None,
                 drag: float = 0.0, gravity: float = 0.0, turbulence: float = 0.0,
                 size_growth: float = 0.0, render_scale: float = 0.5,
                 buffer_budget: int = 90_000, max_blur: float = 6.0, size_bins: int = 3,
                 max_dt: float = 0.05, seed: int | None = None) -> None:
        """Allocate every array once; the hot loop never creates per-particle objects."""
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity, self.blend, self.ramp = capacity, blend, ramp
        self.drag, self.gravity, self.turbulence = drag, gravity, turbulence
        self.size_growth, self.render_scale = size_growth, render_scale
        self.buffer_budget, self.max_blur, self.size_bins = buffer_budget, max_blur, size_bins
        self.max_dt = max_dt
        self.rng = np.random.default_rng(seed)
        self.pos = np.zeros((capacity, 2), np.float32)
        self.vel = np.zeros((capacity, 2), np.float32)
        self.life = np.zeros(capacity, np.float32)  # Seconds left; <= 0 means free.
        self.max_life = np.ones(capacity, np.float32)
        self.size = np.ones(capacity, np.float32)  # Gaussian sigma (or square half-size) in px.
        self.color_seed = np.zeros(capacity, np.float32)  # 0..1 per-particle variation.
        self.color = np.zeros((capacity, 3), np.uint8)  # Used by SQUARES only.
        self._head = 0  # Next slot in birth order, so overwriting it recycles the oldest.
        self.limit = capacity  # Usable slots; adaptive quality lowers it under load.

    @property
    def count(self) -> int:
        """Number of live particles."""
        return int(np.count_nonzero(self.life > 0))

    def clear(self) -> None:
        """Kill every particle."""
        self.life[:] = 0

    def set_limit(self, limit: int) -> None:
        """Use only the first ``limit`` slots (the arrays stay allocated, so raising
        it again later costs nothing); particles beyond the new limit vanish."""
        self.limit = max(1, min(self.capacity, int(limit)))
        self.life[self.limit:] = 0
        self._head %= self.limit

    def _slots(self, count: int) -> np.ndarray:
        count = min(count, self.limit)
        slots = (self._head + np.arange(count)) % self.limit
        self._head = (self._head + count) % self.limit
        return slots

    def emit(self, count: int, position: Point | np.ndarray, velocity: Point | np.ndarray = (0.0, 0.0),
             spread: float = 0.0, radius: float = 0.0, life: tuple[float, float] = (0.4, 0.8),
             size: tuple[float, float] = (4.0, 8.0), color: np.ndarray | None = None) -> np.ndarray:
        """Emit ``count`` particles; ``position``/``velocity`` may be per-particle (n, 2) arrays."""
        if count <= 0:
            return np.empty(0, np.int64)
        slots = self._slots(count)
        n = len(slots)
        rng = self.rng
        position = np.broadcast_to(np.asarray(position, np.float32), (n, 2))
        velocity = np.broadcast_to(np.asarray(velocity, np.float32), (n, 2))
        offsets = rng.normal(0.0, radius, (n, 2)) if radius else 0.0
        self.pos[slots] = position + offsets
        self.vel[slots] = velocity + (rng.normal(0.0, spread, (n, 2)) if spread else 0.0)
        lifetimes = rng.uniform(life[0], life[1], n)
        self.life[slots] = lifetimes
        self.max_life[slots] = lifetimes
        self.size[slots] = rng.uniform(size[0], size[1], n)
        self.color_seed[slots] = rng.random(n)
        if color is not None:
            palette = np.asarray(color, np.uint8).reshape(-1, 3)
            self.color[slots] = palette[rng.integers(0, len(palette), n)]
        return slots

    def emit_radial(self, count: int, center: Point, speed: tuple[float, float],
                    life: tuple[float, float], size: tuple[float, float], radius: float = 0.0,
                    color: np.ndarray | None = None, upward_bias: float = 0.0) -> np.ndarray:
        """Emit outward in all directions (explosions, bursts)."""
        if count <= 0:
            return np.empty(0, np.int64)
        n = min(count, self.limit)
        angles = self.rng.uniform(0, 2 * pi, n)
        speeds = self.rng.uniform(speed[0], speed[1], n)
        directions = np.stack((np.cos(angles), np.sin(angles)), axis=1)
        velocity = directions * speeds[:, None] - np.array((0.0, upward_bias), np.float32)
        position = np.asarray(center, np.float32) + directions * self.rng.uniform(0, radius, (n, 1))
        return self.emit(n, position, velocity, life=life, size=size, color=color)

    def update(self, dt: float) -> None:
        """Advance live particles, independent of frame rate.

        Drag plus gravity is integrated in closed form (exponential approach to
        terminal velocity), so 60 steps of 1/60 s land exactly where 30 steps of
        1/30 s do. Turbulence is a random walk scaled by sqrt(dt), so its spread
        is the same at any frame rate.
        """
        dt = min(max(dt, 0.0), self.max_dt)
        alive = np.flatnonzero(self.life > 0)
        if not alive.size or dt == 0:
            return
        velocity = self.vel[alive]
        position = self.pos[alive]
        if self.drag:
            decay = np.exp(-self.drag * dt)
            terminal = np.array((0.0, self.gravity / self.drag), np.float32)
            position += terminal * dt + (velocity - terminal) * ((1 - decay) / self.drag)
            velocity = terminal + (velocity - terminal) * np.float32(decay)
        else:
            position += velocity * dt
            position[:, 1] += 0.5 * self.gravity * dt * dt
            velocity[:, 1] += self.gravity * dt
        if self.turbulence:
            velocity += self.rng.normal(0.0, self.turbulence * sqrt(dt), velocity.shape).astype(np.float32)
        self.vel[alive] = velocity
        self.pos[alive] = position
        self.life[alive] -= dt

    def render(self, frame: np.ndarray) -> None:
        """Composite live particles onto the BGR frame in place."""
        alive = np.flatnonzero(self.life > 0)
        if not alive.size:
            return
        if self.blend is Blend.SQUARES:
            self._render_squares(frame, alive)
        elif self.blend is Blend.DUST:
            self._render_dust(frame, alive)
        else:
            self._render_splat(frame, alive)

    def _render_squares(self, frame: np.ndarray, alive: np.ndarray) -> None:
        # Debris is few and crisp, so a short loop of filled squares is cheapest.
        remaining = np.sqrt(np.clip(self.life[alive] / self.max_life[alive], 0, 1))
        half = np.maximum(1, (self.size[alive] * remaining)).astype(np.int32)
        points = self.pos[alive].astype(np.int32)
        for (x, y), h, color in zip(points, half, self.color[alive]):
            cv2.rectangle(frame, (int(x - h), int(y - h)), (int(x + h), int(y + h)),
                          (int(color[0]), int(color[1]), int(color[2])), -1)

    def _render_dust(self, frame: np.ndarray, alive: np.ndarray) -> None:
        # Thousands of 1-3 px motes: blend them all at once with fancy indexing,
        # one pass per pixel offset, instead of a Python loop of rectangles.
        height, width = frame.shape[:2]
        fade = np.sqrt(np.clip(self.life[alive] / self.max_life[alive], 0, 1))[:, None]
        colors = self.color[alive].astype(np.float32)
        base = self.pos[alive].astype(np.int32)
        sizes = self.size[alive]
        for dx, dy, need in ((0, 0, 0.0), (1, 0, 1.5), (0, 1, 1.5), (1, 1, 2.2)):
            use = sizes >= need
            xs, ys = base[use, 0] + dx, base[use, 1] + dy
            inside = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
            xs, ys = xs[inside], ys[inside]
            weight = fade[use][inside]
            current = frame[ys, xs].astype(np.float32)
            frame[ys, xs] = (current + (colors[use][inside] - current) * weight).astype(np.uint8)

    def _render_splat(self, frame: np.ndarray, alive: np.ndarray) -> None:
        height, width = frame.shape[:2]
        age = 1.0 - np.clip(self.life[alive] / self.max_life[alive], 0, 1)
        sigma = self.size[alive] * (1.0 + self.size_growth * age)
        rgb, alpha = self.ramp.sample(age)
        rgb *= (0.8 + 0.4 * self.color_seed[alive])[:, None]  # Per-particle brightness variety.
        positions = self.pos[alive]
        pad = 3.0 * float(sigma.max())
        x0 = max(0, floor(float(positions[:, 0].min()) - pad))
        y0 = max(0, floor(float(positions[:, 1].min()) - pad))
        x1 = min(width, ceil(float(positions[:, 0].max()) + pad))
        y1 = min(height, ceil(float(positions[:, 1].max()) + pad))
        if x0 >= x1 or y0 >= y1:
            return
        roi_w, roi_h = x1 - x0, y1 - y0
        # Coarser buffers for big blurs and big areas keep the cost bounded.
        scale = min(self.render_scale, sqrt(self.buffer_budget / (roi_w * roi_h)),
                    self.max_blur / float(sigma.max()))
        scale = max(scale, 0.05)
        buf_w, buf_h = max(1, ceil(roi_w * scale)), max(1, ceil(roi_h * scale))
        bx = np.clip(((positions[:, 0] - x0) * scale).astype(np.int32), 0, buf_w - 1)
        by = np.clip(((positions[:, 1] - y0) * scale).astype(np.int32), 0, buf_h - 1)
        inside = (positions[:, 0] >= x0) & (positions[:, 0] < x1) & \
                 (positions[:, 1] >= y0) & (positions[:, 1] < y1)
        flat = by * buf_w + bx
        channels = 3 if self.blend is Blend.ADD else 4
        values = np.empty((len(alive), channels), np.float32)
        values[:, :3] = rgb * alpha[:, None]  # Premultiplied color.
        if channels == 4:
            values[:, 3] = alpha
        values[~inside] = 0
        buffer_sigma = np.maximum(sigma * scale, 0.7)
        low, high = float(buffer_sigma.min()), float(buffer_sigma.max())
        if high / low < 1.3 or self.size_bins < 2:
            bins = np.zeros(len(alive), np.int32)
            centers = [float(np.sqrt(low * high))]
        else:
            edges = np.geomspace(low, high, self.size_bins + 1)
            bins = np.clip(np.digitize(buffer_sigma, edges[1:-1]), 0, self.size_bins - 1)
            centers = [float(np.sqrt(edges[i] * edges[i + 1])) for i in range(self.size_bins)]
        total = np.zeros((buf_h, buf_w, channels), np.float32)
        cells = buf_w * buf_h
        for index, bin_sigma in enumerate(centers):
            members = np.flatnonzero(bins == index)
            if not members.size:
                continue
            # A unit-sum Gaussian peaks at 1/(2 pi sigma^2); pre-scale so each
            # particle's blurred peak equals its own color and alpha.
            weight = 2 * pi * bin_sigma ** 2
            layer = np.empty((cells, channels), np.float32)
            for channel in range(channels):
                layer[:, channel] = np.bincount(flat[members], values[members, channel] * weight,
                                                minlength=cells)
            layer = layer.reshape(buf_h, buf_w, channels)
            total += cv2.GaussianBlur(layer, (0, 0), bin_sigma)
        roi = frame[y0:y1, x0:x1]
        if self.blend is Blend.ADD:
            # Quantize at low resolution, then upsample 8-bit: 4x less data to move.
            low = np.clip(total, 0, 255).astype(np.uint8)
            cv2.add(roi, cv2.resize(low, (roi_w, roi_h), interpolation=cv2.INTER_LINEAR), dst=roi)
        else:
            # Summed premultiplied color would turn dense smoke white. Use the
            # density-weighted average color, and coverage 1 - exp(-density), so
            # overlapping puffs get thicker and stay smoke-colored.
            density = total[..., 3:4]
            average = total[..., :3] / np.maximum(density, 1e-6)
            coverage = 1.0 - np.exp(-density)
            total[..., :3] = average * coverage
            total[..., 3:4] = coverage * 255.0
            low = np.clip(total, 0, 255).astype(np.uint8)
            full = cv2.resize(low, (roi_w, roi_h), interpolation=cv2.INTER_LINEAR)
            # roi * (1 - alpha) + premultiplied color, all in saturating 8-bit OpenCV ops.
            keep = cv2.cvtColor(255 - full[..., 3], cv2.COLOR_GRAY2BGR)
            cv2.multiply(roi, keep, dst=roi, scale=1 / 255)
            cv2.add(roi, full[..., :3], dst=roi)


class ParticleSet:
    """The shared systems: fire, sparks, spirit, ember (light); smoke and chunks (matter)."""

    def __init__(self, config: EffectsConfig, seed: int | None = None) -> None:
        """Create the four systems from EffectsConfig."""
        c = config
        common = dict(render_scale=c.particle_render_scale, buffer_budget=c.particle_buffer_budget,
                      max_blur=c.particle_max_blur_px, size_bins=c.particle_size_bins, max_dt=c.max_dt)
        rng = np.random.default_rng(seed)
        seeds = rng.integers(0, 2**31, 7)
        self.fire = ParticleSystem(c.fire_capacity, Blend.ADD, FIRE_RAMP, c.fire_drag, c.fire_gravity,
                                   c.fire_turbulence, size_growth=0.5, seed=int(seeds[0]), **common)
        self.smoke = ParticleSystem(c.smoke_capacity, Blend.ALPHA, SMOKE_RAMP, c.smoke_drag,
                                    c.smoke_gravity, c.smoke_turbulence, size_growth=1.8,
                                    seed=int(seeds[1]), **common)
        self.sparks = ParticleSystem(c.spark_capacity, Blend.ADD, SPARK_RAMP, c.spark_drag,
                                     c.spark_gravity, 0.0, seed=int(seeds[2]), **common)
        self.chunks = ParticleSystem(c.chunk_capacity, Blend.SQUARES, None, c.chunk_drag,
                                     c.chunk_gravity, 0.0, seed=int(seeds[3]), max_dt=c.max_dt)
        self.spirit = ParticleSystem(c.spirit_capacity, Blend.ADD, SPIRIT_RAMP, c.fire_drag,
                                     c.fire_gravity, c.fire_turbulence, size_growth=0.4,
                                     seed=int(seeds[4]), **common)
        self.ember = ParticleSystem(c.ember_capacity, Blend.ADD, EMBER_RAMP, c.spark_drag,
                                    -120.0, 260.0, seed=int(seeds[5]), **common)
        self.dust = ParticleSystem(c.dust_capacity, Blend.DUST, None, 0.6, -30.0, 120.0,
                                   seed=int(seeds[6]), max_dt=c.max_dt)

    @property
    def systems(self) -> dict[str, ParticleSystem]:
        """Name -> system, for timing readouts and adaptive quality."""
        return {"fire": self.fire, "smoke": self.smoke, "sparks": self.sparks, "chunks": self.chunks,
                "spirit": self.spirit, "ember": self.ember, "dust": self.dust}

    @property
    def count(self) -> int:
        """Live particles across all systems."""
        return sum(system.count for system in self.systems.values())

    def update(self, dt: float) -> None:
        """Advance every system."""
        for system in self.systems.values():
            system.update(dt)

    def clear(self) -> None:
        """Kill every particle."""
        for system in self.systems.values():
            system.clear()

    def set_budget(self, fraction: float) -> None:
        """Scale every system's particle cap (1.0 = configured capacity)."""
        for system in self.systems.values():
            system.set_limit(round(system.capacity * fraction))
