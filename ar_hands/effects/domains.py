"""Domain Expansions: Gojo's Unlimited Void and Sukuna's Malevolent Shrine.

A domain replaces the world around the user. It expands as a sphere from
the casting hand, holds for up to ``domain_max_s`` seconds (or until cast
again), then collapses. With person segmentation the user stays in front,
rim-lit in the domain's color. Without it, the camera shows through faintly
so the user never vanishes.

Both scenes are procedural and animated, rendered at half resolution (the
backgrounds are soft anyway) and upscaled:
  Unlimited Void: rotating nebula and star field, swirled by a gravitational
    lens around a black hole with a blazing accretion ring, plus streaks of
    "infinite information" flying out of it.
  Malevolent Shrine: blood-red sky with drifting clouds, a horned shrine on a
    mound of skulls, red water reflecting it all with ripples, falling red
    light pillars, and invisible slashes that cut across the screen (and
    through any blocks they cross).
"""
from __future__ import annotations
from math import cos, hypot, pi, sin
from typing import TYPE_CHECKING
import cv2
import numpy as np
from ..config import EffectsConfig
from ..gestures.engine import GestureEngine
from ..gestures.events import BOTH, EventBus, GestureEvent
from ..gestures.types import Gesture
from .base import Effect, EffectContext, EffectState, ModeController
from .sprites import _fractal_noise
from .susanoo import _bezier

if TYPE_CHECKING:
    from .manager import EffectManager

Point = tuple[float, float]


class DomainEffect(Effect):
    """Shared expand / hold / collapse and person compositing; subclasses paint scenes."""

    name = "domain"
    layer = -100  # Below everything: the domain replaces the background.
    needs_person = True
    rim_color = (255, 255, 255)

    def __init__(self, origin: Point, now: float, expand_s: float = 0.9,
                 collapse_s: float = 0.6, max_s: float = 15.0) -> None:
        """Start expanding from ``origin`` (the casting hand)."""
        super().__init__()
        self.origin, self.born = origin, now
        self.expand_s, self.collapse_s, self.max_s = expand_s, collapse_s, max_s
        self.collapsing_since: float | None = None
        self._size: tuple[int, int] = (0, 0)

    def collapse(self, now: float) -> None:
        """Start shrinking back into the origin."""
        if self.collapsing_since is None:
            self.collapsing_since, self.state = now, EffectState.DYING

    def coverage(self, now: float) -> float:
        """0..1: how far the domain sphere has expanded over the screen."""
        grow = min(1.0, (now - self.born) / self.expand_s)
        grow = 1 - (1 - grow) ** 3
        if self.collapsing_since is None:
            return grow
        return grow * max(0.0, 1 - (now - self.collapsing_since) / self.collapse_s)

    def update(self, ctx: EffectContext) -> None:
        """Advance the scene; collapse on timeout; die when fully collapsed."""
        if (ctx.width, ctx.height) != self._size:
            self._size = (ctx.width, ctx.height)
            self.prepare(ctx.width // 2, ctx.height // 2, ctx.rng)
        if self.state is EffectState.SPAWN and ctx.now - self.born >= self.expand_s:
            self.state = EffectState.ACTIVE
        if ctx.now - self.born >= self.max_s:
            self.collapse(ctx.now)
        if self.collapsing_since is not None and self.coverage(ctx.now) <= 0:
            self.kill()
            return
        self.age += ctx.dt
        self.animate(ctx)

    def prepare(self, width: int, height: int, rng: np.random.Generator) -> None:
        """Subclasses build static layers at half resolution."""

    def animate(self, ctx: EffectContext) -> None:
        """Subclasses advance per-frame state (particles, slashes)."""

    def scene(self, ctx: EffectContext) -> np.ndarray:
        """Subclasses return the half-resolution BGR scene for this frame."""
        raise NotImplementedError

    def overlay(self, frame: np.ndarray, ctx: EffectContext) -> None:
        """Subclasses draw things that belong in front of the user (slashes)."""

    def render(self, frame: np.ndarray, ctx: EffectContext) -> None:
        """Domain inside the expanding sphere; the user in front, rim-lit."""
        reach = self.coverage(ctx.now)
        if reach <= 0:
            return
        height, width = frame.shape[:2]
        scene = cv2.resize(self.scene(ctx), (width, height), interpolation=cv2.INTER_LINEAR)
        if reach >= 1.0:
            frame[:] = scene
        else:
            radius = reach * hypot(width, height)
            mask = np.zeros((height // 4, width // 4), np.float32)
            cv2.circle(mask, (int(self.origin[0] / 4), int(self.origin[1] / 4)), int(radius / 4), 1.0, -1,
                       cv2.LINE_AA)
            mask = cv2.resize(cv2.GaussianBlur(mask, (0, 0), 2), (width, height))
            cv2.blendLinear(scene, frame, mask, 1.0 - mask, dst=frame)
            ring = np.zeros((height // 2, width // 2, 3), np.uint8)
            cv2.circle(ring, (int(self.origin[0] / 2), int(self.origin[1] / 2)), int(radius / 2),
                       self.rim_color, max(2, int(6 * (1 - reach) + 2)), cv2.LINE_AA)
            ring = cv2.GaussianBlur(ring, (0, 0), 3)
            cv2.add(frame, cv2.resize(ring, (width, height)), dst=frame)
        if ctx.person is not None and ctx.camera is not None:
            # Grade the user toward the domain's light so they sit IN it, not on it.
            tint = tuple(1.0 - reach * (1.0 - t) for t in self.tint)
            camera = cv2.multiply(ctx.camera, np.array([*tint, 0.0]), dtype=cv2.CV_8U)
            cv2.blendLinear(camera, frame, ctx.person, 1.0 - ctx.person, dst=frame)
            self._rim_light(frame, ctx.person, reach)
        elif ctx.camera is not None:
            cv2.addWeighted(frame, 1 - 0.35 * reach, ctx.camera, 0.35 * reach, 0, dst=frame)
        self.overlay(frame, ctx)

    tint = (1.0, 1.0, 1.0)  # Per-channel BGR grade applied to the user inside the domain.

    def _rim_light(self, frame: np.ndarray, person: np.ndarray, reach: float) -> None:
        """A glow just outside the user's silhouette, lit by the domain.

        Built at quarter resolution in 8 bits and upscaled: it is soft anyway,
        and doing it in full-resolution float cost 14 ms per frame.
        """
        height, width = frame.shape[:2]
        small = cv2.resize(person, (width // 4, height // 4), interpolation=cv2.INTER_AREA)
        # Blur minus the mask is positive only outside the edge: a halo, not a tint.
        rim = np.clip((cv2.GaussianBlur(small, (0, 0), 2.5) - small) * 2.2 * reach, 0, 1)
        light = cv2.merge([np.clip(rim * c, 0, 255).astype(np.uint8) for c in self.rim_color])
        cv2.add(frame, cv2.resize(light, (width, height), interpolation=cv2.INTER_LINEAR), dst=frame)


class UnlimitedVoid(DomainEffect):
    """Gojo Satoru's domain: infinite space around a blazing black hole."""

    name = "unlimited_void"
    rim_color = (255, 220, 170)
    tint = (1.12, 0.95, 0.85)  # Cold blue-white starlight (BGR).

    def prepare(self, width: int, height: int, rng: np.random.Generator) -> None:
        """Nebula, stars, the lens field, and the accretion ring in polar form."""
        self.w, self.h = width, height
        self.center = (width * 0.5, height * 0.44)
        self.hole = height * 0.24
        side = int(hypot(width, height)) + 4
        cloud_a = _fractal_noise(rng, side, 6)
        cloud_b = _fractal_noise(rng, side, 5)
        base = np.zeros((side, side, 3), np.float32)
        base += np.array((40, 12, 6), np.float32)  # Deep space navy.
        wisps = np.clip((cloud_a - 0.45) * 2.2, 0, 1) ** 2
        violet = np.clip((cloud_b - 0.55) * 2.5, 0, 1) ** 2
        base += wisps[..., None] * np.array((255, 215, 170), np.float32) * 0.55
        base += violet[..., None] * np.array((170, 60, 90), np.float32) * 0.6
        self.nebula = np.clip(base, 0, 255).astype(np.uint8)
        # Stars: polar coordinates around the hole, so they can orbit cheaply.
        count = 900
        self.star_r = (rng.random(count) ** 0.6) * side * 0.55 + self.hole * 0.9
        self.star_a = rng.uniform(0, 2 * pi, count)
        self.star_b = rng.uniform(0.3, 1.0, count) ** 2
        self.star_twinkle = rng.uniform(0, 2 * pi, count)
        # Lens: rotate each pixel around the hole by an angle that grows near it.
        ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
        dx, dy = xs - self.center[0], ys - self.center[1]
        distance = np.maximum(np.hypot(dx, dy), 1.0)
        twist = 0.9 * (self.hole / distance) ** 1.5
        cos_t, sin_t = np.cos(twist), np.sin(twist)
        self.lens_x = (self.center[0] + dx * cos_t - dy * sin_t).astype(np.float32)
        self.lens_y = (self.center[1] + dx * sin_t + dy * cos_t).astype(np.float32)
        # Accretion ring texture in polar form: (angle, radius) with hot streaks.
        streaks = cv2.resize(rng.random((24, 360)).astype(np.float32), (720, 160), interpolation=cv2.INTER_CUBIC)
        radial = np.linspace(0, 1, 160, dtype=np.float32)[:, None]
        profile = np.exp(-((radial - 0.22) / 0.07) ** 2) + 0.55 * np.exp(-((radial - 0.45) / 0.25) ** 2)
        self.ring_polar = np.clip(profile * (0.55 + 0.6 * streaks), 0, 1.6)  # (160 radius, 720 angle)
        extent = int(self.hole * 2.6)
        gy, gx = np.mgrid[-extent:extent, -extent:extent].astype(np.float32)
        self.ring_extent = extent
        self.ring_r = np.clip((np.hypot(gx, gy) - self.hole * 0.95) / (self.hole * 1.6) * 159, 0, 159).astype(np.int32)
        self.ring_a = ((np.arctan2(gy, gx) + pi) / (2 * pi) * 719).astype(np.int32)
        self.ring_alpha = np.clip((np.hypot(gx, gy) - self.hole * 0.95) / 3.0, 0, 1)
        self.streaks: list[list[float]] = []

    def animate(self, ctx: EffectContext) -> None:
        """Information streaks fly out of the accretion disk."""
        rng = ctx.rng
        for _ in range(int(rng.poisson(30 * ctx.dt * 10))):
            angle = rng.uniform(0, 2 * pi)
            self.streaks.append([angle, self.hole * rng.uniform(1.3, 1.7), rng.uniform(250, 600), 0.0])
        for streak in self.streaks:
            streak[1] += streak[2] * ctx.dt
            streak[3] += ctx.dt
        self.streaks = [s for s in self.streaks if s[3] < 0.9][-120:]

    def scene(self, ctx: EffectContext) -> np.ndarray:
        """Nebula + stars (lensed) + black hole + accretion ring + streaks."""
        t = ctx.now - self.born
        w, h = self.w, self.h
        side = self.nebula.shape[0]
        rotation = cv2.getRotationMatrix2D((side / 2, side / 2), t * 3.0, 1.0)
        rotation[0, 2] += self.center[0] - side / 2
        rotation[1, 2] += self.center[1] - side / 2
        sky = cv2.warpAffine(self.nebula, rotation, (w, h), flags=cv2.INTER_LINEAR)
        angles = self.star_a + t * 0.25 * (self.hole * 3 / self.star_r)  # Inner stars orbit faster.
        xs = (self.center[0] + self.star_r * np.cos(angles)).astype(np.int32)
        ys = (self.center[1] + self.star_r * np.sin(angles)).astype(np.int32)
        visible = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        twinkle = self.star_b * (0.7 + 0.3 * np.sin(t * 3 + self.star_twinkle))
        stars = np.zeros((h, w), np.float32)
        np.add.at(stars, (ys[visible], xs[visible]), twinkle[visible] * 255)
        stars = cv2.GaussianBlur(stars, (0, 0), 0.7) * 2.2
        sky = cv2.add(sky, cv2.cvtColor(np.clip(stars, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR))
        sky = cv2.remap(sky, self.lens_x, self.lens_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        # Accretion ring: shift the polar texture's angle over time, then look it up.
        cx, cy, e = int(self.center[0]), int(self.center[1]), self.ring_extent
        shift = int(t * 120) % 720
        heat = self.ring_polar[self.ring_r, (self.ring_a + shift) % 720] * self.ring_alpha
        ring = np.dstack((heat * 255, heat * 235, heat * 190))
        x0, y0, x1, y1 = cx - e, cy - e, cx + e, cy + e
        sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
        roi = sky[sy0:sy1, sx0:sx1]
        light = np.clip(ring[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0], 0, 255).astype(np.uint8)
        cv2.add(roi, light, dst=roi)
        cv2.circle(sky, (cx, cy), int(self.hole * 0.95), (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(sky, (cx, cy), int(self.hole), (255, 250, 235), 2, cv2.LINE_AA)  # Photon ring.
        for angle, radius, _, age in self.streaks:
            fade = 1 - age / 0.9
            a = (int(cx + cos(angle) * radius), int(cy + sin(angle) * radius))
            b = (int(cx + cos(angle) * (radius + 40 * fade + 10)), int(cy + sin(angle) * (radius + 40 * fade + 10)))
            cv2.line(sky, a, b, tuple(int(c * fade) for c in (255, 245, 230)), 1, cv2.LINE_AA)
        return sky


class MalevolentShrine(DomainEffect):
    """Ryomen Sukuna's domain: a shrine of skulls under a blood-red sky, and slashes."""

    name = "malevolent_shrine"
    rim_color = (60, 60, 255)
    tint = (0.72, 0.78, 1.12)  # Red light falls on the user (BGR).

    def prepare(self, width: int, height: int, rng: np.random.Generator) -> None:
        """Sky and clouds, the shrine, and the water maps, all precomputed in 8 bits."""
        self.w, self.h = width, height
        self.horizon = horizon = int(height * 0.62)
        g = np.linspace(0, 1, horizon, dtype=np.float32)[:, None, None]
        top, mid, low = (np.array(c, np.float32) for c in ((8, 2, 20), (22, 10, 100), (55, 65, 240)))
        sky = np.where(g < 0.6, top + (mid - top) * (g / 0.6), mid + (low - mid) * ((g - 0.6) / 0.4))
        self.sky_base = np.ascontiguousarray(np.repeat(sky, width, axis=1).clip(0, 255).astype(np.uint8))
        noise = _fractal_noise(rng, max(width * 2, horizon * 2), 6)[:horizon, : width * 2]
        cover = np.clip((noise - 0.42) * 2.2, 0, 1)
        self.cloud_weight = (cover * 0.85).astype(np.float32)
        self.cloud_color = np.full((horizon, width, 3), (6, 2, 14), np.uint8)
        edge = np.clip(cover * (1 - cover) * 4, 0, 1) * np.linspace(0.3, 1, horizon, dtype=np.float32)[:, None]
        self.cloud_glow = (edge[..., None] * np.array((25, 35, 190), np.float32)).astype(np.uint8)
        shrine = self._paint_shrine(width, height, rng)[:horizon]
        self.shrine_bgr = np.ascontiguousarray(shrine[..., :3])
        self.shrine_alpha = shrine[..., 3].astype(np.float32) / 255
        self.shrine_keep = 1.0 - self.shrine_alpha
        water_h = height - horizon
        self.ripple_y = np.arange(water_h, dtype=np.float32)
        self.water_map_y = np.repeat(np.clip(horizon - 1 - self.ripple_y * 0.9, 0, horizon - 1)[:, None],
                                     width, axis=1).astype(np.float32)
        self.water_base_x = np.repeat(np.arange(width, dtype=np.float32)[None, :], water_h, axis=0)
        self.rain = np.stack((rng.uniform(0, width, 70), rng.uniform(0, height, 70),
                              rng.uniform(40, 160, 70), rng.uniform(250, 650, 70)), axis=1)
        self.slashes: list[list[float]] = []
        self.next_slash = 0.6

    def _paint_shrine(self, width: int, height: int, rng: np.random.Generator) -> np.ndarray:
        """BGRA shrine: skull mound, stepped platform, pillars and crossbeam, two tiled
        roofs with upturned eaves and fangs, sweeping horns, and a burning green eye."""
        canvas = np.zeros((height, width, 4), np.uint8)
        cx, base = width * 0.5, self.horizon
        dark, edge = (12, 6, 20, 255), (50, 55, 210, 255)

        def poly(points, color):
            cv2.fillPoly(canvas, [np.asarray(points, np.int32)], color, cv2.LINE_AA)

        def shaded(points, light_top: bool = True, tiles: int = 0):
            """Fill with a vertical red gradient (lit from below by the red sky) and tile lines."""
            mask = np.zeros((height, width), np.uint8)
            cv2.fillPoly(mask, [np.asarray(points, np.int32)], 255, cv2.LINE_AA)
            ys = np.nonzero(mask.any(axis=1))[0]
            if not len(ys):
                return
            g = np.clip((np.arange(height, dtype=np.float32) - ys[0]) / max(1, ys[-1] - ys[0]), 0, 1)[:, None]
            g = g if light_top is False else 1 - g
            color = np.array((10, 6, 18), np.float32) + np.array((30, 30, 120), np.float32) * (g ** 2)[..., None]
            if tiles:
                stripes = np.where((np.arange(width) // tiles) % 2 == 0, 0.7, 1.0).astype(np.float32)
                color = color * stripes[None, :, None]
            alpha = mask.astype(np.float32)[..., None] / 255
            canvas[..., :3] = (canvas[..., :3] * (1 - alpha) + color * alpha).astype(np.uint8)
            canvas[..., 3] = np.maximum(canvas[..., 3], mask)

        # Mound of skulls, drawn back to front so nearer skulls overlap farther ones.
        cv2.ellipse(canvas, (int(cx), base), (int(width * 0.32), int(height * 0.11)), 0, 180, 360, dark, -1, cv2.LINE_AA)
        skulls = []
        for _ in range(260):
            angle = rng.uniform(pi * 1.04, pi * 1.96)
            reach = rng.uniform(0.15, 1.0) ** 0.7
            x = cx + cos(angle) * width * 0.31 * reach
            y = base + sin(angle) * height * 0.1 * reach
            skulls.append((y, x, rng.uniform(0.006, 0.012) * width * (0.6 + 0.4 * (y - base + height * 0.1) / (height * 0.1))))
        for y, x, r in sorted(skulls):
            shade = 0.55 + 0.45 * rng.random()
            cv2.circle(canvas, (int(x), int(y)), int(r), (int(70 * shade), int(80 * shade), int(150 * shade), 255), -1, cv2.LINE_AA)
            cv2.circle(canvas, (int(x - r * 0.25), int(y - r * 0.3)), max(1, int(r * 0.45)),
                       (int(95 * shade), int(110 * shade), int(200 * shade), 255), -1, cv2.LINE_AA)
            for dx in (-0.35, 0.35):
                cv2.ellipse(canvas, (int(x + dx * r), int(y + r * 0.1)), (max(1, int(r * 0.25)), max(1, int(r * 0.3))),
                            0, 0, 360, (8, 4, 12, 255), -1)
        # Stepped platform.
        for k, (half, y0, y1) in enumerate(((0.21, 0.07, 0.035), (0.18, 0.105, 0.07), (0.15, 0.135, 0.105))):
            shaded([(cx - width * half, base - height * y1), (cx + width * half, base - height * y1),
                    (cx + width * (half - 0.01), base - height * y0), (cx - width * (half - 0.01), base - height * y0)])
            cv2.line(canvas, (int(cx - width * half), int(base - height * y1)), (int(cx + width * half), int(base - height * y1)),
                     edge, 1, cv2.LINE_AA)
        # Pillars and a crossbeam (torii style).
        for k in (-0.115, -0.045, 0.045, 0.115):
            x = cx + k * width
            shaded([(x - width * 0.013, base - height * 0.36), (x + width * 0.013, base - height * 0.36),
                    (x + width * 0.013, base - height * 0.135), (x - width * 0.013, base - height * 0.135)], light_top=False)
            cv2.line(canvas, (int(x - width * 0.013), int(base - height * 0.36)), (int(x - width * 0.013), int(base - height * 0.135)),
                     edge, 1, cv2.LINE_AA)
        shaded([(cx - width * 0.15, base - height * 0.3), (cx + width * 0.15, base - height * 0.3),
                (cx + width * 0.15, base - height * 0.28), (cx - width * 0.15, base - height * 0.28)])
        # Two curved, tiled roofs with upturned eaves; fangs under the lower one.
        for level, (half, y_top, depth) in enumerate(((0.28, 0.43, 0.075), (0.17, 0.56, 0.065))):
            y = base - height * y_top
            bottom = _bezier([(cx - width * half, y - height * 0.02), (cx - width * half * 0.55, y + height * depth * 1.15),
                              (cx + width * half * 0.55, y + height * depth * 1.15), (cx + width * half, y - height * 0.02)], 40)
            ridge = _bezier([(cx + width * half, y - height * 0.02), (cx + width * half * 0.45, y - height * 0.03),
                             (cx - width * half * 0.45, y - height * 0.03), (cx - width * half, y - height * 0.02)], 40)
            ridge[:, 1] -= height * 0.03 * np.sin(np.linspace(0, pi, len(ridge)))
            outline = np.concatenate((bottom, ridge))
            shaded(outline, tiles=max(3, width // 90))
            cv2.polylines(canvas, [ridge.astype(np.int32).reshape(-1, 1, 2)], False, edge, 2, cv2.LINE_AA)
            cv2.polylines(canvas, [bottom.astype(np.int32).reshape(-1, 1, 2)], False, (30, 30, 140, 255), 1, cv2.LINE_AA)
            if level == 0:
                for k in range(-10, 11):
                    t = (k + 10) / 20
                    px, py = _bezier([tuple(bottom[0]), tuple(bottom[13]), tuple(bottom[26]), tuple(bottom[-1])], 21)[k + 10]
                    size = width * 0.007 * (1.2 - abs(k) / 14)
                    poly([(px - size, py), (px + size, py), (px, py + height * 0.035 * (1.2 - abs(k) / 14))],
                         (190, 200, 225, 255))
        # Horns and the burning green eye at the top.
        top = base - height * 0.62
        for side in (-1, 1):
            outer = _bezier([(cx + side * width * 0.11, top + height * 0.07), (cx + side * width * 0.24, top - height * 0.02),
                             (cx + side * width * 0.19, top - height * 0.24)], 24)
            inner = _bezier([(cx + side * width * 0.05, top + height * 0.07), (cx + side * width * 0.15, top),
                             (cx + side * width * 0.19, top - height * 0.24)], 24)
            shaded(np.concatenate((outer, inner[::-1])), light_top=False)
            cv2.polylines(canvas, [outer.astype(np.int32).reshape(-1, 1, 2)], False, edge, 2, cv2.LINE_AA)
        eye_c = (int(cx), int(top + height * 0.035))
        cv2.ellipse(canvas, eye_c, (int(width * 0.035), int(height * 0.022)), 0, 0, 360, (40, 120, 20, 255), -1, cv2.LINE_AA)
        cv2.ellipse(canvas, eye_c, (int(width * 0.012), int(height * 0.02)), 0, 0, 360, (150, 255, 170, 255), -1, cv2.LINE_AA)
        self.emblem = (float(eye_c[0]), float(eye_c[1]))
        return canvas

    def animate(self, ctx: EffectContext) -> None:
        """Rain falls; slashes appear at random and cut through blocks."""
        self.rain[:, 1] += self.rain[:, 3] * ctx.dt
        wrapped = self.rain[:, 1] > self.h
        self.rain[wrapped, 1] = -self.rain[wrapped, 2]
        self.next_slash -= ctx.dt
        if self.next_slash <= 0 and self.state is not EffectState.DYING:
            self.next_slash = float(ctx.rng.uniform(0.25, 0.7))
            angle = ctx.rng.uniform(-0.8, 0.8) + (pi / 2 if ctx.rng.random() < 0.3 else 0)
            x, y = ctx.rng.uniform(0.1, 0.9) * ctx.width, ctx.rng.uniform(0.1, 0.8) * ctx.height
            half = ctx.rng.uniform(0.25, 0.55) * ctx.width
            a = (x - cos(angle) * half, y - sin(angle) * half)
            b = (x + cos(angle) * half, y + sin(angle) * half)
            self.slashes.append([a[0], a[1], b[0], b[1], 0.0])
            ctx.manager.screen.shake(5.0)
            ctx.emit("domain_slash", position=(x, y))
            self._cut_blocks(ctx, a, b)
        for slash in self.slashes:
            slash[4] += ctx.dt
        self.slashes = [s for s in self.slashes if s[4] < 0.28]
        count = int(ctx.rng.poisson(50 * ctx.dt))
        if count:
            xs = ctx.rng.uniform(0, ctx.width, count)
            ctx.particles.ember.emit(count, np.stack((xs, np.full(count, ctx.height * 0.95)), 1),
                                     (0, -180), spread=60, life=(0.8, 1.6), size=(1.5, 3.5))

    def _cut_blocks(self, ctx: EffectContext, a: Point, b: Point) -> None:
        from .jjk import find_world
        world = find_world(ctx)
        if world is None:
            return
        ax, ay, bx, by = *a, *b
        for block in list(world.blocks):
            if block.held_by is not None or block.fading_since is not None:
                continue
            px, py = block.center
            t = max(0.0, min(1.0, ((px - ax) * (bx - ax) + (py - ay) * (by - ay)) / ((bx - ax) ** 2 + (by - ay) ** 2)))
            if hypot(px - (ax + t * (bx - ax)), py - (ay + t * (by - ay))) < block.width * 0.5:
                world.smash(block, ctx.now, "dismantle")

    def scene(self, ctx: EffectContext) -> np.ndarray:
        """Sky with drifting clouds, shrine, rippling red reflection, falling light pillars.
        Every step is a saturating 8-bit OpenCV operation on precomputed layers."""
        t = ctx.now - self.born
        w, h, horizon = self.w, self.h, self.horizon
        offset = int(t * 10) % w
        weight = np.ascontiguousarray(self.cloud_weight[:, offset:offset + w])
        upper = cv2.blendLinear(self.cloud_color, self.sky_base, weight, 1.0 - weight)
        cv2.add(upper, np.ascontiguousarray(self.cloud_glow[:, offset:offset + w]), dst=upper)
        upper = cv2.blendLinear(self.shrine_bgr, upper, self.shrine_alpha, self.shrine_keep)
        pulse = 0.6 + 0.4 * sin(t * 4)
        cv2.circle(upper, (int(self.emblem[0]), int(self.emblem[1])), int(w * 0.05),
                   (int(40 * pulse), int(255 * pulse), int(60 * pulse)), 2, cv2.LINE_AA)
        # Water: the sky and shrine mirrored, rippled, and stained red.
        water_h = h - horizon
        ripple = 3.0 * np.sin(self.ripple_y[:, None] * 0.35 + t * 5) * (0.3 + self.ripple_y[:, None] / water_h)
        water = cv2.remap(upper, (self.water_base_x + ripple).astype(np.float32), self.water_map_y,
                          cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        water = cv2.multiply(water, np.array((0.35, 0.35, 0.8, 0.0)), dtype=cv2.CV_8U)
        cv2.add(water, np.array((5.0, 5.0, 30.0, 0.0)), dst=water)
        frame = np.vstack((upper, water))
        for x, y, length, _ in self.rain:
            cv2.line(frame, (int(x), int(y)), (int(x), int(y + length * 0.5)), (40, 40, 210), 1, cv2.LINE_AA)
        return frame

    def overlay(self, frame: np.ndarray, ctx: EffectContext) -> None:
        """Slashes are drawn last: they cut across everything, the user included."""
        for ax, ay, bx, by, age in self.slashes:
            fade = 1 - age / 0.28
            reveal = min(1.0, age / 0.05)
            end = (int(ax + (bx - ax) * reveal), int(ay + (by - ay) * reveal))
            cv2.line(frame, (int(ax), int(ay)), end, (60, 60, int(255 * fade)), max(2, int(9 * fade)), cv2.LINE_AA)
            cv2.line(frame, (int(ax), int(ay)), end, tuple(int(255 * fade) for _ in range(3)), 2, cv2.LINE_AA)


class DomainCaster(ModeController):
    """JJK binding: crossed fingers held -> Unlimited Void; prayer held -> Malevolent Shrine.
    Casting the same domain again (or the other one) collapses the current one."""

    def __init__(self, bus: EventBus, engine: GestureEngine, manager: EffectManager,
                 config: EffectsConfig) -> None:
        """Create the controller; attach() subscribes it."""
        super().__init__(bus)
        self.engine, self.manager, self.config = engine, manager, config
        self.hold_rings = ((Gesture.CROSSED, config.domain_hold_seconds),
                           (Gesture.PRAYER, config.domain_hold_seconds))
        self.domain: DomainEffect | None = None

    @property
    def active(self) -> DomainEffect | None:
        """The domain currently up (not collapsing), if any."""
        if self.domain is not None and self.domain.alive and self.domain.collapsing_since is None:
            return self.domain
        return None

    def bind(self) -> None:
        """The JJK domain mapping."""
        self.on_hold(Gesture.CROSSED, self.config.domain_hold_seconds, lambda e: self._cast(UnlimitedVoid, e))
        self.on_hold(Gesture.PRAYER, self.config.domain_hold_seconds, lambda e: self._cast(MalevolentShrine, e),
                     hand=BOTH)

    def _cast(self, kind: type[DomainEffect], event: GestureEvent) -> None:
        current = self.active
        if current is not None:
            current.collapse(event.timestamp)
            self.manager.hooks.emit("domain_end", domain=current.name)
            if isinstance(current, kind):
                return  # Same sign again: just close it.
        origin = event.position
        c = self.config
        self.domain = kind(origin, event.timestamp, c.domain_expand_s, c.domain_collapse_s, c.domain_max_s)
        self.manager.spawn(self.domain)
        self.manager.screen.flash(0.6)
        self.manager.screen.shake(14.0)
        self.manager.hooks.emit("domain_expansion", domain=self.domain.name, position=origin)

    def on_detach(self) -> None:
        """Leaving JJK collapses any open domain."""
        if self.active is not None:
            self.active.collapse(self.manager.now)
            self.manager.hooks.emit("domain_end", domain=self.domain.name)
