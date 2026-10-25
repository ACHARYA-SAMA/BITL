"""Rasenshuriken: a Rasengan that grows four spinning wind blades.

Trigger: the OK sign held 0.6 s. The thumb-index circle is where a Rasengan
forms, so the pose matches the move. The core grows, then the blades extend,
and it rises above the hand. A flick (or swipe) throws it; on impact it bursts
into a sphere of wind needles that shatters blocks. A fist cancels it.

All sprites are procedural: curved, tapered blades with a white core, cyan
edge, and blue halo, plus a swirling blue-white core.
"""
from __future__ import annotations
from collections.abc import Callable
from math import cos, pi, sin
from typing import TYPE_CHECKING
import cv2
import numpy as np
from ..config import EffectsConfig
from ..gestures.engine import GestureEngine
from ..gestures.events import EventBus, EventKind, GestureEvent
from ..gestures.types import Gesture
from .base import EffectContext, ModeController
from .compositing import Sprite, overlay_rgba
from .fireball import SWIPE_DIRECTIONS, SWIPES
from .projectile import FlightParams, HeldProjectiles, Projectile

if TYPE_CHECKING:
    from .manager import EffectManager

Point = tuple[float, float]


def make_shuriken_blades(size: int = 320) -> np.ndarray:
    """Four curved wind blades (BGRA): white core, cyan edges, soft blue halo."""
    big = size * 2  # Draw at 2x and shrink: smooth, anti-aliased blade edges.
    center = big / 2
    mask = np.zeros((big, big), np.uint8)
    for k in range(4):
        base = k * pi / 2
        s = np.linspace(0.0, 1.0, 80)
        radius = (0.17 + 0.80 * s) * center
        angle = base + 0.75 * s ** 1.5  # The blade sweeps back against the spin.
        points = np.stack((center + radius * np.cos(angle), center + radius * np.sin(angle)), axis=1)
        tangent = np.gradient(points, axis=0)
        tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-6)
        normal = np.stack((-tangent[:, 1], tangent[:, 0]), axis=1)
        width = 0.15 * center * np.sin(np.pi * np.clip(0.1 + 0.95 * s, 0, 1)) ** 0.8 * (1 - 0.45 * s)
        leading = points + normal * (width * 0.35)[:, None]  # A thin, sharp leading edge...
        trailing = points - normal * width[:, None]  # ...and a broad trailing edge.
        polygon = np.concatenate((leading, trailing[::-1])).astype(np.int32)
        cv2.fillPoly(mask, [polygon], 255, cv2.LINE_AA)
    body = mask.astype(np.float32) / 255
    core = cv2.GaussianBlur(cv2.erode(body, np.ones((15, 15), np.uint8)), (0, 0), 2)  # Thin white spine.
    edge = cv2.GaussianBlur(body, (0, 0), big / 160)
    halo = cv2.GaussianBlur(body, (0, 0), big / 40)
    white, cyan, blue = (np.array(v, np.float32) for v in ((255, 255, 255), (255, 235, 90), (255, 110, 10)))
    color = (white * core[..., None] + cyan * np.clip(edge - core, 0, 1)[..., None]
             + blue * np.clip(halo - edge, 0, 1)[..., None])
    alpha = np.clip(0.95 * core + 0.8 * edge + 0.7 * halo, 0, 1)
    color = color / np.maximum(alpha, 1e-3)[..., None]
    bgra = np.dstack((np.clip(color, 0, 255), alpha * 255)).astype(np.uint8)
    return cv2.resize(bgra, (size, size), interpolation=cv2.INTER_AREA)


def make_rasengan_core(size: int = 160) -> np.ndarray:
    """A swirling blue-white sphere (BGRA)."""
    coords = (np.arange(size, dtype=np.float32) + 0.5) / size * 2 - 1
    u, v = np.meshgrid(coords, coords)
    radius = np.hypot(u, v)
    theta = np.arctan2(v, u)
    swirl = 0.5 + 0.5 * np.sin(6 * theta + 10 * radius)  # Spiral bands.
    sphere = np.clip(1 - radius / 0.62, 0, 1)
    glow = np.exp(-(radius / 0.5) ** 2)
    heat = np.clip(0.55 * sphere ** 0.6 + 0.35 * sphere * swirl + 0.5 * np.exp(-(radius / 0.18) ** 2), 0, 1)
    color = np.stack([np.interp(heat, (0, 0.35, 0.7, 1.0), (c0, c1, c2, c3))
                      for c0, c1, c2, c3 in ((200, 255, 255, 255), (60, 180, 240, 255), (10, 60, 200, 255))],
                     axis=-1)
    alpha = np.clip(np.maximum(heat * 1.6, glow * 0.6), 0, 1)
    return np.dstack((color, alpha * 255)).clip(0, 255).astype(np.uint8)


class Rasenshuriken(Projectile):
    """Spinning wind shuriken. ``radius`` is the core; blades reach 3.2x further."""

    name = "rasenshuriken"
    layer = 32

    def __init__(self, owner: str, position: Point, radius: float, now: float) -> None:
        """Start forming (core first, then blades)."""
        super().__init__(owner, position, radius, now)
        self.form = 0.0
        self.blade_angle = 0.0

    def params(self, ctx: EffectContext) -> FlightParams:
        """Rasenshuriken tuning from EffectsConfig."""
        c = ctx.config
        return FlightParams(c.rasen_spring, c.rasen_gravity, c.rasen_growth, c.rasen_max_flight_s,
                            c.fireball_margin_px, c.fireball_lost_hold_s, 0.35)

    def _wind(self, ctx: EffectContext, intensity: float) -> None:
        reach = self.radius * 3.2 * self.form
        count = int(ctx.rng.poisson(140 * intensity * ctx.dt))
        if count and reach > 1:
            angles = ctx.rng.uniform(0, 2 * pi, count)
            ring = np.stack((np.cos(angles), np.sin(angles)), axis=1)
            positions = self.pos + ring * reach * ctx.rng.uniform(0.6, 1.05, (count, 1))
            # Tangential velocity: the sparks orbit with the blades.
            velocity = np.stack((-ring[:, 1], ring[:, 0]), axis=1) * reach * 3.5 - self.vel * 0.1
            ctx.particles.sparks.emit(count, positions, velocity, life=(0.12, 0.3), size=(1.2, 2.4))

    def update(self, ctx: EffectContext) -> None:
        """Spin, form, and fly (the Projectile base handles the life cycle)."""
        self.blade_angle = (self.blade_angle - ctx.config.rasen_spin_dps * ctx.dt) % 360
        if not self.thrown:
            self.form = min(1.0, self.form + ctx.dt / ctx.config.rasen_form_s)
        super().update(ctx)

    def on_held(self, ctx: EffectContext) -> None:
        """Orbiting wind sparks."""
        self._wind(ctx, self.form)

    def on_flight(self, ctx: EffectContext) -> None:
        """A wind trail."""
        self._wind(ctx, 1.4)

    def on_dying(self, ctx: EffectContext, fade: float) -> None:
        """Unravel into sparks."""
        self._wind(ctx, 2.0 * fade)

    def explode(self, ctx: EffectContext) -> None:
        """Burst into a sphere of wind needles that shatters blocks."""
        center = (float(self.pos[0]), float(self.pos[1]))
        ctx.explode(center, self.radius * ctx.config.rasen_blast_ratio, 1.8, style="wind")
        ctx.emit("rasenshuriken_explode", hand=self.owner, position=center, radius=self.radius)
        self.kill()

    def render(self, frame: np.ndarray, ctx: EffectContext) -> None:
        """Core, then blades spinning with a motion-blur smear."""
        sprites = ctx.sprites
        fade = self.fade(ctx.now, self.params(ctx))
        center = (float(self.pos[0]), float(self.pos[1]))
        grow_core = min(1.0, self.form / 0.4)
        grow_blades = float(np.clip((self.form - 0.3) / 0.7, 0, 1))
        blades: Sprite = sprites.rasen_blades
        if grow_blades > 0:
            scale = self.radius * 3.2 * 2 / blades.shape[1] * (0.4 + 0.6 * grow_blades)
            for lag, opacity in ((24, 0.18), (12, 0.35), (0, 0.95)):
                overlay_rgba(frame, blades, center, scale, self.blade_angle + lag,
                             opacity * grow_blades * fade, "add")
        core: Sprite = sprites.rasen_core
        pulse = 1 + 0.06 * sin(ctx.now * 31 + self.phase)
        overlay_rgba(frame, core, center, self.radius * 2.2 / core.shape[1] * grow_core * pulse,
                     -self.blade_angle * 0.3, fade, "add")
        glow = sprites.blue_glow
        overlay_rgba(frame, glow, center, self.radius * 5 / glow.shape[1] * grow_core, 0.0, 0.45 * fade, "add")


class RasenshurikenJutsu(ModeController):
    """JUTSU binding: OK sign held -> form above the hand; flick -> throw; fist -> cancel."""

    def __init__(self, bus: EventBus, engine: GestureEngine, manager: EffectManager,
                 config: EffectsConfig, is_busy: Callable[[str], bool] = lambda hand: False) -> None:
        """``is_busy`` keeps it from forming on a hand already holding another jutsu."""
        super().__init__(bus)
        self.engine, self.manager, self.config, self.is_busy = engine, manager, config, is_busy
        self.held = HeldProjectiles(engine, manager, "rasenshuriken", config.fireball_throw_speed,
                                    config.fireball_launch_gain, config.fireball_min_launch_px)

    def holding(self, hand: str) -> bool:
        """True while ``hand`` holds a Rasenshuriken (blocks fireball summons there)."""
        return hand in self.held.balls

    def bind(self) -> None:
        """The JUTSU Rasenshuriken mapping."""
        self.on_hold(Gesture.OK, self.config.rasen_hold_seconds, self._form)
        for swipe in SWIPES:
            self.subscribe(swipe, EventKind.STARTED, self._swipe_throw)
        self.subscribe(Gesture.FIST, EventKind.STARTED, self._cancel)

    def _anchor(self, hand: str) -> tuple[Point, float] | None:
        center, scale = self.engine.palm_center(hand), self.engine.hand_scale(hand)
        if center is None or scale <= 0:
            return None
        return (center[0], center[1] - self.config.rasen_offset * scale), scale * self.config.rasen_size / 2

    def _form(self, event: GestureEvent) -> None:
        hand = event.hand
        if self.holding(hand) or self.is_busy(hand):
            return
        anchor = self._anchor(hand)
        if anchor is None:
            return
        position, radius = anchor
        point = self.engine.position(hand) or position  # Starts in the OK circle...
        self.held.hold(hand, Rasenshuriken(hand, point, radius, event.timestamp))  # ...and rises.
        self.manager.hooks.emit("rasenshuriken_form", hand=hand, position=position)

    def _swipe_throw(self, event: GestureEvent) -> None:
        if not self.holding(event.hand):
            return
        velocity = np.asarray(self.engine.velocity_px(event.hand), np.float64)
        if np.hypot(*velocity) < 1e-3:
            velocity = np.asarray(SWIPE_DIRECTIONS[event.gesture]) * self.config.fireball_min_launch_px
        self.held.launch(event.hand, tuple(velocity), event.timestamp)

    def _cancel(self, event: GestureEvent) -> None:
        self.held.dissipate(event.hand, event.timestamp, "fist")

    def update(self, ctx: EffectContext) -> None:
        """Follow the hand once formed; throw on a flick; charge glow while forming."""
        self.held.prune()
        for hand in list(self.held.balls):
            anchor = self._anchor(hand) if self.engine.visible(hand) else None
            ball = self.held.balls[hand]
            # Only fully formed shuriken can be thrown: flicking mid-form would waste it.
            min_age = 0.2 if ball.form >= 1.0 else 1e9
            self.held.follow_hand(hand, anchor[0] if anchor else None, anchor[1] if anchor else 0.0,
                                  ctx.now, min_age)

    def render(self, frame: np.ndarray, ctx: EffectContext) -> None:
        """A small blue glow gathers in the OK circle while the sign is held."""
        glow = ctx.sprites.blue_glow
        for hand in self.engine.tracked_hands():
            if self.holding(hand) or not self.engine.is_active(hand, Gesture.OK) \
                    or not self.engine.visible(hand):
                continue
            progress = self.engine.hold_progress(hand, Gesture.OK, self.config.rasen_hold_seconds)
            point = self.engine.position(hand)
            if 0 < progress < 1 and point is not None:
                size = self.engine.hand_scale(hand) * (0.3 + 0.6 * progress)
                overlay_rgba(frame, glow, point, size * 2 / glow.shape[1], 0.0, 0.3 + 0.6 * progress, "add")

    def on_detach(self) -> None:
        """Leaving JUTSU cancels anything still held."""
        self.held.dissipate_all(self.manager.now, "mode_change")
