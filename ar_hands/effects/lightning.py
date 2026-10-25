"""Chidori: crackling lightning around an open hand, charged by holding a fist.

Trigger: FIST held 0.6 s arms it (sparks gather around the fist), then
opening to OPEN_PALM within 1 s releases it; it lasts while the palm stays
open and fades out 0.25 s after.

Why this trigger: a two-step "charge, then release" cannot happen by accident,
which a single pose could. It uses the two most reliably recognized poses (100%
under heavy landmark noise in the Phase 3 tests), and it gives the effect
build-up. While Chidori is armed or active, that hand cannot summon a fireball,
so the open palm never does both.
"""
from __future__ import annotations
from math import cos, pi, sin
from typing import TYPE_CHECKING
import cv2
import numpy as np
from ..config import EffectsConfig
from ..gestures.engine import GestureEngine
from ..gestures.events import BOTH, EventBus, EventKind, GestureEvent
from ..gestures.types import Gesture
from .base import Effect, EffectContext, EffectState, ModeController
from .compositing import overlay_rgba

if TYPE_CHECKING:
    from .manager import EffectManager

Point = tuple[float, float]


def midpoint_bolt(start: Point, end: Point, rng: np.random.Generator, roughness: float,
                  depth: int) -> np.ndarray:
    """Jagged polyline from start to end by recursive midpoint displacement.

    Each level splits every segment and pushes its midpoint sideways by a
    random amount proportional to that segment's length, so detail shrinks
    with scale like a real discharge. Vectorized per level: 2**depth segments.
    """
    points = np.array((start, end), np.float32)
    for _ in range(depth):
        a, b = points[:-1], points[1:]
        segment = b - a
        length = np.hypot(segment[:, 0], segment[:, 1])[:, None]
        normal = np.stack((-segment[:, 1], segment[:, 0]), axis=1) / np.maximum(length, 1e-6)
        middle = (a + b) / 2 + normal * rng.normal(0.0, roughness, (len(a), 1)) * length
        refined = np.empty((len(points) * 2 - 1, 2), np.float32)
        refined[0::2], refined[1::2] = points, middle
        points = refined
    return points


def branching_bolts(start: Point, targets: list[Point], rng: np.random.Generator,
                    roughness: float, depth: int) -> list[np.ndarray]:
    """One main bolt per target, plus an occasional shorter fork off its middle."""
    bolts = []
    for target in targets:
        bolt = midpoint_bolt(start, target, rng, roughness, depth)
        bolts.append(bolt)
        if rng.random() < 0.55 and len(bolt) > 4:
            fork_at = bolt[int(rng.integers(len(bolt) // 4, 3 * len(bolt) // 4))]
            direction = np.asarray(target, np.float32) - np.asarray(start, np.float32)
            angle = rng.uniform(0.4, 0.9) * rng.choice((-1, 1))
            rotated = np.array((direction[0] * cos(angle) - direction[1] * sin(angle),
                                direction[0] * sin(angle) + direction[1] * cos(angle)))
            end = fork_at + rotated * rng.uniform(0.25, 0.45)
            bolts.append(midpoint_bolt(tuple(fork_at), tuple(end), rng, roughness, max(2, depth - 2)))
    return bolts


def draw_bolts(frame: np.ndarray, bolts: list[np.ndarray], colors: tuple, widths: tuple,
               intensity: float) -> None:
    """Three additive passes: wide blurred blue halo, cyan body, thin white core."""
    if not bolts or intensity <= 0:
        return
    stacked = np.concatenate(bolts)
    pad = widths[0] * 3 + 6
    height, width = frame.shape[:2]
    x0 = max(0, int(stacked[:, 0].min()) - pad)
    y0 = max(0, int(stacked[:, 1].min()) - pad)
    x1 = min(width, int(stacked[:, 0].max()) + pad)
    y1 = min(height, int(stacked[:, 1].max()) + pad)
    if x0 >= x1 or y0 >= y1:
        return
    lines = [np.round(bolt - (x0, y0)).astype(np.int32).reshape(-1, 1, 2) for bolt in bolts]
    halo = np.zeros((y1 - y0, x1 - x0, 3), np.uint8)
    cv2.polylines(halo, lines, False, colors[0], widths[0], cv2.LINE_AA)
    cv2.GaussianBlur(halo, (0, 0), widths[0] * 0.5, dst=halo)
    body = np.zeros_like(halo)
    cv2.polylines(body, lines, False, colors[1], widths[1], cv2.LINE_AA)
    cv2.polylines(body, lines, False, colors[2], widths[2], cv2.LINE_AA)
    light = cv2.addWeighted(halo, intensity, body, intensity, 0)
    roi = frame[y0:y1, x0:x1]
    cv2.add(roi, light, dst=roi)


class ChidoriEffect(Effect):
    """Lightning anchored to one hand's palm, re-randomized every few frames."""

    name = "chidori"
    layer = 40

    def __init__(self, owner: str, now: float) -> None:
        """Start crackling on ``owner``'s palm."""
        super().__init__()
        self.owner, self.born = owner, now
        self.center: Point | None = None
        self.scale = 0.0
        self.tips: tuple[Point, ...] = ()
        self.bolts: list[np.ndarray] = []
        self.frames = 0
        self.ending_since: float | None = None
        self.flicker = 1.0

    def end(self, now: float) -> None:
        """Begin the short fade-out."""
        if self.ending_since is None:
            self.ending_since = now
            self.state = EffectState.DYING

    def intensity(self, now: float, fade_s: float) -> float:
        """1 while active, falling to 0 during the fade."""
        if self.ending_since is None:
            return 1.0
        return max(0.0, 1.0 - (now - self.ending_since) / fade_s)

    def update(self, ctx: EffectContext) -> None:
        """Track the palm, regenerate bolts, emit sparks, and fade when ended."""
        c, engine = ctx.config, ctx.engine
        if engine.visible(self.owner):
            features = engine.features(self.owner)
            self.center = features.palm_center
            self.scale = features.scale
            self.tips = features.tips
        elif self.owner not in engine.tracked_hands():
            self.end(ctx.now)  # Hand gone past the engine's grace: no orphaned lightning.
        if self.center is None or self.intensity(ctx.now, c.chidori_fade_s) <= 0:
            self.kill()
            return
        if self.state is EffectState.SPAWN:
            self.state = EffectState.ACTIVE
        self.age += ctx.dt
        if self.frames % max(1, c.chidori_refresh_frames) == 0:
            self._regenerate(ctx)
        self.frames += 1
        self.flicker = float(ctx.rng.uniform(0.75, 1.0))
        count = int(ctx.rng.poisson(160 * ctx.dt * self.intensity(ctx.now, c.chidori_fade_s)))
        ctx.particles.sparks.emit_radial(count, self.center, (self.scale * 2, self.scale * 6),
                                         (0.12, 0.35), (1.2, 2.4), radius=self.scale * 0.4)

    def _regenerate(self, ctx: EffectContext) -> None:
        c, rng = ctx.config, ctx.rng
        cx, cy = self.center
        targets = [tip for tip in self.tips]
        for _ in range(max(0, c.chidori_bolts - len(targets))):
            angle = rng.uniform(0, 2 * pi)
            reach = rng.uniform(*c.chidori_reach) * self.scale
            targets.append((cx + cos(angle) * reach, cy + sin(angle) * reach))
        start = (cx + rng.normal(0, self.scale * 0.05), cy + rng.normal(0, self.scale * 0.05))
        self.bolts = branching_bolts(start, targets[:c.chidori_bolts], rng, c.chidori_roughness,
                                     c.chidori_depth)

    def render(self, frame: np.ndarray, ctx: EffectContext) -> None:
        """Bolts plus a flickering blue-white core glow at the palm."""
        c = ctx.config
        level = self.intensity(ctx.now, c.chidori_fade_s) * self.flicker
        if self.center is None or level <= 0:
            return
        draw_bolts(frame, self.bolts, c.chidori_colors, c.chidori_widths, level)
        glow = ctx.sprites.blue_glow
        overlay_rgba(frame, glow, self.center, self.scale * 2.2 / glow.shape[1] * 2, 0.0, level, "add")
        core = ctx.sprites.white_glow
        overlay_rgba(frame, core, self.center, self.scale * 0.9 / core.shape[1] * 2, 0.0, level, "add")


class Chidori(ModeController):
    """JUTSU-mode lightning bindings: arm with a held fist, release by opening the hand."""

    def __init__(self, bus: EventBus, engine: GestureEngine, manager: EffectManager,
                 config: EffectsConfig) -> None:
        """Create the controller; attach() subscribes it."""
        super().__init__(bus)
        self.engine, self.manager, self.config = engine, manager, config
        self._armed: dict[str, float | None] = {}  # hand -> release deadline (None while fist held)
        self._active: dict[str, ChidoriEffect] = {}

    def bind(self) -> None:
        """The JUTSU lightning mapping."""
        self.on_hold(Gesture.FIST, self.config.chidori_arm_seconds, self._arm)
        self.subscribe(Gesture.FIST, EventKind.ENDED, self._fist_ended)
        self.subscribe(Gesture.OPEN_PALM, EventKind.STARTED, self._release)
        self.subscribe(Gesture.OPEN_PALM, EventKind.ENDED, self._palm_ended)

    def is_busy(self, hand: str) -> bool:
        """True while this hand is charging or casting Chidori (blocks fireball summons)."""
        return hand in self._armed or hand in self._active

    def _arm(self, event: GestureEvent) -> None:
        # Two fists at once are Susanoo's seal, not two Chidori charges.
        if event.hand not in self._active and not self.engine.is_active(BOTH, Gesture.TWO_FISTS):
            self._armed[event.hand] = None
            self.manager.hooks.emit("chidori_charge", hand=event.hand)

    def _fist_ended(self, event: GestureEvent) -> None:
        if event.hand in self._armed and self._armed[event.hand] is None:
            self._armed[event.hand] = event.timestamp + self.config.chidori_release_window_s

    def _release(self, event: GestureEvent) -> None:
        hand = event.hand
        if hand not in self._armed:
            return
        deadline = self._armed.pop(hand)
        if deadline is not None and event.timestamp > deadline:
            return
        effect = ChidoriEffect(hand, event.timestamp)
        self.manager.spawn(effect)
        self._active[hand] = effect
        self.manager.screen.flash(0.25)
        self.manager.hooks.emit("lightning_start", hand=hand, position=self.engine.palm_center(hand))

    def _palm_ended(self, event: GestureEvent) -> None:
        effect = self._active.pop(event.hand, None)
        if effect is not None:
            effect.end(event.timestamp)
            self.manager.hooks.emit("lightning_end", hand=event.hand)

    def update(self, ctx: EffectContext) -> None:
        """Expire missed releases, drop finished effects, and spark around charging fists."""
        for hand, deadline in list(self._armed.items()):
            if deadline is not None and ctx.now > deadline:
                del self._armed[hand]
        for hand, effect in list(self._active.items()):
            if not effect.alive:
                del self._active[hand]
        for hand in self.engine.tracked_hands():
            if not self.engine.visible(hand) or not self.engine.is_active(hand, Gesture.FIST):
                continue
            armed = hand in self._armed
            progress = 1.0 if armed else self.engine.hold_progress(
                hand, Gesture.FIST, self.config.chidori_arm_seconds)
            if progress <= 0 or hand in self._active:
                continue
            scale = self.engine.hand_scale(hand)
            count = int(ctx.rng.poisson((40 + 160 * progress) * ctx.dt))
            ctx.particles.sparks.emit_radial(count, self.engine.palm_center(hand),
                                             (scale * 0.5, scale * 2.5 * progress + scale), (0.1, 0.3),
                                             (1.0, 2.0), radius=scale * 0.6)

    def render(self, frame: np.ndarray, ctx: EffectContext) -> None:
        """Small crackles around an armed fist, so the user knows to open it."""
        for hand in self._armed:
            if not self.engine.visible(hand):
                continue
            center, scale = self.engine.palm_center(hand), self.engine.hand_scale(hand)
            targets = [(center[0] + cos(a) * scale * 0.9, center[1] + sin(a) * scale * 0.9)
                       for a in ctx.rng.uniform(0, 2 * pi, 3)]
            bolts = branching_bolts(center, targets, ctx.rng, 0.3, 3)
            draw_bolts(frame, bolts, self.config.chidori_colors, (5, 2, 1), 0.8)

    def on_detach(self) -> None:
        """Leaving JUTSU fades any active lightning and disarms charges."""
        for hand, effect in self._active.items():
            effect.end(self.manager.now)
            self.manager.hooks.emit("lightning_end", hand=hand)
        self._active.clear()
        self._armed.clear()
