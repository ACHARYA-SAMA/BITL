"""Fireball jutsu: summon by holding an open palm, throw with a flick, explode.

A Fireball is a Projectile (projectile.py has the held -> flying -> impact
life it shares with Rasenshuriken and Hollow Purple). Bonus: PRAYER held
0.8 s, then HANDS_APART -> a Great Fireball sized by the hands.
"""
from __future__ import annotations
from collections.abc import Callable
from math import hypot, inf, pi, sin
from typing import TYPE_CHECKING
import cv2
import numpy as np
from ..config import EffectsConfig
from ..gestures.engine import GestureEngine
from ..gestures.events import BOTH, EventBus, EventKind, GestureEvent
from ..gestures.types import Gesture
from .base import Effect, EffectContext, EffectState, ModeController
from .compositing import overlay_rgba
from .projectile import FlightParams, Projectile

if TYPE_CHECKING:
    from .manager import EffectManager

Point = tuple[float, float]
SWIPES = (Gesture.SWIPE_LEFT, Gesture.SWIPE_RIGHT, Gesture.SWIPE_UP, Gesture.SWIPE_DOWN)
SWIPE_DIRECTIONS = {Gesture.SWIPE_LEFT: (-1.0, 0.0), Gesture.SWIPE_RIGHT: (1.0, 0.0),
                    Gesture.SWIPE_UP: (0.0, -1.0), Gesture.SWIPE_DOWN: (0.0, 1.0)}
SPAWN_SECONDS = 0.15


class Fireball(Projectile):
    """One fireball, held or flying. Radius is in pixels and follows the hand's scale."""

    name = "fireball"
    layer = 30

    def __init__(self, owner: str, position: Point, radius: float, now: float,
                 great: bool = False) -> None:
        """Create a fireball that grows in at ``position``."""
        super().__init__(owner, position, radius, now)
        self.great = great

    def params(self, ctx: EffectContext) -> FlightParams:
        """Fireball tuning from EffectsConfig."""
        c = ctx.config
        return FlightParams(c.fireball_spring, c.fireball_gravity, c.fireball_growth,
                            c.fireball_max_flight_s, c.fireball_margin_px, c.fireball_lost_hold_s,
                            c.fireball_dissipate_s, SPAWN_SECONDS)

    def _emit_flames(self, ctx: EffectContext, intensity: float) -> None:
        rate = ctx.config.fireball_emit_rate * (self.radius / 50.0) ** 1.2 * intensity
        count = int(ctx.rng.poisson(rate * ctx.dt))
        if not count:
            return
        r = self.radius
        drift = (-self.vel * 0.2) if self.thrown else np.array((0.0, -0.9 * r))
        ctx.particles.fire.emit(count, self.pos, drift, spread=r * 1.1, radius=r * 0.35,
                                life=(0.3, 0.65), size=(r * 0.16, r * 0.32))

    def on_held(self, ctx: EffectContext) -> None:
        """Full flames while held."""
        self._emit_flames(ctx, 1.0)

    def on_flight(self, ctx: EffectContext) -> None:
        """Lighter while flying: the smear already reads as speed."""
        self._emit_flames(ctx, 0.5)

    def on_dying(self, ctx: EffectContext, fade: float) -> None:
        """Rise and fade in smoke."""
        self.pos[1] -= 50 * ctx.dt
        ctx.particles.smoke.emit(int(ctx.rng.poisson(90 * ctx.dt)), self.pos, (0, -60),
                                 spread=40, radius=self.radius * 0.5, life=(0.7, 1.3),
                                 size=(self.radius * 0.25, self.radius * 0.45))
        self._emit_flames(ctx, fade)

    def explode(self, ctx: EffectContext) -> None:
        """Burst into a shockwave and remove this fireball."""
        center = (float(self.pos[0]), float(self.pos[1]))
        ctx.explode(center, self.radius * ctx.config.explosion_ratio, 1.6 if self.great else 1.0)
        ctx.emit("fireball_explode", hand=self.owner, position=center, radius=self.radius,
                 great=self.great)
        self.kill()

    def render(self, frame: np.ndarray, ctx: EffectContext) -> None:
        """Flickering additive sprite, plus a motion-blur smear while flying."""
        now = ctx.now
        sprite = ctx.sprites.fireball(int(now * 12))
        flicker = 1 + 0.07 * sin(now * 21 + self.phase) + 0.04 * sin(now * 37 + 2 * self.phase)
        opacity = (0.88 + 0.12 * sin(now * 29 + self.phase)) * self.fade(now, self.params(ctx))
        if self.state is EffectState.SPAWN:
            flicker *= max(0.05, self.age / SPAWN_SECONDS)
        # The sprite's flame body spans about 58% of its half-width.
        scale = self.radius / (0.58 * sprite.shape[1] / 2) * flicker
        center = (float(self.pos[0]), float(self.pos[1]))
        if self.thrown:
            copies = ctx.config.fireball_trail_copies
            for index in range(copies, 0, -1):
                offset = -self.vel * 0.011 * index
                overlay_rgba(frame, sprite, (center[0] + offset[0], center[1] + offset[1]),
                             scale * (1 - 0.08 * index), self.spin, opacity * 0.22 / index, "add")
        overlay_rgba(frame, sprite, center, scale, self.spin, opacity, "add")
        glow = ctx.sprites.white_glow
        overlay_rgba(frame, glow, center, self.radius * 1.3 / glow.shape[1], 0.0, 0.55 * opacity, "add")


# style -> (particle system for the burst, ring color BGR, glow sprite attribute)
EXPLOSION_STYLES = {
    "fire": ("fire", (90, 210, 255), "fire_glow"),
    "wind": ("sparks", (255, 245, 215), "blue_glow"),
    "purple": ("spirit", (255, 110, 230), "purple_glow"),
}


class Explosion(Effect):
    """Particle shockwave plus an expanding radial ring; flash and shake come from the manager.

    ``style`` changes the palette: "fire" (fireballs, TNT), "wind"
    (Rasenshuriken: a sphere of white wind needles), "purple" (Hollow Purple).
    """

    name = "explosion"
    layer = 25

    def __init__(self, center: Point, radius: float, power: float, now: float,
                 style: str = "fire") -> None:
        """Queue a blast; particles are emitted on the first update."""
        super().__init__()
        if style not in EXPLOSION_STYLES:
            raise ValueError(f"style must be one of {sorted(EXPLOSION_STYLES)}")
        self.center, self.radius, self.power, self.born, self.style = center, radius, power, now, style
        self._needles: np.ndarray | None = None

    def lifetime(self, ctx: EffectContext) -> float:
        """The wind sphere lingers twice as long as a fire ring."""
        return ctx.config.explosion_ring_s * (2.0 if self.style == "wind" else 1.0)

    def update(self, ctx: EffectContext) -> None:
        """Burst once, then let the ring expand until it is gone."""
        r, p = self.radius, self.power
        if self.state is EffectState.SPAWN:
            system = getattr(ctx.particles, EXPLOSION_STYLES[self.style][0])
            # Small, numerous particles read as a blast; big ones just sum to white.
            system.emit_radial(int(ctx.config.explosion_particles * p), self.center,
                               (r * 1.2, r * 3.6), (0.3, 0.7), (r * 0.05, r * 0.12),
                               radius=r * 0.15, upward_bias=r * 0.6 if self.style == "fire" else 0.0)
            if self.style == "fire":
                ctx.particles.smoke.emit_radial(int(30 * p), self.center, (r * 0.4, r * 1.2), (0.8, 1.5),
                                                (r * 0.08, r * 0.16), radius=r * 0.3, upward_bias=r * 0.5)
            ctx.particles.sparks.emit_radial(int(90 * p), self.center, (r * 3, r * 8), (0.25, 0.6),
                                             (1.5, 3.0))
            if self.style == "wind":
                angles = ctx.rng.uniform(0, 2 * pi, 90)
                self._needles = np.stack((np.cos(angles), np.sin(angles), ctx.rng.uniform(0.15, 0.6, 90),
                                          ctx.rng.uniform(0.6, 1.05, 90)), axis=1)
            self.state = EffectState.ACTIVE
        self.age += ctx.dt
        if self.age >= self.lifetime(ctx):
            self.kill()

    def render(self, frame: np.ndarray, ctx: EffectContext) -> None:
        """Expanding, thinning, fading ring with a hot core flash (and wind needles)."""
        t = min(1.0, self.age / self.lifetime(ctx))
        eased = 1 - (1 - t) ** 3
        ring = self.radius * (0.3 + 1.1 * eased)
        fade = 1 - t
        thickness = max(1, int(self.radius * 0.22 * fade))
        pad = int(ring + thickness * 3 + 4)
        height, width = frame.shape[:2]
        cx, cy = int(self.center[0]), int(self.center[1])
        x0, y0, x1, y1 = max(0, cx - pad), max(0, cy - pad), min(width, cx + pad), min(height, cy + pad)
        _, ring_color, glow_name = EXPLOSION_STYLES[self.style]
        if x0 < x1 and y0 < y1:
            layer = np.zeros((y1 - y0, x1 - x0, 3), np.uint8)
            color = tuple(int(v * fade) for v in ring_color)
            cv2.circle(layer, (cx - x0, cy - y0), int(ring), color, thickness, cv2.LINE_AA)
            if self._needles is not None:
                # Rasenshuriken: a sphere of countless microscopic wind blades.
                for dx, dy, start, end in self._needles:
                    a = (int(cx - x0 + dx * ring * start), int(cy - y0 + dy * ring * start))
                    b = (int(cx - x0 + dx * ring * end), int(cy - y0 + dy * ring * end))
                    cv2.line(layer, a, b, color, 1, cv2.LINE_AA)
            cv2.GaussianBlur(layer, (0, 0), max(1.0, thickness * 0.6), dst=layer)
            roi = frame[y0:y1, x0:x1]
            cv2.add(roi, layer, dst=roi)
        glow = getattr(ctx.sprites, glow_name)
        overlay_rgba(frame, glow, self.center, self.radius * 1.5 * (0.6 + 0.5 * t) / glow.shape[1] * 2,
                     0.0, 0.7 * fade * fade, "add")


class FireballJutsu(ModeController):
    """JUTSU-mode fireball bindings: summon, throw, dissipate, and the Great Fireball."""

    def __init__(self, bus: EventBus, engine: GestureEngine, manager: EffectManager,
                 config: EffectsConfig, is_busy: Callable[[str], bool] = lambda hand: False) -> None:
        """``is_busy`` lets Chidori block summons on a hand that is casting it."""
        super().__init__(bus)
        self.engine, self.manager, self.config, self.is_busy = engine, manager, config, is_busy
        self.hold_rings = ((Gesture.PRAYER, config.seal_hold_seconds),)
        self._balls: dict[str, Fireball] = {}
        self._last_summon: dict[str, float] = {}
        self._seal_until = -inf

    def bind(self) -> None:
        """The JUTSU fireball mapping, one line per gesture."""
        self.on_hold(Gesture.OPEN_PALM, self.config.fireball_hold_seconds, self._summon)
        for swipe in SWIPES:
            self.subscribe(swipe, EventKind.STARTED, self._swipe_throw)
        self.subscribe(Gesture.FIST, EventKind.STARTED, self._close)
        self.on_hold(Gesture.PRAYER, self.config.seal_hold_seconds, self._seal)
        self.subscribe(Gesture.HANDS_APART, EventKind.STARTED, self._great)

    @property
    def held(self) -> dict[str, Fireball]:
        """Fireballs currently held, by owner ("Left", "Right", or "Both")."""
        return dict(self._balls)

    def _summon(self, event: GestureEvent) -> None:
        hand, now, c = event.hand, event.timestamp, self.config
        if (hand == BOTH or hand in self._balls or self.is_busy(hand)
                or now - self._last_summon.get(hand, -inf) < c.fireball_cooldown_s):
            return
        center, scale = self.engine.palm_center(hand), self.engine.hand_scale(hand)
        if center is None or scale <= 0:
            return
        position = (center[0], center[1] - c.fireball_offset * scale)
        ball = Fireball(hand, position, scale * c.fireball_size / 2, now)
        self.manager.spawn(ball)
        self._balls[hand] = ball
        self._last_summon[hand] = now
        self.manager.hooks.emit("fireball_summon", hand=hand, position=position, radius=ball.radius)

    def _launch(self, owner: str, velocity_px: Point, now: float) -> None:
        ball = self._balls.pop(owner, None)
        if ball is None:
            return
        c = self.config
        velocity = np.asarray(velocity_px, np.float64) * c.fireball_launch_gain
        speed = float(hypot(*velocity))
        if speed < c.fireball_min_launch_px:
            velocity = (velocity / speed if speed > 1e-6 else np.array((0.0, -1.0))) * c.fireball_min_launch_px
        ball.throw((float(velocity[0]), float(velocity[1])), now)
        self.manager.hooks.emit("fireball_throw", hand=owner, velocity=(float(velocity[0]), float(velocity[1])),
                                great=ball.great)

    def _swipe_throw(self, event: GestureEvent) -> None:
        owner = event.hand if event.hand in self._balls else (BOTH if BOTH in self._balls else None)
        if owner is None:
            return
        velocity = np.asarray(self.engine.velocity_px(event.hand), np.float64)
        if hypot(*velocity) < 1e-3:
            velocity = np.asarray(SWIPE_DIRECTIONS[event.gesture]) * self.config.fireball_min_launch_px
        self._launch(owner, (float(velocity[0]), float(velocity[1])), event.timestamp)

    def _close(self, event: GestureEvent) -> None:
        for owner in (event.hand, BOTH):
            ball = self._balls.pop(owner, None)
            if ball is not None:
                ball.dissipate(event.timestamp)
                self.manager.hooks.emit("fireball_dissipate", hand=owner, reason="fist")

    def _seal(self, event: GestureEvent) -> None:
        self._seal_until = event.timestamp + self.config.great_fireball_window_s
        self.manager.screen.flash(0.18)
        self.manager.hooks.emit("jutsu_seal", position=self.engine.palm_center(BOTH))

    def _great(self, event: GestureEvent) -> None:
        two = self.engine.two_hands
        if event.timestamp > self._seal_until or BOTH in self._balls or not two.present:
            return
        self._seal_until = -inf
        radius = max(20.0, two.distance_px * self.config.great_fireball_ratio / 2)
        ball = Fireball(BOTH, two.midpoint, radius, event.timestamp, great=True)
        self.manager.spawn(ball)
        self._balls[BOTH] = ball
        self.manager.hooks.emit("great_fireball", position=two.midpoint, radius=radius)

    def update(self, ctx: EffectContext) -> None:
        """Keep held fireballs on their hands; throw on a fast palm; charge-up sparks."""
        c, engine, now = self.config, self.engine, ctx.now
        for owner, ball in list(self._balls.items()):
            if not ball.held:
                del self._balls[owner]
                continue
            if owner == BOTH:
                two = engine.two_hands
                if not two.present:
                    ball.mark_lost(now)
                    continue
                ball.follow(two.midpoint, max(20.0, two.distance_px * c.great_fireball_ratio / 2))
                velocity = [np.asarray(engine.velocity_px(hand)) for hand in ("Left", "Right")]
                speeds = [hypot(*engine.velocity(hand)) for hand in ("Left", "Right")]
                if ball.age > 0.3 and min(speeds) >= c.fireball_throw_speed:
                    self._launch(owner, tuple((velocity[0] + velocity[1]) / 2), now)
                continue
            if not engine.visible(owner):
                ball.mark_lost(now)
                continue
            center, scale = engine.palm_center(owner), engine.hand_scale(owner)
            ball.follow((center[0], center[1] - c.fireball_offset * scale), scale * c.fireball_size / 2)
            if ball.age > 0.2 and hypot(*engine.velocity(owner)) >= c.fireball_throw_speed:
                self._launch(owner, engine.velocity_px(owner), now)
        for hand in engine.tracked_hands():
            progress = self._charge(hand)
            if 0 < progress < 1 and engine.visible(hand):
                center, scale = engine.palm_center(hand), engine.hand_scale(hand)
                point = (center[0], center[1] - c.fireball_offset * scale)
                ctx.particles.fire.emit(int(ctx.rng.poisson(120 * progress * ctx.dt)), point,
                                        (0, -scale), spread=scale * 0.8, radius=scale * 0.4 * (1 - progress),
                                        life=(0.2, 0.4), size=(scale * 0.05, scale * 0.12))

    def _charge(self, hand: str) -> float:
        """0..1 summon progress while the palm is held open (0 if a summon is not possible)."""
        if (hand in self._balls or self.is_busy(hand)
                or not self.engine.is_active(hand, Gesture.OPEN_PALM)):
            return 0.0
        return self.engine.hold_progress(hand, Gesture.OPEN_PALM, self.config.fireball_hold_seconds)

    def render(self, frame: np.ndarray, ctx: EffectContext) -> None:
        """Growing charge glow above the palm, and a pulsing seal while it is armed."""
        c, engine = self.config, self.engine
        glow = ctx.sprites.fire_glow
        for hand in engine.tracked_hands():
            progress = self._charge(hand)
            if 0 < progress < 1 and engine.visible(hand):
                center, scale = engine.palm_center(hand), engine.hand_scale(hand)
                point = (center[0], center[1] - c.fireball_offset * scale)
                size = scale * c.fireball_size * (0.3 + 1.1 * progress)
                overlay_rgba(frame, glow, point, size / glow.shape[1] * 2, 0.0, 0.35 + 0.6 * progress, "add")
        midpoint = engine.palm_center(BOTH)
        if ctx.now <= self._seal_until and midpoint is not None:
            pulse = 0.6 + 0.4 * sin(ctx.now * 12)
            size = engine.hand_scale(BOTH) * 2.5
            overlay_rgba(frame, glow, midpoint, size / glow.shape[1] * 2, 0.0, 0.5 * pulse, "add")

    def on_detach(self) -> None:
        """Leaving JUTSU dissipates held fireballs; ones already flying finish their flight."""
        for owner, ball in self._balls.items():
            ball.dissipate(self.manager.now)
            self.manager.hooks.emit("fireball_dissipate", hand=owner, reason="mode_change")
        self._balls.clear()
        self._seal_until = -inf
