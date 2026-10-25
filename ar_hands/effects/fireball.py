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
from ..gestures.events import EventBus, EventKind, GestureEvent, both_of, is_both, pair_of
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
RING_SCALE = 4  # Explosion rings are blurred at 1/4 size.


class Fireball(Projectile):
    """One fireball, held or flying. Radius is in pixels and follows the hand's scale.

    Drawn as a turbulent flame field (flame.py) into the light buffer, so it
    blooms and lights the hand and face; the air above it shimmers."""

    name = "fireball"
    layer = 30
    emissive = True

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

    def flow(self) -> tuple[float, float]:
        """Where the flames stream: up, bent back by motion (trailing behind when thrown)."""
        r = max(self.radius, 1.0)
        drag = 1.0 if self.thrown else 0.5
        fx, fy = -self.vel[0] * drag, -self.vel[1] * drag
        if not self.thrown:
            fy -= r * 6.0  # Hot gas rises.
        length = hypot(fx, fy)
        return (0.0, -1.0) if length < 1e-6 else (fx / length, fy / length)

    def _emit_flames(self, ctx: EffectContext, intensity: float) -> None:
        r = self.radius
        flow = np.asarray(self.flow())
        embers = int(ctx.rng.poisson(26 * (r / 50.0) * intensity * ctx.dt))
        if embers:  # Glowing sparks carried up on the hot air.
            ctx.particles.ember.emit(embers, self.pos + flow * r * 0.6, flow * r * 2.4, spread=r * 1.4,
                                     radius=r * 0.6, life=(0.6, 1.3), size=(0.9, 1.9))
        smoke = int(ctx.rng.poisson(9 * (r / 50.0) * intensity * ctx.dt))
        if smoke and not self.thrown:  # Thin smoke off the flame tips.
            ctx.particles.smoke.emit(smoke, self.pos + flow * r * 2.2, flow * r * 1.1, spread=r * 0.3,
                                     radius=r * 0.4, life=(0.9, 1.6), size=(r * 0.18, r * 0.3))

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

    def _size(self, ctx: EffectContext) -> tuple[float, float]:
        """(radius with flicker and spawn growth, brightness)."""
        now = ctx.now
        flicker = 1 + 0.05 * sin(now * 21 + self.phase) + 0.03 * sin(now * 37 + 2 * self.phase)
        brightness = (0.92 + 0.08 * sin(now * 29 + self.phase)) * self.fade(now, self.params(ctx))
        if self.state is EffectState.SPAWN:
            flicker *= max(0.05, self.age / SPAWN_SECONDS)
        return self.radius * flicker, brightness

    def render(self, light: np.ndarray, ctx: EffectContext) -> None:
        """Turbulent flames into the light buffer; a fading streak behind a thrown one."""
        radius, brightness = self._size(ctx)
        flames = ctx.manager.flames
        flow = self.flow()
        speed = 2.2 if self.thrown else 1.0  # Faster turbulence while rushing through the air.
        center = (float(self.pos[0]), float(self.pos[1]))
        if self.thrown:
            for index in (2, 1):  # Motion streak: older, cooler, coarser copies along the path.
                offset = -self.vel * 0.02 * index
                flames.draw(light, (center[0] + offset[0], center[1] + offset[1]), radius * (1 - 0.15 * index),
                            ctx.now * speed + index * 0.07, flow, brightness * 0.4 / index, 1.6,
                            self.phase + index, cool=0.3 * index, detail=0.5)
        # Hotter than a candle flame: a ball of fire with a big bright body.
        flames.draw(light, center, radius, ctx.now * speed, flow, brightness * 1.3, 1.6 if self.thrown else 1.0,
                    self.phase)

    def distort(self, frame: np.ndarray, ctx: EffectContext) -> None:
        """Heat haze above a held fireball."""
        if not self.thrown:
            radius, brightness = self._size(ctx)
            ctx.manager.distortion.haze(frame, (float(self.pos[0]), float(self.pos[1])), radius, ctx.now,
                                        brightness)


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
    emissive = True

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
            # The fire burst is drawn as a flame field; a few flying flames are enough.
            fire = self.style == "fire"
            system.emit_radial(int(ctx.config.explosion_particles * p * (0.15 if fire else 1.0)), self.center,
                               (r * 1.2, r * 3.6), (0.3, 0.7), (r * 0.05, r * 0.12),
                               radius=r * 0.15, upward_bias=r * 0.6 if fire else 0.0)
            if fire:
                # Dark smoke rolls up out of the fireball; hot embers fly out of it.
                ctx.particles.smoke.emit_radial(int(45 * p), self.center, (r * 0.4, r * 1.2), (1.0, 2.0),
                                                (r * 0.1, r * 0.2), radius=r * 0.35, upward_bias=r * 0.7)
                ctx.particles.ember.emit_radial(int(70 * p), self.center, (r * 2, r * 6), (0.4, 1.0),
                                                (1.0, 2.0), radius=r * 0.3)
            else:
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

    def distort(self, frame: np.ndarray, ctx: EffectContext) -> None:
        """The shockwave: a ring of compressed air that bends the picture as it expands."""
        t = min(1.0, self.age / (self.lifetime(ctx) * 0.8))
        if t < 1.0:
            ring = self.radius * (0.3 + 2.2 * (1 - (1 - t) ** 2))
            ctx.manager.distortion.shockwave(frame, self.center, ring, max(4.0, self.radius * 0.18),
                                             (1 - t) * min(1.5, self.power))

    def render(self, frame: np.ndarray, ctx: EffectContext) -> None:
        """A fire blast billows and cools; every blast has a ring, a core flash (and wind needles)."""
        t = min(1.0, self.age / self.lifetime(ctx))
        if self.style == "fire":
            # Grows fast, then rises and cools from yellow to dull red.
            grow = 1 - (1 - min(1.0, t * 1.6)) ** 2
            heat = max(0.0, 1 - t) ** 1.3
            center = (self.center[0], self.center[1] - self.radius * 0.35 * t)
            ctx.manager.flames.draw(frame, center, self.radius * (0.35 + 0.85 * grow), self.born * 3 + t * 2.5,
                                    (0.0, -1.0), 1.25 * heat * min(1.4, self.power), 0.35, self.born, cool=t,
                                    turbulence=1.4, billow=0.85)
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
            color = tuple(int(v * fade * 0.7) for v in ring_color)
            roi = frame[y0:y1, x0:x1]
            # The soft ring is drawn and blurred at quarter size: its wide blur at
            # full size cost about 20 ms a frame on a big blast.
            q = RING_SCALE
            small = np.zeros(((y1 - y0) // q + 1, (x1 - x0) // q + 1, 3), np.uint8)
            cv2.circle(small, ((cx - x0) // q, (cy - y0) // q), int(ring / q), color, max(1, thickness // q),
                       cv2.LINE_AA)
            cv2.GaussianBlur(small, (0, 0), max(0.6, thickness * 0.6 / q), dst=small)
            cv2.add(roi, cv2.resize(small, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR), dst=roi)
            if self._needles is not None:
                # Rasenshuriken: a sphere of countless microscopic wind blades (sharp, full size).
                needles = np.zeros_like(roi)
                for dx, dy, start, end in self._needles:
                    a = (int(cx - x0 + dx * ring * start), int(cy - y0 + dy * ring * start))
                    b = (int(cx - x0 + dx * ring * end), int(cy - y0 + dy * ring * end))
                    cv2.line(needles, a, b, color, 1, cv2.LINE_AA)
                cv2.add(roi, needles, dst=roi)
        glow = getattr(ctx.sprites, glow_name)
        # The fire burst already glows (and blooms); other blasts need the sprite's halo.
        strength = 0.18 if self.style == "fire" else 0.5
        overlay_rgba(frame, glow, self.center, self.radius * 1.5 * (0.6 + 0.5 * t) / glow.shape[1] * 2,
                     0.0, strength * fade * fade, "add")


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
        self._seal_until: dict[str, float] = {}  # Pair id -> when its seal expires.

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
        """Fireballs currently held, by owner (a hand id, or a pair id for a Great Fireball)."""
        return dict(self._balls)

    def _summon(self, event: GestureEvent) -> None:
        hand, now, c = event.hand, event.timestamp, self.config
        if (is_both(hand) or hand in self._balls or self.is_busy(hand)
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
        both = both_of(event.hand)
        owner = event.hand if event.hand in self._balls else (both if both in self._balls else None)
        if owner is None:
            return
        velocity = np.asarray(self.engine.velocity_px(event.hand), np.float64)
        if hypot(*velocity) < 1e-3:
            velocity = np.asarray(SWIPE_DIRECTIONS[event.gesture]) * self.config.fireball_min_launch_px
        self._launch(owner, (float(velocity[0]), float(velocity[1])), event.timestamp)

    def _close(self, event: GestureEvent) -> None:
        for owner in (event.hand, both_of(event.hand)):
            ball = self._balls.pop(owner, None)
            if ball is not None:
                ball.dissipate(event.timestamp)
                self.manager.hooks.emit("fireball_dissipate", hand=owner, reason="fist")

    def _seal(self, event: GestureEvent) -> None:
        self._seal_until[event.hand] = event.timestamp + self.config.great_fireball_window_s
        self.manager.screen.flash(0.18)
        self.manager.hooks.emit("jutsu_seal", position=self.engine.palm_center(event.hand))

    def _great(self, event: GestureEvent) -> None:
        both = event.hand
        two = self.engine.two_hands_of(both)
        if event.timestamp > self._seal_until.get(both, -inf) or both in self._balls or not two.present:
            return
        self._seal_until.pop(both, None)
        radius = max(20.0, two.distance_px * self.config.great_fireball_ratio / 2)
        ball = Fireball(both, two.midpoint, radius, event.timestamp, great=True)
        self.manager.spawn(ball)
        self._balls[both] = ball
        self.manager.hooks.emit("great_fireball", position=two.midpoint, radius=radius)

    def update(self, ctx: EffectContext) -> None:
        """Keep held fireballs on their hands; throw on a fast palm; charge-up sparks."""
        c, engine, now = self.config, self.engine, ctx.now
        for owner, ball in list(self._balls.items()):
            if not ball.held:
                del self._balls[owner]
                continue
            if is_both(owner):
                two = engine.two_hands_of(owner)
                if not two.present:
                    ball.mark_lost(now)
                    continue
                ball.follow(two.midpoint, max(20.0, two.distance_px * c.great_fireball_ratio / 2))
                velocity = [np.asarray(engine.velocity_px(hand)) for hand in pair_of(owner)]
                speeds = [hypot(*engine.velocity(hand)) for hand in pair_of(owner)]
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
        for both, until in self._seal_until.items():
            midpoint = engine.palm_center(both)
            if ctx.now <= until and midpoint is not None:
                pulse = 0.6 + 0.4 * sin(ctx.now * 12)
                size = engine.hand_scale(both) * 2.5
                overlay_rgba(frame, glow, midpoint, size / glow.shape[1] * 2, 0.0, 0.5 * pulse, "add")

    def on_detach(self) -> None:
        """Leaving JUTSU dissipates held fireballs; ones already flying finish their flight."""
        for owner, ball in self._balls.items():
            ball.dissipate(self.manager.now)
            self.manager.hooks.emit("fireball_dissipate", hand=owner, reason="mode_change")
        self._balls.clear()
        self._seal_until.clear()
