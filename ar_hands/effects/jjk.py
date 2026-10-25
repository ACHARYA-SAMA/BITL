"""Jujutsu Kaisen techniques: Lapse Blue, Reversal Red, and Hollow Purple.

  POINT held 0.4 s  -> Blue at the index fingertip: a gravity singularity. A
                       lens warps the picture around it, sparks spiral inward,
                       and nearby blocks are pulled in.
  PEACE held 0.4 s  -> Red between the two raised fingertips: repulsion, with
                       red lightning and outward pulses; blocks are pushed away.
  Blue + Red close  -> they fuse into Hollow Purple between the hands. Flick to
                       fire it: it erases a dark tunnel across the screen and
                       destroys every block it touches, then detonates off-screen.
The orb ends when its gesture ends. The Domain Expansions live in domains.py.
"""
from __future__ import annotations
from math import cos, hypot, pi, sin
from typing import TYPE_CHECKING
import cv2
import numpy as np
from ..config import EffectsConfig
from ..gestures.engine import GestureEngine
from ..gestures.events import BOTH, EventBus, EventKind, GestureEvent, both_of, pair_of
from ..gestures.features import Finger
from ..gestures.types import Gesture
from .base import Effect, EffectContext, EffectState, ModeController
from .compositing import Sprite, overlay_rgba
from .fireball import SWIPES
from .lightning import branching_bolts, draw_bolts
from .projectile import FlightParams, HeldProjectiles, Projectile

if TYPE_CHECKING:
    from .blocks import BlockWorld
    from .domains import DomainEffect
    from .manager import EffectManager

Point = tuple[float, float]
RED_BOLTS = ((40, 30, 230), (90, 110, 255), (230, 235, 255))


def make_purple_core(size: int = 192) -> np.ndarray:
    """Hollow Purple: red and blue halves swirling into a magenta-white center (BGRA)."""
    coords = (np.arange(size, dtype=np.float32) + 0.5) / size * 2 - 1
    u, v = np.meshgrid(coords, coords)
    radius = np.hypot(u, v)
    theta = np.arctan2(v, u) + 2.2 * radius  # The halves twist into each other.
    red_side = 0.5 + 0.5 * np.cos(theta)
    red = np.array((40, 30, 255), np.float32)
    blue = np.array((255, 70, 40), np.float32)
    magenta = np.array((255, 90, 255), np.float32)
    color = red * red_side[..., None] + blue * (1 - red_side)[..., None]
    center = np.exp(-(radius / 0.45) ** 2)[..., None]
    color = color * (1 - center) + magenta * center
    color = color * (1 - np.exp(-(radius / 0.16) ** 2))[..., None] + 255 * np.exp(-(radius / 0.16) ** 2)[..., None]
    alpha = np.clip(1.15 - radius / 0.85, 0, 1) ** 0.7
    return np.dstack((np.clip(color, 0, 255), alpha * 255)).astype(np.uint8)


def find_world(ctx: EffectContext) -> BlockWorld | None:
    """The persistent block world, if one exists (techniques interact with blocks)."""
    for effect in ctx.manager.effects:
        if effect.name == "blocks":
            return effect
    return None


class _Lens:
    """Cached remap fields that pinch the picture toward a center (Blue's gravity)."""

    def __init__(self) -> None:
        self._size = 0
        self._dx: np.ndarray | None = None
        self._dy: np.ndarray | None = None

    def fields(self, size: int, strength: float) -> tuple[np.ndarray, np.ndarray]:
        if size != self._size:
            coords = np.arange(size, dtype=np.float32) - size / 2
            x, y = np.meshgrid(coords, coords)
            d = np.hypot(x, y) / (size / 2)
            # Sample farther out than the pixel itself: content looks sucked inward.
            pull = np.clip(1 - d, 0, 1) ** 2
            self._dx, self._dy = x * pull, y * pull
            self._size = size
        return self._dx * strength, self._dy * strength

    def apply(self, frame: np.ndarray, center: Point, radius: float, strength: float) -> None:
        size = int(radius * 2) | 1
        x0, y0 = int(center[0] - size / 2), int(center[1] - size / 2)
        height, width = frame.shape[:2]
        if x0 < 0 or y0 < 0 or x0 + size > width or y0 + size > height or size < 9:
            return  # Near an edge: skip the lens rather than smear border pixels.
        dx, dy = self.fields(size, strength)
        grid = np.arange(size, dtype=np.float32)
        map_x = grid[None, :] + dx
        map_y = grid[:, None] + dy
        roi = frame[y0:y0 + size, x0:x0 + size]
        roi[:] = cv2.remap(roi.copy(), map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


class CursedOrb(Effect):
    """Blue (attract) or Red (repel), held at the fingertips."""

    layer = 34
    glows = True

    def __init__(self, kind: str, owner: str, position: Point, radius: float, now: float) -> None:
        """``kind`` is "blue" or "red"."""
        super().__init__()
        if kind not in ("blue", "red"):
            raise ValueError("kind must be 'blue' or 'red'")
        self.kind, self.owner, self.born = kind, owner, now
        self.name = kind
        self.pos = np.array(position, np.float64)
        self.target = self.pos.copy()
        self.radius = radius
        self.dying_since: float | None = None
        self.bolts: list[np.ndarray] = []
        self._lens = _Lens()

    def follow(self, target: Point, radius: float) -> None:
        """Track the fingertips."""
        self.target[:] = target
        self.radius = radius

    def release(self, now: float) -> None:
        """Collapse (blue implodes, red bursts)."""
        if self.dying_since is None:
            self.dying_since, self.state = now, EffectState.DYING

    def level(self, now: float) -> float:
        """0..1 presence: grows in, then fades on release."""
        grow = min(1.0, (now - self.born) / 0.25)
        fade = 1.0 if self.dying_since is None else max(0.0, 1 - (now - self.dying_since) / 0.3)
        return grow * fade

    def update(self, ctx: EffectContext) -> None:
        """Follow, emit particles, and push or pull nearby blocks."""
        level = self.level(ctx.now)
        if level <= 0 and self.dying_since is not None:
            self.kill()
            return
        if self.state is EffectState.SPAWN and ctx.now - self.born > 0.25:
            self.state = EffectState.ACTIVE
        self.age += ctx.dt
        self.pos += (self.target - self.pos) * min(1.0, 18 * ctx.dt)
        r = self.radius
        count = int(ctx.rng.poisson(220 * level * ctx.dt))
        if count:
            angles = ctx.rng.uniform(0, 2 * pi, count)
            ring = np.stack((np.cos(angles), np.sin(angles)), axis=1)
            if self.kind == "blue":
                # Spawn on a wide ring and fall inward with a swirl: gravity.
                start = self.pos + ring * r * ctx.rng.uniform(2.5, 4.0, (count, 1))
                velocity = -ring * r * 7 + np.stack((-ring[:, 1], ring[:, 0]), 1) * r * 5
                ctx.particles.sparks.emit(count, start, velocity, life=(0.2, 0.32), size=(1.2, 2.6))
            else:
                velocity = ring * r * ctx.rng.uniform(4, 9, (count, 1))
                ctx.particles.ember.emit(count, self.pos + ring * r * 0.6, velocity,
                                         life=(0.2, 0.45), size=(r * 0.05, r * 0.12))
        if self.kind == "red" and int(self.age * 30) % 2 == 0:
            targets = [(self.pos[0] + cos(a) * r * 2.2, self.pos[1] + sin(a) * r * 2.2)
                       for a in ctx.rng.uniform(0, 2 * pi, 5)]
            self.bolts = branching_bolts(tuple(self.pos), targets, ctx.rng, 0.3, 4)
        world = find_world(ctx)
        if world is not None and self.dying_since is None:
            reach = r * 7
            sign = 1.0 if self.kind == "blue" else -1.0
            for block in world.blocks:
                if block.held_by is not None or block.fading_since is not None:
                    continue
                dx, dy = self.pos[0] - block.center[0], self.pos[1] - block.center[1]
                distance = hypot(dx, dy)
                if 1 < distance < reach:
                    pull = sign * ctx.config.orb_force * (1 - distance / reach) * ctx.dt
                    block.vx += dx / distance * pull
                    block.vy += dy / distance * pull - (ctx.config.gravity * ctx.dt if sign > 0 else 0)
                    block.asleep = False

    def render(self, frame: np.ndarray, ctx: EffectContext) -> None:
        """Blue: space bends toward a dark singularity (light cannot make darkness, so
        this part is drawn on the picture). Red's look is all light, in glow()."""
        level = self.level(ctx.now)
        if level <= 0 or self.kind != "blue":
            return
        center = (float(self.pos[0]), float(self.pos[1]))
        self._lens.apply(frame, center, self.radius * 4.5, self.radius * 1.1 * level)

    def glow(self, light: np.ndarray, ctx: EffectContext) -> None:
        """Blue: halo and spinning rings around the core. Red: lightning, white-hot core, pulses."""
        level = self.level(ctx.now)
        if level <= 0:
            return
        center = (float(self.pos[0]), float(self.pos[1]))
        r = self.radius
        sprites = ctx.sprites
        if self.kind == "blue":
            overlay_rgba(light, sprites.blue_glow, center, r * 7 / sprites.blue_glow.shape[1], 0, 0.8 * level, "add")
            spin = ctx.now * 540
            for k, (ring_r, color) in enumerate(((1.1, (255, 255, 255)), (1.7, (255, 230, 120)), (2.4, (255, 160, 40)))):
                cv2.ellipse(light, (int(center[0]), int(center[1])), (int(r * ring_r), int(r * ring_r * 0.85)),
                            spin * (1 + k * 0.4) % 360, 0, 230 - k * 40,
                            tuple(int(c * level) for c in color), max(1, int(r * 0.08)), cv2.LINE_AA)
        else:
            draw_bolts(light, self.bolts, RED_BOLTS, (7, 3, 1), level)
            pulse = (ctx.now * 1.6) % 1.0
            cv2.circle(light, (int(center[0]), int(center[1])), int(r * (1.2 + 2.5 * pulse)),
                       tuple(int(c * (1 - pulse) * level) for c in (60, 60, 255)), max(1, int(r * 0.1)), cv2.LINE_AA)
            overlay_rgba(light, sprites.red_glow, center, r * 6 / sprites.red_glow.shape[1], 0, level, "add")
            overlay_rgba(light, sprites.white_glow, center, r * 2.2 / sprites.white_glow.shape[1], 0, level, "add")

    def distort(self, frame: np.ndarray, ctx: EffectContext) -> None:
        """Blue's singularity: a black core the light around it cannot fill."""
        level = self.level(ctx.now)
        if level > 0 and self.kind == "blue":
            cv2.circle(frame, (int(self.pos[0]), int(self.pos[1])), max(2, int(self.radius * 0.55 * level)),
                       (30, 6, 2), -1, cv2.LINE_AA)


class HollowPurple(Projectile):
    """The fused technique: held between the hands, then fired. It pierces."""

    name = "hollow_purple"
    layer = 36
    glows = True

    def __init__(self, position: Point, radius: float, now: float, owner: str = BOTH) -> None:
        """Form between one person's hands (``owner`` is their pair id)."""
        super().__init__(owner, position, radius, now)
        self.trail: list[tuple[float, float, float, float]] = []  # (x, y, radius, time)

    def params(self, ctx: EffectContext) -> FlightParams:
        """Hollow Purple tuning: no gravity, grows enormous as it travels."""
        c = ctx.config
        return FlightParams(140.0, 0.0, c.purple_growth, 3.0, 0, c.fireball_lost_hold_s, 0.5, 0.25)

    def on_held(self, ctx: EffectContext) -> None:
        """Violet sparks crackle around it while charging."""
        count = int(ctx.rng.poisson(160 * ctx.dt))
        ctx.particles.spirit.emit_radial(count, tuple(self.pos), (self.radius * 1.5, self.radius * 4),
                                         (0.15, 0.35), (self.radius * 0.05, self.radius * 0.12),
                                         radius=self.radius * 0.8)

    def on_flight(self, ctx: EffectContext) -> None:
        """Record the erased tunnel."""
        self.trail.append((float(self.pos[0]), float(self.pos[1]), self.radius, ctx.now))
        self.trail = [p for p in self.trail if ctx.now - p[3] < ctx.config.purple_trail_s]

    def on_hit(self, ctx: EffectContext) -> None:
        """It does not stop: everything it touches is erased."""
        world = find_world(ctx)
        if world is None:
            return
        for block in list(world.blocks):
            if block.held_by is None and hypot(block.center[0] - self.pos[0],
                                               block.center[1] - self.pos[1]) < self.radius + block.width * 0.6:
                world.smash(block, ctx.now, "hollow_purple")

    def on_exit(self, ctx: EffectContext) -> None:
        """Leaves the screen: a violet detonation at the edge."""
        x = float(np.clip(self.pos[0], 0, ctx.width))
        y = float(np.clip(self.pos[1], 0, ctx.height))
        ctx.explode((x, y), max(ctx.width, ctx.height) * 0.35, 1.6, style="purple")
        ctx.emit("hollow_purple_impact", position=(x, y))
        self.kill()

    def render(self, frame: np.ndarray, ctx: EffectContext) -> None:
        """Erased tunnel behind it, then the swirling sphere and its glow."""
        fade = self.fade(ctx.now, self.params(ctx))
        if len(self.trail) > 1:
            self._render_trail(frame, ctx)
        form = min(1.0, self.age / 0.8)
        center = (float(self.pos[0]), float(self.pos[1]))
        core: Sprite = ctx.sprites.purple_core
        overlay_rgba(frame, core, center, self.radius * 2.4 / core.shape[1] * (0.4 + 0.6 * form),
                     -ctx.now * 120 % 360, fade, "screen")

    def glow(self, light: np.ndarray, ctx: EffectContext) -> None:
        """The violet glow it casts (and the white heart of the sphere)."""
        fade = self.fade(ctx.now, self.params(ctx))
        center = (float(self.pos[0]), float(self.pos[1]))
        overlay_rgba(light, ctx.sprites.purple_glow, center, self.radius * 5 / 128, 0, 0.8 * fade, "add")
        overlay_rgba(light, ctx.sprites.white_glow, center, self.radius * 1.4 / 64, 0, 0.6 * fade, "add")

    def _render_trail(self, frame: np.ndarray, ctx: EffectContext) -> None:
        height, width = frame.shape[:2]
        small = np.zeros((height // 2, width // 2), np.float32)
        for (x0, y0, r0, t0), (x1, y1, r1, _) in zip(self.trail, self.trail[1:]):
            age = (ctx.now - t0) / ctx.config.purple_trail_s
            cv2.line(small, (int(x0 / 2), int(y0 / 2)), (int(x1 / 2), int(y1 / 2)), 1.0 - age,
                     max(1, int(r0 * 0.9)), cv2.LINE_AA)
        small = cv2.GaussianBlur(small, (0, 0), 3)
        edge = np.clip(cv2.GaussianBlur(small, (0, 0), 6) - small * 0.7, 0, 1)
        dark = cv2.resize(small, (width, height))[..., None]
        glow = cv2.resize(edge, (width, height))[..., None]
        # Erased space: darkened, violet-tinted, with a glowing rim.
        out = frame.astype(np.float32) * (1 - 0.8 * dark) + np.array((60, 10, 50), np.float32) * dark
        out += np.array((255, 80, 220), np.float32) * glow * 0.9
        frame[:] = np.clip(out, 0, 255).astype(np.uint8)


class CursedTechniques(ModeController):
    """JJK mode: Blue, Red, Hollow Purple (domains are DomainCaster in domains.py)."""

    def __init__(self, bus: EventBus, engine: GestureEngine, manager: EffectManager,
                 config: EffectsConfig) -> None:
        """Create the controller; attach() subscribes it."""
        super().__init__(bus)
        self.engine, self.manager, self.config = engine, manager, config
        self.orbs: dict[str, CursedOrb] = {}
        self.purple = HeldProjectiles(engine, manager, "hollow_purple", config.fireball_throw_speed,
                                      config.fireball_launch_gain, config.fireball_min_launch_px)

    def bind(self) -> None:
        """The JJK technique mapping."""
        self.on_hold(Gesture.POINT, self.config.orb_hold_seconds, lambda e: self._summon("blue", e))
        self.on_hold(Gesture.PEACE, self.config.orb_hold_seconds, lambda e: self._summon("red", e))
        self.subscribe(Gesture.POINT, EventKind.ENDED, lambda e: self._release("blue", e))
        self.subscribe(Gesture.PEACE, EventKind.ENDED, lambda e: self._release("red", e))
        for swipe in SWIPES:
            self.subscribe(swipe, EventKind.STARTED, self._fire)
        self.subscribe(Gesture.FIST, EventKind.STARTED, self._cancel)

    def _anchor(self, kind: str, hand: str) -> tuple[Point, float] | None:
        features = self.engine.features(hand)
        if features is None:
            return None
        if kind == "blue":
            point = features.tips[Finger.INDEX]
        else:
            index, middle = features.tips[Finger.INDEX], features.tips[Finger.MIDDLE]
            point = ((index[0] + middle[0]) / 2, (index[1] + middle[1]) / 2)
        return point, features.scale * self.config.orb_size / 2

    def _summon(self, kind: str, event: GestureEvent) -> None:
        hand = event.hand
        if hand in self.orbs or both_of(hand) in self.purple.balls:
            return
        anchor = self._anchor(kind, hand)
        if anchor is None:
            return
        orb = CursedOrb(kind, hand, anchor[0], anchor[1], event.timestamp)
        self.manager.spawn(orb)
        self.orbs[hand] = orb
        self.manager.hooks.emit(f"{kind}_start", hand=hand, position=anchor[0])

    def _release(self, kind: str, event: GestureEvent) -> None:
        orb = self.orbs.get(event.hand)
        if orb is not None and orb.kind == kind:
            del self.orbs[event.hand]
            orb.release(event.timestamp)
            self.manager.hooks.emit(f"{kind}_end", hand=event.hand)

    def _fire(self, event: GestureEvent) -> None:
        both = both_of(event.hand)
        if both in self.purple.balls and self.purple.balls[both].age > 0.6:
            self.purple.launch(both, self.engine.velocity_px(event.hand), event.timestamp)

    def _cancel(self, event: GestureEvent) -> None:
        self.purple.dissipate(both_of(event.hand), event.timestamp, "fist")

    def update(self, ctx: EffectContext) -> None:
        """Track orbs, fuse Blue + Red into Purple, and aim or fire Purple."""
        for hand, orb in list(self.orbs.items()):
            anchor = self._anchor(orb.kind, hand) if self.engine.visible(hand) else None
            if anchor is None and hand not in self.engine.tracked_hands():
                del self.orbs[hand]
                orb.release(ctx.now)
            elif anchor is not None:
                orb.follow(*anchor)
        people: dict[str, dict[str, CursedOrb]] = {}
        for orb in self.orbs.values():
            people.setdefault(both_of(orb.owner), {})[orb.kind] = orb
        for both, kinds in people.items():
            if "blue" in kinds and "red" in kinds:
                blue, red = kinds["blue"], kinds["red"]
                gap = hypot(*(blue.pos - red.pos))
                scale = (self.engine.hand_scale(blue.owner) + self.engine.hand_scale(red.owner)) / 2
                if gap < self.config.purple_fuse_distance * scale:
                    self._fuse(both, blue, red, scale, ctx)
        self.purple.prune()
        for both, purple in list(self.purple.balls.items()):
            hands = [h for h in pair_of(both) if self.engine.visible(h)]
            if not hands:
                purple.mark_lost(ctx.now)
                continue
            centers = [self.engine.palm_center(h) for h in hands]
            purple.follow((sum(c[0] for c in centers) / len(centers), sum(c[1] for c in centers) / len(centers)))
            fastest = max(hands, key=lambda h: hypot(*self.engine.velocity(h)))
            if purple.age > 0.6 and hypot(*self.engine.velocity(fastest)) >= self.config.fireball_throw_speed:
                self.purple.launch(both, self.engine.velocity_px(fastest), ctx.now)

    def _fuse(self, both: str, blue: CursedOrb, red: CursedOrb, scale: float, ctx: EffectContext) -> None:
        middle = tuple((blue.pos + red.pos) / 2)
        for orb in (blue, red):
            orb.release(ctx.now)
            self.orbs.pop(orb.owner, None)
        self.purple.hold(both, HollowPurple(middle, scale * self.config.purple_size / 2, ctx.now, both))
        self.manager.screen.flash(0.5)
        self.manager.screen.shake(12.0)
        ctx.particles.spirit.emit_radial(160, middle, (scale * 3, scale * 9), (0.2, 0.5), (2.0, 5.0))
        self.manager.hooks.emit("hollow_purple_form", position=middle)

    def render(self, frame: np.ndarray, ctx: EffectContext) -> None:
        """A small glow gathers at the fingertips while POINT or PEACE is held."""
        for hand in self.engine.tracked_hands():
            if hand in self.orbs or not self.engine.visible(hand):
                continue
            for gesture, kind, glow in ((Gesture.POINT, "blue", ctx.sprites.blue_glow),
                                        (Gesture.PEACE, "red", ctx.sprites.red_glow)):
                if not self.engine.is_active(hand, gesture):
                    continue
                progress = self.engine.hold_progress(hand, gesture, self.config.orb_hold_seconds)
                anchor = self._anchor(kind, hand)
                if anchor and 0 < progress < 1:
                    overlay_rgba(frame, glow, anchor[0], anchor[1] * 3 * progress / 128 * 2, 0, progress, "add")

    def on_detach(self) -> None:
        """Leaving JJK releases orbs and cancels a held Purple."""
        for orb in self.orbs.values():
            orb.release(self.manager.now)
        self.orbs.clear()
        self.purple.dissipate_all(self.manager.now, "mode_change")
