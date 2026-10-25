"""Shared life of hand-held, throwable orbs (fireball, Rasenshuriken, Hollow Purple).

  SPAWN (grows in) -> ACTIVE held: a critically damped spring follows the hand
     |-- flick / swipe --> ACTIVE in flight: gravity, growth, then impact
     |                     (off-screen, timeout, or touching blocks/beams)
     |-- cancel gesture, or hand lost past the grace --> DYING (fade) -> DEAD
Subclasses supply the look (render, on_held / on_flight / on_dying particles)
and what an impact does (explode, or pierce for Hollow Purple).
"""
from __future__ import annotations
from dataclasses import dataclass
from math import hypot, sqrt
from typing import TYPE_CHECKING
import numpy as np
from ..gestures.engine import GestureEngine
from .base import Effect, EffectContext, EffectState

if TYPE_CHECKING:
    from .manager import EffectManager

Point = tuple[float, float]


@dataclass(frozen=True)
class FlightParams:
    """Per-projectile physics and timing."""

    spring: float  # Follow stiffness (1/s^2).
    gravity: float  # px/s^2 while flying.
    growth: float  # Relative radius growth per second in flight (fake depth).
    max_flight_s: float
    margin_px: int  # Explodes this far past the frame edge.
    lost_hold_s: float  # Holds in place this long after its hand vanishes.
    dissipate_s: float
    spawn_s: float = 0.15


class Projectile(Effect):
    """Held, then thrown. Radius is in pixels and follows the hand's scale."""

    name = "projectile"
    layer = 30

    def __init__(self, owner: str, position: Point, radius: float, now: float) -> None:
        """Create a projectile that grows in at ``position``."""
        super().__init__()
        self.owner, self.born = owner, now
        self.pos = np.array(position, np.float64)
        self.vel = np.zeros(2, np.float64)
        self.target = self.pos.copy()
        self.radius = self.target_radius = float(radius)
        self.thrown_at: float | None = None
        self.lost_since: float | None = None
        self.dying_since: float | None = None
        self.phase = (now * 7.3) % 6.28  # Desynchronizes flicker between instances.
        self.spin = 0.0

    def params(self, ctx: EffectContext) -> FlightParams:
        """Subclasses return their tuning from EffectsConfig."""
        raise NotImplementedError

    @property
    def thrown(self) -> bool:
        """True once launched."""
        return self.thrown_at is not None

    @property
    def held(self) -> bool:
        """True while it follows a hand."""
        return not self.thrown and self.state in (EffectState.SPAWN, EffectState.ACTIVE)

    def follow(self, target: Point, radius: float | None = None) -> None:
        """Set where the spring pulls it (and its size); clears any lost-hand timer."""
        self.target[:] = target
        if radius is not None:
            self.target_radius = radius
        self.lost_since = None

    def mark_lost(self, now: float) -> None:
        """The owner hand vanished: hold position, then dissipate after the grace."""
        if self.lost_since is None:
            self.lost_since = now

    def throw(self, velocity: Point, now: float) -> None:
        """Launch along ``velocity`` (px/s)."""
        self.vel[:] = velocity
        self.thrown_at = now
        self.state = EffectState.ACTIVE

    def dissipate(self, now: float) -> None:
        """Fade out instead of exploding."""
        if self.dying_since is None:
            self.dying_since = now
            self.state = EffectState.DYING

    def fade(self, now: float, p: FlightParams) -> float:
        """1 while alive, 1 -> 0 while dissipating."""
        if self.dying_since is None:
            return 1.0
        return max(0.0, 1.0 - (now - self.dying_since) / p.dissipate_s)

    def update(self, ctx: EffectContext) -> None:
        """Spring-follow, fly, or fade; impact on contact or exit."""
        p, dt = self.params(ctx), ctx.dt
        self.age += dt
        self.spin = (self.spin + 50 * dt) % 360
        if self.state is EffectState.SPAWN and self.age >= p.spawn_s:
            self.state = EffectState.ACTIVE
        if self.state is EffectState.DYING:
            fade = self.fade(ctx.now, p)
            if fade <= 0:
                self.kill()
                return
            self.on_dying(ctx, fade)
            return
        if self.thrown:
            self.vel[1] += p.gravity * dt
            self.pos += self.vel * dt
            self.radius *= 1 + p.growth * dt
            flight = ctx.now - self.thrown_at
            margin = p.margin_px + self.radius
            x, y = float(self.pos[0]), float(self.pos[1])
            outside = x < -margin or y < -margin or x > ctx.width + margin or y > ctx.height + margin
            if outside or flight >= p.max_flight_s:
                self.on_exit(ctx)
                return
            if flight > 0.05 and ctx.hit((x, y), self.radius * 0.6):
                self.on_hit(ctx)
                if not self.alive:
                    return
            self.on_flight(ctx)
            return
        if self.lost_since is not None and ctx.now - self.lost_since >= p.lost_hold_s:
            self.dissipate(ctx.now)
            ctx.emit(f"{self.name}_dissipate", hand=self.owner, reason="hand_lost")
            return
        if self.lost_since is None:
            # Critically damped spring: smooth but never overshooting.
            k = p.spring
            acceleration = k * (self.target - self.pos) - 2 * sqrt(k) * self.vel
            self.vel += acceleration * dt
            self.pos += self.vel * dt
        else:
            self.vel[:] = 0  # Hand lost: hold in place.
        self.radius += (self.target_radius - self.radius) * min(1.0, 8 * dt)
        self.on_held(ctx)

    # Hooks for subclasses -------------------------------------------------
    def on_held(self, ctx: EffectContext) -> None:
        """Called each frame while held."""

    def on_flight(self, ctx: EffectContext) -> None:
        """Called each frame while flying."""

    def on_dying(self, ctx: EffectContext, fade: float) -> None:
        """Called each frame while dissipating (fade 1 -> 0)."""

    def on_hit(self, ctx: EffectContext) -> None:
        """Touched something solid in flight; default: explode."""
        self.explode(ctx)

    def on_exit(self, ctx: EffectContext) -> None:
        """Left the screen or timed out; default: explode."""
        self.explode(ctx)

    def explode(self, ctx: EffectContext) -> None:
        """Default impact: remove silently (subclasses add the spectacle)."""
        self.kill()


class HeldProjectiles:
    """The "one projectile per hand until thrown" bookkeeping shared by controllers."""

    def __init__(self, engine: GestureEngine, manager: EffectManager, hook: str,
                 throw_speed: float, launch_gain: float, min_launch_px: float) -> None:
        """``hook`` prefixes the emitted names: ``<hook>_throw``, ``<hook>_dissipate``."""
        self.engine, self.manager, self.hook = engine, manager, hook
        self.throw_speed, self.launch_gain, self.min_launch_px = throw_speed, launch_gain, min_launch_px
        self.balls: dict[str, Projectile] = {}

    def hold(self, owner: str, projectile: Projectile) -> Projectile:
        """Spawn and start following ``owner``."""
        self.manager.spawn(projectile)
        self.balls[owner] = projectile
        return projectile

    def launch(self, owner: str, velocity_px: Point, now: float) -> Projectile | None:
        """Throw ``owner``'s projectile along the hand velocity (never slower than the minimum)."""
        ball = self.balls.pop(owner, None)
        if ball is None:
            return None
        velocity = np.asarray(velocity_px, np.float64) * self.launch_gain
        speed = float(hypot(*velocity))
        if speed < self.min_launch_px:
            velocity = (velocity / speed if speed > 1e-6 else np.array((0.0, -1.0))) * self.min_launch_px
        ball.throw((float(velocity[0]), float(velocity[1])), now)
        self.manager.hooks.emit(f"{self.hook}_throw", hand=owner,
                                velocity=(float(velocity[0]), float(velocity[1])))
        return ball

    def dissipate(self, owner: str, now: float, reason: str) -> bool:
        """Cancel ``owner``'s projectile; True if there was one."""
        ball = self.balls.pop(owner, None)
        if ball is None:
            return False
        ball.dissipate(now)
        self.manager.hooks.emit(f"{self.hook}_dissipate", hand=owner, reason=reason)
        return True

    def dissipate_all(self, now: float, reason: str) -> None:
        """Cancel everything held (mode change)."""
        for owner in list(self.balls):
            self.dissipate(owner, now, reason)

    def prune(self) -> None:
        """Forget projectiles that were thrown or died."""
        for owner, ball in list(self.balls.items()):
            if not ball.held:
                del self.balls[owner]

    def follow_hand(self, owner: str, target: Point | None, radius: float, now: float,
                    min_age: float = 0.2) -> None:
        """Keep a single-hand projectile on its target; a fast palm throws it."""
        ball = self.balls.get(owner)
        if ball is None:
            return
        if target is None or not self.engine.visible(owner):
            ball.mark_lost(now)
            return
        ball.follow(target, radius)
        if ball.age > min_age and hypot(*self.engine.velocity(owner)) >= self.throw_speed:
            self.launch(owner, self.engine.velocity_px(owner), now)
