"""Minecraft-style blocks and two-hand neon beams with simple, stable 2D physics.

Interactions (BLOCKS mode):
  * Pinch empty space        -> a new block appears in your fingers; release drops it.
  * Pinch a block or beam    -> grab and move it (beam ends stretch, the middle moves it).
  * Pinch with BOTH hands, touch the pinches together, pull apart -> a neon beam
    stretches between them; release to place it. Beams are girders blocks land on.
  * Point at a block 0.5 s   -> it cracks in 4 stages and breaks (TNT ignites instead).
  * Peace sign               -> next block type.   Thumbs-up held 1 s -> clear all.

Physics in brief: gravity, a floor at 92% of the frame, axis-aligned box
collisions resolved lowest-block-first so stacks settle in one pass, a small
bounce, ground friction, and sleeping. A block that has been still for a few
frames stops being simulated, which removes resting jitter entirely; any
change to the world wakes everything so supports are re-checked.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from math import atan2, ceil, degrees, exp, hypot, inf
from typing import TYPE_CHECKING
import cv2
import numpy as np
from ..config import EffectsConfig
from ..gestures.engine import GestureEngine
from ..gestures.events import EventBus, EventKind, GestureEvent, pair_of
from ..gestures.types import Gesture
from .base import Effect, EffectContext, EffectState, HookEmitter, ModeController
from .compositing import overlay_rgba
from .particles import ParticleSet
from .sprites import SpriteLibrary

if TYPE_CHECKING:
    from .manager import EffectManager

Point = tuple[float, float]
PINCHES = (Gesture.PINCH, Gesture.OK)  # OK is a pinch with the other fingers out.


@dataclass(eq=False)
class Block:
    """One cube. (x, y) is the bottom-center of its collision box, which is the
    lower half of the isometric sprite, so stacked cubes line up face to face."""

    kind: str
    x: float
    y: float
    width: int
    born: float
    vx: float = 0.0
    vy: float = 0.0
    held_by: str | None = None
    fresh: bool = False  # Spawned by the pinch that still holds it.
    asleep: bool = False
    still: int = 0
    grounded: bool = False
    crack: float = 0.0
    fading_since: float | None = None
    fuse_at: float | None = None

    @property
    def height(self) -> float:
        """Collision box height: half the sprite."""
        return self.width / 2

    @property
    def left(self) -> float:
        """Collision box left edge."""
        return self.x - self.width / 2

    @property
    def right(self) -> float:
        """Collision box right edge."""
        return self.x + self.width / 2

    @property
    def top(self) -> float:
        """Collision box top edge."""
        return self.y - self.height

    @property
    def center(self) -> Point:
        """Center of the whole sprite (where the pinch holds it)."""
        return (self.x, self.y - self.width / 2)

    def contains(self, point: Point, pad: float = 0.0) -> bool:
        """True if the point lies on the sprite (its full width x width square)."""
        return (self.x - self.width / 2 - pad <= point[0] <= self.x + self.width / 2 + pad
                and self.y - self.width - pad <= point[1] <= self.y + pad)


@dataclass(eq=False)
class Beam:
    """A neon girder between two points; solid once placed."""

    p1: np.ndarray
    p2: np.ndarray
    thickness: float
    born: float
    placed: bool = False
    holders: set[str] = field(default_factory=set)
    cache: tuple[int, int, np.ndarray] | None = None  # (x0, y0, light) for static beams.

    @property
    def length(self) -> float:
        """Length in pixels."""
        return float(hypot(*(self.p2 - self.p1)))

    @property
    def slope_deg(self) -> float:
        """0 = horizontal, 90 = vertical."""
        dx, dy = self.p2 - self.p1
        return abs(degrees(atan2(abs(dy), abs(dx)))) if (dx or dy) else 0.0

    def moved(self) -> None:
        """Geometry changed: drop the cached glow."""
        self.cache = None

    def surface_y(self, x: float, max_slope: float) -> float | None:
        """Top surface y at x, or None if x is off the beam or it is too steep to stand on."""
        if not self.placed or self.slope_deg > max_slope:
            return None
        (x1, y1), (x2, y2) = self.p1, self.p2
        low, high = min(x1, x2), max(x1, x2)
        if not low <= x <= high or high - low < 1e-6:
            return None
        return float(y1 + (x - x1) * (y2 - y1) / (x2 - x1) - self.thickness / 2)

    def distance(self, point: Point) -> float:
        """Distance from a point to the segment."""
        p = np.asarray(point, np.float64)
        segment = self.p2 - self.p1
        denominator = float(segment @ segment)
        t = 0.0 if denominator == 0 else float(np.clip((p - self.p1) @ segment / denominator, 0, 1))
        return float(hypot(*(self.p1 + t * segment - p)))


@dataclass(eq=False)
class Grab:
    """What one pinching hand holds: a block, or a beam end ("p1"/"p2") or body."""

    block: Block | None = None
    beam: Beam | None = None
    part: str = ""
    last: np.ndarray | None = None  # Previous pinch point, for moving a beam body.


def draw_neon_segment(frame: np.ndarray, p1: Point, p2: Point, thickness: float,
                      color: tuple[int, int, int], intensity: float = 1.0,
                      cache: bool = False) -> tuple[int, int, np.ndarray] | None:
    """Additive neon bar (blurred halo, colored body, white core). With ``cache``,
    the computed light layer is returned (and not drawn) so it can be reused."""
    height, width = frame.shape[:2]
    pad = int(thickness * 3) + 6
    xs, ys = (p1[0], p2[0]), (p1[1], p2[1])
    x0, y0 = max(0, int(min(xs)) - pad), max(0, int(min(ys)) - pad)
    x1, y1 = min(width, int(max(xs)) + pad), min(height, int(max(ys)) + pad)
    if x0 >= x1 or y0 >= y1:
        return None
    a = (int(round(p1[0] - x0)), int(round(p1[1] - y0)))
    b = (int(round(p2[0] - x0)), int(round(p2[1] - y0)))
    t = max(2, int(thickness))
    # The halo is soft, so draw and blur it at half resolution (a quarter of the
    # pixels, half the blur radius) and upscale. A big close-up beam went from
    # ~50 ms to a few ms per frame while being stretched.
    roi_w, roi_h = x1 - x0, y1 - y0
    small = np.zeros((max(1, roi_h // 2), max(1, roi_w // 2), 3), np.uint8)
    cv2.line(small, (a[0] // 2, a[1] // 2), (b[0] // 2, b[1] // 2),
             tuple(int(v * 0.6) for v in color), max(1, int(t * 1.3)), cv2.LINE_AA)
    cv2.GaussianBlur(small, (0, 0), min(12.0, max(1.0, t * 0.4)), dst=small)
    halo = cv2.resize(small, (roi_w, roi_h), interpolation=cv2.INTER_LINEAR)
    body = np.zeros_like(halo)
    cv2.line(body, a, b, color, t, cv2.LINE_AA)
    core = tuple(int(v * 0.25 + 255 * 0.75) for v in color)
    cv2.line(body, a, b, core, max(1, int(t * 0.35)), cv2.LINE_AA)
    for end in (a, b):
        cv2.circle(body, end, int(t * 0.75), core, -1, cv2.LINE_AA)
    light = cv2.add(halo, body)
    if intensity != 1.0:
        light = cv2.convertScaleAbs(light, alpha=intensity)
    if cache:
        return (x0, y0, light)
    roi = frame[y0:y1, x0:x1]
    cv2.add(roi, light, dst=roi)
    return None


class BlockWorld(Effect):
    """All blocks and beams: physics, picking, breaking, explosions, rendering.

    It is persistent: it survives mode switches, so a build stays on screen and
    can be blown up from JUTSU mode with a fireball.
    """

    name = "blocks"
    layer = 0
    persistent = True

    def __init__(self, config: EffectsConfig, sprites: SpriteLibrary, particles: ParticleSet,
                 hooks: HookEmitter) -> None:
        """Create an empty world that uses the shared particles and hooks."""
        super().__init__()
        self.config, self.sprites, self.particles, self.hooks = config, sprites, particles, hooks
        self.blocks: list[Block] = []
        self.beams: list[Beam] = []
        self.state = EffectState.ACTIVE
        self.now = 0.0
        # One size per build, set by the first block (from that hand's distance),
        # so blocks tile into columns and stack face to face like Minecraft.
        self.block_width: int | None = None

    # ----------------------------------------------------------------- queries
    @property
    def live_blocks(self) -> list[Block]:
        """Blocks that are not fading out."""
        return [b for b in self.blocks if b.fading_since is None]

    def block_at(self, point: Point, include_held: bool = False, pad: float = 0.0) -> Block | None:
        """Topmost block under (or within ``pad`` pixels of) a point; exact hits win."""
        order = [b for b in reversed(self._draw_order())
                 if b.fading_since is None and (include_held or b.held_by is None)]
        for block in order:
            if block.contains(point):
                return block
        if pad > 0:
            near = [b for b in order if b.contains(point, pad)]
            if near:
                return min(near, key=lambda b: hypot(b.center[0] - point[0], b.center[1] - point[1]))
        return None

    def pick(self, point: Point, radius: float, block_pad: float = 0.0) -> Grab | None:
        """What a pinch at ``point`` grabs: a beam end, then a block, then a beam body."""
        p = np.asarray(point, np.float64)
        for beam in reversed(self.beams):
            for part in ("p1", "p2"):
                if hypot(*(getattr(beam, part) - p)) <= radius:
                    return Grab(beam=beam, part=part, last=p)
        block = self.block_at(point, pad=block_pad)
        if block is not None:
            return Grab(block=block)
        for beam in reversed(self.beams):
            if beam.distance(point) <= beam.thickness / 2 + radius * 0.5:
                return Grab(beam=beam, part="body", last=p)
        return None

    def hit_test(self, center: Point, radius: float) -> bool:
        """Circle vs. block sprites and beams (fireball collisions)."""
        cx, cy = center
        for block in self.blocks:
            if block.fading_since is not None or block.held_by is not None:
                continue
            nearest_x = min(max(cx, block.left), block.right)
            nearest_y = min(max(cy, block.y - block.width), block.y)
            if hypot(cx - nearest_x, cy - nearest_y) <= radius:
                return True
        return any(beam.placed and beam.distance(center) <= radius + beam.thickness / 2
                   for beam in self.beams)

    # --------------------------------------------------------------- mutation
    def wake(self) -> None:
        """Re-simulate every block (after anything that could remove a support)."""
        for block in self.blocks:
            block.asleep, block.still = False, 0

    def spawn_block(self, kind: str, center: Point, width: int, now: float) -> Block:
        """Add a block whose sprite is centered on ``center``; the oldest fades past the cap."""
        block = Block(kind, float(center[0]), float(center[1] + width / 2), int(width), now)
        live = [b for b in self.live_blocks if b.held_by is None]
        while len(live) >= self.config.max_blocks:
            oldest = min(live, key=lambda b: b.born)
            oldest.fading_since = now
            live.remove(oldest)
        self.blocks.append(block)
        return block

    def hold_block(self, block: Block, hand: str) -> None:
        """Start following a hand."""
        block.held_by, block.asleep, block.crack = hand, False, 0.0
        self.wake()  # Whatever stood on it must fall.

    def drag(self, grab: Grab, point: Point) -> None:
        """Move what a hand holds toward the pinch point."""
        p = np.asarray(point, np.float64)
        if grab.beam is not None:
            beam = grab.beam
            if grab.part == "body" and grab.last is not None:
                delta = p - grab.last
                beam.p1, beam.p2 = beam.p1 + delta, beam.p2 + delta
            elif grab.part in ("p1", "p2"):
                setattr(beam, grab.part, p.copy())
            grab.last = p
            beam.moved()
            self.wake()
        elif grab.block is not None:
            grab.last = p

    def release_block(self, block: Block, frame_width: int | None = None) -> None:
        """Let go: a gentle release snaps into the nearest grid column and drops
        straight; a fast one is a throw and keeps its velocity. Overlaps settle."""
        c = self.config
        block.held_by, block.fresh = None, False
        speed = hypot(block.vx, block.vy)
        if speed > c.max_throw_px:
            block.vx *= c.max_throw_px / speed
            block.vy *= c.max_throw_px / speed
        elif c.snap_to_grid and speed <= c.grid_snap_max_speed:
            w = block.width
            column = (int(block.x // w) + 0.5) * w
            if frame_width is not None:
                column = min(max(column, w / 2), frame_width - w / 2)
            block.x, block.vx, block.vy = column, 0.0, max(0.0, block.vy)
        # Dropped inside another block? Rest it on top instead of inside.
        for _ in range(8):
            others = [o for o in self.blocks if o is not block and o.held_by is None
                      and o.fading_since is None and self._overlap(block, o)]
            if not others:
                break
            block.y = min(o.top for o in others)
            block.vy = min(block.vy, 0.0)
        self.wake()

    def remove_block(self, block: Block, debris: bool = True) -> None:
        """Delete a block, optionally bursting into pixel chunks of its colors."""
        if block in self.blocks:
            self.blocks.remove(block)
            if debris:
                count = max(8, int(block.width * 0.25))
                self.particles.chunks.emit_radial(count, block.center, (150, 650), (0.5, 1.0),
                                                  (block.width / 24, block.width / 12),
                                                  radius=block.width * 0.3,
                                                  color=self.sprites.palette(block.kind),
                                                  upward_bias=350)
            self.wake()

    def break_block(self, block: Block, now: float) -> None:
        """Pointing broke it: TNT ignites, everything else shatters."""
        if block.kind == "tnt":
            if block.fuse_at is None:
                block.fuse_at = now + self.config.tnt_fuse_s
                self.hooks.emit("tnt_ignite", position=block.center)
            block.crack = 0.0
            return
        self.remove_block(block)
        self.hooks.emit("block_break", kind=block.kind, position=block.center, reason="pointed")

    def add_beam(self, p1: Point, p2: Point, thickness: float, now: float) -> Beam:
        """Create an unplaced (still stretching) beam; the oldest goes past the cap."""
        while len(self.beams) >= self.config.max_beams:
            self.beams.pop(0)
        beam = Beam(np.asarray(p1, np.float64), np.asarray(p2, np.float64), thickness, now)
        self.beams.append(beam)
        return beam

    def remove_beam(self, beam: Beam) -> None:
        """Delete a beam; blocks resting on it fall."""
        if beam in self.beams:
            self.beams.remove(beam)
            self.wake()

    def clear(self) -> None:
        """Remove everything with a burst of debris; the next block sets a new size."""
        for block in list(self.blocks):
            self.remove_block(block, debris=block.fading_since is None)
        self.beams.clear()
        self.block_width = None
        self.hooks.emit("blocks_clear")

    def width_for(self, scale: float) -> int:
        """Block width for a new block: the build's size, or a new one from the hand."""
        if self.block_width is None or not self.live_blocks:
            c = self.config
            width = max(c.block_min_px, scale * c.block_size)
            self.block_width = int(round(width / c.block_size_quantum) * c.block_size_quantum)
        return self.block_width

    def smash(self, block: Block, now: float, reason: str) -> None:
        """Break a block immediately (a punch): TNT goes off almost at once."""
        if block.kind == "tnt":
            block.fuse_at = min(block.fuse_at or now + 0.12, now + 0.12)
            return
        self.remove_block(block)
        self.hooks.emit("block_break", kind=block.kind, position=block.center, reason=reason)

    @staticmethod
    def _overlap(a: Block, b: Block) -> bool:
        return (min(a.right, b.right) - max(a.left, b.left) > 1
                and min(a.y, b.y) - max(a.top, b.top) > 1)

    # ------------------------------------------------------------- simulation
    def update(self, ctx: EffectContext) -> None:
        """Held springs, physics substeps, fades, and TNT fuses."""
        c = self.config
        self.now = ctx.now
        self.age += ctx.dt
        for block in list(self.blocks):
            if block not in self.blocks:
                continue  # Already removed by a chain reaction earlier in this loop.
            if block.fading_since is not None and ctx.now - block.fading_since >= c.fade_out_s:
                self.blocks.remove(block)
                self.wake()
            elif block.fuse_at is not None and ctx.now >= block.fuse_at:
                self.remove_block(block)
                ctx.explode(block.center, block.width * c.tnt_radius, 1.2)
                self.hooks.emit("tnt_explode", position=block.center)
        steps = max(1, ceil(ctx.dt / 0.02))
        for _ in range(steps):
            self._step(ctx.dt / steps, ctx)

    def update_held(self, block: Block, target: Point, dt: float) -> None:
        """Critically damped spring from the block center toward the pinch point."""
        k = self.config.grab_spring
        cx, cy = block.center
        ax = k * (target[0] - cx) - 2 * k ** 0.5 * block.vx
        ay = k * (target[1] - cy) - 2 * k ** 0.5 * block.vy
        block.vx += ax * dt
        block.vy += ay * dt
        block.x += block.vx * dt
        block.y += block.vy * dt

    def _step(self, h: float, ctx: EffectContext) -> None:
        c = self.config
        floor = ctx.height * c.floor_ratio
        solids = [b for b in self.blocks if b.held_by is None and b.fading_since is None]
        movers = [b for b in solids if not b.asleep]
        # Lowest first: a support settles before the block resting on it moves.
        movers.sort(key=lambda b: -b.y)
        for block in movers:
            block.vy += c.gravity * h
            if block.vx:
                self._move_horizontal(block, block.x + block.vx * h, solids, ctx.width)
            previous = block.y
            target_y = block.y + block.vy * h
            support, beam = self._support(block, previous, target_y, solids, floor)
            landed = target_y >= support
            if landed:
                target_y = support
                impact = block.vy
                block.vy = -impact * c.restitution if impact > c.bounce_min_speed else 0.0
                if impact > 900:
                    self.hooks.emit("block_land", kind=block.kind, speed=impact)
                    self.particles.smoke.emit(6, (block.x, support), (0, -40), spread=60,
                                              radius=block.width * 0.25, life=(0.3, 0.6),
                                              size=(block.width * 0.06, block.width * 0.12))
            block.y = target_y
            block.grounded = landed
            self._settle(block, beam, h)

    def _move_horizontal(self, block: Block, target_x: float, solids: list[Block], width: int) -> None:
        for other in solids:
            if other is block or not (block.top < other.y - 1 and block.y > other.top + 1):
                continue
            if target_x + block.width / 2 > other.left and target_x - block.width / 2 < other.right:
                target_x = (other.left - block.width / 2 if block.x <= other.x
                            else other.right + block.width / 2)
                block.vx = 0.0
        half = block.width / 2
        if target_x < half or target_x > width - half:
            target_x = min(max(target_x, half), width - half)
            block.vx = -block.vx * 0.3
        block.x = target_x

    def _support(self, block: Block, previous: float, target: float, solids: list[Block],
                 floor: float) -> tuple[float, Beam | None]:
        """Highest surface the block's bottom crosses this step (floor, block tops, beams)."""
        support, support_beam = floor, None
        if block.vy < 0:
            return inf, None  # Moving up: nothing to land on.
        for other in solids:
            if other is block:
                continue
            overlap = min(block.right, other.right) - max(block.left, other.left)
            if overlap <= 0 or not previous - 0.5 <= other.top <= target + 0.5:
                continue
            if overlap < 0.25 * min(block.width, other.width):
                # Barely on the corner: slide off the edge instead of balancing on it.
                block.x += (overlap + 0.5) * (1 if block.x >= other.x else -1)
                continue
            if other.top < support:
                support = other.top
        for beam in self.beams:
            y = beam.surface_y(block.x, self.config.beam_platform_max_slope_deg)
            if y is not None and previous - 0.5 <= y <= target + 0.5 and y < support:
                support, support_beam = y, beam
        return support, support_beam

    def _settle(self, block: Block, beam: Beam | None, h: float) -> None:
        """Friction, sliding down sloped beams, and sleeping."""
        c = self.config
        if not block.grounded:
            block.still = 0
            return
        sliding = beam is not None and beam.slope_deg >= c.beam_slide_slope_deg
        if sliding:
            dx, dy = beam.p2 - beam.p1
            downhill = 1.0 if (dy / dx if dx else 0.0) > 0 else -1.0
            angle = atan2(abs(dy), abs(dx))
            block.vx += downhill * c.gravity * np.sin(angle) * np.cos(angle) * h
            block.vx *= exp(-c.ground_friction * 0.15 * h)
            block.still = 0
            return
        block.vx *= exp(-c.ground_friction * h)
        if abs(block.vx) < c.sleep_speed and abs(block.vy) < c.sleep_speed:
            block.still += 1
            if block.still >= c.sleep_frames:
                block.asleep, block.vx, block.vy = True, 0.0, 0.0
        else:
            block.still = 0

    def on_explosion(self, center: Point, radius: float, power: float, ctx: EffectContext) -> None:
        """Shatter blocks near the center, fling the rest, and set off nearby TNT."""
        c = self.config
        for block in list(self.blocks):
            if block.held_by is not None or block.fading_since is not None:
                continue
            bx, by = block.center
            distance = hypot(bx - center[0], by - center[1])
            reach = radius + block.width / 2
            if distance > reach:
                continue
            if block.kind == "tnt":
                fuse = ctx.now + c.tnt_chain_fuse_s
                if block.fuse_at is None or block.fuse_at > fuse:
                    block.fuse_at = fuse
            elif distance < radius * 0.45:
                self.remove_block(block)
                self.hooks.emit("block_break", kind=block.kind, position=block.center, reason="explosion")
            else:
                falloff = 1 - distance / reach
                direction = ((bx - center[0]) / max(distance, 1e-6), (by - center[1]) / max(distance, 1e-6))
                speed = 1500 * power * falloff
                block.vx += direction[0] * speed
                block.vy += direction[1] * speed - 400 * falloff
        self.wake()

    # -------------------------------------------------------------- rendering
    def _draw_order(self) -> list[Block]:
        # Iso stacking: a higher cube's bottom faces cover the lower cube's top,
        # so lower cubes draw first; held blocks draw last, above everything.
        resting = sorted((b for b in self.blocks if b.held_by is None), key=lambda b: -b.y)
        return resting + [b for b in self.blocks if b.held_by is not None]

    def render(self, frame: np.ndarray, ctx: EffectContext) -> None:
        """Beams (static ones from cache), energy pulses, then blocks back to front."""
        c = self.config
        for beam in self.beams:
            if beam.placed and not beam.holders:
                if beam.cache is None:
                    beam.cache = draw_neon_segment(frame, tuple(beam.p1), tuple(beam.p2),
                                                   beam.thickness, c.beam_color, cache=True)
                if beam.cache is not None:
                    x0, y0, light = beam.cache
                    roi = frame[y0:y0 + light.shape[0], x0:x0 + light.shape[1]]
                    if roi.shape == light.shape:
                        cv2.add(roi, light, dst=roi)
            else:
                draw_neon_segment(frame, tuple(beam.p1), tuple(beam.p2), beam.thickness,
                                  c.beam_color, 0.75 if not beam.placed else 1.0)
            if beam.placed:
                glow = ctx.sprites.white_glow
                for offset in (0.0, 0.5):
                    t = (ctx.now * c.beam_pulse_speed + offset) % 1.0
                    point = beam.p1 + (beam.p2 - beam.p1) * t
                    overlay_rgba(frame, glow, (float(point[0]), float(point[1])),
                                 beam.thickness * 2.2 / glow.shape[1] * 2, 0.0, 0.8, "add")
        for block in self._draw_order():
            stage = min(4, int(block.crack * 4) + 1) if block.crack > 0 else 0  # 4 crack stages.
            sprite = self.sprites.block(block.kind, block.width, stage)
            opacity = 1.0
            if block.fading_since is not None:
                opacity = max(0.0, 1 - (ctx.now - block.fading_since) / c.fade_out_s)
            if block.held_by is not None:
                glow = ctx.sprites.blue_glow
                overlay_rgba(frame, glow, block.center, block.width * 1.7 / glow.shape[1], 0.0, 0.45, "add")
            overlay_rgba(frame, sprite, block.center, opacity=opacity)
            if block.fuse_at is not None and int(ctx.now * 8) % 2 == 0:
                overlay_rgba(frame, sprite, block.center, opacity=0.6, mode="add")


class BlockBuilder(ModeController):
    """BLOCKS-mode bindings: pinch to spawn or grab, two-hand pinch for beams,
    point to break, peace to change type, thumbs-up hold to clear."""

    def __init__(self, bus: EventBus, engine: GestureEngine, manager: EffectManager,
                 world: BlockWorld, config: EffectsConfig) -> None:
        """Create the controller; attach() subscribes it."""
        super().__init__(bus)
        self.engine, self.manager, self.world, self.config = engine, manager, world, config
        self.hold_rings = ((Gesture.THUMBS_UP, config.clear_hold_seconds),)
        self.kind_index = 0
        self._grabs: dict[str, Grab] = {}
        self._pinching: dict[str, bool] = {}
        self._live_beams: dict[str, Beam] = {}  # Pair id -> beam being stretched.
        self._last_punch: dict[str, float] = {}

    @property
    def kind(self) -> str:
        """Block type the next pinch will create."""
        return self.config.block_kinds[self.kind_index % len(self.config.block_kinds)]

    @property
    def dragging(self) -> bool:
        """True while any hand holds a block or a beam."""
        return bool(self._grabs)

    def bind(self) -> None:
        """The BLOCKS mapping; pinches are polled in update() because PINCH and OK
        are one "grab" here and a PINCH->OK change must not drop the block."""
        self.subscribe(Gesture.PEACE, EventKind.STARTED, self._next_kind)
        self.on_hold(Gesture.THUMBS_UP, self.config.clear_hold_seconds, self._clear)

    def _next_kind(self, event: GestureEvent) -> None:
        self.kind_index = (self.kind_index + 1) % len(self.config.block_kinds)
        self.manager.hooks.emit("block_select", kind=self.kind)

    def _clear(self, event: GestureEvent) -> None:
        self._release_all()
        self.world.clear()

    def update(self, ctx: EffectContext) -> None:
        """Pinch edges -> grab/spawn/release; drags; two-hand beams; cracks; punches."""
        engine = self.engine
        tracked = engine.tracked_hands()
        for hand in sorted(set(tracked) | set(self._pinching)):
            pinching = hand in tracked and engine.active(hand) in PINCHES
            was = self._pinching.get(hand, False)
            point = engine.position(hand)
            if pinching and not was and point is not None:
                self._pinch_start(hand, point, ctx)
            elif was and not pinching:
                self._pinch_end(hand, ctx)
            elif pinching and point is not None and engine.visible(hand):
                self._pinch_move(hand, point, ctx)
            self._pinching[hand] = pinching
            if not pinching and hand not in tracked:
                del self._pinching[hand]
        for both in engine.tracked_pairs():
            self._try_connect(both, ctx)
        self._update_cracks(ctx)
        self._update_punches(ctx)

    def _pinch_start(self, hand: str, point: Point, ctx: EffectContext) -> None:
        scale = max(1.0, self.engine.hand_scale(hand))
        c = self.config
        grab = self.world.pick(point, c.grab_radius * scale, c.block_pick_pad * scale)
        if grab is None:
            block = self.world.spawn_block(self.kind, point, self.world.width_for(scale), ctx.now)
            block.fresh = True
            grab = Grab(block=block)
            self.manager.hooks.emit("block_place", kind=block.kind, position=point, hand=hand)
        else:
            self.manager.hooks.emit("block_grab" if grab.block else "beam_grab", hand=hand)
        if grab.block is not None:
            self.world.hold_block(grab.block, hand)
        if grab.beam is not None:
            grab.beam.holders.add(hand)
        self._grabs[hand] = grab

    def _pinch_move(self, hand: str, point: Point, ctx: EffectContext) -> None:
        grab = self._grabs.get(hand)
        if grab is None:
            return
        if self.engine.raw(hand) not in PINCHES:
            # Fingers are opening: the stable PINCH lasts a few grace frames, and the
            # thumb-index midpoint drifts as they open. Hold still so release is in place.
            grab.last = np.asarray(point, np.float64)
            if grab.block is not None:
                self.world.update_held(grab.block, grab.block.center, ctx.dt)
            return
        if grab.block is not None:
            self.world.update_held(grab.block, point, ctx.dt)
        else:
            self.world.drag(grab, point)

    def _pinch_end(self, hand: str, ctx: EffectContext) -> None:
        grab = self._grabs.pop(hand, None)
        if grab is None:
            return
        if grab.block is not None and grab.block in self.world.blocks:
            self.world.release_block(grab.block, ctx.width)
            self.manager.hooks.emit("block_release", kind=grab.block.kind, hand=hand)
        if grab.beam is not None:
            grab.beam.holders.discard(hand)
            grab.beam.moved()
            for both, beam in list(self._live_beams.items()):
                if grab.beam is beam:
                    self._finish_beam(both, ctx)

    def _can_connect(self, both: str) -> bool:
        """Both of one person's hands pinch, each empty-handed or holding the block it just spawned."""
        hands = pair_of(both)
        if both in self._live_beams or not all(self._pinching.get(h) for h in hands):
            return False
        return all(g is None or (g.block is not None and g.block.fresh)
                   for g in (self._grabs.get(hand) for hand in hands))

    def _try_connect(self, both: str, ctx: EffectContext) -> None:
        """Pinches touching -> the fresh blocks fuse into a live beam held at both ends."""
        if not self._can_connect(both):
            return
        hands = pair_of(both)
        grabs = [self._grabs.get(hand) for hand in hands]
        points = [self.engine.position(hand) for hand in hands]
        if any(p is None for p in points):
            return
        scale = (self.engine.hand_scale(hands[0]) + self.engine.hand_scale(hands[1])) / 2
        gap = hypot(points[0][0] - points[1][0], points[0][1] - points[1][1])
        if gap > self.config.beam_connect_distance * scale:
            return
        for grab in grabs:
            if grab is not None and grab.block is not None:
                self.world.remove_block(grab.block, debris=False)
        beam = self.world.add_beam(points[0], points[1], scale * self.config.beam_thickness, ctx.now)
        for hand, part in zip(hands, ("p1", "p2")):
            self._grabs[hand] = Grab(beam=beam, part=part, last=np.asarray(self.engine.position(hand)))
            beam.holders.add(hand)
        self._live_beams[both] = beam
        middle = ((points[0][0] + points[1][0]) / 2, (points[0][1] + points[1][1]) / 2)
        ctx.particles.sparks.emit_radial(70, middle, (scale * 2, scale * 7), (0.2, 0.5), (1.5, 3.0))
        self.manager.screen.flash(0.15)
        self.manager.hooks.emit("beam_connect", position=middle)

    def _finish_beam(self, both: str, ctx: EffectContext) -> None:
        """First hand released: place the beam (or let a too-short one fizzle)."""
        beam = self._live_beams.pop(both)
        scale = max(self.engine.hand_scale(h) for h in pair_of(both)) or 100.0
        if beam.length >= self.config.beam_min_length * scale:
            beam.placed = True
            self.world.wake()
            self.manager.hooks.emit("beam_place", length=beam.length, p1=tuple(beam.p1), p2=tuple(beam.p2))
        else:
            middle = tuple((beam.p1 + beam.p2) / 2)
            ctx.particles.sparks.emit_radial(30, middle, (100, 400), (0.2, 0.4), (1.5, 2.5))
            for hand in list(beam.holders):
                self._grabs.pop(hand, None)
            self.world.remove_beam(beam)
            self.manager.hooks.emit("beam_fizzle", position=middle)

    def _update_cracks(self, ctx: EffectContext) -> None:
        """POINT at a block for break_seconds to break it, in 4 crack stages.

        A crack fades slowly instead of resetting when the fingertip slips off
        for a moment, which jittery real landmarks do all the time.
        """
        c = self.config
        targeted: set[int] = set()
        for hand in self.engine.tracked_hands():
            if not (self.engine.is_active(hand, Gesture.POINT) and self.engine.visible(hand)):
                continue
            pad = c.block_pick_pad * 0.5 * self.engine.hand_scale(hand)
            block = self.world.block_at(self.engine.position(hand), pad=pad)
            if block is None or id(block) in targeted:
                continue
            targeted.add(id(block))
            block.crack += ctx.dt / c.break_seconds
            if block.crack >= 1.0:
                self.world.break_block(block, ctx.now)
        for block in self.world.blocks:
            if id(block) not in targeted and block.crack > 0:
                block.crack = max(0.0, block.crack - ctx.dt / (c.break_seconds * c.crack_decay))

    def _update_punches(self, ctx: EffectContext) -> None:
        """A fist moving fast through a block smashes it at once."""
        c = self.config
        for hand in self.engine.tracked_hands():
            if not self.engine.visible(hand) or Gesture.FIST not in (self.engine.raw(hand),
                                                                     self.engine.active(hand)):
                continue
            if hypot(*self.engine.velocity(hand)) < c.punch_speed:
                continue
            if ctx.now - self._last_punch.get(hand, -inf) < c.punch_cooldown_s:
                continue
            pad = 0.35 * self.engine.hand_scale(hand)
            block = self.world.block_at(self.engine.palm_center(hand), pad=pad)
            if block is not None:
                self._last_punch[hand] = ctx.now
                self.world.smash(block, ctx.now, "punch")
                self.manager.screen.shake(7.0)

    def _release_all(self) -> None:
        for hand in list(self._grabs):
            grab = self._grabs.pop(hand)
            if grab.block is not None and grab.block in self.world.blocks:
                self.world.release_block(grab.block, self.manager.size[0])
            if grab.beam is not None:
                grab.beam.holders.discard(hand)
                grab.beam.moved()
        for beam in self._live_beams.values():
            self.world.remove_beam(beam)
        self._live_beams.clear()

    def on_detach(self) -> None:
        """Leaving BLOCKS drops held blocks and cancels an unfinished beam; the build stays."""
        self._release_all()
        for block in self.world.blocks:
            block.crack = 0.0
        self._pinching.clear()

    def render(self, frame: np.ndarray, ctx: EffectContext) -> None:
        """Pinch cursors, the "touch to connect" preview, and a crosshair while pointing."""
        engine = self.engine
        color = self.config.beam_color
        points = {}
        for hand in engine.tracked_hands():
            if not engine.visible(hand):
                continue
            features = engine.features(hand)
            point = features.pinch_point
            pinching = self._pinching.get(hand, False)
            # Show the grab cursor only as thumb and index close in; an open hand's
            # thumb-index midpoint floats in mid-air and would just be clutter.
            if pinching or features.pinch_strength > 0.2:
                p = (int(point[0]), int(point[1]))
                radius = max(6, int(features.scale * 0.12))
                cv2.circle(frame, p, radius + 2, (20, 20, 24), 2, cv2.LINE_AA)
                cv2.circle(frame, p, radius, color, -1 if pinching else 2, cv2.LINE_AA)
            if pinching:
                points[hand] = point
            if engine.is_active(hand, Gesture.POINT):
                tip = features.tips[1]
                t = (int(tip[0]), int(tip[1]))
                for dx, dy in ((1, 0), (0, 1)):
                    cv2.line(frame, (t[0] - 10 * dx, t[1] - 10 * dy), (t[0] + 10 * dx, t[1] + 10 * dy),
                             (255, 255, 255), 2, cv2.LINE_AA)
        for both in engine.tracked_pairs():
            left, right = pair_of(both)
            if left in points and right in points and self._can_connect(both):
                self._render_connect_hint(frame, np.asarray(points[left]), np.asarray(points[right]), color)

    @staticmethod
    def _render_connect_hint(frame: np.ndarray, a: np.ndarray, b: np.ndarray, color: tuple[int, int, int]) -> None:
        """Dashed line between one person's pinches: touch them to make a beam."""
        length = float(hypot(*(b - a)))
        segments = max(2, int(length / 18))
        for i in range(0, segments, 2):
            p1 = a + (b - a) * i / segments
            p2 = a + (b - a) * min(1.0, (i + 1) / segments)
            cv2.line(frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, 2, cv2.LINE_AA)
        middle = ((a + b) / 2).astype(int)
        cv2.putText(frame, "touch pinches to connect", (int(middle[0]) - 95, int(middle[1]) - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
