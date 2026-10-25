"""Susanoo: a giant violet spirit warrior that rises up behind the user.

Trigger: both fists held 1 s (toggles it on and off). It rises from below,
breathes violet flames, and its eyes burn yellow. With person segmentation
the user stands IN FRONT of it, like the reference art; without it, it is
drawn as a translucent overlay.

Everything is procedural. The skeleton is drawn as filled shapes and bone
strokes; a distance transform turns that silhouette into "spirit energy"
(bright at the edges, deep violet inside, glowing outline); an animated
noise field scrolls upward through a band around the silhouette to make
the flames.
"""
from __future__ import annotations
from math import pi, sin
from typing import TYPE_CHECKING
import cv2
import numpy as np
from ..config import EffectsConfig
from ..gestures.engine import GestureEngine
from ..gestures.events import BOTH, EventBus, GestureEvent, pair_of
from ..gestures.types import Gesture
from .base import Effect, EffectContext, EffectState, ModeController
from .compositing import Sprite, overlay_rgba
from .sprites import _fractal_noise

if TYPE_CHECKING:
    from .manager import EffectManager

Point = tuple[float, float]
ASPECT = 1.3  # Canvas width / height.


def _bezier(points: list[Point], count: int = 24) -> np.ndarray:
    """Points along a quadratic or cubic Bezier curve."""
    t = np.linspace(0.0, 1.0, count)[:, None]
    p = np.asarray(points, np.float64)
    if len(p) == 3:
        return (1 - t) ** 2 * p[0] + 2 * (1 - t) * t * p[1] + t ** 2 * p[2]
    return ((1 - t) ** 3 * p[0] + 3 * (1 - t) ** 2 * t * p[1]
            + 3 * (1 - t) * t ** 2 * p[2] + t ** 3 * p[3])


class _Painter:
    """Draws in normalized coordinates (height = 1) onto solid / line / dark masks."""

    def __init__(self, height: int) -> None:
        self.h = height
        self.w = int(height * ASPECT)
        self.solid = np.zeros((height, self.w), np.uint8)
        self.lines = np.zeros_like(self.solid)
        self.dark = np.zeros_like(self.solid)
        self.shell = np.zeros_like(self.solid)

    def px(self, points: np.ndarray | list) -> np.ndarray:
        return np.round(np.asarray(points, np.float64) * self.h).astype(np.int32)

    def poly(self, points, layer: str = "solid") -> None:
        cv2.fillPoly(getattr(self, layer), [self.px(points)], 255, cv2.LINE_AA)

    def ellipse(self, center: Point, axes: tuple[float, float], angle: float = 0.0,
                layer: str = "solid", thickness: int = -1, arc: tuple[int, int] = (0, 360)) -> None:
        c = tuple(int(v) for v in self.px(center))
        a = tuple(max(1, int(v)) for v in self.px(axes))
        cv2.ellipse(getattr(self, layer), c, a, angle, arc[0], arc[1], 255, thickness, cv2.LINE_AA)

    def stroke(self, points, width: float, layer: str = "solid") -> None:
        cv2.polylines(getattr(self, layer), [self.px(points).reshape(-1, 1, 2)], False, 255,
                      max(1, int(width * self.h)), cv2.LINE_AA)

    def bone(self, a: Point, b: Point, width: float, knob: float) -> None:
        """A long bone: shaft plus rounded knobs, with a highlight line."""
        self.stroke([a, b], width)
        for end in (a, b):
            self.ellipse(end, (knob, knob))
        self.stroke([a, b], 0.003, "lines")


def paint_susanoo(height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[Point]]:
    """Return (body BGRA, glow BGR, shadow BGRA, flame band (quarter res, 0..1), eye centers px)."""
    p = _Painter(height)
    cx = ASPECT / 2
    # Spirit-armor volume behind the bones: chest, shoulders, and head.
    p.ellipse((cx, 0.5), (0.26, 0.33), layer="shell")
    p.poly([(cx - 0.40, 0.33), (cx + 0.40, 0.33), (cx + 0.30, 0.52), (cx - 0.30, 0.52)], layer="shell")
    p.ellipse((cx, 0.17), (0.11, 0.12), layer="shell")
    for side in (-1, 1):
        def m(u: float) -> float:
            return cx + side * u  # Mirror helper: u is the distance from the center line.
        # Horns: two curved polygons sweeping up and out from the crown.
        outer = _bezier([(m(0.072), 0.14), (m(0.17), 0.11), (m(0.205), 0.0)])
        inner = _bezier([(m(0.028), 0.095), (m(0.11), 0.07), (m(0.205), 0.0)])
        p.poly(np.concatenate((outer, inner[::-1])))
        p.stroke(outer, 0.003, "lines")
        # Collar plates, pauldrons with ridges, and pauldron spikes.
        p.poly([(m(0.03), 0.30), (m(0.29), 0.335), (m(0.315), 0.378), (m(0.04), 0.348)])
        p.ellipse((m(0.33), 0.378), (0.115, 0.072), angle=-side * 14)
        for k in range(3):
            p.ellipse((m(0.325), 0.39 + k * 0.012), (0.085 - k * 0.016, 0.045 - k * 0.008),
                      angle=-side * 14, layer="lines", thickness=max(1, height // 400), arc=(180, 360))
        p.poly([(m(0.36), 0.338), (m(0.47), 0.262), (m(0.40), 0.352)])
        p.poly([(m(0.30), 0.325), (m(0.36), 0.248), (m(0.335), 0.333)])
        # Ribs: seven curved pairs; the lower two are short "floating" ribs.
        for i in range(7):
            y = 0.372 + i * 0.041
            w = 0.17 + 0.075 * sin(pi * (i + 1) / 8)
            outer_rib = _bezier([(m(0.014), y), (m(w * 0.62), y - 0.022), (m(w * 1.06), y + 0.018),
                                 (m(w * 0.96), y + 0.062)])
            p.stroke(outer_rib, 0.02)
            p.stroke(outer_rib, 0.003, "lines")
            if i < 5:
                front = _bezier([(m(w * 0.96), y + 0.062), (m(w * 0.7), y + 0.09),
                                 (m(0.035), y + 0.072)], 16)
                p.stroke(front, 0.014)
        # Arms: humerus, a double forearm, armored bracer, and a clawed hand.
        shoulder, elbow, wrist = (m(0.34), 0.405), (m(0.475), 0.61), (m(0.44), 0.815)
        p.bone(shoulder, elbow, 0.045, 0.038)
        p.bone(elbow, (m(0.425), 0.81), 0.024, 0.022)
        p.bone(elbow, (m(0.462), 0.815), 0.024, 0.022)
        p.poly([(m(0.45), 0.66), (m(0.505), 0.675), (m(0.475), 0.77), (m(0.425), 0.76)])
        for k, spread in enumerate((-0.05, -0.018, 0.016, 0.05)):
            knuckle = (wrist[0] + side * spread * 0.6, wrist[1] + 0.035)
            tip = (wrist[0] + side * spread * 1.4, wrist[1] + 0.13 - abs(spread) * 0.4)
            p.bone(wrist, knuckle, 0.016, 0.015)
            p.stroke(_bezier([knuckle, (tip[0], knuckle[1] + 0.05), tip]), 0.013)
        # Pelvis wings.
        p.ellipse((m(0.1), 0.95), (0.105, 0.05), angle=side * 10, thickness=max(2, height // 90), arc=(180, 330))
    # Skull: cranium, face, eye sockets, nose, teeth.
    p.ellipse((cx, 0.165), (0.083, 0.09))
    p.poly([(cx - 0.081, 0.17), (cx - 0.076, 0.222), (cx - 0.056, 0.262), (cx - 0.024, 0.277),
            (cx + 0.024, 0.277), (cx + 0.056, 0.262), (cx + 0.076, 0.222), (cx + 0.081, 0.17)])
    p.ellipse((cx, 0.165), (0.083, 0.09), layer="lines", thickness=max(1, height // 360), arc=(180, 360))
    eyes = [(cx - 0.034, 0.182), (cx + 0.034, 0.182)]
    for eye in eyes:
        p.ellipse(eye, (0.025, 0.018), layer="dark")
    p.poly([(cx, 0.2), (cx - 0.013, 0.225), (cx + 0.013, 0.225)], layer="dark")
    p.stroke([(cx - 0.047, 0.245), (cx + 0.047, 0.245)], 0.0035, "dark")
    for i in range(9):
        x = cx - 0.04 + i * 0.01
        p.stroke([(x, 0.236), (x, 0.256)], 0.0025, "dark")
    # Neck, spine, and sternum.
    for i in range(3):
        p.ellipse((cx, 0.288 + i * 0.017), (0.018, 0.008))
    for i in range(19):
        v = 0.34 + i * 0.033
        p.ellipse((cx, v), (0.021, 0.013))
        p.ellipse((cx, v), (0.016, 0.011), layer="lines", thickness=1)
    p.poly([(cx - 0.021, 0.36), (cx + 0.021, 0.36), (cx + 0.016, 0.6), (cx, 0.632), (cx - 0.016, 0.6)])

    # Style: distance from the silhouette edge drives brightness and opacity.
    solid = p.solid.astype(np.float32) / 255
    inside = cv2.distanceTransform((p.solid > 127).astype(np.uint8), cv2.DIST_L2, 5) / height
    rim = np.exp(-inside / 0.012) * solid
    deep, bright = np.array((135, 30, 80), np.float32), np.array((255, 185, 225), np.float32)
    shell = cv2.GaussianBlur(p.shell.astype(np.float32) / 255, (0, 0), height * 0.01)
    shell_inside = cv2.distanceTransform((p.shell > 127).astype(np.uint8), cv2.DIST_L2, 5) / height
    shell_rim = np.exp(-shell_inside / 0.02) * shell
    color = deep + (bright - deep) * rim[..., None]
    alpha = (0.34 + 0.62 * rim) * solid
    # The shell is faint violet mass with a brighter rim, under the bones.
    shell_alpha = (0.16 * shell + 0.34 * shell_rim) * (1 - solid)
    color = color * (1 - shell_alpha[..., None]) + (deep * 0.8 + bright * 0.2) * shell_alpha[..., None]
    alpha = alpha + shell_alpha
    lines = cv2.GaussianBlur(p.lines.astype(np.float32) / 255, (0, 0), 0.8)
    color = color * (1 - lines[..., None]) + np.array((255, 225, 255), np.float32) * lines[..., None]
    alpha = np.maximum(alpha, lines * 0.9 * np.clip(solid * 2, 0, 1))
    dark = cv2.GaussianBlur(p.dark.astype(np.float32) / 255, (0, 0), 1.0)[..., None]
    color = color * (1 - dark) + np.array((40, 0, 28), np.float32) * dark
    alpha = np.maximum(alpha, dark[..., 0] * 0.9)
    body = np.dstack((np.clip(color, 0, 255), np.clip(alpha, 0, 1) * 255)).astype(np.uint8)
    halo = cv2.GaussianBlur(np.maximum(solid, shell * 0.6), (0, 0), height * 0.02)
    glow = np.clip(halo[..., None] * np.array((255, 60, 150), np.float32) * 0.75, 0, 255).astype(np.uint8)
    # The flame band hugs the silhouette from outside; built at quarter resolution.
    quarter = cv2.resize(np.maximum(solid, shell), (p.w // 4, height // 4), interpolation=cv2.INTER_AREA)
    # Smear the silhouette upward so the noise cuts it into rising tongues.
    rising = quarter.copy()
    for k in range(1, 14):
        shifted = np.zeros_like(quarter)
        shifted[:-k * 2] = quarter[k * 2:]
        rising = np.maximum(rising, shifted * (1 - k / 14))
    grown = cv2.dilate(rising, np.ones((5, 5), np.uint8))
    band = np.clip(cv2.GaussianBlur(grown, (0, 0), 2.0) * 1.5 - quarter * 0.35, 0, 1)
    # A soft dark aura behind the figure: in a bright room the violet needs contrast.
    shade = cv2.GaussianBlur(np.maximum(solid, shell), (0, 0), height * 0.05) * 0.5
    shadow = np.dstack((np.zeros_like(shade), np.zeros_like(shade), np.zeros_like(shade),
                        np.clip(shade, 0, 1) * 255)).astype(np.uint8)
    eye_px = [(e[0] * height, e[1] * height) for e in eyes]
    return body, glow, shadow, band.astype(np.float32), eye_px


class SusanooEffect(Effect):
    """The rising, breathing spirit warrior. Persistent until dismissed."""

    name = "susanoo"
    layer = -50  # Behind blocks and every other effect, in front of the camera.
    needs_person = True

    def __init__(self, now: float, rise_s: float = 1.0, fade_s: float = 0.8) -> None:
        """Begin rising."""
        super().__init__()
        self.born, self.rise_s, self.fade_s = now, rise_s, fade_s
        self.dying_since: float | None = None
        self.anchor_x: float | None = None
        self._height = 0
        self._body: Sprite | None = None
        self._glow: Sprite | None = None
        self._shadow: Sprite | None = None
        self._band: np.ndarray | None = None
        self._eyes: list[Point] = []
        self._noise: np.ndarray | None = None

    def dismiss(self, now: float) -> None:
        """Sink and fade away."""
        if self.dying_since is None:
            self.dying_since = now
            self.state = EffectState.DYING

    def _prepare(self, frame_height: int, rng: np.random.Generator) -> None:
        height = int(frame_height * 1.05)
        if height == self._height:
            return
        body, glow, shadow, band, eyes = paint_susanoo(height)
        self._height = height
        self._body, self._glow = Sprite.from_image(body), Sprite.from_image(glow)
        self._shadow = Sprite.from_image(shadow)
        self._band, self._eyes = band, eyes
        bh, bw = band.shape
        # Taller than the band so it can scroll; stretched vertically into flame shapes.
        noise = _fractal_noise(rng, max(bh, bw) * 2)
        self._noise = cv2.resize(noise, (bw, bh * 2), interpolation=cv2.INTER_LINEAR).astype(np.float32)

    def presence(self, now: float) -> float:
        """0..1 visibility from rising in and fading out."""
        rise = min(1.0, (now - self.born) / self.rise_s)
        fade = 1.0 if self.dying_since is None else max(0.0, 1 - (now - self.dying_since) / self.fade_s)
        return (rise * (2 - rise)) * fade

    def _origin(self, ctx: EffectContext) -> tuple[float, float]:
        """Top-left of the sprite: centered on the user, rising from below."""
        target = ctx.width / 2
        if ctx.person is not None:
            columns = ctx.person[::8, ::8].sum(axis=0)
            if columns.sum() > 5:
                target = float(np.average(np.arange(len(columns)) * 8, weights=columns))
        else:
            centers = [ctx.engine.palm_center(h) for h in pair_of(getattr(self, "owner", BOTH))
                       if ctx.engine.visible(h)]
            if len(centers) == 2:
                target = (centers[0][0] + centers[1][0]) / 2
        self.anchor_x = target if self.anchor_x is None else self.anchor_x + (target - self.anchor_x) * 0.12
        rise = min(1.0, (ctx.now - self.born) / self.rise_s)
        sink = 0.0 if self.dying_since is None else min(1.0, (ctx.now - self.dying_since) / self.fade_s)
        y = ctx.height * 0.55 * ((1 - rise) ** 2 + sink * 0.5) - ctx.height * 0.03
        y += ctx.height * 0.006 * sin(ctx.now * 2.2)  # Slow breathing bob.
        return self.anchor_x - self._height * ASPECT / 2, y

    def update(self, ctx: EffectContext) -> None:
        """Animate, emit spirit flames, and die after the fade-out."""
        self._prepare(ctx.height, ctx.rng)
        if self.state is EffectState.SPAWN and ctx.now - self.born >= self.rise_s:
            self.state = EffectState.ACTIVE
        if self.presence(ctx.now) <= 0 and self.dying_since is not None:
            self.kill()
            return
        self.age += ctx.dt

    def render(self, frame: np.ndarray, ctx: EffectContext) -> None:
        """Glow, translucent body, rising flames, burning eyes, then the user in front."""
        if self._body is None:
            return
        level = self.presence(ctx.now)
        x0, y0 = self._origin(ctx)
        width = self._body.shape[1]
        center = (x0 + width / 2, y0 + self._height / 2)
        overlay_rgba(frame, self._shadow, center, opacity=level, mode="alpha")
        overlay_rgba(frame, self._glow, center, opacity=0.9 * level, mode="add")
        overlay_rgba(frame, self._body, center, opacity=0.95 * level, mode="alpha")
        self._render_flames(frame, ctx, x0, y0, level)
        flicker = 0.85 + 0.15 * sin(ctx.now * 17) * sin(ctx.now * 7.3)
        for ex, ey in self._eyes:
            point = (x0 + ex, y0 + ey)
            overlay_rgba(frame, ctx.sprites.fire_glow, point, self._height * 0.09 / 128 * 2, 0, level * flicker, "add")
            overlay_rgba(frame, ctx.sprites.white_glow, point, self._height * 0.035 / 64 * 2, 0, level, "add")
        ctx.paste_person(frame)

    def _render_flames(self, frame: np.ndarray, ctx: EffectContext, x0: float, y0: float,
                       level: float) -> None:
        band, noise = self._band, self._noise
        bh, bw = band.shape
        offset = int(ctx.now * bh * 0.6) % bh  # Scroll the noise upward: flames rise.
        field = noise[offset:offset + bh]
        if field.shape[0] < bh:
            field = np.vstack((field, noise[: bh - field.shape[0]]))
        flames = band * np.clip((field - 0.38) * 2.4, 0, 1) ** 1.3 * level
        color = flames[..., None] * np.array((255, 90, 190), np.float32) * 1.1
        color += (flames ** 2.5)[..., None] * np.array((255, 220, 255), np.float32) * 0.7
        light = cv2.resize(np.clip(color, 0, 255).astype(np.uint8), (bw * 4, bh * 4),
                           interpolation=cv2.INTER_LINEAR)
        height, width = frame.shape[:2]
        ox, oy = int(x0), int(y0)
        fx0, fy0, fx1, fy1 = max(0, ox), max(0, oy), min(width, ox + bw * 4), min(height, oy + bh * 4)
        if fx0 < fx1 and fy0 < fy1:
            roi = frame[fy0:fy1, fx0:fx1]
            cv2.add(roi, light[fy0 - oy:fy1 - oy, fx0 - ox:fx1 - ox], dst=roi)


class Susanoo(ModeController):
    """JUTSU binding: both fists held 1 s toggles Susanoo."""

    def __init__(self, bus: EventBus, engine: GestureEngine, manager: EffectManager,
                 config: EffectsConfig) -> None:
        """Create the controller; attach() subscribes it."""
        super().__init__(bus)
        self.engine, self.manager, self.config = engine, manager, config
        self.hold_rings = ((Gesture.TWO_FISTS, config.susanoo_hold_seconds),)
        self.effect: SusanooEffect | None = None

    @property
    def active(self) -> bool:
        """True while Susanoo is up (or rising)."""
        return self.effect is not None and self.effect.alive and self.effect.dying_since is None

    def bind(self) -> None:
        """The JUTSU Susanoo mapping."""
        self.on_hold(Gesture.TWO_FISTS, self.config.susanoo_hold_seconds, self._toggle, hand=BOTH)

    def _toggle(self, event: GestureEvent) -> None:
        if self.active:
            self.effect.dismiss(event.timestamp)
            self.manager.hooks.emit("susanoo_dismiss")
            return
        self.effect = SusanooEffect(event.timestamp, self.config.susanoo_rise_s, self.config.susanoo_fade_s)
        self.effect.owner = event.hand
        self.manager.spawn(self.effect)
        self.manager.screen.flash(0.35)
        self.manager.screen.shake(10.0)
        self.manager.hooks.emit("susanoo_summon")

    def on_detach(self) -> None:
        """Leaving JUTSU lets Susanoo fade away."""
        if self.active:
            self.effect.dismiss(self.manager.now)
            self.manager.hooks.emit("susanoo_dismiss")
