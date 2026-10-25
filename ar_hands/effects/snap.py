"""The snap: the user turns to dust (and back), Infinity War style.

On a finger snap the user, as the camera sees them at that moment, freezes
into a "ghost" that crumbles away along a ragged diagonal front. Every
crumbling pixel becomes a dust mote with that pixel's own color, carried off
by the wind. Behind, the user is replaced by the background, learned from
frames where those pixels were visible and inpainted where they never were.
They stay gone until the next snap (or ``snap_gone_s``), then re-form with
the front sweeping back.

Needs person segmentation; without it the snap is ignored with a warning.
"""
from __future__ import annotations
import logging
from typing import TYPE_CHECKING
import cv2
import numpy as np
from ..config import EffectsConfig
from ..gestures.events import EventBus, EventKind, GestureEvent
from ..gestures.types import Gesture
from .base import Effect, EffectContext, EffectState, ModeController
from .sprites import _fractal_noise

if TYPE_CHECKING:
    from .manager import EffectManager

LOGGER = logging.getLogger(__name__)


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


class DustEffect(Effect):
    """Vanish -> gone -> reappear, for the person in front of the camera."""

    name = "dust"
    layer = -60  # Hides the user before blocks and other effects are drawn.
    needs_person = True

    def __init__(self, now: float, config: EffectsConfig) -> None:
        """Start vanishing; the ghost is captured on the first update with a mask."""
        super().__init__()
        self.config = config
        self.phase, self.phase_start = "vanish", now
        self.ghost: np.ndarray | None = None
        self.ghost_alpha: np.ndarray | None = None
        self.front: np.ndarray | None = None  # 0..1 per pixel: when it crumbles.
        self.plate: np.ndarray | None = None
        self._last_progress = 0.0
        self._frames = 0
        self._waiting_since = now

    def reappear(self, now: float) -> None:
        """Re-form (the second snap)."""
        if self.phase in ("vanish", "gone"):
            self.phase, self.phase_start = "reappear", now
            self.state = EffectState.DYING

    def progress(self, now: float, seconds: float) -> float:
        """0..1 through the current phase, eased."""
        t = min(1.0, max(0.0, (now - self.phase_start) / seconds))
        return t * t * (3 - 2 * t)

    def _capture(self, ctx: EffectContext) -> None:
        height, width = ctx.camera.shape[:2]
        self.ghost = ctx.camera.copy()
        self.ghost_alpha = np.clip(ctx.person * 1.2, 0, 1).astype(np.float32)
        ys, xs = np.nonzero(self.ghost_alpha > 0.3)
        x0, x1 = (xs.min(), xs.max()) if len(xs) else (0, width)
        y0, y1 = (ys.min(), ys.max()) if len(ys) else (0, height)
        gy, gx = np.mgrid[0:height, 0:width].astype(np.float32)
        # A diagonal front from the upper right, made ragged by noise.
        sweep = 0.6 * (x1 - gx) / max(1, x1 - x0) + 0.4 * (gy - y0) / max(1, y1 - y0)
        noise = cv2.resize(_fractal_noise(ctx.rng, 256, 5), (width, height))
        front = 0.75 * np.clip(sweep, 0, 1) + 0.25 * noise
        # Rescale over the person's own pixels, so crumbling starts the moment
        # they snap and finishes exactly as the phase ends.
        body = self.ghost_alpha > 0.5
        low, high = (float(front[body].min()), float(front[body].max())) if body.any() else (0.0, 1.0)
        self.front = np.clip((front - low) / max(1e-6, high - low), 0, 1).astype(np.float32)
        self._refresh_plate(ctx)
        ctx.emit("snap_dust", position=(float((x0 + x1) / 2), float((y0 + y1) / 2)))

    def _refresh_plate(self, ctx: EffectContext) -> None:
        if ctx.background is not None:
            self.plate = ctx.background.fill((ctx.width, ctx.height))
        if self.plate is None and ctx.camera is not None:
            # Nothing learned yet: a heavy blur of the scene beats a hole.
            self.plate = cv2.GaussianBlur(ctx.camera, (0, 0), 25)

    def update(self, ctx: EffectContext) -> None:
        """Advance phases, crumble pixels into dust, keep the background fresh."""
        c = self.config
        self._frames += 1
        if self.ghost is None:
            if ctx.person is None or ctx.camera is None:
                if ctx.now - self._waiting_since > 0.6:
                    LOGGER.warning("The snap needs person segmentation, which is unavailable.")
                    self.kill()
                return
            self._capture(ctx)
            self.state = EffectState.ACTIVE
        if self._frames % 15 == 0:
            self._refresh_plate(ctx)
        if self.phase == "vanish":
            p = self.progress(ctx.now, c.snap_vanish_s) * 1.05
            self._crumble(ctx, self._last_progress, p, outward=True)
            self._last_progress = p
            if p >= 1.05:
                self.phase, self.phase_start = "gone", ctx.now
        elif self.phase == "gone":
            if ctx.now - self.phase_start >= c.snap_gone_s:
                self.reappear(ctx.now)
        else:
            q = self.progress(ctx.now, c.snap_reappear_s) * 1.05
            self._crumble(ctx, 1.05 - q - 0.03, 1.05 - q, outward=False)
            if q >= 1.05:
                self.kill()

    def _crumble(self, ctx: EffectContext, start: float, end: float, outward: bool) -> None:
        """Turn pixels whose front lies in [start, end) into dust motes."""
        step = 2  # Sample every other pixel: plenty of motes, a quarter of the work.
        front = self.front[::step, ::step]
        alpha = self.ghost_alpha[::step, ::step]
        band = (front >= min(start, end)) & (front < max(start, end)) & (alpha > 0.5)
        ys, xs = np.nonzero(band)
        if not len(ys):
            return
        limit = self.config.dust_per_frame
        if len(ys) > limit:
            pick = ctx.rng.choice(len(ys), limit, replace=False)
            ys, xs = ys[pick], xs[pick]
        positions = np.stack((xs * step, ys * step), axis=1).astype(np.float32)
        colors = self.ghost[ys * step, xs * step]
        wind = np.array(self.config.dust_wind, np.float32)
        count = len(ys)
        if outward:
            velocity = wind * ctx.rng.uniform(0.5, 1.5, (count, 1)) + ctx.rng.normal(0, 40, (count, 2))
            ctx.particles.dust.emit(count, positions, velocity, life=self.config.dust_life,
                                    size=(1.0, 2.6), color=colors)
        else:
            # Re-forming: motes fly in from downwind and land on the front.
            travel = ctx.rng.uniform(0.3, 0.6, (count, 1))
            start_points = positions + wind * travel
            ctx.particles.dust.emit(count, start_points, -wind, life=(0.3, 0.6), size=(1.0, 2.2), color=colors)

    def render(self, frame: np.ndarray, ctx: EffectContext) -> None:
        """Replace the live person with the background, then draw what is left of them."""
        if self.ghost is None or self.plate is None:
            return
        height, width = frame.shape[:2]
        live = ctx.person if ctx.person is not None else self.ghost_alpha
        small = cv2.resize(live, (width // 4, height // 4), interpolation=cv2.INTER_AREA)
        hide = cv2.GaussianBlur(cv2.dilate(small, np.ones((5, 5), np.uint8)), (0, 0), 1.5)
        hide = np.clip(cv2.resize(hide, (width, height)) * 1.3, 0, 1)
        plate = self.plate if self.plate.shape == frame.shape else cv2.resize(self.plate, (width, height))
        cv2.blendLinear(plate, frame, hide, 1.0 - hide, dst=frame)
        if self.phase == "vanish":
            p = self.progress(ctx.now, self.config.snap_vanish_s) * 1.05
            intact = self.ghost_alpha * _smoothstep(p, p + 0.03, self.front)
            cv2.blendLinear(self.ghost, frame, intact, 1.0 - intact, dst=frame)
            ember = (np.abs(self.front - p) < 0.012) & (self.ghost_alpha > 0.5)
            frame[ember] = np.clip(frame[ember].astype(np.int16) + (30, 60, 90), 0, 255).astype(np.uint8)
        elif self.phase == "reappear" and ctx.camera is not None:
            q = 1.05 - self.progress(ctx.now, self.config.snap_reappear_s) * 1.05
            back = live * _smoothstep(q, q + 0.03, self.front)
            cv2.blendLinear(ctx.camera, frame, back, 1.0 - back, dst=frame)


class SnapVanish(ModeController):
    """Draw-mode binding: snap your fingers to turn to dust; snap again to come back."""

    def __init__(self, bus: EventBus, manager: EffectManager, config: EffectsConfig) -> None:
        """Create the controller; attach() subscribes it."""
        super().__init__(bus)
        self.manager, self.config = manager, config
        self.effect: DustEffect | None = None

    @property
    def active(self) -> bool:
        """True from the snap until the user has fully re-formed."""
        return self.effect is not None and self.effect.alive

    def bind(self) -> None:
        """The snap mapping."""
        self.subscribe(Gesture.SNAP, EventKind.STARTED, self._snap)

    def _snap(self, event: GestureEvent) -> None:
        self.trigger(event.timestamp)

    def trigger(self, now: float) -> None:
        """Vanish, or re-form if already gone (also bound to a key for demos)."""
        if self.active and self.effect.phase in ("vanish", "gone"):
            self.effect.reappear(now)
            self.manager.hooks.emit("snap_return")
            return
        if self.active:
            return
        self.effect = DustEffect(now, self.config)
        self.manager.spawn(self.effect)
        self.manager.hooks.emit("snap")

    def on_detach(self) -> None:
        """Leaving Draw mode brings the user back."""
        if self.active:
            self.effect.reappear(self.manager.now)
