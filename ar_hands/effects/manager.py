"""EffectManager (lifecycle, layering, timing, screen FX) and ModeManager (modes).

Per frame, after GestureEngine.update() has dispatched its events:

    ModeManager.update(ctx)    active mode's controllers poll hands (drag, aim, charge)
    EffectManager.update(...)  every effect + particles + screen FX advance by dt
    EffectManager.render()     alpha layer:    effects with layer < 20 (blocks)
                               distortion:     heat haze, shockwaves, lensing bend the picture
                               smoke, chunks, dust
                               light:          emissive effects, glows, fire/sparks into the
                                               light buffer, then bloom + light spill (lighting.py)
    ModeManager.render()       interaction overlays (charge glows, pinch cursors)
    EffectManager.post()       flash, then camera shake (after the skeleton is drawn)
"""
from __future__ import annotations
from collections.abc import Callable, Mapping, Sequence
from enum import Enum
from math import exp, inf
import time
import cv2
import numpy as np
from ..config import EffectsConfig
from ..gestures.engine import GestureEngine
from ..gestures.events import BOTH, EventBus, GestureEvent
from ..gestures.types import Gesture
from ..hand_tracker import HandData
from ..segmentation import BackgroundModel
from .base import Effect, EffectContext, HookEmitter, ModeController
from .fireball import Explosion
from .flame import Distortion, FlameField
from .lighting import LightCompositor
from .particles import ParticleSet
from .sprites import SpriteLibrary

Point = tuple[float, float]
ALPHA_LAYER_LIMIT = 20  # Effects below this layer are opaque matter, drawn before smoke.


class ScreenFX:
    """Full-frame flash (a fixed number of frames) and decaying camera shake."""

    def __init__(self, config: EffectsConfig, rng: np.random.Generator) -> None:
        """Start calm."""
        self.config, self.rng = config, rng
        self.flash_level = 0.0
        self.flash_left = 0
        self.shake_amplitude = 0.0
        self._buffer: np.ndarray | None = None

    def flash(self, strength: float) -> None:
        """White flash fading over ``flash_frames`` frames."""
        self.flash_level = max(self.flash_level if self.flash_left else 0.0, min(1.0, strength))
        self.flash_left = self.config.flash_frames

    def shake(self, amplitude_px: float) -> None:
        """Random translation that decays exponentially."""
        self.shake_amplitude = max(self.shake_amplitude, amplitude_px)

    def update(self, dt: float) -> None:
        """Decay the shake in real time."""
        self.shake_amplitude *= exp(-dt / self.config.shake_decay_s)
        if self.shake_amplitude < 0.5:
            self.shake_amplitude = 0.0

    def apply(self, frame: np.ndarray) -> None:
        """Shake (reflected borders, no black edges), then flash, in place."""
        if self.shake_amplitude > 0:
            dx, dy = self.rng.uniform(-self.shake_amplitude, self.shake_amplitude, 2)
            matrix = np.float32([[1, 0, dx], [0, 1, dy]])
            if self._buffer is None or self._buffer.shape != frame.shape:
                self._buffer = np.empty_like(frame)
            cv2.warpAffine(frame, matrix, (frame.shape[1], frame.shape[0]), dst=self._buffer,
                           borderMode=cv2.BORDER_REFLECT)
            frame[:] = self._buffer
        if self.flash_left > 0:
            alpha = self.flash_level * self.flash_left / self.config.flash_frames
            cv2.convertScaleAbs(frame, dst=frame, alpha=1 - alpha, beta=255 * alpha)
            self.flash_left -= 1


class EffectManager:
    """Owns live effects, the shared particle systems, screen FX, and hook listeners."""

    def __init__(self, config: EffectsConfig, engine: GestureEngine, sprites: SpriteLibrary,
                 seed: int | None = None) -> None:
        """Create an empty scene."""
        self.config, self.engine, self.sprites = config, engine, sprites
        self.rng = np.random.default_rng(seed)
        self.hooks = HookEmitter()
        self.particles = ParticleSet(config, seed)
        self.screen = ScreenFX(config, self.rng)
        self.flames = FlameField()
        self.distortion = Distortion(self.flames.noise)
        self.lighting = LightCompositor()
        self.effects: list[Effect] = []
        self.timings: dict[str, float] = {}  # Smoothed ms per effect type, update + render.
        self.now = 0.0
        self.size = (1280, 720)
        self._ctx: EffectContext | None = None
        self._frame_times: dict[str, float] = {}

    def spawn(self, effect: Effect) -> Effect:
        """Add an effect; it updates from the next update() on."""
        self.effects.append(effect)
        return effect

    def context(self, now: float, dt: float, hands: Sequence[HandData] = (),
                events: Sequence[GestureEvent] = (), size: tuple[int, int] | None = None,
                camera: np.ndarray | None = None, person: np.ndarray | None = None,
                background: BackgroundModel | None = None) -> EffectContext:
        """Build (and remember) this frame's context; dt is clamped to max_dt."""
        if size is not None:
            self.size = size
        self.now = now
        self._ctx = EffectContext(now, min(max(dt, 0.0), self.config.max_dt), self.size[0],
                                  self.size[1], self.engine, hands, events, self.config,
                                  self.sprites, self.particles, self.rng, self,
                                  camera, person, background)
        return self._ctx

    @property
    def wants_person(self) -> bool:
        """True while any live effect needs the person mask (domains, Susanoo, the snap)."""
        return any(effect.needs_person for effect in self.effects)

    def find(self, name: str) -> Effect | None:
        """The first live effect with this name (e.g. "blocks")."""
        return next((effect for effect in self.effects if effect.name == name), None)

    def _timed(self, key: str, start: float) -> None:
        """Accumulate this frame's milliseconds for one effect type."""
        self._frame_times[key] = self._frame_times.get(key, 0.0) + (time.perf_counter() - start) * 1000

    def _commit_timings(self) -> None:
        """Fold the frame's totals into smoothed readings; idle types decay away."""
        for key in set(self.timings) | set(self._frame_times):
            value = 0.9 * self.timings.get(key, 0.0) + 0.1 * self._frame_times.get(key, 0.0)
            if value < 0.005 and key not in self._frame_times:
                self.timings.pop(key, None)
            else:
                self.timings[key] = value
        self._frame_times.clear()

    def update(self, dt: float, hands: Sequence[HandData], events: Sequence[GestureEvent],
               now: float | None = None, size: tuple[int, int] | None = None,
               camera: np.ndarray | None = None, person: np.ndarray | None = None,
               background: BackgroundModel | None = None) -> EffectContext:
        """Advance every effect and particle by dt (real seconds, clamped).

        ``camera``/``person``/``background`` are passed on frames where
        segmentation ran (see ``wants_person``)."""
        ctx = self.context(self.now + dt if now is None else now, dt, hands, events, size,
                           camera, person, background)
        for effect in list(self.effects):
            start = time.perf_counter()
            effect.update(ctx)
            self._timed(effect.name, start)
        start = time.perf_counter()
        self.particles.update(ctx.dt)
        self._timed("particles", start)
        self.screen.update(ctx.dt)
        self.effects = [effect for effect in self.effects if effect.alive]
        return ctx

    def render(self, frame: np.ndarray, extra_light: np.ndarray | None = None) -> None:
        """Matter first, then distortion and smoke, then everything that emits light.

        ``extra_light`` (the neon drawing) joins the light buffer, so it too
        blooms and lights the scene."""
        ctx = self._ctx or self.context(self.now, 0.0, size=(frame.shape[1], frame.shape[0]))
        ordered = sorted(self.effects, key=lambda effect: effect.layer)
        matter = [effect for effect in ordered if effect.layer < ALPHA_LAYER_LIMIT]
        overlay = [effect for effect in ordered if effect.layer >= ALPHA_LAYER_LIMIT and not effect.emissive]
        for effect in matter:
            start = time.perf_counter()
            effect.render(frame, ctx)
            self._timed(effect.name, start)
        for effect in ordered:
            start = time.perf_counter()
            effect.distort(frame, ctx)
            self._timed(effect.name, start)
        for system in (self.particles.smoke, self.particles.chunks, self.particles.dust):
            start = time.perf_counter()
            system.render(frame)
            self._timed("particles", start)
        for effect in overlay:
            start = time.perf_counter()
            effect.render(frame, ctx)
            self._timed(effect.name, start)
        self._render_light(frame, ctx, ordered, extra_light)
        self._commit_timings()

    def _render_light(self, frame: np.ndarray, ctx: EffectContext, ordered: list[Effect],
                      extra_light: np.ndarray | None) -> None:
        """Light sources into the light buffer, then bloom and spill (skipped when dark)."""
        sources = [effect for effect in ordered if effect.emissive or effect.glows]
        glowing = any(np.any(system.life > 0) for system in self.particles.lights)
        if not sources and not glowing and extra_light is None:
            return
        light = self.lighting.begin(frame)
        if extra_light is not None:
            cv2.add(light, extra_light, dst=light)
        for effect in sources:
            start = time.perf_counter()
            if effect.emissive:
                effect.render(light, ctx)
            else:
                effect.glow(light, ctx)
            self._timed(effect.name, start)
        start = time.perf_counter()
        self.particles.render_light(light)
        self._timed("particles", start)
        start = time.perf_counter()
        self.lighting.lit = True
        self.lighting.compose(frame)
        self._timed("lighting", start)

    def post(self, frame: np.ndarray) -> None:
        """Screen flash and shake, applied after everything that should shake."""
        self.screen.apply(frame)

    def explode(self, center: Point, radius: float, power: float = 1.0, style: str = "fire") -> None:
        """Blast at ``center``: effect, flash, shake, and on_explosion for every effect."""
        c = self.config
        self.spawn(Explosion(center, radius, power, self.now, style))
        self.screen.flash(c.flash_strength * min(1.5, power))
        self.screen.shake(c.shake_px * radius / 60 * power)
        ctx = self._ctx or self.context(self.now, 0.0)
        for effect in list(self.effects):
            effect.on_explosion(center, radius, power, ctx)
        self.hooks.emit("explosion", position=center, radius=radius, power=power, style=style)

    def hit_test(self, center: Point, radius: float) -> bool:
        """True if any solid effect touches the circle."""
        return any(effect.hit_test(center, radius) for effect in self.effects)

    def clear(self) -> None:
        """Remove transient effects and particles; persistent ones (the block world) stay."""
        self.effects = [effect for effect in self.effects if effect.persistent]
        self.particles.clear()

    def count(self, name: str) -> int:
        """Live effects of one type, e.g. count("fireball")."""
        return sum(1 for effect in self.effects if effect.name == name)

    def stats(self) -> list[str]:
        """Timing lines for the debug overlay: smoothed update+render ms per effect type."""
        lines = [f"{key:<10} {value:5.2f} ms" for key, value in sorted(self.timings.items())]
        total = sum(self.timings.values())
        lines.append(f"{'effects':<10} {total:5.2f} ms  ({self.particles.count} particles)")
        return lines


class Mode(Enum):
    """Top-level app modes, switched with keys 1-4, the menu, or a held ROCK sign."""

    DRAW = "Draw"
    JUTSU = "Jutsu"
    BLOCKS = "Blocks"
    JJK = "Jujutsu"


class ModeSwitcher(ModeController):
    """Always-on binding: ROCK held mode_hold_seconds -> next mode."""

    def __init__(self, bus: EventBus, seconds: float, on_switch: Callable[[float], None]) -> None:
        """``on_switch(timestamp)`` is called when the hold completes."""
        super().__init__(bus)
        self.seconds, self.on_switch = seconds, on_switch
        self.hold_rings = ((Gesture.ROCK, seconds),)

    def bind(self) -> None:
        """Subscribe the hold."""
        self.on_hold(Gesture.ROCK, self.seconds, lambda event: self.on_switch(event.timestamp))


class MenuOpener(ModeController):
    """Always-on binding: both open palms held -> open the menu.

    ``can_open`` vetoes it while something is held (a fireball in each palm is
    also "two open palms"), so the menu never steals an effect mid-cast."""

    def __init__(self, bus: EventBus, seconds: float, on_open: Callable[[float], None],
                 can_open: Callable[[], bool] = lambda: True) -> None:
        """``on_open(timestamp)`` is called when the hold completes and ``can_open()`` agrees."""
        super().__init__(bus)
        self.seconds, self.on_open, self.can_open = seconds, on_open, can_open
        self.hold_rings = ((Gesture.TWO_PALMS, seconds),)

    def bind(self) -> None:
        """Subscribe the two-hand hold."""
        self.on_hold(Gesture.TWO_PALMS, self.seconds, self._open, hand=BOTH)

    def _open(self, event: GestureEvent) -> None:
        if self.can_open():
            self.on_open(event.timestamp)


class ModeManager:
    """Attach exactly one mode's controllers at a time (plus global ones).

    Switching detaches the old mode first, so its bus subscriptions are gone
    before the new mode subscribes: no leaked callbacks, which len(bus) proves.
    """

    def __init__(self, controllers: Mapping[Mode, Sequence[ModeController]],
                 global_controllers: Sequence[ModeController] = (), hooks: HookEmitter | None = None,
                 initial: Mode = Mode.DRAW) -> None:
        """Attach the initial mode and the global controllers."""
        self.controllers = {mode: tuple(controllers.get(mode, ())) for mode in Mode}
        self.global_controllers = tuple(global_controllers)
        self.hooks = hooks
        self.mode = initial
        self.changed_at = -inf
        self.suspended = False
        for controller in self.global_controllers + self.controllers[initial]:
            controller.attach()

    @property
    def active_controllers(self) -> tuple[ModeController, ...]:
        """The current mode's controllers followed by the global ones (none while suspended)."""
        if self.suspended:
            return ()
        return self.controllers[self.mode] + self.global_controllers

    def suspend(self) -> None:
        """Input focus moves to an overlay (menu, guide): detach every controller,
        so no gesture can draw, cast, or switch modes until resume()."""
        if self.suspended:
            return
        for controller in self.controllers[self.mode] + self.global_controllers:
            controller.detach()
        self.suspended = True

    def resume(self) -> None:
        """Give input back to the current mode (which may have changed meanwhile)."""
        if not self.suspended:
            return
        self.suspended = False
        for controller in self.global_controllers + self.controllers[self.mode]:
            controller.attach()

    def switch(self, mode: Mode, now: float) -> bool:
        """Detach the old mode, attach the new one; False if already in that mode.
        While suspended only the choice changes; resume() attaches it."""
        if mode is self.mode:
            return False
        if not self.suspended:
            for controller in self.controllers[self.mode]:
                controller.detach()
        previous, self.mode = self.mode, mode
        if not self.suspended:
            for controller in self.controllers[mode]:
                controller.attach()
        self.changed_at = now
        if self.hooks is not None:
            self.hooks.emit("mode_change", mode=mode.value, previous=previous.value)
        return True

    def next(self, now: float) -> None:
        """Cycle Draw -> Jutsu -> Blocks -> Jujutsu -> Draw."""
        modes = list(Mode)
        self.switch(modes[(modes.index(self.mode) + 1) % len(modes)], now)

    def hold_rings(self) -> list[tuple[Gesture, float]]:
        """(gesture, seconds) pairs whose hold progress should be shown as rings."""
        return [ring for controller in self.active_controllers for ring in controller.hold_rings]

    def update(self, ctx: EffectContext) -> None:
        """Per-frame interaction logic of the active controllers."""
        for controller in self.active_controllers:
            controller.update(ctx)

    def render(self, frame: np.ndarray, ctx: EffectContext) -> None:
        """Interaction overlays of the active controllers."""
        for controller in self.active_controllers:
            controller.render(frame, ctx)

    def close(self) -> None:
        """Detach everything (app shutdown)."""
        for controller in self.controllers[self.mode] + self.global_controllers:
            controller.detach()
