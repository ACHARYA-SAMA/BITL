"""Effect lifecycle, the per-frame context, named hooks, and mode controllers."""
from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
import logging
from typing import TYPE_CHECKING, Any
import cv2
import numpy as np
from ..config import EffectsConfig
from ..gestures.engine import GestureEngine
from ..gestures.events import Callback, EventBus, EventKind, GestureEvent, Subscription
from ..gestures.types import Gesture
from ..hand_tracker import HandData

if TYPE_CHECKING:  # Only for type hints: the manager imports this module.
    from ..segmentation import BackgroundModel
    from .manager import EffectManager
    from .particles import ParticleSet
    from .sprites import SpriteLibrary

LOGGER = logging.getLogger(__name__)
Point = tuple[float, float]
HookListener = Callable[[str, Mapping[str, Any]], None]


class EffectState(Enum):
    """spawn -> active -> dying -> dead; the manager drops dead effects."""

    SPAWN = "spawn"
    ACTIVE = "active"
    DYING = "dying"
    DEAD = "dead"


class HookEmitter:
    """Named effect hooks ("fireball_throw", "block_break", ...) for audio and recording.

    Phase 5 subscribes a listener here; effects never import audio code.
    Like the gesture bus, a failing listener is logged and isolated.
    """

    def __init__(self) -> None:
        """Start with no listeners."""
        self._listeners: list[HookListener] = []

    def subscribe(self, listener: HookListener) -> HookListener:
        """Call ``listener(name, payload)`` for every hook; returns it for unsubscribe()."""
        self._listeners.append(listener)
        return listener

    def unsubscribe(self, listener: HookListener) -> bool:
        """Remove a listener; True if it was subscribed."""
        if listener in self._listeners:
            self._listeners.remove(listener)
            return True
        return False

    def emit(self, name: str, **payload: Any) -> None:
        """Notify every listener; exceptions never reach the effect that emitted."""
        for listener in tuple(self._listeners):
            try:
                listener(name, payload)
            except Exception:  # A broken listener must not break the effect.
                LOGGER.exception("Effect hook listener failed on %s", name)


@dataclass
class EffectContext:
    """Everything an effect may read or request during one frame."""

    now: float
    dt: float
    width: int
    height: int
    engine: GestureEngine
    hands: Sequence[HandData]
    events: Sequence[GestureEvent]
    config: EffectsConfig
    sprites: SpriteLibrary
    particles: ParticleSet
    rng: np.random.Generator
    manager: EffectManager
    # Set only on frames where an effect asked for segmentation (needs_person):
    camera: np.ndarray | None = None  # The clean mirrored camera frame (before any drawing).
    person: np.ndarray | None = None  # float32 0..1 person mask at frame size.
    background: BackgroundModel | None = None  # What is behind the person.

    def paste_person(self, frame: np.ndarray, strength: float = 1.0) -> bool:
        """Put the real person back on top of whatever was drawn; False if no mask."""
        if self.person is None or self.camera is None or strength <= 0:
            return False
        weight = self.person if strength >= 1 else self.person * strength
        cv2.blendLinear(self.camera, frame, weight, 1.0 - weight, dst=frame)
        return True

    def emit(self, name: str, **payload: Any) -> None:
        """Fire a named hook (see HookEmitter)."""
        self.manager.hooks.emit(name, **payload)

    def spawn(self, effect: Effect) -> Effect:
        """Add an effect to the scene."""
        return self.manager.spawn(effect)

    def explode(self, center: Point, radius: float, power: float = 1.0, style: str = "fire") -> None:
        """Blast with particles, ring, flash, shake, and impulses on other effects."""
        self.manager.explode(center, radius, power, style)

    def hit(self, center: Point, radius: float) -> bool:
        """True if a circle touches any solid effect (blocks, beams)."""
        return self.manager.hit_test(center, radius)


class Effect(ABC):
    """A visual element with a lifecycle. ``layer`` orders alpha-blended effects."""

    name = "effect"
    layer = 10
    persistent = False  # Survives EffectManager.clear() (e.g. the block world).
    needs_person = False  # True asks the app to segment the person on frames it is alive.
    # Light sources (lighting.py): an emissive effect's render() draws into the black
    # light buffer, which blooms and lights up the scene; it must only add color.
    # Others draw on the frame and may add light separately in glow() (glows = True).
    emissive = False
    glows = False

    def __init__(self) -> None:
        """Start in SPAWN with age 0."""
        self.state = EffectState.SPAWN
        self.age = 0.0

    @property
    def alive(self) -> bool:
        """False once the effect is DEAD."""
        return self.state is not EffectState.DEAD

    def kill(self) -> None:
        """Remove at the end of this frame."""
        self.state = EffectState.DEAD

    @abstractmethod
    def update(self, ctx: EffectContext) -> None:
        """Advance by ctx.dt seconds."""

    @abstractmethod
    def render(self, frame: np.ndarray, ctx: EffectContext) -> None:
        """Draw onto the BGR frame in place."""

    def glow(self, light: np.ndarray, ctx: EffectContext) -> None:
        """Add light for a non-emissive effect (set ``glows``); default none."""

    def distort(self, frame: np.ndarray, ctx: EffectContext) -> None:
        """Bend the camera picture before light is drawn (heat haze, lensing); default none."""

    def hit_test(self, center: Point, radius: float) -> bool:
        """Solid effects return True when the circle touches them."""
        return False

    def on_explosion(self, center: Point, radius: float, power: float, ctx: EffectContext) -> None:
        """React to a blast (knockback, chain reactions); default is no reaction."""


class ModeController:
    """Owns one mode's gesture subscriptions and per-frame interaction logic.

    Every subscription goes through subscribe()/on_hold() so that detach()
    removes exactly those, which is what guarantees no leaked callbacks on a
    mode switch.
    """

    hold_rings: tuple[tuple[Gesture, float], ...] = ()  # (gesture, seconds) shown as rings.

    def __init__(self, bus: EventBus) -> None:
        """Bind to a bus without subscribing yet."""
        self.bus = bus
        self._subscriptions: list[Subscription] = []

    @property
    def attached(self) -> bool:
        """True while this controller's subscriptions are live."""
        return bool(self._subscriptions)

    def attach(self) -> None:
        """Subscribe this mode's bindings (idempotent)."""
        self.detach()
        self.bind()

    def detach(self) -> None:
        """Unsubscribe everything this controller subscribed, then clean up."""
        for subscription in self._subscriptions:
            self.bus.unsubscribe(subscription)
        if self._subscriptions:
            self._subscriptions.clear()
            self.on_detach()

    def subscribe(self, gesture: Gesture | None, kind: EventKind | None, callback: Callback,
                  hand: str | None = None) -> Subscription:
        """Tracked EventBus.subscribe."""
        subscription = self.bus.subscribe(gesture, kind, callback, hand)
        self._subscriptions.append(subscription)
        return subscription

    def on_hold(self, gesture: Gesture, seconds: float, callback: Callback,
                hand: str | None = None) -> Subscription:
        """Tracked EventBus.on_hold_for."""
        subscription = self.bus.on_hold_for(gesture, seconds, callback, hand)
        self._subscriptions.append(subscription)
        return subscription

    def bind(self) -> None:
        """Subclasses subscribe their gesture-to-effect mapping here."""

    def on_detach(self) -> None:
        """Subclasses end their transient effects here."""

    def update(self, ctx: EffectContext) -> None:
        """Per-frame polling logic while the mode is active."""

    def render(self, frame: np.ndarray, ctx: EffectContext) -> None:
        """Per-frame interaction overlays (charge glows, previews, cursors)."""
