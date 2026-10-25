"""Draw mode: the Phase 2/3 drawing gestures packaged as a mode controller."""
from __future__ import annotations
from collections.abc import Callable, Mapping
from functools import partial
from .actions import Action, GestureAction, Trigger
from .effects.base import ModeController
from .gestures.events import EventBus, EventKind, GestureEvent
from .gestures.types import Gesture


class DrawMode(ModeController):
    """Subscribes Config.gesture_actions while Draw mode is active.

    ACTIVE bindings (POINT = draw, FIST = hover) are states that the pen
    adapter polls each frame, so only STARTED and HOLD bindings subscribe.
    """

    def __init__(self, bus: EventBus, bindings: Mapping[Gesture, GestureAction],
                 perform: Callable[[Action, bool], None]) -> None:
        """``perform(action, from_gesture)`` runs an action (App.perform)."""
        super().__init__(bus)
        self.bindings, self.perform = bindings, perform
        self.hold_rings = tuple((gesture, binding.hold_seconds) for gesture, binding in bindings.items()
                                if binding.trigger is Trigger.HOLD)

    @property
    def draw_gestures(self) -> frozenset[Gesture]:
        """Gestures that put the pen down."""
        return frozenset(gesture for gesture, binding in self.bindings.items()
                         if binding.action is Action.DRAW and binding.trigger is Trigger.ACTIVE)

    def bind(self) -> None:
        """Subscribe each one-shot binding."""
        for gesture, binding in self.bindings.items():
            handler = partial(self._fire, binding.action)
            if binding.trigger is Trigger.HOLD:
                self.on_hold(gesture, binding.hold_seconds, handler)
            elif binding.trigger is Trigger.STARTED:
                self.subscribe(gesture, EventKind.STARTED, handler)

    def _fire(self, action: Action, event: GestureEvent) -> None:
        self.perform(action, True)
