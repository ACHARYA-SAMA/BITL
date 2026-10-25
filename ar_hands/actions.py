"""Application actions and how a gesture binding triggers them.

Kept separate from the gesture engine: the engine only reports what hands do,
while this vocabulary says what the drawing app does about it. Phase 4 modes can
define their own bindings with the same types.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum


class Action(Enum):
    """Something the app can do in response to a gesture or a key."""

    DRAW = "draw"
    HOVER = "hover"
    CLEAR = "clear"
    NEXT_COLOR = "next_color"
    SAVE_PNG = "save_png"
    UNDO = "undo"
    REDO = "redo"


class Trigger(Enum):
    """When a binding fires."""

    ACTIVE = "active"    # Polled every frame while the gesture is stable.
    STARTED = "started"  # Once, when the gesture becomes stable.
    HOLD = "hold"        # Once per activation, after hold_seconds of holding.


@dataclass(frozen=True)
class GestureAction:
    """One gesture binding: which action, and when it fires."""

    action: Action
    trigger: Trigger = Trigger.STARTED
    hold_seconds: float = 0.0

    def __post_init__(self) -> None:
        """Reject a hold without a duration, which would fire on the first frame."""
        if self.trigger is Trigger.HOLD and self.hold_seconds <= 0:
            raise ValueError("A HOLD binding needs hold_seconds > 0.")
