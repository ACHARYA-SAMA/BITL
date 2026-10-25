"""Temporal filtering: turn flickery per-frame guesses into stable gestures.

Two layers, because they reject different noise:
1. A sliding-window plurality vote absorbs isolated wrong frames INSIDE a run,
   so one glitch does not restart a gesture's entry count.
2. Hysteresis on the vote's output: a winner must hold for ``enter_frames`` in
   a row to activate, and the active gesture survives ``exit_grace_frames`` of
   losing before it ends. Unequal thresholds stop chatter at a boundary.
With the defaults (window 5, enter 4), a 2-frame flicker can never even win
the vote, so a single misdetected frame can never fire an action.
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from ..config import GestureConfig
from .types import Gesture


class SchmittTrigger:
    """Boolean latch with separate on/off thresholds (on below, off above).

    A pinch ratio hovering at a single threshold would toggle every frame. With
    a gap (default on < 0.25, off > 0.35) noise must cross the whole band.
    """

    def __init__(self, enter_below: float, exit_above: float) -> None:
        """Create an open (False) latch."""
        if not enter_below < exit_above:
            raise ValueError("enter_below must be smaller than exit_above.")
        self.enter_below, self.exit_above = enter_below, exit_above
        self.state = False

    def update(self, value: float) -> bool:
        """Feed one measurement and return the latched state."""
        if self.state:
            if value > self.exit_above:
                self.state = False
        elif value < self.enter_below:
            self.state = True
        return self.state

    def reset(self) -> None:
        """Release the latch."""
        self.state = False


@dataclass(frozen=True)
class Transition:
    """What changed this frame; both fields are set for a direct hand-over."""

    ended: Gesture | None = None
    started: Gesture | None = None


NO_CHANGE = Transition()


class GestureStabilizer:
    """Per-hand vote + hysteresis + lost-hand grace. Feed exactly one call per frame."""

    def __init__(self, config: GestureConfig) -> None:
        """Start with no active gesture."""
        self.config = config
        self._window: deque[tuple[Gesture, float]] = deque(maxlen=config.vote_window)
        self.reset()

    def reset(self) -> None:
        """Forget everything without emitting a transition."""
        self._window.clear()
        self.active = self.winner = self.candidate = Gesture.NONE
        self.streak = self.misses = self.missing = 0

    @property
    def lost(self) -> bool:
        """True once the hand has been missing longer than the grace period."""
        return self.missing > self.config.lost_grace_frames

    @property
    def confidence(self) -> float:
        """Mean raw confidence of the window frames that agree with the active gesture."""
        values = [score for gesture, score in self._window if gesture is self.active]
        return sum(values) / len(values) if values else 0.0

    def _vote(self) -> Gesture:
        counts: dict[Gesture, int] = {}
        for gesture, _ in self._window:
            counts[gesture] = counts.get(gesture, 0) + 1
        best = max(counts.values())
        # Ties go to the most recent of the tied gestures: newer evidence wins.
        for gesture, _ in reversed(self._window):
            if counts[gesture] == best:
                return gesture
        return Gesture.NONE

    def update(self, raw: Gesture, confidence: float) -> Transition:
        """Feed this frame's raw recognition for a visible hand."""
        self.missing = 0
        self._window.append((raw, confidence))
        self.winner = self._vote()
        if self.winner is self.candidate:
            self.streak += 1
        else:
            self.candidate, self.streak = self.winner, 1
        self.misses = 0 if self.winner is self.active else self.misses + 1
        return self._settle()

    def update_missing(self) -> Transition:
        """Advance one frame without a hand; end the gesture once grace runs out."""
        self.missing += 1
        if self.lost and self.active is not Gesture.NONE:
            ended = self.active
            self.reset()
            self.missing = self.config.lost_grace_frames + 1
            return Transition(ended=ended)
        return NO_CHANGE

    def _settle(self) -> Transition:
        ended = started = None
        if self.active is not Gesture.NONE and self.misses > self.config.exit_grace_frames:
            ended, self.active = self.active, Gesture.NONE
        if (self.candidate is not Gesture.NONE and self.candidate is not self.active
                and self.streak >= self.config.enter_frames):
            if self.active is not Gesture.NONE:
                ended = self.active  # Possible only when enter_frames <= exit_grace_frames.
            self.active = started = self.candidate
            self.misses = 0
        if ended is None and started is None:
            return NO_CHANGE
        return Transition(ended, started)
