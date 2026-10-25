"""Adaptive quality: trade visual detail for frame rate when the pipeline cannot keep up."""
from __future__ import annotations
from collections.abc import Callable, Sequence
from dataclasses import dataclass
import logging

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class QualityStep:
    """One reversible degradation, applied in list order and restored in reverse."""

    name: str
    lower: Callable[[], None]
    restore: Callable[[], None]
    applicable: Callable[[], bool] = lambda: True  # False = nothing to gain (e.g. glow already off).


class AdaptiveQuality:
    """Step down after ``down_after_s`` below target, back up after ``up_after_s`` with headroom.

    The input is processing time per frame, not the displayed frame rate: a
    15 FPS webcam caps the display, and lowering quality would not help it.
    The gap between "below target" and "target x headroom" is a dead zone, and
    a step that has to be undone soon after it was restored doubles the wait
    before the next restore, so quality settles instead of oscillating.
    """

    def __init__(self, steps: Sequence[QualityStep], target_fps: float = 24.0,
                 down_after_s: float = 2.0, up_after_s: float = 5.0, headroom: float = 1.4,
                 enabled: bool = True, smoothing: float = 0.9) -> None:
        """No step applied yet."""
        if target_fps <= 0 or headroom < 1:
            raise ValueError("target_fps must be positive and headroom at least 1.")
        self.steps = tuple(steps)
        self.target_fps, self.down_after_s, self.headroom = target_fps, down_after_s, headroom
        self.base_up_after_s = self.up_after_s = up_after_s
        self.enabled = enabled
        self.smoothing = smoothing
        self.applied: list[QualityStep] = []
        self.busy_ms = 0.0
        self._below_since: float | None = None
        self._above_since: float | None = None
        self._last_restore = -1e9

    @property
    def level(self) -> int:
        """Number of steps currently applied (0 = full quality)."""
        return len(self.applied)

    @property
    def achievable_fps(self) -> float:
        """Frame rate the processing alone would allow."""
        return 1000.0 / self.busy_ms if self.busy_ms > 0 else 0.0

    def update(self, now: float, busy_ms: float) -> str | None:
        """Feed one processed frame's busy time; returns a message when quality changes."""
        if busy_ms <= 0:
            return None
        self.busy_ms = busy_ms if self.busy_ms == 0 else (
            self.smoothing * self.busy_ms + (1 - self.smoothing) * busy_ms)
        if not self.enabled:
            return None
        fps = self.achievable_fps
        if fps < self.target_fps:
            self._above_since = None
            self._below_since = now if self._below_since is None else self._below_since
            if now - self._below_since >= self.down_after_s:
                self._below_since = now  # The next step waits a full period on fresh numbers.
                return self._step_down(now)
        elif fps > self.target_fps * self.headroom and self.applied:
            self._below_since = None
            self._above_since = now if self._above_since is None else self._above_since
            if now - self._above_since >= self.up_after_s:
                self._above_since = now
                return self._step_up(now)
        else:
            self._below_since = self._above_since = None
        return None

    def _step_down(self, now: float) -> str | None:
        for step in self.steps:
            if step in self.applied or not step.applicable():
                continue
            step.lower()
            self.applied.append(step)
            if now - self._last_restore < 10.0:
                self.up_after_s *= 2  # Restored too eagerly last time: be slower to restore again.
            message = f"Quality down: {step.name} ({self.achievable_fps:.0f} FPS possible)"
            LOGGER.info(message)
            return message
        return None

    def _step_up(self, now: float) -> str:
        step = self.applied.pop()
        step.restore()
        self._last_restore = now
        message = f"Quality up: {step.name} restored"
        LOGGER.info(message)
        return message

    def set_enabled(self, enabled: bool) -> None:
        """Turning adaptation off restores full quality immediately."""
        self.enabled = enabled
        if not enabled:
            self.reset()

    def reset(self) -> None:
        """Undo every applied step (newest first)."""
        while self.applied:
            self.applied.pop().restore()
        self.up_after_s = self.base_up_after_s
        self._below_since = self._above_since = None
