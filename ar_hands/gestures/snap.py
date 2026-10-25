"""Finger-snap detection: a two-step motion, not a pose.

A snap is too fast to see as a pose: the release takes 30-60 ms, one frame
at 15-30 FPS. It is detected as a sequence instead:
  1. ARM: the thumb tip presses the middle fingertip (and is clearly closer
     to it than to the index tip, so an ordinary pinch never arms it).
  2. FIRE: within ``snap_window_s`` of the last armed frame, the thumb has
     left (ratio above ``snap_release_ratio``) AND the middle tip moved
     toward the wrist by ``snap_drop``, the finger slamming into the palm.
Both steps use hand-scale ratios, so a snap reads the same near or far.
"""
from __future__ import annotations
from math import inf
from ..config import GestureConfig
from .features import HandFeatures


class SnapDetector:
    """Per-hand snap state machine with a cooldown."""

    def __init__(self, config: GestureConfig) -> None:
        """Start disarmed."""
        self.config = config
        self.armed_at: float | None = None
        self.armed_reach = 0.0
        self._last_fire = -inf

    def reset(self) -> None:
        """Disarm (hand lost); the cooldown is kept."""
        self.armed_at = None

    def update(self, t: float, features: HandFeatures | None) -> bool:
        """Feed one frame; True on the frame a snap completes."""
        c = self.config
        if features is None:
            self.reset()
            return False
        ratio = features.middle_pinch_ratio
        if ratio < c.snap_arm_ratio and ratio + c.snap_margin < features.pinch_ratio:
            self.armed_at, self.armed_reach = t, features.middle_reach
            return False
        if self.armed_at is None:
            return False
        if t - self.armed_at > c.snap_window_s:
            self.armed_at = None
            return False
        released = ratio > c.snap_release_ratio
        slammed = self.armed_reach - features.middle_reach >= c.snap_drop
        if released and slammed and t - self._last_fire >= c.snap_cooldown_s:
            self.armed_at, self._last_fire = None, t
            return True
        return False
