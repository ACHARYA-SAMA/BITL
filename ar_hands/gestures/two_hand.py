"""Two-hand gestures: PRAYER, TWO_FISTS, HANDS_APART, plus a continuous distance.

Raw two-hand poses are built from each hand's RAW per-frame result, then
stabilized once by the engine's "Both" stabilizer. Starting from the per-hand
STABLE gestures would stack two stabilizers and double the latency.
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from math import hypot, inf
from ..config import GestureConfig
from .features import LONG_FINGERS, FingerState, HandFeatures
from .motion import slope
from .types import Gesture

Point = tuple[float, float]


@dataclass(frozen=True)
class TwoHandState:
    """Continuous two-hand measurements; zeros while fewer than two hands are visible."""

    present: bool = False
    distance_px: float = 0.0  # Wrist to wrist.
    distance: float = 0.0  # Wrist to wrist in mean hand scales.
    separation_speed: float = 0.0  # Hand scales per second; positive = moving apart.
    midpoint: Point = (0.0, 0.0)  # Between the palm centers.
    scale: float = 0.0  # Mean hand scale.


def _gap(a: Point, b: Point) -> float:
    return hypot(a[0] - b[0], a[1] - b[1])


def _open(features: HandFeatures) -> bool:
    """Four long fingers extended. The thumb is ignored: in a prayer seal it is
    usually pressed alongside the index finger, so it never reads as extended."""
    return all(features.state(finger) is FingerState.EXTENDED for finger in LONG_FINGERS)


def is_prayer(left: HandFeatures, right: HandFeatures, config: GestureConfig) -> bool:
    """Both hands open, palm centers together, and matching fingertips together."""
    if not (_open(left) and _open(right)):
        return False
    scale = (left.scale + right.scale) / 2
    if _gap(left.palm_center, right.palm_center) / scale > config.prayer_palm_distance:
        return False
    tip_gap = sum(_gap(left.tips[f], right.tips[f]) for f in LONG_FINGERS) / len(LONG_FINGERS)
    return tip_gap / scale <= config.prayer_tip_distance


def recognize_two_hand(left: HandFeatures | None, right: HandFeatures | None,
                       left_raw: Gesture, right_raw: Gesture,
                       config: GestureConfig) -> Gesture:
    """Return this frame's raw two-hand pose (NONE unless both hands are visible)."""
    if left is None or right is None:
        return Gesture.NONE
    if is_prayer(left, right, config):
        return Gesture.PRAYER
    if left_raw is Gesture.FIST and right_raw is Gesture.FIST:
        return Gesture.TWO_FISTS
    if left_raw is Gesture.OPEN_PALM and right_raw is Gesture.OPEN_PALM:
        return Gesture.TWO_PALMS  # Both palms shown apart (pressed together is PRAYER).
    return Gesture.NONE


class TwoHandTracker:
    """Track wrist separation over time and detect HANDS_APART with a cooldown."""

    def __init__(self, config: GestureConfig) -> None:
        """Start with no history."""
        self.config = config
        self._samples: deque[tuple[float, float]] = deque(maxlen=config.motion_history)
        self._last_apart = -inf
        self.state = TwoHandState()

    def reset(self) -> None:
        """Forget separation history; keep the cooldown."""
        self._samples.clear()
        self.state = TwoHandState()

    def update(self, t: float, left: HandFeatures | None,
               right: HandFeatures | None) -> bool:
        """Update measurements; return True when HANDS_APART should fire now."""
        c = self.config
        if left is None or right is None:
            self.reset()
            return False
        scale = (left.scale + right.scale) / 2
        distance_px = _gap(left.wrist, right.wrist)
        distance = distance_px / scale
        if self._samples:
            last_t, last_distance = self._samples[-1]
            dt = t - last_t
            if dt <= 0:
                return False
            if abs(distance - last_distance) / dt > 2 * c.motion_max_speed:
                self._samples.clear()  # A label swap, not real motion.
        self._samples.append((t, distance))
        recent = [s for s in self._samples if t - s[0] <= c.motion_window_s]
        if len(recent) < c.motion_min_samples:
            recent = list(self._samples)[-c.motion_min_samples:]
        speed = slope([s[0] for s in recent], [s[1] for s in recent]) if len(recent) >= 2 else 0.0
        midpoint = ((left.palm_center[0] + right.palm_center[0]) / 2,
                    (left.palm_center[1] + right.palm_center[1]) / 2)
        self.state = TwoHandState(True, distance_px, distance, speed, midpoint, scale)
        if (len(self._samples) >= c.swipe_min_samples and speed >= c.apart_speed
                and t - self._last_apart >= c.apart_cooldown_s):
            self._last_apart = t
            return True
        return False
