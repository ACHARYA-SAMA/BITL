"""Palm velocity, swipes, and push toward the camera, all distance-invariant.

Velocities are in hand scales per second. A swipe covers about the same number
of hand-lengths per second whether the hand is 40 cm or 1 m from the camera,
while its pixel speed differs by more than 2x. Pixel velocity is also exposed
for effects that need it (Phase 4 throws a fireball along it).

Directions are in the mirrored display: moving your hand to your right moves
it to the screen-right and gives SWIPE_RIGHT, like a mirror.
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from math import hypot, inf
from ..config import GestureConfig
from .types import Gesture

Point = tuple[float, float]


@dataclass(frozen=True)
class MotionSample:
    """One timestamped palm observation."""

    t: float
    x: float
    y: float
    scale: float


def slope(ts: list[float], values: list[float]) -> float:
    """Least-squares d(value)/dt over the samples.

    A regression over ~0.15 s uses every sample, so one noisy landmark frame
    shifts it far less than a two-point finite difference would. It also uses
    real timestamps, so a dropped frame does not inflate the speed.
    """
    n = len(ts)
    mean_t = sum(ts) / n
    mean_v = sum(values) / n
    denominator = sum((t - mean_t) ** 2 for t in ts)
    if denominator <= 0:
        return 0.0
    return sum((t - mean_t) * (v - mean_v) for t, v in zip(ts, values)) / denominator


class MotionTracker:
    """Short palm history for one hand, with swipe detection and a cooldown."""

    def __init__(self, config: GestureConfig) -> None:
        """Create an empty history."""
        self.config = config
        self._samples: deque[MotionSample] = deque(maxlen=config.motion_history)
        self._last_swipe = -inf
        self.velocity: Point = (0.0, 0.0)
        self.velocity_px: Point = (0.0, 0.0)
        self.push_amount = 0.0

    @property
    def sample_count(self) -> int:
        """Number of samples since the last reset."""
        return len(self._samples)

    @property
    def speed(self) -> float:
        """Current speed in hand scales per second."""
        return hypot(*self.velocity)

    def reset(self) -> None:
        """Forget history (hand lost or teleported); the swipe cooldown is kept."""
        self._samples.clear()
        self.velocity = self.velocity_px = (0.0, 0.0)
        self.push_amount = 0.0

    def update(self, t: float, point: Point, scale: float) -> None:
        """Add a palm sample at perf_counter time ``t``."""
        if self._samples:
            last = self._samples[-1]
            dt = t - last.t
            if dt <= 0:
                return
            jump = hypot(point[0] - last.x, point[1] - last.y) / max(scale, 1e-6) / dt
            if jump > self.config.motion_max_speed:
                # No real hand moves this fast: it is a label swap or a bad
                # frame. Restart history so the jump never becomes a swipe.
                self.reset()
        self._samples.append(MotionSample(t, point[0], point[1], scale))
        self._estimate(t)

    def _estimate(self, now: float) -> None:
        c = self.config
        recent = [s for s in self._samples if now - s.t <= c.motion_window_s]
        if len(recent) < c.motion_min_samples:
            recent = list(self._samples)[-c.motion_min_samples:]
        if len(recent) < 2:
            self.velocity = self.velocity_px = (0.0, 0.0)
            self.push_amount = 0.0
            return
        ts = [s.t for s in recent]
        mean_scale = sum(s.scale for s in recent) / len(recent)
        vx, vy = slope(ts, [s.x for s in recent]), slope(ts, [s.y for s in recent])
        self.velocity_px = (vx, vy)
        self.velocity = (vx / mean_scale, vy / mean_scale)
        # The hand looks bigger as it approaches, so relative growth rate of the
        # scale is a distance-independent "push toward the camera" signal.
        self.push_amount = slope(ts, [s.scale for s in recent]) / mean_scale

    def detect_swipe(self, now: float, allowed: bool = True) -> tuple[Gesture, float] | None:
        """Return (SWIPE_*, confidence) at most once per cooldown, else None.

        The cooldown is shared by all directions on purpose: the hand coming
        back after a fast right swipe must not register as a left swipe.
        """
        c = self.config
        if (not allowed or len(self._samples) < c.swipe_min_samples
                or now - self._last_swipe < c.swipe_cooldown_s):
            return None
        vx, vy = self.velocity
        speed = hypot(vx, vy)
        if speed < c.swipe_speed:
            return None
        if abs(vx) >= c.swipe_dominance * abs(vy):
            gesture = Gesture.SWIPE_RIGHT if vx > 0 else Gesture.SWIPE_LEFT
        elif abs(vy) >= c.swipe_dominance * abs(vx):
            gesture = Gesture.SWIPE_DOWN if vy > 0 else Gesture.SWIPE_UP  # Image y grows down.
        else:
            return None  # Diagonal: ambiguous, so ignore it rather than guess.
        self._last_swipe = now
        return gesture, min(1.0, speed / (2 * c.swipe_speed))
