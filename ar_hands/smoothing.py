"""Timestamp-aware, low-latency smoothing for noisy fingertip measurements."""
from __future__ import annotations
from math import isfinite, pi


class OneEuroFilter:
    """Filter one axis, adapting its cutoff to the estimated movement speed."""

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float) -> None:
        """Initialize this component and its independent runtime state."""
        if min_cutoff <= 0 or beta < 0 or d_cutoff <= 0:
            raise ValueError("One Euro cutoffs must be positive; beta must be nonnegative.")
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self) -> None:
        """Forget history so a reappearing hand starts exactly at its new position."""
        self._time: float | None = None
        self._raw = self._value = self._derivative = 0.0

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        return 1.0 / (1.0 + 1.0 / (2.0 * pi * cutoff * dt))

    def filter(self, value: float, timestamp: float) -> float:
        """Return a smoothed value using strictly increasing seconds."""
        if not isfinite(value) or not isfinite(timestamp):
            raise ValueError("Filter input and timestamp must be finite.")
        if self._time is None:
            self._time, self._raw, self._value = timestamp, value, value
            return value
        dt = timestamp - self._time
        if dt <= 0:
            raise ValueError("One Euro timestamps must strictly increase.")
        # Landmarks jitter even for a stationary finger. Unlike a fixed moving
        # average, a speed-adaptive cutoff smooths pauses without lagging fast arcs.
        derivative = (value - self._raw) / dt
        alpha_d = self._alpha(self.d_cutoff, dt)
        self._derivative += alpha_d * (derivative - self._derivative)
        cutoff = self.min_cutoff + self.beta * abs(self._derivative)
        self._value += self._alpha(cutoff, dt) * (value - self._value)
        self._raw, self._time = value, timestamp
        return self._value


class OneEuroPointFilter:
    """An independent x/y filter pair for one tracked fingertip."""

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float) -> None:
        """Initialize this component and its independent runtime state."""
        self.x = OneEuroFilter(min_cutoff, beta, d_cutoff)
        self.y = OneEuroFilter(min_cutoff, beta, d_cutoff)

    def reset(self) -> None:
        """Reset both axes together after tracking loss."""
        self.x.reset()
        self.y.reset()

    def filter(self, point: tuple[float, float], timestamp: float) -> tuple[float, float]:
        """Smooth a pixel coordinate using a real frame timestamp."""
        return self.x.filter(point[0], timestamp), self.y.filter(point[1], timestamp)
