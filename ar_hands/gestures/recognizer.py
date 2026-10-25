"""Single-frame recognition: hand features -> raw gesture + confidence.

Stateless by design. The only temporal input is ``is_pinched``, the Schmitt
trigger state the stabilizer keeps; everything else about time (voting,
hysteresis, grace periods) belongs to stabilizer.py.
"""
from __future__ import annotations
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from ..config import GestureConfig
from .features import FingerReading, FingerState, HandFeatures
from .types import Gesture

__all__ = ["Gesture", "Need", "Rule", "RULES", "Recognition", "recognize"]


class Need(Enum):
    """What a rule requires of one finger."""

    EXT = "extended"
    CURL = "curled"
    NOT_EXT = "curled or partial"
    ANY = "any"


# A condition returns 0.0 to reject the rule, or a positive confidence term
# that is averaged with the per-finger agreement scores.
Condition = Callable[[HandFeatures, bool, GestureConfig], float]


@dataclass(frozen=True)
class Rule:
    """One row of the gesture table: per-finger needs plus an optional condition."""

    gesture: Gesture
    fingers: tuple[Need, Need, Need, Need, Need]  # Thumb, index, middle, ring, pinky.
    condition: Condition | None = None


@dataclass(frozen=True)
class Recognition:
    """A raw, unstabilized per-frame result."""

    gesture: Gesture
    confidence: float


NOTHING = Recognition(Gesture.NONE, 0.0)
_DEFAULT_CONFIG = GestureConfig()


def pinched(features: HandFeatures, is_pinched: bool, config: GestureConfig) -> float:
    """Pinch latched AND fingertips away from the palm, which rules out a tucked fist."""
    if is_pinched and features.pinch_reach >= config.pinch_min_reach:
        return max(features.pinch_strength, 0.05)
    return 0.0


def thumb_points_up(features: HandFeatures, is_pinched: bool, config: GestureConfig) -> float:
    """THUMBS_UP means up on screen, so unlike every other pose it is NOT roll-invariant."""
    tilt = abs(features.thumb_direction)
    if tilt > config.thumbs_up_max_tilt_deg:
        return 0.0
    return 1.0 - 0.5 * tilt / config.thumbs_up_max_tilt_deg


def fingers_crossed(features: HandFeatures, is_pinched: bool, config: GestureConfig) -> float:
    """Index tip lying on the middle finger's last bones (Gojo's sign)."""
    if features.cross_gap > config.crossed_max_gap:
        return 0.0
    return 1.0 - 0.5 * features.cross_gap / config.crossed_max_gap


def fingers_spread(features: HandFeatures, is_pinched: bool, config: GestureConfig) -> float:
    """A real V: the index tip clearly away from the middle finger."""
    return 1.0 if features.cross_gap >= config.peace_min_gap else 0.0


E, C, N, A = Need.EXT, Need.CURL, Need.NOT_EXT, Need.ANY
# First match wins, so a specific pose precedes any general pose it overlaps:
# OK before PINCH, THUMBS_UP before FIST, CROSSED before PEACE. Adding a
# gesture is one line here.
RULES: tuple[Rule, ...] = (
    #    gesture             thumb index middle ring pinky   extra condition
    Rule(Gesture.OK,        (A, A, E, E, E), pinched),
    Rule(Gesture.PINCH,     (A, A, A, A, A), pinched),
    Rule(Gesture.OPEN_PALM, (E, E, E, E, E)),
    Rule(Gesture.ROCK,      (A, E, C, C, E)),
    Rule(Gesture.THREE,     (A, E, E, E, C)),
    Rule(Gesture.CROSSED,   (A, E, E, C, C), fingers_crossed),
    Rule(Gesture.PEACE,     (A, E, E, C, C), fingers_spread),
    Rule(Gesture.POINT,     (A, E, C, C, C)),
    Rule(Gesture.THUMBS_UP, (E, C, C, C, C), thumb_points_up),
    Rule(Gesture.FIST,      (N, C, C, C, C)),
)


def _agreement(need: Need, reading: FingerReading) -> float | None:
    """Return how strongly a finger meets a need (0..1), or None if it fails."""
    if need is Need.EXT:
        return reading.openness if reading.state is FingerState.EXTENDED else None
    if need is Need.CURL:
        return 1.0 - reading.openness if reading.state is FingerState.CURLED else None
    return 1.0 - reading.openness if reading.state is not FingerState.EXTENDED else None


def recognize(features: HandFeatures | None, is_pinched: bool | None = None,
              config: GestureConfig | None = None,
              rules: Sequence[Rule] = RULES) -> Recognition:
    """Return the first matching rule's gesture and a 0..1 confidence.

    Without an ``is_pinched`` latch the plain enter threshold is used, which
    keeps this function usable on single images and in tests.
    """
    if features is None:
        return NOTHING
    config = config or _DEFAULT_CONFIG
    if is_pinched is None:
        is_pinched = features.pinch_ratio < config.pinch_enter
    for rule in rules:
        terms: list[float] = []
        for need, reading in zip(rule.fingers, features.fingers):
            if need is Need.ANY:
                continue
            term = _agreement(need, reading)
            if term is None:
                break
            terms.append(term)
        else:
            if rule.condition is not None:
                extra = rule.condition(features, is_pinched, config)
                if extra <= 0.0:
                    continue
                terms.append(extra)
            confidence = sum(terms) / len(terms) if terms else 1.0
            return Recognition(rule.gesture, min(1.0, max(0.0, confidence)))
    return NOTHING
