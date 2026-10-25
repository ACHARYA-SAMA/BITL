"""Thin adapter from gesture-engine decisions to per-hand canvas strokes.

Phase 2 decided pen up/down here with its own pointing test and frame
debounce. The gesture engine now owns that decision, already stabilized, so a
second debounce would only add latency. What remains is drawing-specific:
fingertip smoothing, splitting strokes on teleports or far resumptions, and
backfilling the stroke start that the stabilizer's entry delay would lose.
"""
from __future__ import annotations
from collections import deque
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from .canvas import AirCanvas
from .config import Config
from .gestures.engine import GestureEngine
from .gestures.features import Finger, FingerState
from .gestures.types import Gesture
from .hand_tracker import HandData, Landmark, distance
from .smoothing import OneEuroPointFilter

Point = tuple[float, float]


@dataclass(frozen=True)
class PenSignal:
    """The gesture layer's verdict for one hand this frame.

    ``down`` keeps a stroke open (stable draw gesture). ``ink`` says this
    frame's fingertip may be drawn (index visibly extended right now), which
    stops the line the instant the finger curls, before the stable gesture ends.
    """

    down: bool
    ink: bool


PEN_UP = PenSignal(False, False)


def pen_signals(engine: GestureEngine, draw_gestures: Collection[Gesture]) -> dict[str, PenSignal]:
    """Ask the engine whether each tracked hand is drawing.

    This replaces Phase 2's is_pen_down(). Hands within the engine's lost-hand
    grace stay ``down`` without ``ink``, so a brief dropout pauses rather
    than ends a stroke.
    """
    signals: dict[str, PenSignal] = {}
    for name in engine.tracked_hands():
        features = engine.features(name)
        ink = features is not None and (engine.raw(name) in draw_gestures
                                        or features.state(Finger.INDEX) is FingerState.EXTENDED)
        signals[name] = PenSignal(engine.active(name) in draw_gestures, ink)
    return signals


@dataclass
class Cursor:
    """Smoothed cursor feedback for one visible hand."""

    hand: str
    point: tuple[float, float]
    drawing: bool


@dataclass
class _PenState:
    """Private per-hand smoothing and gap state, never shared between hands."""

    smoother: OneEuroPointFilter
    recent: deque[tuple[Point, bool]] = field(default_factory=deque)  # (point, ink)
    missing: int = 0
    dry: int = 0
    last_raw: tuple[int, int] | None = None
    last_ink: Point | None = None


class PenController:
    """Turn per-hand PenSignals into smooth, uninterrupted, independent strokes."""

    def __init__(self, config: Config) -> None:
        """Initialize this component and its independent runtime state."""
        self.config = config
        self._states: dict[str, _PenState] = {}

    def _new_state(self) -> _PenState:
        c = self.config
        return _PenState(OneEuroPointFilter(c.smoothing_min_cutoff, c.smoothing_beta,
                                            c.smoothing_d_cutoff),
                         deque(maxlen=c.pen_backfill_frames))

    def reset(self, canvas: AirCanvas) -> None:
        """Finish strokes and reset filters after brush/history operations."""
        canvas.end_all()
        self._states.clear()

    def update(self, hands: list[HandData], canvas: AirCanvas, timestamp: float,
               color: tuple[int, int, int], thickness: int,
               signals: Mapping[str, PenSignal]) -> list[Cursor]:
        """Update from one frame; timestamp is monotonic seconds, not a frame count."""
        visible: dict[str, HandData] = {}
        for hand in hands:
            if hand.handedness not in visible or hand.score > visible[hand.handedness].score:
                visible[hand.handedness] = hand
        for name in tuple(self._states):
            if name in visible:
                continue
            state = self._states[name]
            state.missing += 1
            state.smoother.reset()  # Never smooth across a gap: no streak on return.
            state.recent.clear()
            if not signals.get(name, PEN_UP).down:
                canvas.end_stroke(name)
                del self._states[name]
        return [self._update_hand(name, hand, canvas, timestamp, color, thickness,
                                  signals.get(name, PEN_UP))
                for name, hand in visible.items()]

    def _update_hand(self, name: str, hand: HandData, canvas: AirCanvas, timestamp: float,
                     color: tuple[int, int, int], thickness: int, signal: PenSignal) -> Cursor:
        c = self.config
        state = self._states.get(name)
        if state is None:
            state = self._states[name] = self._new_state()
        raw = hand.landmarks_px[Landmark.INDEX_TIP]
        size = max(c.min_hand_size, distance(hand.landmarks_px[Landmark.WRIST],
                                             hand.landmarks_px[Landmark.MIDDLE_MCP]))
        if (state.missing == 0 and state.last_raw is not None
                and distance(raw, state.last_raw) > size * c.max_jump_hand_sizes):
            # No fingertip moves this far in one frame: the label switched
            # hands. Split, and restart smoothing at the new position.
            canvas.end_stroke(name)
            state = self._states[name] = self._new_state()
        point = state.smoother.filter(raw, timestamp)
        resumed_after_gap = state.missing > 0 or state.dry >= c.pen_gap_frames
        state.missing, state.last_raw = 0, raw
        state.recent.append((point, signal.ink))
        if not signal.down:
            canvas.end_stroke(name)
            state.last_ink = None
        elif name not in canvas.active:
            canvas.begin_stroke(name, color, thickness)
            # The stabilizer confirms POINT a few frames late. Replay the trailing
            # frames where the finger was already extended, so the line starts
            # where the user started pointing, not ~130 ms later.
            backfill: list[Point] = []
            for past, inked in reversed(state.recent):
                if not inked:
                    break
                backfill.append(past)
            for past in reversed(backfill):
                canvas.add_point(name, past)
            state.last_ink = backfill[0] if backfill else None
        elif signal.ink:
            if (resumed_after_gap and state.last_ink is not None
                    and distance(point, state.last_ink) > size * c.resume_max_jump_hand_sizes):
                # Ink resumed somewhere else after a pause: start a new stroke
                # instead of bridging the gap with a straight line.
                canvas.end_stroke(name)
                canvas.begin_stroke(name, color, thickness)
            canvas.add_point(name, point)
            state.last_ink = point
        state.dry = 0 if signal.ink else state.dry + 1
        return Cursor(name, point, signal.down)
