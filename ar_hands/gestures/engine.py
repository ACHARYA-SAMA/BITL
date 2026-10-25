"""GestureEngine: one ``update()`` per frame turns hands into stable gestures and events.

    HandData ─► features ─► recognizer ─► stabilizer ─► STARTED/HELD/ENDED ─┐
         │                 ▲ pinch latch                                    │
         ├─► snap ─► SNAP (arm, then release + slam) ───────────────────────┤
         ├─► motion ─► SWIPE_* (cooldown) ──────────────────────────────────┼─► EventBus
         └─► two_hand ─► "Both" stabilizer ─► PRAYER / TWO_FISTS ───────────┤
                         separation speed ─► HANDS_APART ───────────────────┘

State is kept per handedness label ("Left"/"Right") plus "Both". Everything is
plain Python on 21 points per hand: no numpy, no allocations beyond small
tuples, about 0.1 ms per frame for two hands (see tests/test_gestures.py).
"""
from __future__ import annotations
from collections.abc import Iterable
from dataclasses import dataclass, field
from ..config import GestureConfig
from ..hand_tracker import HandData
from .events import BOTH, EventBus, EventKind, GestureEvent
from .features import HandFeatures, extract_features
from .motion import MotionTracker
from .snap import SnapDetector
from .recognizer import NOTHING, recognize
from .stabilizer import GestureStabilizer, SchmittTrigger, Transition
from .two_hand import TwoHandState, TwoHandTracker, recognize_two_hand
from .types import Gesture

Point = tuple[float, float]


@dataclass
class _Lifecycle:
    """Turns one stabilizer's transitions into lifecycle events for one hand id."""

    name: str
    stabilizer: GestureStabilizer
    started_at: float = 0.0
    position: Point = (0.0, 0.0)
    confidence: float = 0.0

    def emit(self, transition: Transition, t: float, position: Point | None,
             batch: _Batch) -> None:
        """Record ENDED with the old anchor, then STARTED or HELD with the new one."""
        if transition.ended is not None:
            batch.ended.append(GestureEvent(EventKind.ENDED, transition.ended, self.name,
                                            self.position, t - self.started_at,
                                            self.confidence, t))
        if position is not None:
            self.position = position
        active = self.stabilizer.active
        if active is Gesture.NONE:
            return
        self.confidence = self.stabilizer.confidence or self.confidence
        if transition.started is not None:
            self.started_at = t
            batch.started.append(GestureEvent(EventKind.STARTED, active, self.name,
                                              self.position, 0.0, self.confidence, t))
        else:
            batch.held.append(GestureEvent(EventKind.HELD, active, self.name, self.position,
                                           t - self.started_at, self.confidence, t))


@dataclass
class _Batch:
    """Events of one frame, kept in dispatch order: ENDED, STARTED, HELD, instants."""

    ended: list[GestureEvent] = field(default_factory=list)
    started: list[GestureEvent] = field(default_factory=list)
    held: list[GestureEvent] = field(default_factory=list)
    instant: list[GestureEvent] = field(default_factory=list)

    def add_instant(self, gesture: Gesture, hand: str, position: Point,
                    confidence: float, t: float) -> None:
        """Instant gestures get STARTED and ENDED together, so lifecycles stay paired."""
        for kind in (EventKind.STARTED, EventKind.ENDED):
            self.instant.append(GestureEvent(kind, gesture, hand, position, 0.0, confidence, t))

    def ordered(self) -> list[GestureEvent]:
        """Return the frame's events in dispatch order."""
        return self.ended + self.started + self.held + self.instant


@dataclass
class _HandTrack:
    """Everything the engine remembers about one handedness label."""

    lifecycle: _Lifecycle
    pinch: SchmittTrigger
    motion: MotionTracker
    snap: SnapDetector
    features: HandFeatures | None = None
    raw: Gesture = Gesture.NONE
    raw_confidence: float = 0.0
    scale: float = 0.0
    palm: Point | None = None


def _anchor(gesture: Gesture, features: HandFeatures) -> Point:
    """Where a gesture "is": pinch midpoint, index tip for POINT, else the palm."""
    if gesture in (Gesture.PINCH, Gesture.OK):
        return features.pinch_point
    if gesture is Gesture.POINT:
        return features.tips[1]
    return features.palm_center


class GestureEngine:
    """Per-frame gesture pipeline with an event bus and a polling API."""

    def __init__(self, config: GestureConfig | None = None, bus: EventBus | None = None) -> None:
        """Create an engine; pass a shared bus to let other systems subscribe."""
        self.config = config or GestureConfig()
        # Not "bus or EventBus()": a bus with no subscribers yet has len 0 and is falsy.
        self.bus = bus if bus is not None else EventBus()
        self._tracks: dict[str, _HandTrack] = {}
        self._both = _Lifecycle(BOTH, GestureStabilizer(self.config))
        self._two = TwoHandTracker(self.config)
        self.last_events: tuple[GestureEvent, ...] = ()

    # ----------------------------------------------------------------- update
    def update(self, hands: Iterable[HandData], timestamp: float) -> list[GestureEvent]:
        """Process one frame (perf_counter seconds), dispatch its events, and return them."""
        visible: dict[str, HandData] = {}
        for hand in hands:
            # The tracker already makes labels unique; keep the most confident
            # candidate anyway so a duplicate can never merge two hands' state.
            if hand.handedness not in visible or hand.score > visible[hand.handedness].score:
                visible[hand.handedness] = hand
        batch = _Batch()
        for name, hand in visible.items():
            self._update_hand(name, hand, timestamp, batch)
        for name in tuple(self._tracks):
            if name not in visible:
                self._update_missing(name, timestamp, batch)
        self._update_both(timestamp, batch)
        events = batch.ordered()
        self.last_events = tuple(events)
        self.bus.dispatch_all(events)
        return events

    def _new_track(self, name: str) -> _HandTrack:
        c = self.config
        return _HandTrack(_Lifecycle(name, GestureStabilizer(c)),
                          SchmittTrigger(c.pinch_enter, c.pinch_exit), MotionTracker(c), SnapDetector(c))

    def _update_hand(self, name: str, hand: HandData, t: float, batch: _Batch) -> None:
        track = self._tracks.get(name)
        if track is None:
            track = self._tracks[name] = self._new_track(name)
        features = extract_features(hand, self.config)
        if features is None:
            recognition = NOTHING
            track.pinch.reset()
        else:
            recognition = recognize(features, track.pinch.update(features.pinch_ratio), self.config)
            track.motion.update(t, features.palm_center, features.scale)
            track.scale, track.palm = features.scale, features.palm_center
        track.features, track.raw, track.raw_confidence = (features, recognition.gesture,
                                                           recognition.confidence)
        stabilizer = track.lifecycle.stabilizer
        transition = stabilizer.update(recognition.gesture, recognition.confidence)
        anchor = None
        if features is not None:
            anchor = _anchor(stabilizer.active if stabilizer.active is not Gesture.NONE
                             else recognition.gesture, features)
        track.lifecycle.emit(transition, t, anchor, batch)
        # Judge the pose on raw, voted, and active gestures: fast motion blurs
        # landmarks, and the frames before POINT activates must count as drawing.
        poses = {recognition.gesture, stabilizer.winner, stabilizer.active}
        c = self.config
        allowed = (not poses & c.swipe_suppressed_by
                   and (not c.swipe_requires or bool(poses & c.swipe_requires)))
        swipe = track.motion.detect_swipe(t, allowed)
        if swipe is not None and track.palm is not None:
            batch.add_instant(swipe[0], name, track.palm, swipe[1], t)
        if track.snap.update(t, features):
            batch.add_instant(Gesture.SNAP, name, features.tips[0], 1.0, t)

    def _update_missing(self, name: str, t: float, batch: _Batch) -> None:
        track = self._tracks[name]
        track.features, track.raw, track.raw_confidence = None, Gesture.NONE, 0.0
        # A returning hand must not produce a velocity spike from its old position.
        track.motion.reset()
        track.snap.reset()
        track.lifecycle.emit(track.lifecycle.stabilizer.update_missing(), t, None, batch)
        if track.lifecycle.stabilizer.lost:
            del self._tracks[name]

    def _update_both(self, t: float, batch: _Batch) -> None:
        left, right = self._tracks.get("Left"), self._tracks.get("Right")
        left_features = left.features if left else None
        right_features = right.features if right else None
        apart = self._two.update(t, left_features, right_features)
        state = self._two.state
        stabilizer = self._both.stabilizer
        if left_features is None or right_features is None:
            transition = stabilizer.update_missing()
            if stabilizer.lost:
                stabilizer.missing = self.config.lost_grace_frames + 1  # Cap the counter.
        else:
            raw = recognize_two_hand(left_features, right_features, left.raw, right.raw,
                                     self.config)
            confidence = (left.raw_confidence + right.raw_confidence) / 2
            transition = stabilizer.update(raw, confidence if raw is not Gesture.NONE else 0.0)
        self._both.emit(transition, t, state.midpoint if state.present else None, batch)
        if apart:
            batch.add_instant(Gesture.HANDS_APART, BOTH, state.midpoint,
                              min(1.0, state.separation_speed / (2 * self.config.apart_speed)), t)

    def reset(self, timestamp: float) -> list[GestureEvent]:
        """End every active gesture with ENDED events (dispatched), then forget all state."""
        batch = _Batch()
        for lifecycle in [track.lifecycle for track in self._tracks.values()] + [self._both]:
            active = lifecycle.stabilizer.active
            lifecycle.stabilizer.reset()  # Before emitting, so no HELD follows the ENDED.
            if active is not Gesture.NONE:
                lifecycle.emit(Transition(ended=active), timestamp, None, batch)
        self._tracks.clear()
        self._two.reset()
        events = batch.ordered()
        self.bus.dispatch_all(events)
        return events

    # ---------------------------------------------------------------- polling
    def _track(self, hand: str) -> _HandTrack | None:
        return self._tracks.get(hand)

    def tracked_hands(self) -> tuple[str, ...]:
        """Single-hand labels with state: visible now or within the lost grace."""
        return tuple(self._tracks)

    def visible(self, hand: str) -> bool:
        """True if this hand was detected (and measurable) this frame."""
        track = self._track(hand)
        return track is not None and track.features is not None

    def active(self, hand: str) -> Gesture:
        """Stable gesture of "Left", "Right", or "Both" (NONE if untracked)."""
        if hand == BOTH:
            return self._both.stabilizer.active
        track = self._track(hand)
        return track.lifecycle.stabilizer.active if track else Gesture.NONE

    def is_active(self, hand: str, gesture: Gesture) -> bool:
        """Polling shortcut for ``active(hand) is gesture``."""
        return self.active(hand) is gesture

    def raw(self, hand: str) -> Gesture:
        """This frame's unstabilized gesture (NONE while the hand is missing)."""
        if hand == BOTH:
            return self._both.stabilizer.winner
        track = self._track(hand)
        return track.raw if track else Gesture.NONE

    def confidence(self, hand: str) -> float:
        """Confidence of the active gesture, 0..1."""
        if hand == BOTH:
            return self._both.confidence
        track = self._track(hand)
        return track.lifecycle.confidence if track else 0.0

    def active_duration(self, hand: str, now: float) -> float:
        """Seconds the active gesture has been held (0 when none)."""
        if hand == BOTH:
            lifecycle = self._both
        else:
            track = self._track(hand)
            if track is None:
                return 0.0
            lifecycle = track.lifecycle
        if lifecycle.stabilizer.active is Gesture.NONE:
            return 0.0
        return now - lifecycle.started_at

    def features(self, hand: str) -> HandFeatures | None:
        """This frame's geometry, or None while the hand is missing."""
        track = self._track(hand)
        return track.features if track else None

    def position(self, hand: str) -> Point | None:
        """Anchor of the active gesture (see GestureEvent.position)."""
        if hand == BOTH:
            return self._both.position if self._two.state.present else None
        track = self._track(hand)
        return track.lifecycle.position if track else None

    def palm_center(self, hand: str) -> Point | None:
        """Last known palm center in pixels ("Both" = midpoint of the palms)."""
        if hand == BOTH:
            return self._two.state.midpoint if self._two.state.present else None
        track = self._track(hand)
        return track.palm if track else None

    def hand_scale(self, hand: str) -> float:
        """Last known hand scale in pixels ("Both" = mean of the two)."""
        if hand == BOTH:
            return self._two.state.scale
        track = self._track(hand)
        return track.scale if track else 0.0

    def velocity(self, hand: str) -> Point:
        """Smoothed palm velocity in hand scales per second (screen axes, y down)."""
        track = self._track(hand)
        return track.motion.velocity if track else (0.0, 0.0)

    def velocity_px(self, hand: str) -> Point:
        """Smoothed palm velocity in pixels per second, e.g. to throw something."""
        track = self._track(hand)
        return track.motion.velocity_px if track else (0.0, 0.0)

    def push_amount(self, hand: str) -> float:
        """Relative growth rate of hand scale per second; positive = toward the camera."""
        track = self._track(hand)
        return track.motion.push_amount if track else 0.0

    @property
    def two_hands(self) -> TwoHandState:
        """Continuous two-hand measurements for this frame."""
        return self._two.state

    @property
    def hands_distance(self) -> float:
        """Wrist-to-wrist distance in mean hand scales (0 unless both are visible)."""
        return self._two.state.distance

    def hold_progress(self, hand: str, gesture: Gesture, seconds: float | None = None) -> float:
        """0..1 progress toward an on_hold_for() (or ``seconds``), for progress rings."""
        return self.bus.hold_progress(hand, gesture, seconds)
