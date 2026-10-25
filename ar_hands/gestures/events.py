"""Gesture events and a publish/subscribe bus that a broken subscriber cannot crash.

Lifecycle per (hand, gesture): one STARTED, a HELD every frame while active,
then one ENDED. Instant gestures (swipes, HANDS_APART) send STARTED and ENDED
in the same frame, so every STARTED is always paired with an ENDED.
"""
from __future__ import annotations
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import Enum
import logging
from .types import Gesture

LOGGER = logging.getLogger(__name__)
BOTH = "Both"
# Hand ids: the first person's hands are "Left"/"Right" and their pair is "Both";
# person 2 is "Left 2"/"Right 2"/"Both 2", and so on (see identity.py).


def person_of(hand: str) -> int:
    """1 for "Left"/"Right"/"Both", 2 for "Left 2", ..."""
    _, _, number = hand.rpartition(" ")
    return int(number) if number.isdigit() else 1


def hand_id(handedness: str, person: int) -> str:
    """The id of ``person``'s "Left" or "Right" hand."""
    return handedness if person == 1 else f"{handedness} {person}"


def both_of(hand: str) -> str:
    """The two-hand id of the person this hand (or pair) belongs to."""
    return hand_id(BOTH, person_of(hand))


def pair_of(hand: str) -> tuple[str, str]:
    """The (left, right) hand ids of this hand's or pair's person."""
    person = person_of(hand)
    return hand_id("Left", person), hand_id("Right", person)


def is_both(hand: str) -> bool:
    """True for a two-hand id: "Both", "Both 2", ..."""
    return hand == BOTH or hand.startswith(BOTH + " ")


class EventKind(Enum):
    """Where an event sits in a gesture's lifecycle."""

    STARTED = "started"
    HELD = "held"
    ENDED = "ended"


@dataclass(frozen=True)
class GestureEvent:
    """One immutable lifecycle event; positions are mirrored-frame pixels."""

    kind: EventKind
    gesture: Gesture
    hand: str  # "Left", "Right", "Both", or another person's "Left 2", "Both 2"...
    position: tuple[float, float]  # Palm center, pinch midpoint, or index tip for POINT.
    duration: float  # Seconds since STARTED.
    confidence: float
    timestamp: float  # perf_counter seconds.


Callback = Callable[[GestureEvent], None]


@dataclass(eq=False)
class Subscription:
    """Handle returned by subscribe()/on_hold_for(); pass it to unsubscribe()."""

    gesture: Gesture | None  # None = every gesture.
    kind: EventKind | None  # None = every kind.
    callback: Callback
    hand: str | None = None  # None = every hand.
    hold_seconds: float | None = None
    errors: int = 0
    active: bool = True
    # Hold bookkeeping per hand: STARTED seen since subscribing, and already fired.
    armed: set[str] = field(default_factory=set, repr=False)
    fired: set[str] = field(default_factory=set, repr=False)

    def accepts(self, event: GestureEvent) -> bool:
        """Return True if the gesture and hand filters match (kind is checked separately)."""
        return ((self.gesture is None or self.gesture is event.gesture)
                and _hand_matches(self.hand, event.hand))


def _hand_matches(wanted: str | None, hand: str) -> bool:
    """A BOTH filter accepts every person's pair, so two-hand moves work for everyone."""
    return wanted is None or wanted == hand or (wanted == BOTH and is_both(hand))


class EventBus:
    """Route events to callbacks; isolate, count, and log callback failures."""

    def __init__(self) -> None:
        """Create a bus with no subscribers."""
        self._subscriptions: list[Subscription] = []
        self._durations: dict[tuple[str, Gesture], float] = {}

    def __len__(self) -> int:
        """Number of live subscriptions, which makes leaked callbacks easy to test for."""
        return len(self._subscriptions)

    def subscribe(self, gesture: Gesture | None, kind: EventKind | None,
                  callback: Callback, hand: str | None = None) -> Subscription:
        """Call ``callback(event)`` for matching events; None filters match everything."""
        subscription = Subscription(gesture, kind, callback, hand)
        self._subscriptions.append(subscription)
        return subscription

    def on_hold_for(self, gesture: Gesture, seconds: float, callback: Callback,
                    hand: str | None = None) -> Subscription:
        """Call ``callback`` once per activation after ``gesture`` is held ``seconds``.

        Only activations that START after subscribing count. Otherwise a mode
        switch while the user is already holding a pose would fire instantly.
        """
        if seconds <= 0:
            raise ValueError("Hold duration must be positive.")
        subscription = Subscription(gesture, EventKind.HELD, callback, hand, seconds)
        self._subscriptions.append(subscription)
        return subscription

    def unsubscribe(self, target: Subscription | Callback) -> int:
        """Remove a subscription, or every subscription of a callback; return the count."""
        removed = [s for s in self._subscriptions if s is target or s.callback == target]
        for subscription in removed:
            subscription.active = False  # Also skips it if a dispatch is in progress.
        self._subscriptions = [s for s in self._subscriptions if s.active]
        return len(removed)

    def clear(self) -> None:
        """Remove every subscription."""
        for subscription in self._subscriptions:
            subscription.active = False
        self._subscriptions.clear()

    def dispatch(self, event: GestureEvent) -> None:
        """Deliver one event. A raising callback is logged and never propagates."""
        self._track(event)
        # Iterate over a snapshot so callbacks may subscribe or unsubscribe.
        for subscription in tuple(self._subscriptions):
            if subscription.active and subscription.accepts(event) and self._due(subscription, event):
                self._call(subscription, event)

    def dispatch_all(self, events: Iterable[GestureEvent]) -> None:
        """Deliver events in order."""
        for event in events:
            self.dispatch(event)

    def hold_progress(self, hand: str, gesture: Gesture, seconds: float | None = None) -> float:
        """Return 0..1 progress of a hold, for UI rings.

        With ``seconds`` the progress is measured against that duration. Without
        it, the shortest registered on_hold_for that has not fired yet is used:
        1.0 once all have fired, 0.0 if none is registered or armed.
        """
        duration = self._durations.get((hand, gesture))
        if duration is None:
            return 0.0
        if seconds is None:
            holds = [s for s in self._subscriptions
                     if s.hold_seconds is not None and s.gesture is gesture
                     and _hand_matches(s.hand, hand) and hand in s.armed]
            pending = [s.hold_seconds for s in holds if hand not in s.fired]
            if not pending:
                return 1.0 if holds else 0.0
            seconds = min(pending)
        return min(1.0, duration / seconds)

    def _track(self, event: GestureEvent) -> None:
        key = (event.hand, event.gesture)
        if event.kind is EventKind.ENDED:
            self._durations.pop(key, None)
        else:
            self._durations[key] = event.duration

    @staticmethod
    def _due(subscription: Subscription, event: GestureEvent) -> bool:
        if subscription.hold_seconds is None:
            return subscription.kind is None or subscription.kind is event.kind
        hand = event.hand
        if event.kind is EventKind.STARTED:
            subscription.armed.add(hand)
            subscription.fired.discard(hand)
            return False
        if event.kind is EventKind.ENDED:
            subscription.armed.discard(hand)
            subscription.fired.discard(hand)
            return False
        if (hand in subscription.armed and hand not in subscription.fired
                and event.duration >= subscription.hold_seconds):
            subscription.fired.add(hand)
            return True
        return False

    @staticmethod
    def _call(subscription: Subscription, event: GestureEvent) -> None:
        try:
            subscription.callback(event)
        except Exception:  # A broken subscriber must never stop the frame loop.
            subscription.errors += 1
            # HELD fires every frame, so log the first traceback, then every 100th.
            if subscription.errors == 1 or subscription.errors % 100 == 0:
                LOGGER.exception("Gesture callback %s failed (%d errors so far) on %s %s %s",
                                 getattr(subscription.callback, "__qualname__", subscription.callback),
                                 subscription.errors, event.hand, event.gesture.name, event.kind.name)
