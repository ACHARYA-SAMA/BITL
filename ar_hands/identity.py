"""Stable ids for many hands: which hand is which, and whose hands belong together.

MediaPipe returns up to ``num_hands`` hands per frame in no particular order,
each with a Left/Right guess. Gestures, pens and held effects keep state per
hand, so with a group of friends every hand needs an id that survives from
frame to frame, and two-hand moves (prayer, both fists, beams) need to know
which left and right hand belong to the same person.

Each frame the detections are matched to the previous frame's hands by palm
distance (in hand sizes). A matched hand keeps its id and its handedness, so a
one-frame Left/Right flip from the classifier changes nothing. A new hand joins
the nearest person who is missing that hand, otherwise it starts a new person.
Person 1's hands are "Left"/"Right" (and their pair "Both"), so one person
playing alone sees exactly the ids the rest of the app always used; person 2
gets "Left 2"/"Right 2"/"Both 2", and so on.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
from math import hypot
from .gestures.events import hand_id
from .hand_tracker import HandData, Landmark, distance

Point = tuple[float, float]
PALM = (Landmark.WRIST, Landmark.INDEX_MCP, Landmark.MIDDLE_MCP, Landmark.PINKY_MCP)
OTHER = {"Left": "Right", "Right": "Left"}


@dataclass
class _Track:
    id: str
    handedness: str
    person: int
    palm: Point
    scale: float
    seen: float


def _palm(hand: HandData) -> Point:
    points = [hand.landmarks_px[i] for i in PALM]
    return sum(p[0] for p in points) / len(points), sum(p[1] for p in points) / len(points)


def _scale(hand: HandData) -> float:
    return max(1.0, distance(hand.landmarks_px[Landmark.WRIST], hand.landmarks_px[Landmark.MIDDLE_MCP]))


class HandIdentities:
    """Assigns ``track_id`` (and a stable ``handedness``) to every detected hand."""

    def __init__(self, keep_s: float = 0.6, match_ratio: float = 2.5, pair_ratio: float = 6.0,
                 duplicate_ratio: float = 0.25) -> None:
        """``keep_s``: how long a vanished hand keeps its id. Ratios are in hand sizes:
        how far a hand may move between frames, how far apart one person's hands
        may be, and how close two detections must be to count as one hand."""
        self.keep_s, self.match_ratio, self.pair_ratio = keep_s, match_ratio, pair_ratio
        self.duplicate_ratio = duplicate_ratio
        self._tracks: dict[str, _Track] = {}

    def reset(self) -> None:
        """Forget every hand (the camera changed)."""
        self._tracks.clear()

    @property
    def people(self) -> int:
        """People with at least one hand remembered."""
        return len({track.person for track in self._tracks.values()})

    def assign(self, hands: list[HandData], now: float) -> list[HandData]:
        """Return the hands with ids, in the order given (duplicates removed)."""
        hands = self._without_duplicates(hands)
        self._tracks = {key: track for key, track in self._tracks.items() if now - track.seen <= self.keep_s}
        palms = [_palm(hand) for hand in hands]
        scales = [_scale(hand) for hand in hands]
        # Greedy matching, closest pairs first; a classifier label change costs a little.
        candidates = []
        for index, hand in enumerate(hands):
            for track in self._tracks.values():
                size = (scales[index] + track.scale) / 2
                gap = hypot(palms[index][0] - track.palm[0], palms[index][1] - track.palm[1]) / size
                if gap <= self.match_ratio:
                    candidates.append((gap + (0.5 if hand.handedness != track.handedness else 0.0),
                                       index, track.id))
        assigned: dict[int, _Track] = {}
        taken: set[str] = set()
        for _, index, key in sorted(candidates):
            if index not in assigned and key not in taken:
                assigned[index], taken = self._tracks[key], taken | {key}
        for index in sorted(set(range(len(hands))) - set(assigned), key=lambda i: palms[i][0]):
            assigned[index] = self._new_track(hands[index], palms[index], scales[index], now, taken)
            taken.add(assigned[index].id)
        result = []
        for index, hand in enumerate(hands):
            track = assigned[index]
            track.palm, track.scale, track.seen = palms[index], scales[index], now
            result.append(replace(hand, handedness=track.handedness, track_id=track.id))
        return result

    def _without_duplicates(self, hands: list[HandData]) -> list[HandData]:
        """One hand detected twice (nearly identical landmarks): keep the more confident."""
        kept: list[HandData] = []
        for hand in sorted(hands, key=lambda h: -h.score):
            scale = _scale(hand)
            if all(sum(distance(a, b) for a, b in zip(hand.landmarks_px, other.landmarks_px))
                   / len(Landmark) >= self.duplicate_ratio * scale for other in kept):
                kept.append(hand)
        return [hand for hand in hands if any(hand is k for k in kept)]

    def _new_track(self, hand: HandData, palm: Point, scale: float, now: float, taken: set[str]) -> _Track:
        """Join the nearest person missing this hand, or start a new person."""
        people: dict[int, dict[str, _Track]] = {}
        for track in self._tracks.values():
            people.setdefault(track.person, {})[track.handedness] = track
        label, best = hand.handedness, None
        for person, owned in people.items():
            for wanted in (label, OTHER.get(label, label)):
                if wanted in owned or not owned:
                    continue
                partner = next(iter(owned.values()))
                gap = hypot(palm[0] - partner.palm[0], palm[1] - partner.palm[1]) / max(scale, partner.scale)
                # In the mirrored view a person's left hand is normally left of their right hand.
                sideways = (palm[0] < partner.palm[0]) == (wanted == "Left")
                cost = gap + (0.0 if wanted == label else 1.5) + (0.0 if sideways else 1.0)
                if gap <= self.pair_ratio and (best is None or cost < best[0]):
                    best = (cost, person, wanted)
        if best is not None:
            _, person, label = best
        else:
            person = next(n for n in range(1, len(people) + 2) if n not in people)
        track = _Track(hand_id(label, person), label, person, palm, scale, now)
        if track.id in taken or track.id in self._tracks:  # Both of a person's slots are in use.
            person = next(n for n in range(1, len(people) + 2) if n not in people)
            track = _Track(hand_id(label, person), label, person, palm, scale, now)
        self._tracks[track.id] = track
        return track
