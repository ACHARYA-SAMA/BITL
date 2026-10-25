"""Hand geometry that is invariant to scale, in-plane rotation, and mirroring.

Inputs are pixel landmarks of the MIRRORED frame: the app flips the camera
image before detection, so screen-right is the user's right, like a mirror.
Recognition uses only distance ratios and internal joint angles. Those do not
change when the image is mirrored or the hand rolls in the image plane, so one
set of thresholds serves both hands at any rotation. Only roll_angle,
thumb_direction, and palm_normal_z depend on orientation, by definition, and
each says so where it is computed.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, IntEnum
from math import acos, atan2, degrees, hypot
from ..config import GestureConfig
from ..hand_tracker import HandData, Landmark, distance

Point = tuple[float, float]


class FingerState(Enum):
    """Discrete finger state; PARTIAL is the deliberate dead zone between."""

    EXTENDED = "extended"
    PARTIAL = "partial"
    CURLED = "curled"


class Finger(IntEnum):
    """Finger order used by HandFeatures.fingers and the recognizer rule table."""

    THUMB = 0
    INDEX = 1
    MIDDLE = 2
    RING = 3
    PINKY = 4


LONG_FINGERS = (Finger.INDEX, Finger.MIDDLE, Finger.RING, Finger.PINKY)
_FINGER_JOINTS = {  # (MCP, PIP, TIP)
    Finger.INDEX: (Landmark.INDEX_MCP, Landmark.INDEX_PIP, Landmark.INDEX_TIP),
    Finger.MIDDLE: (Landmark.MIDDLE_MCP, Landmark.MIDDLE_PIP, Landmark.MIDDLE_TIP),
    Finger.RING: (Landmark.RING_MCP, Landmark.RING_PIP, Landmark.RING_TIP),
    Finger.PINKY: (Landmark.PINKY_MCP, Landmark.PINKY_PIP, Landmark.PINKY_TIP),
}
TIPS = (Landmark.THUMB_TIP, Landmark.INDEX_TIP, Landmark.MIDDLE_TIP,
        Landmark.RING_TIP, Landmark.PINKY_TIP)
PALM_POINTS = (Landmark.WRIST, Landmark.INDEX_MCP, Landmark.MIDDLE_MCP,
               Landmark.RING_MCP, Landmark.PINKY_MCP)


@dataclass(frozen=True)
class FingerReading:
    """One finger's state plus a continuous 0 (curled) to 1 (extended) openness."""

    state: FingerState
    openness: float


@dataclass(frozen=True)
class HandFeatures:
    """Everything the recognizer, motion, and two-hand logic need from one hand."""

    scale: float
    fingers: tuple[FingerReading, ...]  # Indexed by Finger.
    pinch_ratio: float
    pinch_strength: float
    pinch_point: Point
    pinch_reach: float
    palm_center: Point
    roll: float
    thumb_direction: float
    palm_normal_z: float
    tips: tuple[Point, ...]  # Thumb to pinky.
    wrist: Point
    middle_pinch_ratio: float = 9.0  # Thumb tip to middle tip / scale (finger snaps).
    middle_reach: float = 0.0  # Middle tip to wrist / scale.
    cross_gap: float = 9.0  # Index tip to the middle finger's PIP-TIP segment / scale.

    def state(self, finger: Finger) -> FingerState:
        """Return the discrete state of one finger."""
        return self.fingers[finger].state

    def openness(self, finger: Finger) -> float:
        """Return the continuous openness of one finger."""
        return self.fingers[finger].openness


def _ramp(value: float, low: float, high: float) -> float:
    """Map low..high onto 0..1, clamped; it works for decreasing ramps too."""
    return min(1.0, max(0.0, (value - low) / (high - low)))


def _heading(start: Point, end: Point) -> float:
    """Degrees from screen-up to start->end, positive clockwise on screen."""
    # Image y grows downward, so "up" is -y; atan2(dx, -dy) is 0 for up, 90 for right.
    return degrees(atan2(end[0] - start[0], start[1] - end[1]))


def hand_scale(hand: HandData) -> float:
    """Wrist to middle-finger MCP in pixels: the unit for every gesture threshold.

    Palm length barely changes as fingers move, unlike the hand's bounding box,
    so ratios of it are stable while a pose changes and at any camera distance.
    """
    return distance(hand.landmarks_px[Landmark.WRIST], hand.landmarks_px[Landmark.MIDDLE_MCP])


def joint_angle(a: Point, b: Point, c: Point) -> float | None:
    """Return the angle at b in degrees (180 = straight), or None if a bone has no length."""
    v1x, v1y = a[0] - b[0], a[1] - b[1]
    v2x, v2y = c[0] - b[0], c[1] - b[1]
    n1, n2 = hypot(v1x, v1y), hypot(v2x, v2y)
    if n1 <= 0 or n2 <= 0:
        return None
    cosine = (v1x * v2x + v1y * v2y) / (n1 * n2)
    return degrees(acos(min(1.0, max(-1.0, cosine))))


def _angle_trust(a: Point, b: Point, c: Point, scale: float, config: GestureConfig) -> float:
    """0..1 trust in the angle at b, fading out as either bone foreshortens.

    A bone pointing at the camera projects to a few pixels, and its direction
    is then mostly landmark noise. Fading its weight smoothly (instead of
    dropping the angle below a cutoff) keeps openness continuous, so a slowly
    bending finger cannot jump straight across the PARTIAL band.
    """
    shortest = min(distance(a, b), distance(c, b)) / scale
    return _ramp(shortest, config.min_bone_ratio, 2 * config.min_bone_ratio)


def classify(openness: float, config: GestureConfig) -> FingerState:
    """Threshold openness, leaving a PARTIAL band so borderline fingers match nothing."""
    if openness >= config.extended_threshold:
        return FingerState.EXTENDED
    if openness <= config.curled_threshold:
        return FingerState.CURLED
    return FingerState.PARTIAL


def finger_extension(hand: HandData, finger: Finger, scale: float,
                     config: GestureConfig) -> FingerReading:
    """Score one long finger from its PIP angle and its tip-vs-PIP reach.

    The two cues fail differently, so blending them is robust: the angle is
    unreliable when a bone points at the camera and projects to a few pixels,
    while reach is fooled by a straight finger folded at the knuckle. Both use
    only internal geometry, so in-plane rotation and mirroring cannot change them.
    """
    points = hand.landmarks_px
    mcp, pip, tip = (points[index] for index in _FINGER_JOINTS[finger])
    wrist = points[Landmark.WRIST]
    reach = (distance(tip, wrist) - distance(pip, wrist)) / scale
    reach_score = _ramp(reach, config.finger_reach_curled, config.finger_reach_extended)
    angle = joint_angle(mcp, pip, tip)
    if angle is None:
        return FingerReading(classify(reach_score, config), reach_score)
    angle_score = _ramp(angle, config.finger_angle_curled_deg, config.finger_angle_extended_deg)
    weight = config.finger_angle_weight * _angle_trust(mcp, pip, tip, scale, config)
    openness = weight * angle_score + (1 - weight) * reach_score
    return FingerReading(classify(openness, config), openness)


def thumb_extension(hand: HandData, scale: float, config: GestureConfig) -> FingerReading:
    """Score the thumb, which folds ACROSS the palm instead of curling toward the wrist.

    Its tip-to-wrist distance barely changes when it folds, so instead measure
    how far the tip is from the index and pinky knuckles. A folded thumb lies
    near them; an extended one points away. Distances to both knuckles are the
    same for left and right hands, mirrored or not, so no handedness is needed.
    """
    points = hand.landmarks_px
    mcp, ip, tip = points[Landmark.THUMB_MCP], points[Landmark.THUMB_IP], points[Landmark.THUMB_TIP]
    index_score = _ramp(distance(tip, points[Landmark.INDEX_MCP]) / scale,
                        config.thumb_index_gap_curled, config.thumb_index_gap_extended)
    pinky_score = _ramp(distance(tip, points[Landmark.PINKY_MCP]) / scale,
                        config.thumb_pinky_gap_curled, config.thumb_pinky_gap_extended)
    w_angle, w_index, w_pinky = config.thumb_weights
    angle = joint_angle(mcp, ip, tip)
    if angle is None:
        w_angle, angle_score = 0.0, 0.0
    else:
        w_angle *= _angle_trust(mcp, ip, tip, scale, config)
        angle_score = _ramp(angle, config.thumb_angle_curled_deg, config.thumb_angle_extended_deg)
    # Weight the angle by how much it can be trusted; the gaps share the rest.
    gap_share = (1.0 - w_angle) / (w_index + w_pinky)
    openness = (w_angle * angle_score
                + gap_share * (w_index * index_score + w_pinky * pinky_score))
    return FingerReading(classify(openness, config), openness)


def pinch_ratio(hand: HandData, scale: float | None = None) -> float:
    """Thumb tip to index tip distance divided by hand scale."""
    scale = hand_scale(hand) if scale is None else scale
    points = hand.landmarks_px
    return distance(points[Landmark.THUMB_TIP], points[Landmark.INDEX_TIP]) / max(scale, 1e-6)


def pinch_strength(ratio: float, config: GestureConfig) -> float:
    """Map pinch_ratio to 0 (open) .. 1 (fingertips touching)."""
    return 1.0 - _ramp(ratio, config.pinch_full, config.pinch_open)


def palm_center(hand: HandData) -> Point:
    """Mean of the wrist and four knuckles: stable while fingers move."""
    points = [hand.landmarks_px[index] for index in PALM_POINTS]
    return (sum(p[0] for p in points) / len(points), sum(p[1] for p in points) / len(points))


def roll_angle(hand: HandData) -> float:
    """In-plane rotation of wrist->middle MCP: 0 = fingers up, positive clockwise.

    Measured on the mirrored display, so turning your hand clockwise as you see
    it on screen gives a positive roll. Range is -180..180 degrees.
    """
    return _heading(hand.landmarks_px[Landmark.WRIST], hand.landmarks_px[Landmark.MIDDLE_MCP])


def thumb_direction(hand: HandData) -> float:
    """Screen heading of the thumb (MCP to tip): 0 = straight up. Orientation-dependent."""
    return _heading(hand.landmarks_px[Landmark.THUMB_MCP], hand.landmarks_px[Landmark.THUMB_TIP])


def palm_normal_z(hand: HandData) -> float:
    """Signed palm-normal component toward the camera: +1 palm faces you, -1 back of hand.

    It is the sine of the angle between wrist->index MCP and wrist->pinky MCP.
    That 2D cross product flips sign between left and right hands, and it
    would flip again if the frame were not mirrored. In the mirrored view a
    right palm facing the camera has its thumb on the screen-left, which gives a
    positive cross product. Near 0 means the hand is edge-on, as in a prayer
    seal or a sideways thumbs-up.
    """
    points = hand.landmarks_px
    wrist, index, pinky = (points[Landmark.WRIST], points[Landmark.INDEX_MCP],
                           points[Landmark.PINKY_MCP])
    ax, ay = index[0] - wrist[0], index[1] - wrist[1]
    bx, by = pinky[0] - wrist[0], pinky[1] - wrist[1]
    norms = hypot(ax, ay) * hypot(bx, by)
    if norms <= 0:
        return 0.0
    sine = (ax * by - ay * bx) / norms
    return sine if hand.handedness == "Right" else -sine


def segment_distance(point: Point, a: Point, b: Point) -> float:
    """Distance from a point to the segment a-b."""
    abx, aby = b[0] - a[0], b[1] - a[1]
    length2 = abx * abx + aby * aby
    t = 0.0 if length2 <= 0 else min(1.0, max(0.0, ((point[0] - a[0]) * abx + (point[1] - a[1]) * aby) / length2))
    return hypot(point[0] - (a[0] + t * abx), point[1] - (a[1] + t * aby))


def extract_features(hand: HandData, config: GestureConfig) -> HandFeatures | None:
    """Compute all features once per hand per frame; None if the hand is too small."""
    scale = hand_scale(hand)
    if scale < config.min_hand_scale_px:
        return None
    points = hand.landmarks_px
    fingers = (thumb_extension(hand, scale, config),) + tuple(
        finger_extension(hand, finger, scale, config) for finger in LONG_FINGERS)
    thumb_tip, index_tip, wrist = (points[Landmark.THUMB_TIP], points[Landmark.INDEX_TIP],
                                   points[Landmark.WRIST])
    ratio = pinch_ratio(hand, scale)
    pinch_point = ((thumb_tip[0] + index_tip[0]) / 2, (thumb_tip[1] + index_tip[1]) / 2)
    return HandFeatures(
        scale=scale,
        fingers=fingers,
        pinch_ratio=ratio,
        pinch_strength=pinch_strength(ratio, config),
        pinch_point=pinch_point,
        pinch_reach=distance(pinch_point, wrist) / scale,
        palm_center=palm_center(hand),
        roll=roll_angle(hand),
        thumb_direction=thumb_direction(hand),
        palm_normal_z=palm_normal_z(hand),
        tips=tuple((float(points[i][0]), float(points[i][1])) for i in TIPS),
        wrist=(float(wrist[0]), float(wrist[1])),
        middle_pinch_ratio=distance(thumb_tip, points[Landmark.MIDDLE_TIP]) / scale,
        middle_reach=distance(points[Landmark.MIDDLE_TIP], wrist) / scale,
        cross_gap=segment_distance(index_tip, points[Landmark.MIDDLE_PIP],
                                   points[Landmark.MIDDLE_TIP]) / scale,
    )
