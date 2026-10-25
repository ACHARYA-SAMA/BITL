"""A small anatomical hand model that produces MediaPipe-style landmarks.

Used by the unit tests (synthetic hands for every gesture) and by the gesture
guide, which draws its pose icons from the same model, so the manual always
shows exactly the shapes the recognizer is tested against.

Hand-local model: wrist at the origin, image axes (y down), fingers toward -y,
wrist->middle MCP = 1 unit. A right hand with its palm toward the camera in the
mirrored view has its thumb on the -x side; a left hand is the x-mirror.
"""
from __future__ import annotations
import math
from .hand_tracker import HandData, Landmark

Point = tuple[float, float]
FRAME = (1280, 720)

_MCPS = {5: (-0.35, -0.95), 9: (0.0, -1.0), 13: (0.30, -0.93), 17: (0.55, -0.80)}
_BONES = {5: (0.42, 0.25, 0.22), 9: (0.47, 0.28, 0.23),
          13: (0.43, 0.27, 0.22), 17: (0.33, 0.20, 0.20)}
_SPLAY = {5: -8.0, 9: 0.0, 13: 7.0, 17: 15.0}
# MCP, PIP, DIP flexion toward the camera in degrees. A palm-facing camera sees
# each bone foreshortened to length * cos(cumulative flexion), as MediaPipe does.
EXT, CURL, PINCHED, RELAXED = (5, 5, 3), (70, 100, 60), (45, 60, 30), (40, 50, 30)
PARTIAL = (28, 43, 23)  # Lands in the PARTIAL band between curled and extended.
SNAP_BENT = (35, 50, 25)  # A middle finger pressed against the thumb, ready to snap.
_THUMBS = {
    "spread": ((-0.25, -0.20), (-0.45, -0.42), (-0.68, -0.60), (-0.90, -0.75)),
    "folded": ((-0.25, -0.20), (-0.42, -0.40), (-0.30, -0.62), (0.00, -0.72)),
    "up": ((-0.25, -0.20), (-0.50, -0.40), (-0.78, -0.50), (-1.05, -0.55)),
    "flicked": ((-0.25, -0.20), (-0.48, -0.45), (-0.62, -0.72), (-0.70, -1.00)),
}
# Recognizable poses: name -> (thumb shape, flexion of index, middle, ring, pinky).
# Each name matches a Gesture member (Gesture[name.upper()]).
POSES = {
    "open_palm": ("spread", (EXT, EXT, EXT, EXT)),
    "fist": ("folded", (CURL, CURL, CURL, CURL)),
    "point": ("folded", (EXT, CURL, CURL, CURL)),
    "peace": ("folded", (EXT, EXT, CURL, CURL)),
    "crossed": ("folded", (EXT, EXT, CURL, CURL)),
    "thumbs_up": ("up", (CURL, CURL, CURL, CURL)),
    "rock": ("folded", (EXT, CURL, CURL, EXT)),
    "three": ("folded", (EXT, EXT, EXT, CURL)),
    "pinch": ("pinch", (PINCHED, RELAXED, RELAXED, RELAXED)),
    "ok": ("pinch", (PINCHED, EXT, EXT, EXT)),
}
# The two halves of a finger snap (motion, not static gestures).
EXTRA_POSES = {
    "snap_ready": ("on_middle", (RELAXED, SNAP_BENT, CURL, CURL)),
    "snapped": ("flicked", (RELAXED, CURL, CURL, CURL)),
}


def _finger(mcp: int, flexion: tuple, splay: float) -> list[Point]:
    """Return PIP, DIP, TIP for one finger from its knuckle, flexion, and splay."""
    x, y = _MCPS[mcp]
    angle = math.radians(splay)
    ux, uy = math.sin(angle), -math.cos(angle)
    points, bend = [], 0.0
    for length, flex in zip(_BONES[mcp], flexion):
        bend += flex
        projected = length * math.cos(math.radians(bend))
        x, y = x + ux * projected, y + uy * projected
        points.append((x, y))
    return points


def local_pose(pose: str, flexions: tuple | None = None) -> list[Point]:
    """21 hand-local points for a named pose (before scaling and placement)."""
    thumb, default_flexions = POSES.get(pose) or EXTRA_POSES[pose]
    points = [(0.0, 0.0)] * 21
    for mcp, flexion in zip((5, 9, 13, 17), flexions or default_flexions):
        points[mcp] = _MCPS[mcp]
        # A pinching index turns toward the thumb to meet it.
        splay = -15.0 if mcp == 5 and flexion == PINCHED else _SPLAY[mcp]
        points[mcp + 1:mcp + 4] = _finger(mcp, flexion, splay)
    if pose == "crossed":
        # The index tip rests across the middle finger's last bone (Gojo's sign).
        pip, tip = points[Landmark.MIDDLE_PIP], points[Landmark.MIDDLE_TIP]
        cross = (pip[0] + 0.75 * (tip[0] - pip[0]) + 0.02, pip[1] + 0.75 * (tip[1] - pip[1]))
        index_pip = points[Landmark.INDEX_PIP]
        points[Landmark.INDEX_DIP] = ((index_pip[0] + cross[0]) / 2, (index_pip[1] + cross[1]) / 2)
        points[Landmark.INDEX_TIP] = cross
    if thumb == "pinch":
        tip_x, tip_y = points[Landmark.INDEX_TIP]
        points[1:5] = [(-0.25, -0.20), (-0.45, -0.48), (-0.50, -0.78), (tip_x + 0.02, tip_y + 0.03)]
    elif thumb == "on_middle":
        tip_x, tip_y = points[Landmark.MIDDLE_TIP]
        points[1:5] = [(-0.25, -0.20), (-0.42, -0.50), (-0.30, -0.95), (tip_x - 0.03, tip_y + 0.03)]
    else:
        points[1:5] = list(_THUMBS[thumb])
    return points


def make_hand(pose: str, scale: float = 100, roll: float = 0, offset: Point = (0, 0),
              hand: str = "Right", score: float = 0.95, flexions: tuple | None = None,
              local: list[Point] | None = None, frame: tuple[int, int] = FRAME) -> HandData:
    """Return 21 plausible pixel landmarks for a named pose.

    ``scale`` is the wrist-to-middle-MCP length in pixels, ``roll`` rotates the
    hand clockwise on screen around its palm center, and ``offset`` moves that
    center away from the frame center. Thumbs-up is a fist turned so the thumb
    points up, and roll is measured from that natural upright pose.
    """
    points = local or local_pose(pose, flexions)
    mirror = -1.0 if hand == "Left" else 1.0
    base = (90.0 if hand == "Right" else -90.0) if pose == "thumbs_up" else 0.0
    theta = math.radians(base + roll)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    palm = [points[i] for i in (0, 5, 9, 13, 17)]
    palm_x, palm_y = sum(p[0] for p in palm) / 5, sum(p[1] for p in palm) / 5
    center_x, center_y = frame[0] / 2 + offset[0], frame[1] / 2 + offset[1]
    pixels = []
    for x, y in points:
        x, y = (x - palm_x) * mirror, y - palm_y
        # Clockwise on screen, because image y points down.
        rx, ry = x * cos_t - y * sin_t, x * sin_t + y * cos_t
        pixels.append((round(center_x + rx * scale), round(center_y + ry * scale)))
    normalized = [(px / frame[0], py / frame[1], 0.0) for px, py in pixels]
    xs, ys = zip(*pixels)
    return HandData(hand, score, pixels, normalized, (min(xs), min(ys), max(xs), max(ys)))


def edge_on(pose: str = "open_palm") -> list[Point]:
    """A pose seen from its side (palms pressed together read like this)."""
    return [(x * 0.15, y) for x, y in local_pose(pose)]
