"""In-app gesture guide (press ? / H-key menu item) and the first-run welcome page.

Every card shows the hand pose as a drawn skeleton icon, generated from the
same anatomical model the unit tests feed the recognizer (handpose.py), so
the picture is exactly the shape the app expects.
"""
from __future__ import annotations
from dataclasses import dataclass
import cv2
import numpy as np
from .config import Config
from .hand_tracker import HandData
from .handpose import edge_on, make_hand

Color = tuple[int, int, int]
BONES = ((0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
         (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20))
TIPS = (4, 8, 12, 16, 20)


@dataclass(frozen=True)
class Card:
    """One gesture: icon poses (one or two hands), a title, how to do it, what it does."""

    title: str
    how: str
    effect: str
    poses: tuple[str, ...]  # Pose names; two names draw a left and a right hand.
    badge: str = ""  # Short tag on the icon: "HOLD 1s", "FLICK", ...
    accent: Color = (255, 220, 120)


WELCOME = (
    Card("Point", "Index finger up, others curled", "Draws in Draw mode", ("point",), accent=(255, 255, 0)),
    Card("Pinch", "Thumb tip to index tip", "Grab blocks, click menu buttons", ("pinch",), accent=(255, 200, 40)),
    Card("Rock sign", "Index + pinky up, hold 1 s", "Switch to the next mode", ("rock",), "HOLD 1s", (200, 120, 255)),
    Card("Both palms", "Open both hands, hold 1 s", "Open the menu", ("open_palm", "open_palm"), "HOLD 1s",
         (120, 255, 160)),
)

PAGES: dict[str, tuple[Card, ...]] = {
    "Welcome": WELCOME,
    "Draw": (
        Card("Draw", "Point with your index finger", "Neon line follows your fingertip", ("point",),
             accent=(255, 255, 0)),
        Card("Pen up", "Make a fist (or curl the index)", "Stops drawing", ("fist",)),
        Card("Next color", "Peace sign", "Cycles the brush color", ("peace",), accent=(255, 0, 255)),
        Card("Save PNG", "Thumbs up", "Saves your drawing", ("thumbs_up",), accent=(120, 255, 160)),
        Card("Clear", "Open palm, hold still 1 s", "Clears the canvas (not while drawing)", ("open_palm",),
             "HOLD 1s", (120, 160, 255)),
        Card("Snap: turn to dust", "Press thumb on middle finger, snap", "You crumble into dust; snap again to return",
             ("snap_ready",), "SNAP", (90, 160, 255)),
    ),
    "Jutsu": (
        Card("Fireball", "Open palm, hold 0.5 s; flick to throw", "Fireball in your palm, explodes on impact",
             ("open_palm",), "HOLD + FLICK", (40, 150, 255)),
        Card("Chidori", "Fist for 0.6 s, then open the hand", "Lightning crackles around your hand",
             ("fist",), "FIST > OPEN", (255, 230, 120)),
        Card("Rasenshuriken", "OK sign, hold 0.6 s; flick to throw", "Spinning wind shuriken; wind-sphere blast",
             ("ok",), "HOLD + FLICK", (255, 220, 90)),
        Card("Susanoo", "Both fists, hold 1 s (again to dismiss)", "A violet spirit giant rises behind you",
             ("fist", "fist"), "HOLD 1s", (255, 90, 200)),
        Card("Great Fireball", "Palms together 0.8 s, then pull apart", "One huge fireball between your hands",
             ("prayer",), "SEAL > APART", (40, 120, 255)),
        Card("Cancel", "Make a fist while holding a jutsu", "It fizzles out", ("fist",)),
    ),
    "Blocks": (
        Card("New block", "Pinch empty space; open to drop", "Drops into the grid and stacks", ("pinch",),
             accent=(80, 200, 120)),
        Card("Move", "Pinch a block or a beam end", "Drag it anywhere", ("pinch",), accent=(80, 200, 120)),
        Card("Neon beam", "Pinch both hands, touch, pull apart", "Stretch a beam; blocks land on it",
             ("pinch", "pinch"), "TOUCH > PULL", (255, 200, 40)),
        Card("Break", "Point at a block 0.35 s", "It cracks and shatters (TNT explodes)", ("point",),
             "HOLD", (180, 180, 255)),
        Card("Punch", "Swing a fist through a block", "Smashes it instantly", ("fist",), "SWING", (60, 120, 255)),
        Card("Type / Clear", "Peace = next type, thumbs-up 1 s = clear", "Grass, stone, planks, TNT, diamond",
             ("peace",), accent=(120, 255, 200)),
    ),
    "Jujutsu": (
        Card("Blue", "Point and hold 0.4 s", "A singularity at your fingertip pulls things in", ("point",),
             "HOLD", (255, 160, 40)),
        Card("Red", "Peace sign, hold 0.4 s", "Repulsion orb pushes things away", ("peace",), "HOLD", (60, 60, 255)),
        Card("Hollow Purple", "Blue in one hand, Red in the other: bring together", "Fuse them; flick to fire",
             ("point", "peace"), "FUSE + FLICK", (255, 90, 230)),
        Card("Unlimited Void", "Cross index over middle, hold 1 s", "Gojo's domain: infinite space", ("crossed",),
             "HOLD 1s", (255, 220, 170)),
        Card("Malevolent Shrine", "Palms together, hold 1 s", "Sukuna's domain: the shrine and its slashes",
             ("prayer",), "HOLD 1s", (60, 60, 255)),
        Card("Close a domain", "Cast the same domain again", "Or wait 15 s", ("crossed",)),
    ),
}


def wrap(text: str, width_px: int, scale: float, thickness: int = 1) -> list[str]:
    """Greedy word wrap with OpenCV's text metrics."""
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if cv2.getTextSize(trial, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0] <= width_px or not line:
            line = trial
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines


def draw_hand_icon(canvas: np.ndarray, hand: HandData, color: Color) -> None:
    """Skeleton lines and joints, fingertips highlighted."""
    points = hand.landmarks_px
    for a, b in BONES:
        cv2.line(canvas, points[a], points[b], color, 2, cv2.LINE_AA)
    for index, point in enumerate(points):
        cv2.circle(canvas, point, 4 if index in TIPS else 2, (255, 255, 255) if index in TIPS else color,
                   -1, cv2.LINE_AA)


def pose_icon(poses: tuple[str, ...], size: int, color: Color) -> np.ndarray:
    """A size x size BGR icon of one or two hands (prayer = two palms pressed together)."""
    icon = np.zeros((size, size, 3), np.uint8)
    frame = (size, size)
    if poses == ("prayer",):
        for hand in ("Left", "Right"):
            draw_hand_icon(icon, make_hand("open_palm", scale=size * 0.3, offset=(0, size * 0.12), hand=hand,
                                           local=edge_on(), frame=frame), color)
        return icon
    if len(poses) == 2:
        for pose, hand, dx in ((poses[0], "Left", -size * 0.22), (poses[1], "Right", size * 0.22)):
            draw_hand_icon(icon, make_hand(pose, scale=size * 0.2, offset=(dx, size * 0.08), hand=hand,
                                           frame=frame), color)
        return icon
    draw_hand_icon(icon, make_hand(poses[0], scale=size * 0.36, offset=(0, size * 0.06), frame=frame), color)
    return icon


class GestureGuide:
    """Full-screen pages of gesture cards; navigate with arrows/A-D, close with ?/Esc/pinch."""

    def __init__(self, config: Config) -> None:
        """Closed until opened."""
        self.config = config
        self.visible = False
        self.page = 1
        self.onboarding = False
        self._icons: dict[tuple, np.ndarray] = {}

    @property
    def names(self) -> list[str]:
        """Page titles in order."""
        return list(PAGES)

    def open(self, page: str | None = None, onboarding: bool = False) -> None:
        """Show the guide, optionally at a named page (the current mode's)."""
        self.visible, self.onboarding = True, onboarding
        if page in PAGES:
            self.page = self.names.index(page)

    def close(self) -> None:
        """Hide the guide."""
        self.visible = self.onboarding = False

    def turn(self, step: int) -> None:
        """Next (+1) or previous (-1) page, wrapping around."""
        self.page = (self.page + step) % len(PAGES)

    def _icon(self, card: Card, size: int) -> np.ndarray:
        key = (card.poses, size, card.accent)
        if key not in self._icons:
            self._icons[key] = pose_icon(card.poses, size, card.accent)
        return self._icons[key]

    def render(self, frame: np.ndarray) -> None:
        """Dim the scene and draw the current page's cards in a grid."""
        if not self.visible:
            return
        # Laid out at 1280x720, or 720x1280 for portrait (phone) frames, then resized.
        design = (720, 1280) if frame.shape[0] > frame.shape[1] else (1280, 720)
        if frame.shape[1::-1] != design:
            canvas = cv2.resize(frame, design)
            self.render(canvas)
            frame[:] = cv2.resize(canvas, frame.shape[1::-1])
            return
        height, width = frame.shape[:2]
        portrait = height > width
        cv2.addWeighted(frame, 0.25, np.zeros_like(frame), 0.75, 0, dst=frame)
        name = self.names[self.page]
        cards = PAGES[name]
        font, bold = cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX
        title = "AR HANDS" if self.onboarding else f"GESTURE GUIDE  -  {name}"
        cv2.putText(frame, title, (40, 62), bold, 1.0 if portrait else 1.25, (255, 255, 255), 2, cv2.LINE_AA)
        tabs_x, tabs_y = width - 40, 104 if portrait else 56  # Portrait: tabs on their own row.
        for index in reversed(range(len(PAGES))):
            label = self.names[index]
            (tw, _), _ = cv2.getTextSize(label, font, 0.55, 1)
            tabs_x -= tw + 22
            active = index == self.page
            if active:
                cv2.rectangle(frame, (tabs_x - 8, tabs_y - 22), (tabs_x + tw + 8, tabs_y + 8), (80, 70, 60), -1)
            cv2.putText(frame, label, (tabs_x, tabs_y), font, 0.55, (255, 255, 255) if active else (150, 150, 150),
                        1, cv2.LINE_AA)
        columns = 1 if portrait else 2
        rows = (len(cards) + columns - 1) // columns
        top, bottom, margin, gap = 136 if portrait else 92, height - 64, 40, 18
        card_w = (width - 2 * margin - gap * (columns - 1)) // columns
        card_h = min(250, (bottom - top - gap * (rows - 1)) // rows)
        icon = min(card_h - 24, int(card_w * 0.38))
        for index, card in enumerate(cards):
            row, column = divmod(index, columns)
            x = margin + column * (card_w + gap)
            y = top + row * (card_h + gap)
            panel = frame[y:y + card_h, x:x + card_w]
            cv2.addWeighted(panel, 0.4, np.full_like(panel, (40, 32, 28)), 0.6, 0, dst=panel)
            cv2.rectangle(frame, (x, y), (x + card_w, y + card_h), card.accent, 1, cv2.LINE_AA)
            ix, iy = x + 12, y + (card_h - icon) // 2
            frame[iy:iy + icon, ix:ix + icon] = np.maximum(frame[iy:iy + icon, ix:ix + icon], self._icon(card, icon))
            if card.badge:
                (bw, bh), _ = cv2.getTextSize(card.badge, font, 0.45, 1)
                bx, by = ix + (icon - bw) // 2, iy + icon - 6
                cv2.rectangle(frame, (bx - 6, by - bh - 6), (bx + bw + 6, by + 5), card.accent, -1)
                cv2.putText(frame, card.badge, (bx, by), font, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
            tx, text_w = ix + icon + 16, card_w - icon - 44
            cv2.putText(frame, card.title, (tx, y + 38), bold, 0.8, card.accent, 1, cv2.LINE_AA)
            line_y = y + 70
            for line in wrap(card.how, text_w, 0.55):
                cv2.putText(frame, line, (tx, line_y), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
                line_y += 24
            line_y += 6
            for line in wrap(card.effect, text_w, 0.5):
                cv2.putText(frame, line, (tx, line_y), font, 0.5, (190, 190, 190), 1, cv2.LINE_AA)
                line_y += 22
        footer = ("Press any key or pinch to start  -  ? opens this guide any time" if self.onboarding else
                  "Left/Right or A/D: page   -   ? or Esc: close   -   pinch: close")
        scale = min(0.6, 0.6 * (width - 2 * margin) / cv2.getTextSize(footer, font, 0.6, 1)[0][0])
        (fw, _), _ = cv2.getTextSize(footer, font, scale, 1)
        cv2.putText(frame, footer, ((width - fw) // 2, height - 26), font, scale, (220, 220, 220), 1, cv2.LINE_AA)
