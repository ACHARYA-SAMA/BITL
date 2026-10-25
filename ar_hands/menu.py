"""Keyboard, mouse and fingertip controls share one settings menu.

Hand input: the index fingertip is the cursor (the app smooths it with a One
Euro filter). Hovering an item for ``dwell_seconds`` fills a ring and clicks
it, once per visit, so a resting hand never toggles a switch back and forth.
A pinch (or mouse click) clicks instantly; on a slider it grabs the knob and
the value follows the hand until the pinch opens. A pinch drags the index tip
toward the thumb, so a pinch click uses the item that was under the cursor
``pinch_lookback_s`` earlier. The menu owns no app state: ``value(key)``
reads it and ``change(key, value)`` applies a choice, so keyboard shortcuts
and menu labels can never disagree. While it is open the app suspends every
mode controller (input focus).
"""
from __future__ import annotations
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
import cv2
import numpy as np
from .config import Color, Config
from .settings import CAMERA_SOURCES

Point = tuple[float, float]
Box = tuple[int, int, int, int]  # x0, y0, x1, y1 in frame pixels.


@dataclass(frozen=True)
class Item:
    """A menu control and its allowed values."""

    key: str
    label: str
    kind: str = "toggle"  # toggle | choice | slider | action
    choices: tuple = ()
    low: float = 0.0
    high: float = 1.0
    step: float = 0.05


MODES = ("Draw", "Jutsu", "Blocks", "Jujutsu")
# Column-major: the left column is "what you do", the right one "how it runs".
ITEMS: tuple[Item, ...] = (
    Item("mode", "Mode", "choice", MODES),
    Item("color_index", "Brush color", "choice", tuple(range(7))),
    Item("thickness", "Brush thickness", "slider", low=2, high=30, step=1),
    Item("glow", "Skeleton glow"),
    Item("show_skeleton", "Show skeleton"),
    Item("show_hud", "Show HUD"),
    Item("fullscreen", "Fullscreen"),
    Item("guide", "Gesture guide", "action"),
    Item("close", "Return to camera", "action"),
    Item("quit", "Quit", "action"),
    Item("camera_source", "Camera", "choice", CAMERA_SOURCES),
    Item("max_hands", "Max hands", "choice", (2, 4, 6, 8)),
    Item("volume", "Volume", "slider", low=0.0, high=1.0, step=0.05),
    Item("muted", "Mute sound"),
    Item("sensitivity", "Sensitivity", "slider", low=0.3, high=0.9, step=0.05),
    Item("record", "Record video"),
    Item("record_clean", "Clean recording"),
    Item("adaptive_quality", "Adaptive quality"),
    Item("reset", "Reset settings", "action"),
)
DANGER = frozenset({"quit", "reset"})


def rounded_rect(image: np.ndarray, box: Box, radius: int, color: Color, thickness: int = -1) -> None:
    """Anti-aliased rounded rectangle, filled (thickness -1) or outlined."""
    x0, y0, x1, y1 = box
    r = max(1, min(radius, (x1 - x0) // 2, (y1 - y0) // 2))
    if thickness < 0:
        cv2.rectangle(image, (x0 + r, y0), (x1 - r, y1), color, -1)
        cv2.rectangle(image, (x0, y0 + r), (x1, y1 - r), color, -1)
        for cx, cy in ((x0 + r, y0 + r), (x1 - r, y0 + r), (x0 + r, y1 - r), (x1 - r, y1 - r)):
            cv2.circle(image, (cx, cy), r, color, -1, cv2.LINE_AA)
        return
    for start, end in (((x0 + r, y0), (x1 - r, y0)), ((x0 + r, y1), (x1 - r, y1)),
                       ((x0, y0 + r), (x0, y1 - r)), ((x1, y0 + r), (x1, y1 - r))):
        cv2.line(image, start, end, color, thickness, cv2.LINE_AA)
    for center, angle in (((x0 + r, y0 + r), 180), ((x1 - r, y0 + r), 270), ((x1 - r, y1 - r), 0),
                          ((x0 + r, y1 - r), 90)):
        cv2.ellipse(image, center, (r, r), angle, 0, 90, color, thickness, cv2.LINE_AA)


class Menu:
    """An input-focused overlay; every selection goes through one callback."""

    def __init__(self, config: Config, value: Callable[[str], object],
                 change: Callable[[str, object], None],
                 enabled: Callable[[str], bool] = lambda key: True) -> None:
        """``value(key)`` reads live state, ``change(key, value)`` applies a choice."""
        self.config, self.value, self.change, self.enabled = config, value, change, enabled
        self.visible = False
        self.selected = 0
        self.hover: str | None = None
        self.hover_since = 0.0
        self.progress = 0.0  # Dwell ring fill under the cursor, 0..1.
        self.cursor: Point | None = None
        self._pressed = False
        self._fired = False
        self._drag: str | None = None
        self._trail: deque[tuple[float, Point]] = deque(maxlen=40)
        self._boxes: dict[str, Box] = {}
        self._size: tuple[int, int] = (0, 0)
        self._scale = 1.0
        self._panel: Box = (0, 0, 0, 0)

    def reset_pointer(self) -> None:
        """Require a fresh dwell/click after opening, tracking loss, or changing input."""
        self.hover, self._drag, self.cursor = None, None, None
        self._pressed, self._fired, self.progress = True, False, 0.0
        self._trail.clear()

    # Geometry ---------------------------------------------------------------------
    def layout(self, size: tuple[int, int]) -> dict[str, Box]:
        """Frame-coordinate hit boxes; the display window handles letterboxing separately."""
        if size == self._size and self._boxes:
            return self._boxes
        t = self.config.menu
        width, height = size
        # A portrait (phone) frame gets one tall column at full size instead of
        # two columns shrunk to fit its narrow width.
        portrait = height > width
        columns = 1 if portrait else t.columns
        scale = min(width / 720, height / 1280) if portrait else min(width / 1280, height / 720)
        rows = (len(ITEMS) + columns - 1) // columns
        header, footer = int(84 * scale), int(44 * scale)
        gap = max(4, int(t.gap * scale * 0.8))
        room = height - int(24 * scale) - header - footer - (rows - 1) * gap
        row_h = max(20, min(int(t.row_height * scale), room // rows))
        panel_w = min(width - 2 * gap, int(t.width * scale))
        panel_h = header + rows * row_h + (rows - 1) * gap + footer
        x0, y0 = (width - panel_w) // 2, max(0, (height - panel_h) // 2)
        cell_w = (panel_w - (columns + 1) * gap) // columns
        boxes = {}
        for index, item in enumerate(ITEMS):
            column, row = divmod(index, rows)
            x = x0 + gap + column * (cell_w + gap)
            y = y0 + header + row * (row_h + gap)
            boxes[item.key] = (x, y, x + cell_w, y + row_h)
        self._boxes, self._size, self._scale = boxes, size, scale
        self._panel = (x0, y0, x0 + panel_w, y0 + panel_h)
        return boxes

    def _item_at(self, point: Point | None) -> Item | None:
        if point is None:
            return None
        x, y = point
        return next((item for item in ITEMS if self._boxes[item.key][0] <= x <= self._boxes[item.key][2]
                     and self._boxes[item.key][1] <= y <= self._boxes[item.key][3]), None)

    def _track(self, key: str) -> Box:
        x0, y0, x1, y1 = self._boxes[key]
        middle = (y0 + y1) // 2
        return x0 + int((x1 - x0) * 0.48), middle - 4, x1 - int(18 * self._scale), middle + 4

    # Actions ------------------------------------------------------------------------
    def activate(self, key: str, fraction: float | None = None, step: int = 1) -> None:
        """Activate a control, step a choice, or set a slider (``fraction`` 0..1 of its range)."""
        item = next(item for item in ITEMS if item.key == key)
        if not self.enabled(key):
            return
        if item.kind == "action":
            self.change(key, True)
        elif item.kind == "toggle":
            self.change(key, not bool(self.value(key)))
        elif item.kind == "choice":
            current = self.value(key)
            index = item.choices.index(current) if current in item.choices else 0
            self.change(key, item.choices[(index + step) % len(item.choices)])
        elif item.kind == "slider":
            current = self.value(key)
            if fraction is None:
                value = float(current if current is not None else item.low) + step * item.step
            else:
                value = item.low + min(1.0, max(0.0, fraction)) * (item.high - item.low)
            steps = round((min(item.high, max(item.low, value)) - item.low) / item.step)
            value = min(item.high, item.low + steps * item.step)
            self.change(key, int(round(value)) if key == "thickness" else round(value, 4))

    def key(self, name: str) -> bool:
        """Navigate rows with arrows; left/right adjust the selected value; True if handled."""
        item = ITEMS[self.selected]
        if name in ("up", "down"):
            self.selected = (self.selected + (1 if name == "down" else -1)) % len(ITEMS)
        elif name in ("left", "right"):
            step = 1 if name == "right" else -1
            if item.kind in ("choice", "slider"):
                self.activate(item.key, step=step)
            elif item.kind == "toggle" and bool(self.value(item.key)) != (step > 0):
                self.activate(item.key)
        elif name in ("enter", "space"):
            self.activate(item.key)
        else:
            return False
        return True

    def pointer(self, point: Point | None, pressed: bool, now: float, size: tuple[int, int]) -> None:
        """Dwell once per entry; pinch/click instantly; pinch-drag sliders continuously."""
        self.layout(size)
        self.cursor = point
        if point is None:
            self.hover, self._drag, self.progress, self._fired = None, None, 0.0, False
            self._pressed = pressed
            return
        self._trail.append((now, point))
        press = pressed and not self._pressed
        self._pressed = pressed
        if self._drag is not None:
            if pressed:
                self._slide(self._drag, point[0])
                return
            self._drag = None
        item = self._item_at(point)
        key = item.key if item is not None else None
        if key != self.hover:
            self.hover, self.hover_since, self._fired = key, now, False
            if item is not None:
                self.selected = ITEMS.index(item)
        if press:
            target = self._item_at(self._lookback(now)) or item
            if target is not None and self.enabled(target.key):
                self._fired = True  # The click also satisfies this visit's dwell.
                if target.kind == "slider":
                    self._drag = target.key
                    self._slide(target.key, point[0])
                else:
                    self.activate(target.key)
            self.progress = 0.0
            return
        dwell = item is not None and item.kind != "slider" and self.enabled(item.key) and not self._fired
        self.progress = min(1.0, (now - self.hover_since) / self.config.menu.dwell_seconds) if dwell else 0.0
        if dwell and self.progress >= 1.0:
            self._fired, self.progress = True, 0.0
            self.activate(item.key)

    def _lookback(self, now: float) -> Point:
        """Cursor position just before the pinch started closing (only recent samples
        count: after a pause or a jump the current position is the intent)."""
        target = now - self.config.menu.pinch_lookback_s
        earlier = [point for t, point in self._trail if target - 0.25 <= t <= target]
        return earlier[-1] if earlier else self._trail[-1][1]

    def _slide(self, key: str, x: float) -> None:
        x0, _, x1, _ = self._track(key)
        self.activate(key, (x - x0) / max(1, x1 - x0))

    # Drawing -----------------------------------------------------------------------------
    def render(self, frame: np.ndarray) -> None:
        """Draw a consistent two-column panel, current values and dwell feedback."""
        if not self.visible:
            return
        t = self.config.menu
        height, width = frame.shape[:2]
        boxes = self.layout((width, height))
        cv2.convertScaleAbs(frame, dst=frame, alpha=1 - t.backdrop_opacity)
        s = self._scale
        px0, py0, px1, py1 = self._panel
        panel = frame[py0:py1, px0:px1]
        card = panel.copy()
        rounded_rect(card, (0, 0, px1 - px0 - 1, py1 - py0 - 1), int(t.corner_radius * s), t.panel)
        cv2.addWeighted(card, t.panel_opacity, panel, 1 - t.panel_opacity, 0, dst=panel)
        rounded_rect(frame, (px0, py0, px1 - 1, py1 - 1), int(t.corner_radius * s), t.border, 1)
        font, bold = cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, "BITL / CONTROL ROOM", (px0 + int(24 * s), py0 + int(44 * s)), bold,
                    t.title_scale * s, t.text, 2, cv2.LINE_AA)
        cv2.putText(frame, "Point to hover. Dwell or pinch to select. Pinch-drag sliders.",
                    (px0 + int(24 * s), py0 + int(70 * s)), font, 0.5 * s, t.muted, 1, cv2.LINE_AA)
        for index, item in enumerate(ITEMS):
            self._draw_item(frame, index, item, boxes[item.key], s)
        footer = "Arrows + Enter  |  Esc or Tab closes  |  mouse works too"
        (fw, _), _ = cv2.getTextSize(footer, font, 0.48 * s, 1)
        cv2.putText(frame, footer, ((px0 + px1 - fw) // 2, py1 - int(16 * s)), font, 0.48 * s, t.muted, 1,
                    cv2.LINE_AA)
        self._draw_cursor(frame, s)

    def _draw_item(self, frame: np.ndarray, index: int, item: Item, box: Box, s: float) -> None:
        t = self.config.menu
        x0, y0, x1, y1 = box
        enabled = self.enabled(item.key)
        hot = enabled and item.key in (self.hover, self._drag)
        rounded_rect(frame, box, int(10 * s), t.hover if hot else t.item)
        if index == self.selected:
            rounded_rect(frame, box, int(10 * s), t.accent, 2)
        color = t.text if enabled else t.disabled
        font, scale = cv2.FONT_HERSHEY_SIMPLEX, t.label_scale * s
        middle = (y0 + y1) // 2
        (_, th), _ = cv2.getTextSize(item.label, font, scale, 1)
        if item.kind == "action":
            label_color = t.danger if item.key in DANGER and enabled else color
            (tw, _), _ = cv2.getTextSize(item.label, font, scale, 1)
            cv2.putText(frame, item.label, ((x0 + x1 - tw) // 2, middle + th // 2), font, scale, label_color, 1,
                        cv2.LINE_AA)
            return
        cv2.putText(frame, item.label, (x0 + int(16 * s), middle + th // 2), font, scale, color, 1, cv2.LINE_AA)
        value = self.value(item.key)
        right = x1 - int(18 * s)
        if item.kind == "toggle":
            w, h = int(46 * s), int(24 * s)
            track = (right - w, middle - h // 2, right, middle + h // 2)
            on = bool(value)
            rounded_rect(frame, track, h // 2, (t.on if on else (80, 80, 80)) if enabled else t.disabled)
            knob = track[2] - h // 2 if on else track[0] + h // 2
            cv2.circle(frame, (knob, middle), max(2, h // 2 - 3), (250, 250, 250), -1, cv2.LINE_AA)
        elif item.kind == "choice":
            names = self.config.palette_names
            shown = names[value] if item.key == "color_index" and isinstance(value, int) else str(value)
            text = f"<  {shown}  >"
            (tw, _), _ = cv2.getTextSize(text, font, scale, 1)
            cv2.putText(frame, text, (right - tw, middle + th // 2), font, scale,
                        t.accent if enabled else t.disabled, 1, cv2.LINE_AA)
            if item.key == "color_index" and isinstance(value, int):
                chip = int(16 * s)
                cv2.rectangle(frame, (right - tw - chip - 12, middle - chip // 2),
                              (right - tw - 12, middle + chip // 2), self.config.palette[value], -1)
        elif item.kind == "slider":
            tx0, ty0, tx1, ty1 = self._track(item.key)
            current = float(value if value is not None else item.low)
            fraction = (current - item.low) / max(1e-9, item.high - item.low)
            fill = int(tx0 + min(1.0, max(0.0, fraction)) * (tx1 - tx0))
            rounded_rect(frame, (tx0, ty0, tx1, ty1), 4, (80, 80, 80))
            if enabled:
                rounded_rect(frame, (tx0, ty0, max(tx0 + 8, fill), ty1), 4, t.accent)
            cv2.circle(frame, (fill, middle), int(10 * s), (250, 250, 250) if enabled else t.disabled, -1,
                       cv2.LINE_AA)
            text = (f"{int(current)} px" if item.key == "thickness" else
                    f"{current * 100:.0f}%" if item.key == "volume" else f"{current:.2f}") if enabled else "n/a"
            (tw, _), _ = cv2.getTextSize(text, font, 0.5 * s, 1)
            cv2.putText(frame, text, (tx0 - tw - int(12 * s), middle + th // 2), font, 0.5 * s,
                        t.muted if enabled else t.disabled, 1, cv2.LINE_AA)

    def _draw_cursor(self, frame: np.ndarray, s: float) -> None:
        if self.cursor is None:
            return
        t = self.config.menu
        center = (int(self.cursor[0]), int(self.cursor[1]))
        radius = int(t.cursor_radius * s)
        cv2.circle(frame, center, radius + 2, (20, 20, 20), 3, cv2.LINE_AA)
        cv2.circle(frame, center, radius, t.accent, -1 if self._pressed else 2, cv2.LINE_AA)
        if self.progress > 0:
            cv2.ellipse(frame, center, (radius + 7, radius + 7), -90, 0, 360 * self.progress, t.on, 3,
                        cv2.LINE_AA)
