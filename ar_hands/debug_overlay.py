"""Developer view of the gesture pipeline, toggled with V.

Shows, per hand: raw vs stable gesture, hand scale, pinch ratio, roll, palm
facing, per-finger openness bars, and velocity (also as an arrow at the
palm). Below that: two-hand measurements and a scrolling log of the last
events. Drawing happens only while visible, so it costs nothing when hidden.
"""
from __future__ import annotations
from collections import deque
from collections.abc import Iterable
import cv2
import numpy as np
from .config import Config
from .gestures.engine import GestureEngine
from .gestures.events import BOTH, EventKind, GestureEvent
from .gestures.features import Finger, FingerState


class GestureDebugOverlay:
    """Render engine internals; record() every frame, draw() only when visible."""

    PANEL_WIDTH = 330  # Minimum; panels grow to fit their longest line.
    LINE_HEIGHT = 18
    PADDING = 8
    FONT_SCALE = 0.45
    BAR_WIDTH = 150
    LOG_WIDTH = 470

    def __init__(self, config: Config) -> None:
        """Create a hidden-or-shown overlay with an empty event log."""
        self.config = config
        self.visible = config.show_debug
        self._log: deque[str] = deque(maxlen=config.debug_log_size)
        c = config
        self._state_colors = {FingerState.EXTENDED: c.state_colors[0],
                              FingerState.PARTIAL: c.state_colors[1],
                              FingerState.CURLED: c.state_colors[2]}

    def record(self, events: Iterable[GestureEvent]) -> None:
        """Log lifecycle changes; HELD would flood the log at 30 events per second."""
        for event in events:
            if event.kind is not EventKind.HELD:
                self._log.append(f"{event.timestamp:9.2f}  {event.kind.name:<7} "
                                 f"{event.gesture.name:<11} {event.hand:<5} {event.confidence:.2f}")

    def _shade(self, frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> None:
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
        if x1 >= x2 or y1 >= y2:
            return
        roi = frame[y1:y2, x1:x2]
        shade = np.full_like(roi, self.config.dark_color)
        cv2.addWeighted(roi, 1 - self.config.hud_opacity, shade, self.config.hud_opacity, 0, dst=roi)

    def _text(self, frame: np.ndarray, text: str, x: int, y: int,
              color: tuple[int, int, int] | None = None) -> None:
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, self.FONT_SCALE,
                    color or self.config.text_color, 1, cv2.LINE_AA)

    def _width(self, text: str) -> int:
        return cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, self.FONT_SCALE, 1)[0][0]

    def _hand_panel(self, frame: np.ndarray, engine: GestureEngine, hand: str, top: int) -> int:
        features = engine.features(hand)
        header = (f"{hand}: {engine.active(hand).name} {engine.confidence(hand):.2f}"
                  f"  (raw {engine.raw(hand).name})")
        if features is None:
            lines, after = [header, "lost: gesture held during grace"], []
        else:
            vx, vy = engine.velocity(hand)
            lines = [header, f"scale {features.scale:.0f}px  pinch {features.pinch_ratio:.2f}"
                             f"  roll {features.roll:+.0f}"]
            after = [f"vel {vx:+.1f},{vy:+.1f}/s  push {engine.push_amount(hand):+.2f}/s",
                     f"palm {features.palm_normal_z:+.2f}  reach {features.pinch_reach:.2f}"
                     f"  thumb {features.thumb_direction:+.0f}"]
        bars = len(Finger) if features is not None else 0
        width = max([self.PANEL_WIDTH] + [self._width(line) + 2 * self.PADDING
                                          for line in lines + after])
        left = frame.shape[1] - self.config.margin - width
        bottom = top + (len(lines) + bars + len(after)) * self.LINE_HEIGHT + 2 * self.PADDING
        self._shade(frame, left, top, left + width, bottom)
        x, y = left + self.PADDING, top + self.PADDING + 12
        for line in lines:
            self._text(frame, line, x, y)
            y += self.LINE_HEIGHT
        if features is not None:
            for finger in Finger:
                reading = features.fingers[finger]
                self._text(frame, finger.name[0], x, y)
                bar_left, bar_top = x + 18, y - 10
                cv2.rectangle(frame, (bar_left, bar_top),
                              (bar_left + self.BAR_WIDTH, bar_top + 10), (70, 70, 80), -1)
                cv2.rectangle(frame, (bar_left, bar_top),
                              (bar_left + int(self.BAR_WIDTH * reading.openness), bar_top + 10),
                              self._state_colors[reading.state], -1)
                self._text(frame, f"{reading.openness:.2f} {reading.state.name.lower()}",
                           bar_left + self.BAR_WIDTH + 8, y)
                y += self.LINE_HEIGHT
        for line in after:
            self._text(frame, line, x, y)
            y += self.LINE_HEIGHT
        return bottom

    def _arrow(self, frame: np.ndarray, engine: GestureEngine, hand: str) -> None:
        center = engine.palm_center(hand)
        if center is None or not engine.visible(hand):
            return
        vx, vy = engine.velocity_px(hand)
        seconds = self.config.debug_arrow_seconds
        start = (int(round(center[0])), int(round(center[1])))
        end = (int(round(center[0] + vx * seconds)), int(round(center[1] + vy * seconds)))
        cv2.circle(frame, start, 4, self.config.text_color, -1, cv2.LINE_AA)
        if start != end:
            cv2.arrowedLine(frame, start, end, self.config.state_colors[1], 2, cv2.LINE_AA,
                            tipLength=0.25)

    def draw(self, frame: np.ndarray, engine: GestureEngine,
             effect_stats: list[str] | None = None) -> None:
        """Draw panels, arrows, the two-hand line, effect timings, and the event log."""
        c = self.config
        top = c.margin + 3 * c.hud_brush_offset  # Below the three full-width HUD lines.
        for hand in sorted(engine.tracked_hands(), reverse=True):  # Right, then Left.
            top = self._hand_panel(frame, engine, hand, top) + c.margin // 2
            self._arrow(frame, engine, hand)
        two = engine.two_hands
        both_text = (f"Both: {engine.active(BOTH).name}  dist {two.distance:.1f}"
                     f"  sep {two.separation_speed:+.1f}/s" if two.present else "Both: need two hands")
        width = max(self.PANEL_WIDTH, self._width(both_text) + 2 * self.PADDING)
        left = frame.shape[1] - c.margin - width
        self._shade(frame, left, top, left + width, top + self.LINE_HEIGHT + 2 * self.PADDING)
        self._text(frame, both_text, left + self.PADDING, top + self.PADDING + 12)
        if effect_stats:
            # Per-effect update+render cost, smoothed (budget: under 8 ms in total).
            top += self.LINE_HEIGHT + 2 * self.PADDING + c.margin // 2
            width = max([self.PANEL_WIDTH] + [self._width(line) + 2 * self.PADDING for line in effect_stats])
            left = frame.shape[1] - c.margin - width
            self._shade(frame, left, top, left + width,
                        top + len(effect_stats) * self.LINE_HEIGHT + 2 * self.PADDING)
            for row, line in enumerate(effect_stats):
                self._text(frame, line, left + self.PADDING, top + self.PADDING + 12 + row * self.LINE_HEIGHT)
        lines = list(self._log) or ["(no gesture events yet)"]
        height = len(lines) * self.LINE_HEIGHT + 2 * self.PADDING
        # Sit above the key-hint panel at the bottom-left.
        bottom = frame.shape[0] - c.margin - 44
        self._shade(frame, c.margin, bottom - height, c.margin + self.LOG_WIDTH, bottom)
        for row, line in enumerate(lines):
            self._text(frame, line, c.margin + self.PADDING,
                       bottom - height + self.PADDING + 12 + row * self.LINE_HEIGHT)
