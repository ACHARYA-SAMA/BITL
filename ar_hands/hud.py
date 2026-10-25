"""Smoothed FPS and readable overlay panels."""
from __future__ import annotations
import time
import cv2
import numpy as np
from .config import Config
from .effects.compositing import overlay_rgba
from .pen import Cursor


class FPSCounter:
    """Smooth end-to-end frame rate with an exponential moving average."""

    def __init__(self, smoothing: float = 0.9) -> None:
        """Initialize this component and its independent runtime state."""
        if not 0 <= smoothing < 1:
            raise ValueError("FPS smoothing must be in [0, 1).")
        self.smoothing = smoothing
        self._last: float | None = None
        self.value = 0.0

    def update(self) -> float:
        """Sample once per displayed frame using a monotonic high-resolution clock."""
        now = time.perf_counter()
        if self._last is not None and now > self._last:
            instant = 1.0 / (now - self._last)
            self.value = (instant if self.value == 0 else
                          self.smoothing * self.value + (1 - self.smoothing) * instant)
        self._last = now
        return self.value


class HUDRenderer:
    """Render status and controls independently of the skeleton."""

    KEY_HINT = ("1 Draw 2 Jutsu 3 Blocks 4 Jujutsu | TAB menu ? guide | R record M mute F fullscreen | "
                "F1-F7 color [ ] size | f fade P PNG S shot | F10 timing Q quit")

    def __init__(self, config: Config) -> None:
        """Initialize this component and its independent runtime state."""
        self.config = config
        self.show_indices = config.show_indices
        self.glow = config.glow

    def _panel(self, frame: np.ndarray, text: str, bottom: bool = False,
               offset: int = 0, center: bool = False) -> None:
        c = self.config
        height, width = frame.shape[:2]
        margin = min(c.margin, max(0, (min(width, height) - 1) // 4))
        available = max(1, width - 2 * margin)
        font = cv2.FONT_HERSHEY_SIMPLEX
        raw_width = cv2.getTextSize(text, font, c.text_scale, c.text_thickness)[0][0]
        scale = c.text_scale * min(1.0, max(1, available - 16) / max(1, raw_width))
        (tw, th), baseline = cv2.getTextSize(text, font, scale, c.text_thickness)
        panel_height = min(height - 2 * margin, th + baseline + 16)
        panel_width = min(available, tw + 16)
        top = height - margin - panel_height if bottom else min(margin + offset, height - panel_height)
        left = (width - panel_width) // 2 if center else margin
        roi = frame[top:top + panel_height, left:left + panel_width]
        if not roi.size:
            return
        shade = np.full_like(roi, c.dark_color)
        cv2.addWeighted(roi, 1 - c.hud_opacity, shade, c.hud_opacity, 0, dst=roi)
        cv2.putText(frame, text, (left + 8, top + 8 + th), font, scale,
                    c.text_color, c.text_thickness, cv2.LINE_AA)

    def draw(self, frame: np.ndarray, fps: float, hand_count: int,
             color_index: int = 0, thickness: int | None = None,
             fade: bool = False, stroke_count: int = 0, gesture_status: str = "",
             mode: str = "Draw", status: str | None = None, icon: np.ndarray | None = None,
             hint: str = "") -> None:
        """Metrics and mode, a mode status line (with color swatch or block icon),
        a gesture hint line, and the fitted key hint at the bottom."""
        c = self.config
        height, width = frame.shape[:2]
        self._panel(frame, f"FPS {fps:5.1f}  |  Hands {hand_count}  |  {width}x{height}  |  Mode: {mode}")
        swatch = status is None  # The brush swatch belongs to the default (Draw) status.
        if swatch:
            status = (f"    {c.palette_names[color_index]}  {thickness or c.default_thickness}px  |  "
                      f"Fade {'on' if fade else 'off'}  |  Strokes {stroke_count}  |  "
                      f"{gesture_status or 'Point to draw'}")
        self._panel(frame, status, offset=c.hud_brush_offset)
        x, y = c.margin + 8, c.margin + c.hud_brush_offset + 8
        if icon is not None:
            overlay_rgba(frame, icon, (x + 10, y + 8))
        elif swatch:
            cv2.rectangle(frame, (x, y), (x + 16, y + 16), c.palette[color_index], -1)
        if hint:
            self._panel(frame, hint, offset=2 * c.hud_brush_offset)
        self._panel(frame, self.KEY_HINT, bottom=True)

    def draw_toast(self, frame: np.ndarray, text: str) -> None:
        """Confirm a gesture-triggered action, centered below the HUD lines."""
        self._panel(frame, text, offset=3 * self.config.hud_brush_offset, center=True)

    def draw_banner(self, frame: np.ndarray, title: str, progress: float) -> None:
        """Mode-change title: pops in over the first quarter, fades over the last 40%."""
        if not 0 <= progress < 1:
            return
        c = self.config
        height, width = frame.shape[:2]
        opacity = 1.0 if progress < 0.6 else (1 - progress) / 0.4
        scale = 2.2 * (0.75 + 0.25 * min(1.0, progress / 0.25))
        font = cv2.FONT_HERSHEY_DUPLEX
        (tw, th), baseline = cv2.getTextSize(title, font, scale, 3)
        x0, y0 = (width - tw) // 2 - 30, int(height * 0.36) - th - 24
        x1, y1 = x0 + tw + 60, y0 + th + baseline + 48
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(width, x1), min(height, y1)
        if x0 >= x1 or y0 >= y1:
            return
        roi = frame[y0:y1, x0:x1]
        card = roi.copy()
        card[:] = (card * 0.25 + np.asarray(c.dark_color) * 0.75).astype(np.uint8)
        cv2.putText(card, title, (30, 24 + th), font, scale, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.addWeighted(card, opacity, roi, 1 - opacity, 0, dst=roi)

    def draw_hold_ring(self, frame: np.ndarray, center: tuple[float, float], radius: float,
                       progress: float, color: tuple[int, int, int]) -> None:
        """Draw a clockwise progress arc from 12 o'clock so a hold can be aborted in time."""
        c = self.config
        point = (int(round(center[0])), int(round(center[1])))
        axes = (max(1, int(round(radius))),) * 2
        cv2.ellipse(frame, point, axes, 0, 0, 360, c.dark_color, c.hold_ring_thickness + 2, cv2.LINE_AA)
        cv2.ellipse(frame, point, axes, -90, 0, 360 * min(1.0, max(0.0, progress)), color,
                    c.hold_ring_thickness, cv2.LINE_AA)

    def draw_cursors(self, frame: np.ndarray, cursors: list[Cursor],
                     color: tuple[int, int, int]) -> None:
        """Draw hollow hover rings or filled glowing pen-down dots."""
        c = self.config
        height, width = frame.shape[:2]
        for cursor in cursors:
            x, y = (int(round(value)) for value in cursor.point)
            if cursor.drawing:
                # Local patches keep cursor glow independent of full-frame blur.
                radius = c.cursor_halo_radius
                x1, y1 = max(0, x - radius), max(0, y - radius)
                x2, y2 = min(width, x + radius + 1), min(height, y + radius + 1)
                if x1 < x2 and y1 < y2:
                    yy, xx = np.ogrid[y1:y2, x1:x2]
                    alpha = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) /
                                   (2 * c.cursor_halo_sigma ** 2)) * c.cursor_glow_strength
                    glow = (alpha[:, :, None] * np.asarray(color)).astype(np.uint8)
                    cv2.add(frame[y1:y2, x1:x2], glow, dst=frame[y1:y2, x1:x2])
            cv2.circle(frame, (x, y), c.cursor_radius + c.cursor_outline,
                       c.dark_color, c.cursor_outline, cv2.LINE_AA)
            cv2.circle(frame, (x, y), c.cursor_radius, color,
                       -1 if cursor.drawing else c.cursor_outline, cv2.LINE_AA)
