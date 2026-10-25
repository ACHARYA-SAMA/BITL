"""The display window: resizable, fullscreen, letterboxed, with mouse mapping.

OpenCV backends disagree on how a resizable window shows an image of another
size (Win32 stretches it, Qt can keep the ratio). So the window never relies
on that: each frame is letterboxed into a canvas exactly the size of the
window's client area, and the mouse position is mapped back through the same
scale and offset. Menu clicks land where they look, at any window size.
"""
from __future__ import annotations
from dataclasses import dataclass
import logging
import time
import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)
Point = tuple[float, float]


@dataclass(frozen=True)
class Letterbox:
    """frame -> canvas: canvas = frame * scale + offset."""

    scale: float = 1.0
    offset: tuple[int, int] = (0, 0)
    size: tuple[int, int] = (0, 0)  # Canvas width, height.

    def to_frame(self, x: float, y: float, frame_size: tuple[int, int]) -> Point | None:
        """Canvas pixel -> frame pixel, or None on the black bars."""
        fx, fy = (x - self.offset[0]) / self.scale, (y - self.offset[1]) / self.scale
        if 0 <= fx < frame_size[0] and 0 <= fy < frame_size[1]:
            return fx, fy
        return None


def fit(frame_size: tuple[int, int], canvas_size: tuple[int, int]) -> Letterbox:
    """Largest aspect-preserving fit of the frame inside the canvas, centered."""
    fw, fh = frame_size
    cw, ch = canvas_size
    scale = min(cw / fw, ch / fh)
    width, height = round(fw * scale), round(fh * scale)
    return Letterbox(scale, ((cw - width) // 2, (ch - height) // 2), (cw, ch))


@dataclass
class MouseState:
    """Latest mouse position in FRAME coordinates and button state."""

    position: Point | None = None
    pressed: bool = False
    moved_at: float = -1e9


class Window:
    """Owns the HighGUI window; every HighGUI call is guarded (headless tests, closed windows)."""

    def __init__(self, name: str, frame_size: tuple[int, int], fullscreen: bool = False) -> None:
        """Create a resizable window sized to the frame."""
        self.name = name
        self.frame_size = frame_size
        self.fullscreen = False
        self.mouse = MouseState()
        self.letterbox = Letterbox(1.0, (0, 0), frame_size)
        self._canvas: np.ndarray | None = None
        self.shown = False
        try:
            cv2.namedWindow(name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL)
            cv2.resizeWindow(name, *frame_size)
            cv2.setMouseCallback(name, self._on_mouse)
        except cv2.error as exc:
            LOGGER.debug("Window setup: %s", exc)
        if fullscreen:
            self.set_fullscreen(True)

    def _client_size(self) -> tuple[int, int] | None:
        try:
            _, _, width, height = cv2.getWindowImageRect(self.name)
        except (cv2.error, AttributeError):
            return None
        return (width, height) if width > 1 and height > 1 else None

    def present(self, frame: np.ndarray) -> None:
        """Show the frame letterboxed into the window's current client area."""
        size = (frame.shape[1], frame.shape[0])
        self.frame_size = size
        client = self._client_size()
        if client is None or client == size:
            self.letterbox = Letterbox(1.0, (0, 0), size)
            image = frame
        else:
            self.letterbox = box = fit(size, client)
            if self._canvas is None or self._canvas.shape[1::-1] != client:
                self._canvas = np.zeros((client[1], client[0], 3), np.uint8)
            else:
                self._canvas[:] = 0
            width, height = round(size[0] * box.scale), round(size[1] * box.scale)
            x, y = box.offset
            interpolation = cv2.INTER_LINEAR if box.scale > 1 else cv2.INTER_AREA
            cv2.resize(frame, (width, height), dst=self._canvas[y:y + height, x:x + width],
                       interpolation=interpolation)
            image = self._canvas
        cv2.imshow(self.name, image)
        self.shown = True

    def _on_mouse(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        """HighGUI callback (canvas coordinates) -> frame coordinates."""
        self.mouse.position = self.letterbox.to_frame(x, y, self.frame_size)
        self.mouse.moved_at = time.perf_counter()
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouse.pressed = True
        elif event == cv2.EVENT_LBUTTONUP:
            self.mouse.pressed = False

    def set_fullscreen(self, enabled: bool) -> None:
        """Toggle borderless fullscreen; the letterbox follows automatically."""
        try:
            cv2.setWindowProperty(self.name, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if enabled else cv2.WINDOW_NORMAL)
            if not enabled:
                cv2.resizeWindow(self.name, *self.frame_size)
        except cv2.error as exc:
            LOGGER.warning("Fullscreen toggle failed: %s", exc)
            return
        self.fullscreen = enabled

    def is_open(self) -> bool:
        """False once the user closed the window (only meaningful after the first frame)."""
        if not self.shown:
            return True
        try:
            return cv2.getWindowProperty(self.name, cv2.WND_PROP_VISIBLE) >= 1
        except cv2.error:
            return False  # Some backends raise after the X button destroys the window.
