"""Independent stroke history, cached neon rasterization, and transparent export."""
from __future__ import annotations
from dataclasses import dataclass
from math import ceil, floor, hypot
from pathlib import Path
import time
from typing import Callable
import cv2
import numpy as np
from .config import Config


@dataclass
class Stroke:
    """One continuous stroke in mirrored camera pixel coordinates."""

    points: list[tuple[float, float]]
    color: tuple[int, int, int]
    thickness: int
    created_at: float
    hand: str


class AirCanvas:
    """Keep independent active strokes and cache finished/active neon layers."""

    def __init__(self, config: Config, clock: Callable[[], float] = time.perf_counter) -> None:
        """Initialize this component and its independent runtime state."""
        self.config, self.clock = config, clock
        self.finished: list[Stroke] = []
        self.active: dict[str, Stroke] = {}
        self.fade_enabled = config.fade_enabled
        self._redo: list[Stroke] = []
        self._ended: dict[int, float] = {}
        self._shape: tuple[int, int] | None = None
        self._finished_layer: np.ndarray | None = None
        self._active_layer: np.ndarray | None = None
        self._layer: np.ndarray | None = None
        self._finished_dirty = self._active_dirty = True

    def begin_stroke(self, hand: str, color: tuple[int, int, int], thickness: int) -> None:
        """Start a stroke for one hand, leaving the other hand independent."""
        self.end_stroke(hand)
        self.active[hand] = Stroke([], color, thickness, self.clock(), hand)
        self._active_dirty = True

    def add_point(self, hand: str, point: tuple[float, float]) -> None:
        """Append sufficiently separated points and bound each spline's size."""
        stroke = self.active.get(hand)
        if stroke is None:
            return
        if stroke.points and hypot(point[0] - stroke.points[-1][0],
                                   point[1] - stroke.points[-1][1]) < self.config.min_point_spacing:
            return
        if len(stroke.points) >= self.config.max_stroke_points:
            last = stroke.points[-1]
            self.begin_stroke(hand, stroke.color, stroke.thickness)
            stroke = self.active[hand]
            stroke.points.append(last)
        stroke.points.append((float(point[0]), float(point[1])))
        if len(stroke.points) >= 2:
            self._redo.clear()
        self._active_dirty = True

    def end_stroke(self, hand: str) -> None:
        """Finish a stroke; discard taps and single-point tracking noise."""
        stroke = self.active.pop(hand, None)
        if stroke is None:
            return
        if len(stroke.points) >= 2:
            self.finished.append(stroke)
            self._ended[id(stroke)] = self.clock()
            while len(self.finished) > self.config.max_strokes:
                self._ended.pop(id(self.finished.pop(0)), None)
            self._finished_dirty = True
        self._active_dirty = True

    def end_all(self) -> None:
        """Finish all in-progress strokes before a history or brush operation."""
        for hand in tuple(self.active):
            self.end_stroke(hand)

    def undo(self) -> bool:
        """Finish active strokes and undo the most recently completed stroke."""
        self.end_all()
        self._expire(self.clock())
        if not self.finished:
            return False
        stroke = self.finished.pop()
        self._ended.pop(id(stroke), None)
        self._redo.append(stroke)
        self._finished_dirty = True
        return True

    def redo(self) -> bool:
        """Restore an undone stroke, restarting its optional fade lifetime."""
        self.end_all()
        if not self._redo:
            return False
        stroke = self._redo.pop()
        self.finished.append(stroke)
        self._ended[id(stroke)] = self.clock()
        self._finished_dirty = True
        return True

    def clear(self) -> None:
        """Clear active strokes, finished strokes, and undo/redo history."""
        self.finished.clear()
        self.active.clear()
        self._redo.clear()
        self._ended.clear()
        self._finished_dirty = self._active_dirty = True

    def set_fade(self, enabled: bool) -> None:
        """Toggle fading; enabling it gives existing finished strokes a full lifetime."""
        self.fade_enabled = enabled
        if enabled:
            now = self.clock()
            self._ended = {id(stroke): now for stroke in self.finished}
        self._finished_dirty = True

    def _expire(self, now: float) -> None:
        if self.fade_enabled:
            kept = [s for s in self.finished
                    if now - self._ended[id(s)] < self.config.fade_seconds]
            if len(kept) != len(self.finished):
                self.finished[:] = kept
                self._ended = {id(s): self._ended[id(s)] for s in kept}
                self._finished_dirty = True

    def _curve(self, stroke: Stroke) -> np.ndarray:
        points = np.asarray(stroke.points, dtype=np.float32)
        if len(points) == 2:
            return points
        padded = np.concatenate((points[:1], points, points[-1:]))
        p0, p1, p2, p3 = (padded[i:i + len(points) - 1, None, :] for i in range(4))
        t = np.linspace(0, 1, self.config.spline_samples, endpoint=False,
                        dtype=np.float32)[None, :, None]
        # Catmull-Rom passes through the samples, unlike a moving average which
        # cuts corners. Vectorizing all segments avoids a Python loop per sample.
        curve = 0.5 * (2 * p1 + (-p0 + p2) * t +
                       (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t +
                       (-p0 + 3 * p1 - 3 * p2 + p3) * t * t * t)
        return np.concatenate((curve.reshape(-1, 2), points[-1:]))

    def _rasterize(self, strokes: list[Stroke], now: float, fading: bool) -> np.ndarray:
        height, width = self._shape
        c = self.config
        layer = np.zeros((height, width, 3), dtype=np.uint8)
        for stroke in strokes:
            if len(stroke.points) < 2:
                continue
            opacity = max(0.0, 1.0 - (now - self._ended[id(stroke)]) / c.fade_seconds) if fading else 1.0
            color = tuple(round(channel * opacity) for channel in stroke.color)
            hot = tuple(round((channel * (1 - c.core_white_mix) + 255 * c.core_white_mix) * opacity)
                        for channel in stroke.color)
            curve = self._curve(stroke)
            padding = ceil(stroke.thickness * c.halo_width / 2 +
                           c.halo_padding_sigmas * c.halo_sigma / c.halo_scale)
            # Rasterize only this stroke's bounds, including its halo. Each
            # stroke is additive, whether active or finished, so lifting a pen
            # cannot change the brightness/color of an existing crossing.
            x1 = max(0, floor(float(curve[:, 0].min())) - padding)
            y1 = max(0, floor(float(curve[:, 1].min())) - padding)
            x2 = min(width, ceil(float(curve[:, 0].max())) + padding + 1)
            y2 = min(height, ceil(float(curve[:, 1].max())) + padding + 1)
            if x1 >= x2 or y1 >= y2:
                continue
            roi_width, roi_height = x2 - x1, y2 - y1
            body = np.zeros((roi_height, roi_width, 3), dtype=np.uint8)
            small = (max(1, round(roi_width * c.halo_scale)),
                     max(1, round(roi_height * c.halo_scale)))
            halo = np.zeros((small[1], small[0], 3), dtype=np.uint8)
            local = curve - np.array((x1, y1), dtype=np.float32)
            line = np.rint(local).astype(np.int32).reshape(-1, 1, 2)
            low_line = np.rint(local * c.halo_scale).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(halo, [low_line], False, color,
                          max(1, round(stroke.thickness * c.halo_width * c.halo_scale)), cv2.LINE_AA)
            cv2.polylines(body, [line], False, color, stroke.thickness, cv2.LINE_AA)
            cv2.polylines(body, [line], False, hot,
                          max(1, round(stroke.thickness * c.core_width)), cv2.LINE_AA)
            # The blur touches one quarter of this already cropped region.
            cv2.GaussianBlur(halo, (0, 0), c.halo_sigma, dst=halo)
            glow = cv2.resize(halo, (roi_width, roi_height), interpolation=cv2.INTER_LINEAR)
            cv2.addWeighted(body, 1, glow, c.drawing_glow_strength, 0, dst=body)
            target = layer[y1:y2, x1:x2]
            cv2.add(target, body, dst=target)
        return layer

    def render(self, frame_shape: tuple[int, ...]) -> np.ndarray:
        """Return a cached BGR light layer; callers must not modify the returned array."""
        shape = (int(frame_shape[0]), int(frame_shape[1]))
        if min(shape) <= 0:
            raise ValueError("Canvas dimensions must be positive.")
        if shape != self._shape:
            self._shape = shape
            self._finished_dirty = self._active_dirty = True
        now = self.clock()
        self._expire(now)
        fading = self.fade_enabled and bool(self.finished)
        changed = self._finished_dirty or self._active_dirty or fading
        if self._finished_dirty or fading:
            self._finished_layer = self._rasterize(self.finished, now, fading)
        if self._active_dirty:
            self._active_layer = self._rasterize(list(self.active.values()), now, False)
        if changed:
            self._layer = cv2.add(self._finished_layer, self._active_layer)
        self._finished_dirty = self._active_dirty = False
        return self._layer

    def export_png(self, path: Path) -> None:
        """Save the drawing alone as straight-alpha BGRA, including its soft halo."""
        if self._shape is None:
            raise ValueError("A camera frame is needed before exporting a drawing.")
        layer = self.render(self._shape)
        alpha = layer.max(axis=2)
        # Neon RGB is effectively premultiplied on black. Unpremultiply before
        # PNG export so normal alpha compositors do not darken the glow twice.
        rgb = np.zeros_like(layer, dtype=np.float32)
        np.divide(layer.astype(np.float32) * 255, alpha[:, :, None],
                  out=rgb, where=alpha[:, :, None] > 0)
        bgra = np.dstack((np.clip(rgb, 0, 255).astype(np.uint8), alpha))
        ok, encoded = cv2.imencode(".png", bgra)
        if not ok:
            raise OSError("OpenCV could not encode the drawing.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as output:
            output.write(encoded.tobytes())
