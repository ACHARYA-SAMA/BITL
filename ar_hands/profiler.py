"""Per-stage frame timing: a stacked bar and a 120-frame sparkline (F10), and an exit summary."""
from __future__ import annotations
from collections import deque
from collections.abc import Callable, Iterable
from types import TracebackType
import time
import cv2
import numpy as np

# Pipeline order. "capture" is time spent WAITING for the camera thread, so it
# shows when the webcam (not the code) limits the frame rate.
STAGES = ("capture", "detect", "segment", "gestures", "effects", "canvas", "render", "imshow")
STAGE_COLORS = {  # BGR
    "capture": (150, 150, 150), "detect": (60, 180, 255), "segment": (200, 120, 255),
    "gestures": (90, 230, 120), "effects": (60, 90, 255), "canvas": (255, 220, 60),
    "render": (255, 140, 80), "imshow": (120, 120, 200),
}
BUDGET_LINES_MS = (33.3, 66.7)  # 30 and 15 FPS guides on the sparkline.
# Stages that run on the analyzer thread, at the same time as the others (pipeline.py).
PARALLEL = frozenset({"detect", "segment"})


def busy(stage_ms: dict[str, float]) -> float:
    """Processing time per frame: the slower of the render thread and the analyzer
    thread, since they overlap (the wait for the camera is not processing)."""
    render = sum(ms for name, ms in stage_ms.items() if name != "capture" and name not in PARALLEL)
    return max(render, sum(ms for name, ms in stage_ms.items() if name in PARALLEL))


class StageTimer:
    """``with profiler.stage("detect"):`` adds the block's duration to this frame."""

    __slots__ = ("_profiler", "_name", "_start")

    def __init__(self, profiler: Profiler, name: str) -> None:
        """Bind to a profiler and a stage name."""
        self._profiler, self._name, self._start = profiler, name, 0.0

    def __enter__(self) -> StageTimer:
        """Start timing."""
        self._start = self._profiler.clock()
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None,
                 tb: TracebackType | None) -> None:
        """Record the elapsed milliseconds, even when the block raised."""
        self._profiler.add(self._name, (self._profiler.clock() - self._start) * 1000.0)


class Profiler:
    """Collects stage timings per frame; keeps the last ``history`` frames and run totals."""

    def __init__(self, history: int = 120, clock: Callable[[], float] = time.perf_counter,
                 stages: Iterable[str] = STAGES) -> None:
        """Empty until the first frame ends."""
        self.clock = clock
        self.stages = tuple(stages)
        self.history: deque[dict[str, float]] = deque(maxlen=history)
        self.visible = False
        self.frames = 0
        self._current: dict[str, float] = {}
        self._totals: dict[str, float] = {}

    def stage(self, name: str) -> StageTimer:
        """A context manager timing one stage of the current frame."""
        return StageTimer(self, name)

    def add(self, name: str, ms: float) -> None:
        """Add time to a stage (a stage may run several times per frame)."""
        self._current[name] = self._current.get(name, 0.0) + ms

    def end_frame(self) -> dict[str, float]:
        """Close the current frame's record and return it."""
        record, self._current = self._current, {}
        self.history.append(record)
        self.frames += 1
        for name, ms in record.items():
            self._totals[name] = self._totals.get(name, 0.0) + ms
        return record

    def average(self, frames: int | None = None) -> dict[str, float]:
        """Mean ms per stage over the last ``frames`` frames (all kept frames by default)."""
        records = list(self.history)[-frames:] if frames else list(self.history)
        if not records:
            return {}
        names = {name for record in records for name in record}
        return {name: sum(record.get(name, 0.0) for record in records) / len(records)
                for name in self._ordered(names)}

    def busy_ms(self, frames: int = 30) -> float:
        """Mean processing time per frame, excluding the wait for the camera."""
        return busy(self.average(frames))

    def _ordered(self, names: Iterable[str]) -> list[str]:
        names = set(names)
        return [name for name in self.stages if name in names] + sorted(names - set(self.stages))

    def summary(self) -> str:
        """Run-average table for the log on exit."""
        if not self.frames:
            return "Profiler: no frames."
        lines = [f"Profiler: {self.frames} frames, average ms per frame:"]
        total = sum(self._totals.values()) / self.frames
        for name in self._ordered(self._totals):
            ms = self._totals[name] / self.frames
            lines.append(f"  {name:<9} {ms:7.2f} ms  {100 * ms / total if total else 0:5.1f}%")
        processing = busy({name: ms / self.frames for name, ms in self._totals.items()})
        lines.append(f"  {'total':<9} {total:7.2f} ms  (processing {processing:.2f} ms = "
                     f"{1000 / processing if processing > 0 else 0:.0f} FPS achievable; detection runs alongside)")
        return "\n".join(lines)

    def draw(self, frame: np.ndarray) -> None:
        """Bottom-right panel: stacked bar of the last 10 frames' stages, legend, sparkline."""
        if not self.visible or not self.history:
            return
        height, width = frame.shape[:2]
        panel_w, panel_h = min(460, width - 24), 210
        x0, y0 = width - panel_w - 12, height - panel_h - 56
        if x0 < 0 or y0 < 0:
            return
        roi = frame[y0:y0 + panel_h, x0:x0 + panel_w]
        cv2.addWeighted(roi, 0.25, np.full_like(roi, (24, 20, 18)), 0.75, 0, dst=roi)
        font = cv2.FONT_HERSHEY_SIMPLEX
        average = self.average(10)
        total = sum(average.values())
        busy = total - average.get("capture", 0.0)
        cv2.putText(frame, f"Frame {total:5.1f} ms   processing {busy:5.1f} ms "
                    f"({1000 / busy if busy > 0 else 0:4.0f} FPS possible)", (x0 + 10, y0 + 22),
                    font, 0.45, (240, 240, 240), 1, cv2.LINE_AA)
        # Stacked bar: full width = 66.7 ms (15 FPS).
        bar_x, bar_y, bar_w, bar_h = x0 + 10, y0 + 32, panel_w - 20, 16
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
        cursor = float(bar_x)
        for name, ms in average.items():
            span = ms / BUDGET_LINES_MS[1] * bar_w
            end = min(bar_x + bar_w, cursor + span)
            if end > cursor:
                cv2.rectangle(frame, (int(cursor), bar_y), (int(end), bar_y + bar_h),
                              STAGE_COLORS.get(name, (200, 200, 200)), -1)
            cursor = end
        guide = bar_x + int(BUDGET_LINES_MS[0] / BUDGET_LINES_MS[1] * bar_w)
        cv2.line(frame, (guide, bar_y - 3), (guide, bar_y + bar_h + 3), (255, 255, 255), 1)
        # Legend, two columns.
        for index, (name, ms) in enumerate(average.items()):
            column, row = divmod(index, 4)
            lx, ly = bar_x + column * (bar_w // 2), bar_y + 36 + row * 18
            cv2.rectangle(frame, (lx, ly - 9), (lx + 10, ly + 1), STAGE_COLORS.get(name, (200, 200, 200)), -1)
            cv2.putText(frame, f"{name:<9}{ms:6.1f} ms", (lx + 16, ly), font, 0.42, (225, 225, 225), 1,
                        cv2.LINE_AA)
        # Sparkline of total frame time.
        spark_y0, spark_y1 = y0 + 118, y0 + panel_h - 12
        totals = np.array([sum(record.values()) for record in self.history], np.float32)
        top = max(BUDGET_LINES_MS[1] * 1.2, float(totals.max()))
        span_h = spark_y1 - spark_y0

        def y_of(ms: float) -> int:
            return int(spark_y1 - min(ms, top) / top * span_h)

        for ms in BUDGET_LINES_MS:
            cv2.line(frame, (bar_x, y_of(ms)), (bar_x + bar_w, y_of(ms)), (80, 80, 80), 1)
            cv2.putText(frame, f"{1000 / ms:.0f} fps", (bar_x + bar_w - 48, y_of(ms) - 3), font, 0.35,
                        (150, 150, 150), 1, cv2.LINE_AA)
        capacity = self.history.maxlen or len(totals)
        step = bar_w / max(1, capacity - 1)
        points = np.array([(bar_x + i * step, y_of(ms)) for i, ms in enumerate(totals)], np.int32)
        if len(points) > 1:
            cv2.polylines(frame, [points.reshape(-1, 1, 2)], False, (90, 230, 255), 1, cv2.LINE_AA)
