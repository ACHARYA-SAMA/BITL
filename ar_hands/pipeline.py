"""Frame pipelining: hand detection on its own thread while the main thread renders.

Hand detection (MediaPipe, ~11 ms here, more with many hands) and rendering
(effects, HUD, display, ~5-10 ms) used to run one after the other, so a frame
cost their sum. MediaPipe releases the GIL while it infers, so the next frame
can be detected while this one renders: a frame then costs only the slower of
the two. Like the camera, the analyzer keeps one slot in and one slot out, so
when it falls behind it skips stale frames instead of queueing latency.
"""
from __future__ import annotations
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
import ctypes
import logging
import sys
import threading
from typing import Generic, TypeVar
import numpy as np
from .hand_tracker import HandData

LOGGER = logging.getLogger(__name__)
T = TypeVar("T")


@dataclass
class Analysis:
    """One camera frame after detection. ``frame`` is mirrored and never drawn on,
    so the renderer can composite it again between camera frames."""

    frame: np.ndarray
    hands: list[HandData]
    person: np.ndarray | None
    timestamp_ms: int
    timings: dict[str, float] = field(default_factory=dict)  # Worker stage -> ms, for the profiler.


class FrameAnalyzer(Generic[T]):
    """``submit(frame)`` from the main loop; ``result()`` returns the newest analysis.

    An exception in ``analyze`` is handed back by ``result()`` on the main
    thread, where the frame-error policy lives, and the worker keeps going."""

    def __init__(self, analyze: Callable[[np.ndarray], T]) -> None:
        """Nothing runs until start()."""
        self._analyze = analyze
        self._cond = threading.Condition()
        self._input: np.ndarray | None = None
        self._output: T | None = None
        self._error: BaseException | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.busy = False  # True while a frame is being analyzed.

    def start(self) -> FrameAnalyzer[T]:
        """Start the worker thread (idempotent)."""
        if self._thread is None:
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="analyzer", daemon=True)
            self._thread.start()
        return self

    def stop(self, timeout: float = 2.0) -> None:
        """Finish the frame in progress and stop."""
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None

    def submit(self, frame: np.ndarray) -> None:
        """Queue a frame, replacing one that has not started yet (it is stale now)."""
        with self._cond:
            self._input = frame
            self._cond.notify_all()

    def result(self, timeout: float = 0.0) -> T | None:
        """The newest finished analysis, waiting up to ``timeout``; raises the worker's error."""
        with self._cond:
            self._cond.wait_for(lambda: self._output is not None or self._error is not None, timeout)
            if self._error is not None:
                error, self._error = self._error, None
                raise error
            output, self._output = self._output, None
            return output

    @property
    def idle(self) -> bool:
        """No frame queued or in progress."""
        return self._input is None and not self.busy

    def _run(self) -> None:
        while not self._stop.is_set():
            with self._cond:
                self._cond.wait_for(lambda: self._input is not None or self._stop.is_set())
                if self._stop.is_set():
                    return
                frame, self._input = self._input, None
                self.busy = True
            try:
                output = self._analyze(frame)
            except Exception as exc:  # Handed to the main thread's frame-error policy.
                with self._cond:
                    self._error, self.busy = exc, False
                    self._cond.notify_all()
                continue
            with self._cond:
                self._output, self.busy = output, False
                self._cond.notify_all()


@contextmanager
def fast_timer() -> Iterator[None]:
    """1 ms Windows timer resolution while the app runs.

    By default Windows rounds short waits up to its 15.6 ms tick, so the
    ``cv2.waitKeyEx(1)`` every frame needs to pump the window really waited
    about 13 ms: alone enough to cap the app near 40 FPS. At 1 ms it takes
    about 2 ms. Other systems already wait accurately."""
    winmm = None
    if sys.platform == "win32":
        try:
            winmm = ctypes.WinDLL("winmm")
            winmm.timeBeginPeriod(1)
        except (OSError, AttributeError) as exc:
            LOGGER.debug("Timer resolution unchanged: %s", exc)
            winmm = None
    try:
        yield
    finally:
        if winmm is not None:
            winmm.timeEndPeriod(1)
