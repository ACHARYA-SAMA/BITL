"""Webcam opening with fallbacks, and a capture thread that keeps only the newest frame."""
from __future__ import annotations
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
import logging
import sys
import threading
import time
from types import TracebackType
from typing import Protocol
import cv2
import numpy as np
from . import AppError
from .config import Config

LOGGER = logging.getLogger(__name__)


class Capture(Protocol):
    """The part of cv2.VideoCapture the stream uses (tests pass a fake)."""

    def read(self) -> tuple[bool, np.ndarray | None]: ...

    def release(self) -> None: ...


def fourcc_name(code: float) -> str:
    """Decode CAP_PROP_FOURCC into text such as 'MJPG', or hex if unprintable."""
    value = int(code)
    text = "".join(chr((value >> (8 * i)) & 0xFF) for i in range(4))
    return text if text.isprintable() and text.strip() else f"0x{value:08x}"


def open_camera(config: Config) -> cv2.VideoCapture:
    """Try cameras 0-2, preferring DirectShow on Windows with default fallback."""
    c = config
    backends = (cv2.CAP_DSHOW, None) if sys.platform == "win32" else (None,)
    for index in c.camera_indices:
        for backend in backends:
            cap: cv2.VideoCapture | None = None
            accepted = False
            try:
                cap = cv2.VideoCapture(index) if backend is None else cv2.VideoCapture(index, backend)
                if not cap.isOpened():
                    continue
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, c.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, c.height)
                cap.set(cv2.CAP_PROP_FPS, c.camera_fps)
                if c.camera_fourcc:
                    # Ask for camera-side compression AFTER size and FPS: DirectShow
                    # ignores a format requested before them. On the dev laptop this
                    # took 720p capture from 10 to 15 FPS. Unsupported is harmless.
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*c.camera_fourcc))
                LOGGER.info("Camera %d: %.0fx%.0f at %.1f reported FPS, format %s", index,
                            cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
                            cap.get(cv2.CAP_PROP_FPS), fourcc_name(cap.get(cv2.CAP_PROP_FOURCC)))
                accepted = True
                return cap
            except cv2.error as exc:
                LOGGER.warning("Camera %d backend %s: %s", index, backend, exc)
            finally:
                if cap is not None and not accepted:
                    cap.release()
    raise AppError("Camera not found. Check that no other app is using it and that "
                   "camera permissions are enabled. Tried camera indexes "
                   f"{', '.join(str(i) for i in c.camera_indices)}.")


class CameraState(Enum):
    """What the main loop should show."""

    STARTING = "starting"
    LIVE = "live"
    LOST = "lost"  # Disconnected, not found, or stuck: retrying every reconnect_interval_s.


class CameraThreadError(AppError):
    """An unexpected exception inside the capture thread, re-raised on the main thread."""


@dataclass(frozen=True)
class FramePacket:
    """One captured frame; the main thread owns ``frame`` and may modify it in place."""

    frame: np.ndarray
    index: int  # 1, 2, 3, ... in capture order (gaps = frames dropped as stale).
    timestamp: float  # Clock time when read() returned it.


class CameraStream:
    """Capture on a background thread; the main loop always receives the newest frame.

    Why: ``VideoCapture.read()`` blocks until the driver delivers a frame (1/15 s
    on many webcams, and forever when a driver hangs). On the UI thread that
    stalls rendering, and Windows marks the window "Not Responding". A queue
    would remove the stall but build latency whenever processing is slower than
    the camera; a one-frame slot drops stale frames, so the picture is always
    current. Failed reads trigger a reconnect loop, a read stuck in the driver
    is abandoned for a fresh capture, and unexpected exceptions are re-raised by
    ``read()`` on the main thread instead of dying silently.
    """

    def __init__(self, config: Config, opener: Callable[[Config], Capture] = open_camera,
                 clock: Callable[[], float] = time.perf_counter) -> None:
        """Nothing runs until start()."""
        self.config = config
        self._opener, self._clock = opener, clock
        self._cond = threading.Condition()
        self._stop = threading.Event()
        self._packet: FramePacket | None = None
        self._published = 0
        self._consumed = 0
        self._error: BaseException | None = None
        self._state = CameraState.STARTING
        self._message = "Starting camera..."
        self._generation = 0
        self._worker: threading.Thread | None = None
        self._abandoned: list[threading.Thread] = []
        self._read_started: float | None = None  # Set while the current worker is inside read().
        self._last_frame_at: float | None = None
        self._last_restart = -1e9
        self.fps = 0.0  # Measured capture rate (smoothed), independent of processing speed.
        self.size: tuple[int, int] | None = None
        self.reconnects = 0

    # Lifecycle ---------------------------------------------------------------
    def start(self) -> CameraStream:
        """Start the capture thread (idempotent)."""
        if self._worker is None:
            self._stop.clear()
            self._spawn()
        return self

    def stop(self, timeout: float = 2.0) -> None:
        """Stop the thread; its capture is released by the thread itself."""
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        worker, self._worker = self._worker, None
        for thread in (worker, *self._abandoned):
            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout)
                if thread.is_alive():
                    # A driver call that never returns cannot be interrupted; the
                    # thread is a daemon, so it cannot keep the process alive.
                    LOGGER.warning("Camera thread %s is stuck in the driver; leaving it.", thread.name)
        self._abandoned.clear()

    def __enter__(self) -> CameraStream:
        """Start on entering a with block."""
        return self.start()

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None,
                 tb: TracebackType | None) -> None:
        """Always stop, including on exceptions and Ctrl+C."""
        self.stop()

    # Main-thread API -----------------------------------------------------------
    @property
    def state(self) -> CameraState:
        """STARTING, LIVE, or LOST (reconnecting)."""
        return self._state

    @property
    def message(self) -> str:
        """Human-readable status for the reconnect panel."""
        return self._message

    def read(self, timeout: float = 0.1) -> FramePacket | None:
        """The newest unseen frame, waiting up to ``timeout`` seconds; None if none arrived.

        Raises CameraThreadError if the capture thread crashed."""
        with self._cond:
            self._cond.wait_for(self._ready, timeout)
            self._raise_error()
            packet = self._packet
            if packet is None or packet.index <= self._consumed:
                self._check_stall()
                return None
            self._consumed = packet.index
            self._packet = None  # The worker never touches a published frame again.
            return packet

    def _ready(self) -> bool:
        return self._error is not None or (self._packet is not None and self._packet.index > self._consumed)

    def _raise_error(self) -> None:
        if self._error is not None:
            error, self._error = self._error, None
            raise CameraThreadError(f"The camera thread failed: {error!r}") from error

    def _check_stall(self) -> None:
        """Called under the lock: abandon a worker stuck inside read() for too long."""
        started = self._read_started
        now = self._clock()
        if started is None or now - started < self.config.camera_stall_s:
            return
        self._set_state(CameraState.LOST, "Camera stopped responding - reconnecting...")
        self._abandoned = [thread for thread in self._abandoned if thread.is_alive()]
        if self._abandoned or now - self._last_restart < self.config.camera_stall_s:
            return  # One stuck driver call at a time is enough; wait for it.
        LOGGER.warning("Camera read stalled for %.1f s; starting a fresh capture.", now - started)
        if self._worker is not None:
            self._abandoned.append(self._worker)
        self._last_restart = now
        self.reconnects += 1
        self._spawn()

    # Worker thread -------------------------------------------------------------
    def _spawn(self) -> None:
        self._generation += 1
        self._read_started = None
        worker = threading.Thread(target=self._run, args=(self._generation,),
                                  name=f"camera-{self._generation}", daemon=True)
        self._worker = worker
        worker.start()

    def _current(self, generation: int) -> bool:
        return generation == self._generation and not self._stop.is_set()

    def _set_state(self, state: CameraState, message: str = "") -> None:
        if state is not self._state:
            LOGGER.log(logging.INFO if state is CameraState.LIVE else logging.WARNING,
                       "Camera %s%s", state.value, f": {message}" if message else "")
        self._state, self._message = state, message

    def _run(self, generation: int) -> None:
        cap: Capture | None = None
        failures = 0
        try:
            while self._current(generation):
                if cap is None:
                    cap = self._open(generation)
                    failures = 0
                    if cap is None:
                        self._stop.wait(self.config.reconnect_interval_s)
                    continue
                self._read_started = self._clock()
                try:
                    ok, frame = cap.read()
                except cv2.error as exc:
                    LOGGER.debug("Camera read raised: %s", exc)
                    ok, frame = False, None
                if not self._current(generation):
                    break  # Abandoned or stopped while blocked in the driver.
                self._read_started = None
                if ok and frame is not None and frame.size:
                    failures = 0
                    self._publish(frame)
                    continue
                failures += 1
                if failures < self.config.max_read_failures:
                    self._stop.wait(self.config.retry_delay_ms / 1000)
                    continue
                with self._cond:
                    self._set_state(CameraState.LOST, "Camera lost - reconnecting...")
                    self.reconnects += 1
                self._release(cap)
                cap = None
                self._stop.wait(self.config.reconnect_interval_s)
        except Exception as exc:  # Anything else is a bug: hand it to the main thread.
            with self._cond:
                if generation == self._generation:
                    self._error = exc
                    self._cond.notify_all()
        finally:
            if cap is not None:
                self._release(cap)

    def _open(self, generation: int) -> Capture | None:
        try:
            cap = self._opener(self.config)
        except AppError as exc:
            with self._cond:
                self._set_state(CameraState.LOST, str(exc).splitlines()[0])
            return None
        if not self._current(generation):
            self._release(cap)
            return None
        return cap

    def _publish(self, frame: np.ndarray) -> None:
        now = self._clock()
        with self._cond:
            if self._last_frame_at is not None and now > self._last_frame_at:
                instant = 1.0 / (now - self._last_frame_at)
                self.fps = instant if self.fps == 0 else 0.9 * self.fps + 0.1 * instant
            self._last_frame_at = now
            self._published += 1
            self._packet = FramePacket(frame, self._published, now)
            self.size = (frame.shape[1], frame.shape[0])
            self._set_state(CameraState.LIVE)
            self._cond.notify_all()

    @staticmethod
    def _release(cap: Capture) -> None:
        try:
            cap.release()
        except (cv2.error, RuntimeError, OSError) as exc:
            LOGGER.warning("Camera release warning: %s", exc)
