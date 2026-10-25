"""Video recording of the composited output (no audio; see the README for capturing sound).

Why the writer uses the MEASURED frame rate: a video file stores frames plus
one fixed playback rate. If the app produces 15 frames per second but the file
says 30, every second of real time plays back in half a second (sped up).
So the writer opens at the rate actually measured, and frames are then placed
on a wall-clock timeline: when the app runs slower than the file rate a frame
is repeated, when it runs faster one is skipped. Playback speed stays right
even if the frame rate changes mid-recording (a domain expansion, say).

Encoding runs on a background thread, so writing never slows the render loop.
"""
from __future__ import annotations
from collections.abc import Callable
from datetime import datetime
import logging
from pathlib import Path
import queue
import threading
from typing import Any
import cv2
import numpy as np
from . import AppError

LOGGER = logging.getLogger(__name__)
# (fourcc, extension), in order of preference: mp4v is the most portable .mp4
# encoder bundled with OpenCV; XVID .avi is the widely supported fallback.
CODECS = (("mp4v", ".mp4"), ("XVID", ".avi"), ("MJPG", ".avi"))
Writer = Any  # cv2.VideoWriter or a test double with isOpened/write/release.


class Recorder:
    """``start()`` opens a writer, ``write()`` feeds frames, ``stop()`` always releases it."""

    def __init__(self, folder: Path, queue_size: int = 48,
                 writer_factory: Callable[[str, int, float, tuple[int, int]], Writer] = cv2.VideoWriter,
                 min_fps: float = 5.0, max_fps: float = 60.0) -> None:
        """Nothing is opened until start()."""
        self.folder = folder
        self.writer_factory = writer_factory
        self.min_fps, self.max_fps = min_fps, max_fps
        self.path: Path | None = None
        self.codec = ""
        self.fps = 0.0
        self.size: tuple[int, int] | None = None
        self.started_at = 0.0
        self.frames_written = 0
        self.dropped = 0
        self._queue: queue.Queue[tuple[np.ndarray, int] | None] = queue.Queue(queue_size)
        self._writer: Writer | None = None
        self._thread: threading.Thread | None = None
        self._scheduled = 0  # Frames placed on the timeline so far.
        self._carry = 0  # Repeats owed by frames that found the queue full.
        self._error: BaseException | None = None
        self._closing = threading.Event()

    @property
    def active(self) -> bool:
        """True between start() and stop()."""
        return self._writer is not None

    def elapsed(self, now: float) -> float:
        """Seconds recorded so far."""
        return max(0.0, now - self.started_at) if self.active else 0.0

    def start(self, size: tuple[int, int], fps: float, now: float) -> Path:
        """Open ``recordings/ar_hands_YYYYMMDD_HHMMSS.mp4`` (or .avi); raises AppError if no codec works."""
        if self.active:
            return self.path
        if self._thread is not None and self._thread.is_alive():
            raise AppError('Previous recording is still finishing; try again shortly.')
        self._closing.clear()
        self.fps = float(min(self.max_fps, max(self.min_fps, round(fps, 2))))
        self.size = (int(size[0]), int(size[1]))
        stem = f"ar_hands_{datetime.now():%Y%m%d_%H%M%S}"
        try:
            self.folder.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise AppError(f"Cannot create {self.folder}: {exc}") from None
        for fourcc, extension in CODECS:
            path = self._unique(self.folder / f"{stem}{extension}")
            writer = self.writer_factory(str(path), cv2.VideoWriter_fourcc(*fourcc), self.fps, self.size)
            if writer.isOpened():
                self._writer, self.path, self.codec = writer, path, fourcc
                break
            writer.release()
            path.unlink(missing_ok=True)
            LOGGER.info("Video codec %s unavailable; trying the next one.", fourcc)
        else:
            raise AppError("No video codec could be opened (tried mp4v, XVID, MJPG).")
        self.started_at, self.frames_written, self.dropped = now, 0, 0
        self._scheduled = self._carry = 0
        self._error = None
        self._thread = threading.Thread(target=self._drain, name="recorder", daemon=True)
        self._thread.start()
        LOGGER.info("Recording %s with %s at %.2f FPS (measured), %dx%d.", self.path, self.codec,
                    self.fps, *self.size)
        return self.path

    @staticmethod
    def _unique(path: Path) -> Path:
        candidate, suffix = path, 1
        while candidate.exists():
            candidate = path.with_name(f"{path.stem}_{suffix:02d}{path.suffix}")
            suffix += 1
        return candidate

    def write(self, frame: np.ndarray, now: float) -> None:
        """Place ``frame`` on the timeline: written 0, 1 or more times so the file matches real time."""
        if not self.active:
            return
        if self._error is not None:
            error, self._error = self._error, None
            self.stop()
            raise AppError(f"Recording stopped: {error}") from error
        due = int((now - self.started_at) * self.fps) + 1  # Frames that should exist by now.
        count = due - self._scheduled + self._carry
        if count <= 0:
            return  # Running faster than the file rate: this frame is not needed.
        # Bounded: a long stall (window dragged, laptop asleep) must not queue minutes of copies.
        count = min(count, int(self.fps * 2))
        if frame.shape[1::-1] != self.size:  # Camera reconnected at another size.
            image = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
        else:
            image = frame.copy()  # The caller keeps drawing on its frame (REC indicator).
        try:
            self._queue.put_nowait((image, count))
        except queue.Full:
            self._carry = count  # Owed: the next frame that fits is repeated instead.
            self._scheduled = due
            self.dropped += 1
            return
        self._carry = 0
        self._scheduled = due

    def _drain(self) -> None:
        writer = self._writer
        try:
            while True:
                try:
                    item = self._queue.get(timeout=.1)
                except queue.Empty:
                    if self._closing.is_set():
                        return
                    continue
                if item is None:
                    return
                image, count = item
                for _ in range(count):
                    writer.write(image)
                self.frames_written += count
        except Exception as exc:
            self._error = exc
        finally:
            # The encoder thread alone owns release, including error and slow-write paths.
            try:
                writer.release()
            except (cv2.error, OSError, RuntimeError) as exc:
                if self._error is None:
                    self._error = exc
                LOGGER.warning('Video writer release warning: %s', exc)

    def stop(self) -> Path | None:
        """Flush, release the writer, and return the file path (safe to call any time)."""
        writer, self._writer = self._writer, None
        if writer is None:
            return None
        thread = self._thread
        self._closing.set()
        if thread is not None:
            thread.join(10.0)
            if thread.is_alive():
                LOGGER.warning('Encoder is still finishing %s; cleanup remains on its thread.', self.path)
                return self.path
        self._thread = None
        while not self._queue.empty():  # Anything left after a writer error.
            self._queue.get_nowait()
        if self._error is not None:
            error, self._error = self._error, None
            raise AppError(f'Recording could not finish: {error}') from error
        seconds = self.frames_written / self.fps if self.fps else 0.0
        LOGGER.info("Saved recording %s (%d frames, %.1f s%s).", self.path, self.frames_written, seconds,
                    f", {self.dropped} frames skipped under load" if self.dropped else "")
        return self.path
