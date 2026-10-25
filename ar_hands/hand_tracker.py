"""MediaPipe Tasks hand tracking and reusable landmark data."""
from __future__ import annotations
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import IntEnum
from http.client import HTTPException
from math import hypot
from pathlib import Path
from tempfile import NamedTemporaryFile
from types import TracebackType
import logging
import threading
import urllib.error
import urllib.request
import zipfile
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from . import AppError
from .config import Config

LOGGER = logging.getLogger(__name__)


class Landmark(IntEnum):
    """All 21 MediaPipe indices, shared by future drawing and gesture modules."""

    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12
    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20


@dataclass
class HandData:
    """One hand in mirrored image coordinates; bbox endpoints are inclusive."""

    handedness: str
    score: float
    landmarks_px: list[tuple[int, int]]
    landmarks_norm: list[tuple[float, float, float]]
    bbox: tuple[int, int, int, int]
    track_id: str = ""  # Stable id from identity.py ("Left", "Right 2"...); "" = use handedness.

    @property
    def key(self) -> str:
        """The id gestures, pens and effects keep this hand's state under."""
        return self.track_id or self.handedness


def distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """Return Euclidean pixel distance (used by pinch and gesture features)."""
    return hypot(p1[0] - p2[0], p1[1] - p2[1])


def resolve_handedness(hands: list[HandData], duplicate_ratio: float) -> list[HandData]:
    """Drop double detections and make handedness labels unique.

    MediaPipe classifies each hand independently, so both hands occasionally
    get the same label, or one hand is detected twice. Pens and gestures keep
    state per label, so a repeated label would merge two hands into one zigzag.
    Same-label hands whose landmarks nearly coincide are one hand: keep the more
    confident one. Otherwise position decides. In the mirrored selfie view the
    user's right hand appears on the right side of the screen.
    """
    if len(hands) != 2 or hands[0].handedness != hands[1].handedness:
        return hands
    first, second = hands
    scale = max(1.0, distance(first.landmarks_px[Landmark.WRIST],
                              first.landmarks_px[Landmark.MIDDLE_MCP]))
    gap = sum(distance(a, b) for a, b in zip(first.landmarks_px, second.landmarks_px)) / len(Landmark)
    if gap < duplicate_ratio * scale:
        return [max(hands, key=lambda hand: hand.score)]
    left, right = sorted(hands, key=lambda hand: hand.landmarks_px[Landmark.WRIST][0])
    return [replace(left, handedness="Left"), replace(right, handedness="Right")]


def download_model(url: str, path: Path, config: Config, label: str,
                   validate: Callable[[Path], bool]) -> Path:
    """Download once, publishing the file only after a complete, valid transfer."""
    temporary: Path | None = None
    try:
        if path.is_file() and path.stat().st_size > 0:
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        LOGGER.info("Downloading %s to %s", label, path)
        with urllib.request.urlopen(url, timeout=config.download_timeout) as response:
            total = int(response.headers.get("Content-Length", "0"))
            received, last_bucket = 0, -1
            # A separate temporary file prevents an interrupted download from
            # becoming the cached model used on the next launch.
            with NamedTemporaryFile(dir=path.parent, suffix=".part", delete=False) as output:
                temporary = Path(output.name)
                while chunk := response.read(config.download_chunk_size):
                    output.write(chunk)
                    received += len(chunk)
                    bucket = int(received * 10 / total) if total else received // (1024 * 1024)
                    if bucket != last_bucket:
                        status = f"{min(bucket * 10, 100)}%" if total else f"{received // 1024} KiB"
                        LOGGER.info("  %s download: %s", label, status)
                        last_bucket = bucket
            if not received or (total and received != total):
                raise ValueError("The model download was incomplete.")
        if not validate(temporary):
            raise ValueError(f"The response was not a valid {label} file.")
        temporary.replace(path)
        LOGGER.info("%s ready.", label)
        return path
    except (OSError, urllib.error.URLError, HTTPException, ValueError) as exc:
        raise AppError(
            f"{label} download failed: {exc}\nCheck your internet/proxy and folder permissions.\n"
            f"Manual download: {url}\nSave the file as: {path}"
        ) from None
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError as exc:
                LOGGER.warning("Could not remove temporary download %s: %s", temporary, exc)


def ensure_model(config: Config) -> Path:
    """The hand landmarker bundle (a zip), downloaded on first run."""
    return download_model(config.model_url, config.base_dir / "models" / "hand_landmarker.task",
                          config, "Hand model", zipfile.is_zipfile)


class HandTracker:
    """Own a synchronous VIDEO landmarker; accept already mirrored BGR frames."""

    def __init__(self, config: Config | None = None) -> None:
        """Initialize this component and its independent runtime state."""
        self.config = config or Config()
        self._last_timestamp = -1
        self._rgb: np.ndarray | None = None
        self._small: np.ndarray | None = None
        self._landmarker: vision.HandLandmarker | None = None
        self._lock = threading.Lock()  # detect() runs on the analyzer thread; reconfigure() on the UI's.
        # Mutable at runtime: adaptive quality lowers it when frames run slow.
        self.detect_scale = self.config.detect_scale
        self._landmarker = self._create(self.config)

    @staticmethod
    def _create(config: Config) -> vision.HandLandmarker:
        path = ensure_model(config)
        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=config.num_hands,
            min_hand_detection_confidence=config.detection_confidence,
            min_hand_presence_confidence=config.presence_confidence,
            min_tracking_confidence=config.tracking_confidence,
        )
        try:
            return vision.HandLandmarker.create_from_options(options)
        except (OSError, RuntimeError, ValueError) as exc:
            raise AppError(
                f"Cannot initialize hand tracking: {exc}\nIf the model is damaged, "
                f"delete {path} and restart to download it again.\nManual model: {config.model_url}"
            ) from None

    def reconfigure(self, config: Config) -> None:
        """Apply new confidence thresholds (the menu's sensitivity slider).

        MediaPipe fixes them at creation, so build the replacement first and
        swap only once it exists: a failure leaves the working tracker in place."""
        replacement = self._create(config)
        with self._lock:
            previous, self._landmarker = self._landmarker, replacement
            self.config = config
        if previous is not None:
            try:
                previous.close()
            except (RuntimeError, OSError) as exc:
                LOGGER.warning("Tracker cleanup warning: %s", exc)

    def __enter__(self) -> HandTracker:
        """Return the tracker for use in a with block."""
        return self

    def __exit__(self, exc_type: type[BaseException] | None,
                 exc: BaseException | None, tb: TracebackType | None) -> None:
        """Release native resources, including on exceptions and Ctrl+C."""
        self.close()

    def close(self) -> None:
        """Close once; tolerate repeated cleanup without hiding earlier errors."""
        with self._lock:
            landmarker, self._landmarker = self._landmarker, None
        if landmarker is not None:
            try:
                landmarker.close()
            except (RuntimeError, OSError) as exc:
                LOGGER.warning("Tracker cleanup warning: %s", exc)

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int) -> list[HandData]:
        """Detect up to ``num_hands`` hands; timestamp_ms must strictly increase."""
        with self._lock:
            return self._detect(frame_bgr, timestamp_ms)

    def _detect(self, frame_bgr: np.ndarray, timestamp_ms: int) -> list[HandData]:
        if self._landmarker is None:
            raise RuntimeError("HandTracker has been closed.")
        if timestamp_ms <= self._last_timestamp:
            raise ValueError("VIDEO timestamps must strictly increase in milliseconds.")
        if frame_bgr.dtype != np.uint8 or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("Expected an unsigned 8-bit, three-channel BGR frame.")
        height, width = frame_bgr.shape[:2]
        if not width or not height:
            raise ValueError("Cannot detect hands in an empty frame.")
        source = frame_bgr
        if self.detect_scale < 0.999:
            # Landmarks come back normalized (0..1), so they map onto the full
            # frame below without any rescaling: detection size is invisible.
            size = (max(1, round(width * self.detect_scale)), max(1, round(height * self.detect_scale)))
            if self._small is None or self._small.shape[1::-1] != size:
                self._small = np.empty((size[1], size[0], 3), np.uint8)
            cv2.resize(frame_bgr, size, dst=self._small, interpolation=cv2.INTER_AREA)
            source = self._small
        if self._rgb is None or self._rgb.shape != source.shape:
            self._rgb = np.empty_like(source)
        # OpenCV supplies BGR; the model expects RGB. Reuse this destination
        # because VIDEO inference completes before the next frame is processed.
        cv2.cvtColor(source, cv2.COLOR_BGR2RGB, dst=self._rgb)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=self._rgb)
        self._last_timestamp = timestamp_ms
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        hands: list[HandData] = []
        for landmarks, categories in zip(result.hand_landmarks, result.handedness):
            if len(landmarks) != len(Landmark) or not categories:
                continue
            category = max(categories, key=lambda item: item.score)
            if category.category_name not in ("Left", "Right"):
                continue
            normalized = [(float(p.x), float(p.y), float(p.z)) for p in landmarks]
            # Normalized x/y are fractions of the ACTUAL image dimensions, not
            # the requested camera size. Clamp edge/out-of-frame predictions.
            pixels = [(max(0, min(width - 1, int(x * width))),
                       max(0, min(height - 1, int(y * height)))) for x, y, _ in normalized]
            xs, ys = zip(*pixels)
            pad = self.config.bbox_padding
            bbox = (max(0, min(xs) - pad), max(0, min(ys) - pad),
                    min(width - 1, max(xs) + pad), min(height - 1, max(ys) + pad))
            hands.append(HandData(category.category_name, float(category.score),
                                  pixels, normalized, bbox))
        if self.config.num_hands > 2:
            return hands  # Several people: identity.py sorts out duplicates and labels.
        return resolve_handedness(hands, self.config.duplicate_hand_ratio)
