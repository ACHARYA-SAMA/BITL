"""Person segmentation and a learned background plate.

The selfie segmenter gives a soft mask of the person. Effects use it two ways:
  * put the user IN FRONT of something (Susanoo, a Domain Expansion): draw
    the scene, then paste the camera pixels back where the mask says person;
  * make the user DISAPPEAR (snap to dust): paste background where the person was.
For the second, ``BackgroundModel`` keeps a running estimate of what is behind
the person, learned from frames where those pixels were visible, and inpaints
what has never been seen.

Segmentation costs about 4 ms at 640x360, so it runs only on frames where
something asks for it. The model is downloaded at startup (never inside the
render loop, where a download would freeze the picture). If it cannot load,
effects fall back to plain overlays instead of failing.
"""
from __future__ import annotations
import logging
from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from . import AppError
from .config import Config
from .hand_tracker import download_model

LOGGER = logging.getLogger(__name__)


def is_tflite(path: Path) -> bool:
    """TFLite flatbuffers carry the identifier "TFL3" at byte offset 4."""
    try:
        with open(path, "rb") as handle:
            return handle.read(8)[4:8] == b"TFL3"
    except OSError:
        return False


def segmenter_path(config: Config) -> Path:
    """Where the selfie segmenter model lives."""
    return config.base_dir / "models" / "selfie_segmenter.tflite"


def ensure_segmenter_model(config: Config) -> Path:
    """The selfie segmenter (~250 KB), downloaded once."""
    return download_model(config.segmenter_url, segmenter_path(config), config, "Segmentation model", is_tflite)


def keep_people(soft: np.ndarray, min_share: float = 0.2) -> np.ndarray:
    """0/1 mask of the large blobs only: the person (and a second person), not stray
    false positives such as a curtain rod. A blob survives if it has at least
    ``min_share`` of the largest blob's area."""
    binary = (soft > 0.3).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 2:
        return np.ones_like(soft) if count == 2 else np.zeros_like(soft)
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = np.zeros(count, np.float32)
    keep[1:] = (areas >= areas.max() * min_share).astype(np.float32)
    # Grow slightly so the soft edges of kept blobs are not cut off.
    return cv2.dilate(keep[labels], np.ones((5, 5), np.uint8))


class PersonSegmenter:
    """Soft person mask for a frame, computed only when asked; None if unavailable."""

    def __init__(self, config: Config, download: bool = False) -> None:
        """Nothing loads until the first mask() call. With ``download`` False a
        missing model makes segmentation unavailable instead of fetching it."""
        self.config = config
        self.download = download
        self.width = config.segmentation_width  # Adaptive quality may lower it.
        self._segmenter: vision.ImageSegmenter | None = None
        self._failed = False
        self._last_timestamp = -1
        self._smoothed: np.ndarray | None = None
        self._last_call = -1

    @property
    def available(self) -> bool:
        """False after the model failed to load (effects then use overlays)."""
        return not self._failed

    def _open(self) -> bool:
        try:
            path = ensure_segmenter_model(self.config) if self.download else segmenter_path(self.config)
            if not is_tflite(path):
                raise AppError(f"model file missing at {path} (it downloads at startup when online)")
            options = vision.ImageSegmenterOptions(
                base_options=python.BaseOptions(model_asset_path=str(path)),
                running_mode=vision.RunningMode.VIDEO,
                output_confidence_masks=True, output_category_mask=False)
            self._segmenter = vision.ImageSegmenter.create_from_options(options)
            return True
        except (AppError, OSError, RuntimeError, ValueError) as exc:
            LOGGER.warning("Person segmentation unavailable (%s); effects will use overlays.", exc)
            self._failed = True
            return False

    def mask(self, frame_bgr: np.ndarray, timestamp_ms: int) -> np.ndarray | None:
        """float32 person probability (0..1) at the frame's size, temporally smoothed."""
        if self._failed or (self._segmenter is None and not self._open()):
            return None
        c = self.config
        height, width = frame_bgr.shape[:2]
        work_w = min(width, self.width)
        work_h = max(1, round(work_w * height / width))
        small = cv2.resize(frame_bgr, (work_w, work_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        # VIDEO mode needs strictly increasing timestamps, even if we skip frames.
        self._last_timestamp = max(self._last_timestamp + 1, timestamp_ms)
        try:
            result = self._segmenter.segment_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), self._last_timestamp)
        except (RuntimeError, ValueError) as exc:
            LOGGER.warning("Segmentation failed on a frame: %s", exc)
            return None
        raw = np.asarray(result.confidence_masks[0].numpy_view(), np.float32).reshape(work_h, work_w)
        # A soft ramp, not a hard threshold: weak false positives vanish and
        # hair edges stay soft instead of jagged.
        soft = np.clip((raw - c.mask_low) / (c.mask_high - c.mask_low), 0.0, 1.0)
        soft *= keep_people(soft)
        stale = timestamp_ms - self._last_call > 500
        if self._smoothed is None or self._smoothed.shape != soft.shape or stale:
            self._smoothed = soft
        else:
            # Light temporal smoothing removes edge flicker without visible lag.
            self._smoothed = c.mask_smoothing * self._smoothed + (1 - c.mask_smoothing) * soft
        self._last_call = timestamp_ms
        return cv2.resize(self._smoothed, (width, height), interpolation=cv2.INTER_LINEAR)

    def close(self) -> None:
        """Release the native segmenter."""
        segmenter, self._segmenter = self._segmenter, None
        if segmenter is not None:
            try:
                segmenter.close()
            except (RuntimeError, OSError) as exc:
                LOGGER.warning("Segmenter cleanup warning: %s", exc)


class BackgroundModel:
    """A running estimate of the scene behind the person, at half resolution."""

    WORK_SCALE = 0.5

    def __init__(self, learn_rate: float = 0.12) -> None:
        """Empty until the first update()."""
        self.learn_rate = learn_rate
        self.plate: np.ndarray | None = None  # float32 BGR, half resolution.
        self.seen: np.ndarray | None = None  # 0..1: how surely each pixel was observed.
        self._filled: np.ndarray | None = None
        self._dirty = True

    def reset(self) -> None:
        """Forget everything (e.g. the camera moved)."""
        self.plate = self.seen = self._filled = None
        self._dirty = True

    def update(self, frame_bgr: np.ndarray, person: np.ndarray) -> None:
        """Learn from pixels that are clearly not the person."""
        height, width = frame_bgr.shape[:2]
        size = (max(1, int(width * self.WORK_SCALE)), max(1, int(height * self.WORK_SCALE)))
        small = cv2.resize(frame_bgr, size, interpolation=cv2.INTER_AREA).astype(np.float32)
        # Dilate the person generously so hair edges and shadows are never learned
        # as background (they would leave a ghost of the user behind).
        person_small = cv2.dilate(cv2.resize(person, size), np.ones((17, 17), np.uint8))
        visible = np.clip(1.0 - person_small * 3.0, 0.0, 1.0)
        if self.plate is None or self.plate.shape != small.shape:
            self.plate, self.seen = small, visible.copy()
        else:
            weight = (self.learn_rate * visible)[..., None]
            self.plate += weight * (small - self.plate)
            self.seen = np.maximum(self.seen * 0.999, visible)
        self._dirty = True

    def fill(self, size: tuple[int, int]) -> np.ndarray | None:
        """Full-resolution background: learned where seen, inpainted where never seen."""
        if self.plate is None:
            return None
        if self._dirty or self._filled is None or self._filled.shape[1::-1] != size:
            plate = np.clip(self.plate, 0, 255).astype(np.uint8)
            unknown = (self.seen < 0.5).astype(np.uint8) * 255
            if unknown.any():
                # Inpaint at quarter scale: fast, and the result is soft anyway.
                tiny = cv2.resize(plate, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
                tiny_mask = cv2.resize(unknown, tiny.shape[1::-1], interpolation=cv2.INTER_NEAREST)
                patched = cv2.inpaint(tiny, tiny_mask, 5, cv2.INPAINT_TELEA)
                patched = cv2.resize(patched, plate.shape[1::-1], interpolation=cv2.INTER_LINEAR)
                plate = np.where(unknown[..., None] > 0, patched, plate)
            self._filled = cv2.resize(plate, size, interpolation=cv2.INTER_LINEAR)
            self._dirty = False
        return self._filled
