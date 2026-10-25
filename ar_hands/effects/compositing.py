"""Sprite compositing onto BGR frames: clipped, rotated, premultiplied, ROI-only.

Premultiplied alpha matters twice here. Resampling a straight-alpha sprite
blends its colors with the black of fully transparent neighbors, which leaves
a dark fringe around every rotated or scaled sprite. Premultiplying first
makes those neighbors contribute nothing. It also turns "alpha", "add", and
"screen" into a few saturating 8-bit OpenCV operations.

Speed: a sprite is premultiplied once, into a ``Sprite``, and not on every
draw. Blending stays in uint8 with cv2.add/multiply instead of float NumPy
math. Sprites whose alpha is only 0 or 255 (the pixel-art blocks) are drawn
with a single masked copy.
"""
from __future__ import annotations
from dataclasses import dataclass
from math import ceil, cos, radians, sin
import cv2
import numpy as np

MODES = ("alpha", "add", "screen")


def ensure_bgra(sprite: np.ndarray) -> np.ndarray:
    """Return a uint8 BGRA view or copy; gray/BGR inputs are treated as opaque."""
    if sprite.dtype != np.uint8:
        raise ValueError("Sprites must be 8-bit images.")
    if sprite.ndim == 2:
        sprite = cv2.cvtColor(sprite, cv2.COLOR_GRAY2BGR)
    if sprite.ndim != 3 or sprite.shape[2] not in (3, 4):
        raise ValueError("Sprites must be gray, BGR, or BGRA.")
    if sprite.shape[2] == 3:
        alpha = np.full(sprite.shape[:2] + (1,), 255, dtype=np.uint8)
        sprite = np.concatenate((sprite, alpha), axis=2)
    return sprite


def premultiply(bgra: np.ndarray) -> np.ndarray:
    """uint8 BGRA whose color is already multiplied by alpha (rounded)."""
    out = bgra.copy()
    alpha = cv2.cvtColor(bgra[..., 3], cv2.COLOR_GRAY2BGR)
    out[..., :3] = cv2.multiply(bgra[..., :3], alpha, scale=1 / 255)
    return out


@dataclass(frozen=True, eq=False)
class Sprite:
    """A sprite prepared for repeated drawing: premultiplied once, alpha type known."""

    premul: np.ndarray  # uint8 BGRA, color premultiplied by alpha.
    binary: bool  # Alpha is only 0 or 255, so a masked copy is exact.

    @classmethod
    def from_image(cls, image: np.ndarray) -> Sprite:
        """Prepare a gray, BGR (opaque), or straight-alpha BGRA image."""
        bgra = ensure_bgra(image)
        alpha = bgra[..., 3]
        return cls(premultiply(bgra), bool(np.all((alpha == 0) | (alpha == 255))))

    @property
    def shape(self) -> tuple[int, ...]:
        """(height, width, 4), like the image it came from."""
        return self.premul.shape


def transformed_size(width: int, height: int, scale: float, angle_deg: float) -> tuple[int, int]:
    """Size of the canvas that holds the scaled, rotated sprite without cutting corners."""
    rad = radians(angle_deg)
    c, s = abs(cos(rad)), abs(sin(rad))
    sw, sh = width * scale, height * scale
    return max(1, ceil(sw * c + sh * s - 1e-9)), max(1, ceil(sw * s + sh * c - 1e-9))


def overlay_rgba(dst_bgr: np.ndarray, sprite_bgra: np.ndarray | Sprite, center: tuple[float, float],
                 scale: float = 1.0, angle_deg: float = 0.0, opacity: float = 1.0,
                 mode: str = "alpha") -> None:
    """Draw a sprite centered at ``center`` onto ``dst_bgr`` in place.

    ``sprite_bgra`` is a raw image (prepared on the fly) or a ``Sprite`` (fast).
    ``angle_deg`` is counter-clockwise on screen (OpenCV's convention). Only
    the on-screen part of the transformed sprite is computed: a sprite half
    off-screen costs half, and a fully off-screen one returns before any
    resampling.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if opacity <= 0 or scale <= 0:
        return
    sprite = sprite_bgra if isinstance(sprite_bgra, Sprite) else Sprite.from_image(sprite_bgra)
    source = sprite.premul
    height, width = source.shape[:2]
    out_w, out_h = transformed_size(width, height, scale, angle_deg)
    x0 = int(round(center[0] - out_w / 2))
    y0 = int(round(center[1] - out_h / 2))
    frame_h, frame_w = dst_bgr.shape[:2]
    cx0, cy0 = max(0, x0), max(0, y0)
    cx1, cy1 = min(frame_w, x0 + out_w), min(frame_h, y0 + out_h)
    if cx0 >= cx1 or cy0 >= cy1:
        return  # Entirely off-screen.
    roi = dst_bgr[cy0:cy1, cx0:cx1]
    plain = scale == 1.0 and angle_deg == 0.0
    if plain:
        src = source[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]  # A view: no copy, no resampling.
        if sprite.binary and mode == "alpha" and opacity >= 1.0:
            np.copyto(roi, src[..., :3], where=src[..., 3:4] > 0)
            return
    else:
        warp_scale = scale
        if scale < 0.6:
            # warpAffine has no area filter; shrink first so small sprites do not shimmer.
            small = (max(1, round(width * scale)), max(1, round(height * scale)))
            source = cv2.resize(source, small, interpolation=cv2.INTER_AREA)
            height, width = source.shape[:2]
            warp_scale = 1.0
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle_deg, warp_scale)
        # Center the sprite in the enlarged canvas, then shift so the warp only
        # produces the visible sub-rectangle of that canvas.
        matrix[0, 2] += out_w / 2 - width / 2 - (cx0 - x0)
        matrix[1, 2] += out_h / 2 - height / 2 - (cy0 - y0)
        src = cv2.warpAffine(source, matrix, (cx1 - cx0, cy1 - cy0), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    color, alpha = src[..., :3], src[..., 3]
    if opacity < 1.0:
        color = cv2.convertScaleAbs(color, alpha=opacity)
        alpha = cv2.convertScaleAbs(alpha, alpha=opacity)
    if mode == "add":
        cv2.add(roi, color, dst=roi)
    elif mode == "alpha":
        keep = cv2.cvtColor(255 - alpha, cv2.COLOR_GRAY2BGR)
        cv2.multiply(roi, keep, dst=roi, scale=1 / 255)
        cv2.add(roi, color, dst=roi)
    else:  # screen = dst + src * (1 - dst), which never exceeds 255.
        headroom = 255 - roi
        cv2.add(roi, cv2.multiply(color, headroom, scale=1 / 255), dst=roi)
