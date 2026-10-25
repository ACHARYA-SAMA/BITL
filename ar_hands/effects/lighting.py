"""Light that behaves like light: glowing effects bloom and illuminate the scene.

Glowing effects (fire, lightning, orbs, sparks, neon) draw into a separate
black "light" buffer instead of straight onto the camera picture. Composing
that buffer is what makes them read as light sources rather than stickers:

* bloom: a camera lens smears bright light into a soft halo. The buffer is
  blurred at 1/4 and 1/16 resolution (two halo sizes) and added back;
* spill: light hits the person and the room. A very wide blur of the buffer
  is a light map; the camera picture is multiplied by it and added, so skin
  and walls near a fireball turn warm orange and Chidori flashes them blue,
  in proportion to their own color (dark surfaces reflect little);
* the light itself is added last; where it saturates to white it looks like
  an overexposed flame core, which is what a webcam records.

Everything runs at low resolution except three saturating 8-bit adds and one
multiply, and those only inside the region the light reaches (found on the
1/16 image): about 1-2 ms for a fireball, and nothing on frames without light.
"""
from __future__ import annotations
import cv2
import numpy as np


class LightCompositor:
    """Owns the reusable light buffer and composes it onto the frame."""

    def __init__(self, bloom: float = 0.7, wide_bloom: float = 0.5, spill: float = 1.0,
                 spill_softness: float = 55.0, bloom_threshold: int = 90) -> None:
        """Strengths: near halo, wide halo, and the most a light can brighten a surface
        (1.3 = +130%), reached gradually: ``spill_softness`` is the light level that
        gets about 63% of the way there."""
        self.bloom, self.wide_bloom, self.spill = bloom, wide_bloom, spill
        self.bloom_threshold = bloom_threshold  # Only light brighter than this blooms, like a lens.
        # Incoming light -> brightening, as 8-bit fractions of ``spill``. Saturating
        # instead of linear: a big blast lights the room without clipping skin to white.
        levels = np.arange(256, dtype=np.float32)
        curve = 255.0 * (1.0 - np.exp(-levels / spill_softness))
        self._spill_lut = np.clip(curve, 0, 255).astype(np.uint8)
        self._light: np.ndarray | None = None
        self.lit = False  # Something was drawn into the buffer this frame.

    def begin(self, frame: np.ndarray) -> np.ndarray:
        """A cleared light buffer the size of ``frame`` (allocated once per size)."""
        if self._light is None or self._light.shape != frame.shape:
            self._light = np.zeros_like(frame)
        else:
            self._light.fill(0)
        self.lit = False
        return self._light

    def compose(self, frame: np.ndarray) -> None:
        """Spill, light, then bloom onto ``frame`` in place."""
        light = self._light
        if light is None or not self.lit:
            return
        height, width = frame.shape[:2]
        quarter = (max(1, width // 4), max(1, height // 4))
        sixteenth = (max(1, width // 16), max(1, height // 16))
        small = cv2.resize(light, quarter, interpolation=cv2.INTER_AREA)
        tiny = cv2.resize(small, sixteenth, interpolation=cv2.INTER_AREA)
        room = cv2.GaussianBlur(tiny, (0, 0), 3.5)
        brightest = room.max(axis=2)
        # Below ~4/255 the spill brightens surfaces by under 4%: not worth touching those pixels.
        reach = np.flatnonzero(brightest.max(axis=0) > 4), np.flatnonzero(brightest.max(axis=1) > 4)
        if not reach[0].size:
            return  # Too faint to see.
        # The region the light reaches, in 1/16 cells, snapped to whole cells at every scale.
        cx0, cx1, cy0, cy1 = reach[0][0], reach[0][-1] + 1, reach[1][0], reach[1][-1] + 1
        x0, x1 = cx0 * 16, min(width, cx1 * 16)
        y0, y1 = cy0 * 16, min(height, cy1 * 16)
        region = (x1 - x0, y1 - y0)
        # Bloom from highlights only: dim light blooming too would wash big areas out.
        hot = cv2.subtract(small, (self.bloom_threshold,) * 3)
        near = cv2.GaussianBlur(hot, (0, 0), 2.5)[cy0 * 4:cy1 * 4, cx0 * 4:cx1 * 4]
        wide = cv2.resize(hot, sixteenth, interpolation=cv2.INTER_AREA)
        wide = cv2.GaussianBlur(wide, (0, 0), 1.6)[cy0:cy1, cx0:cx1]
        wide = cv2.resize(wide, near.shape[1::-1], interpolation=cv2.INTER_LINEAR)
        glow = cv2.addWeighted(near, self.bloom, wide, self.wide_bloom, 0)
        room = cv2.LUT(room[cy0:cy1, cx0:cx1], self._spill_lut)
        room = cv2.resize(room, region, interpolation=cv2.INTER_LINEAR)
        target = frame[y0:y1, x0:x1]
        # Reflected light = surface color x incoming light.
        cv2.add(target, cv2.multiply(target, room, scale=self.spill / 255), dst=target)
        cv2.add(target, light[y0:y1, x0:x1], dst=target)
        cv2.add(target, cv2.resize(glow, region, interpolation=cv2.INTER_LINEAR), dst=target)
