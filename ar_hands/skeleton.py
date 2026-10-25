"""OpenCV skeleton renderer, independent of drawing strokes."""
from __future__ import annotations
import cv2
import numpy as np
from .config import Config
from .hand_tracker import HandData, Landmark


class SkeletonRenderer:
    """Draw the complete topology, labels, optional glow, and readable HUD."""

    # Immutable connection list: 6 palm edges plus 3 bones for each digit.
    PALM_CONNECTIONS = ((0, 1), (0, 5), (5, 9), (9, 13), (13, 17), (0, 17))
    CONNECTIONS = PALM_CONNECTIONS + (
        (1, 2), (2, 3), (3, 4), (5, 6), (6, 7), (7, 8),
        (9, 10), (10, 11), (11, 12), (13, 14), (14, 15),
        (15, 16), (17, 18), (18, 19), (19, 20),
    )
    TIPS = frozenset((Landmark.THUMB_TIP, Landmark.INDEX_TIP, Landmark.MIDDLE_TIP,
                      Landmark.RING_TIP, Landmark.PINKY_TIP))

    def __init__(self, config: Config) -> None:
        """Initialize this component and its independent runtime state."""
        self.config = config
        self.glow = config.glow
        self.show_indices = config.show_indices
        self._overlay: np.ndarray | None = None
        self._blurred: np.ndarray | None = None

    def _skeleton(self, frame: np.ndarray, hand: HandData) -> None:
        c = self.config
        palette = c.left_palette if hand.handedness == "Left" else c.right_palette
        for a, b in self.CONNECTIONS:
            color = palette[5 if (a, b) in self.PALM_CONNECTIONS else (b - 1) // 4]
            cv2.line(frame, hand.landmarks_px[a], hand.landmarks_px[b], color,
                     c.bone_thickness, cv2.LINE_AA)
        for index, point in enumerate(hand.landmarks_px):
            radius = c.tip_radius if index in self.TIPS else c.joint_radius
            color = palette[5 if index == Landmark.WRIST else (index - 1) // 4]
            cv2.circle(frame, point, radius + c.joint_outline, c.dark_color, -1, cv2.LINE_AA)
            cv2.circle(frame, point, radius, color, -1, cv2.LINE_AA)

    def draw(self, frame: np.ndarray, hands: list[HandData]) -> None:
        """Render skeletons and hand annotations in place on the BGR image."""
        c = self.config
        if self.glow and hands:
            if self._overlay is None or self._overlay.shape != frame.shape:
                self._overlay = np.zeros_like(frame)
                self._blurred = np.empty_like(frame)
            self._overlay.fill(0)
            for hand in hands:
                self._skeleton(self._overlay, hand)
            # Blur only the skeleton layer: blurring the webcam itself would
            # soften the whole scene. Additive blending preserves its brightness.
            cv2.GaussianBlur(self._overlay, (c.glow_kernel, c.glow_kernel),
                             c.glow_sigma, dst=self._blurred)
            cv2.addWeighted(frame, 1.0, self._blurred, c.glow_strength, 0, dst=frame)
        for hand in hands:
            self._skeleton(frame, hand)
            if self.show_indices:
                for index, (x, y) in enumerate(hand.landmarks_px):
                    cv2.putText(frame, str(index), (x + c.joint_radius, max(12, y - 7)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, c.text_color, 1, cv2.LINE_AA)
            x1, y1, x2, y2 = hand.bbox
            color = (c.left_palette if hand.handedness == "Left" else c.right_palette)[5]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
            label = f"{hand.handedness} {hand.score:.2f}"
            label_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0]
            x = max(0, min(x1, frame.shape[1] - label_width - 2))
            y = max(18, y1 - 8)
            cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        c.dark_color, 3, cv2.LINE_AA)
            cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        color, 1, cv2.LINE_AA)
