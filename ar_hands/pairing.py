"""The pairing screen shown while the phone camera waits for a phone: QR code and steps."""
from __future__ import annotations
import sys
import cv2
import numpy as np
from .camera import CameraState
from .config import Config
from .guide import wrap
from .menu import rounded_rect
from .phone import PhoneCamera

STEPS = (
    ("Put the phone on the same Wi-Fi as this computer",
     "Or connect this computer to the phone's hotspot."),
    ("Scan the QR code with the phone's camera",
     "Or type the link below into the phone's browser."),
    ("Accept the security warning",
     "Tap Advanced (iPhone: Show Details) and continue. The certificate is this computer's own."),
    ("Allow camera access, then tap Start",
     "Prop the phone up about an arm's length away, hands in view."),
)
BACKGROUND = ((34, 22, 18), (12, 10, 10))  # BGR, top to bottom.
WAITING, PROBLEM = (60, 190, 255), (90, 90, 235)


class PairingScreen:
    """Draws into a blank frame; the gradient and QR image are cached per size."""

    def __init__(self, config: Config) -> None:
        """Nothing is drawn until render()."""
        self.config = config
        self._backdrop: np.ndarray | None = None
        self._qr: tuple[str, int, np.ndarray] | None = None

    def render(self, frame: np.ndarray, phone: PhoneCamera, now: float) -> None:
        """Fill ``frame`` with the steps, the live status and the QR code (or the startup error)."""
        t = self.config.menu
        height, width = frame.shape[:2]
        s = min(width / 1280, height / 720)
        frame[:] = self._background(width, height)
        font, bold = cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX
        left, text_right = int(80 * s), int(800 * s)

        def text(value: str, y: int, scale: float, color: tuple[int, int, int], x: int = left,
                 face: int = font, thickness: int = 1) -> None:
            cv2.putText(frame, value, (x, int(y * s)), face, scale * s, color, thickness, cv2.LINE_AA)

        text("BITL  /  PHONE CAMERA", 96, .55, t.accent)
        text("Use your phone as the camera", 150, 1.2, t.text, face=bold, thickness=2)
        text("Phones film sharper and smoother than most laptop webcams.", 186, .6, t.muted)
        y = 250
        for number, (title, detail) in enumerate(STEPS, 1):
            center = (left + int(16 * s), int((y - 6) * s))
            cv2.circle(frame, center, int(16 * s), t.accent, 2, cv2.LINE_AA)
            (nw, nh), _ = cv2.getTextSize(str(number), font, .6 * s, 2)
            cv2.putText(frame, str(number), (center[0] - nw // 2, center[1] + nh // 2), font, .6 * s, t.accent, 2,
                        cv2.LINE_AA)
            x = left + int(48 * s)
            text(title, y, .68, t.text, x)
            line_y = y + 26
            for line in wrap(detail, text_right - x, .5 * s):
                text(line, line_y, .5, t.muted, x)
                line_y += 22
            y = line_y + 26
        failed = phone.error is not None or phone.state is CameraState.LOST
        status = phone.message
        if phone.state is CameraState.STARTING:
            status += "." * (1 + int(now * 2) % 3)
        dot = WAITING if not failed else PROBLEM
        if int(now * 2) % 2 == 0 or failed:
            cv2.circle(frame, (left + int(8 * s), int(588 * s)), int(7 * s), dot, -1, cv2.LINE_AA)
        for index, line in enumerate(wrap(status, text_right - left - int(30 * s), .62 * s)[:2]):
            text(line, 594 + 26 * index, .62, dot, left + int(30 * s))
        footer = "O  laptop webcam      TAB  settings      ?  guide      Q  quit"
        if sys.platform == "win32":
            text("Windows asking about network access? Choose Allow, or the phone cannot connect.",
                 652, .48, t.muted)
        text(footer, 690, .5, t.muted)
        if phone.running:
            self._draw_code(frame, phone.url, s)

    def _background(self, width: int, height: int) -> np.ndarray:
        if self._backdrop is None or self._backdrop.shape[1::-1] != (width, height):
            ramp = np.linspace(0, 1, height, dtype=np.float32)[:, None, None]
            top, bottom = (np.asarray(color, np.float32) for color in BACKGROUND)
            column = top * (1 - ramp) + bottom * ramp
            self._backdrop = np.ascontiguousarray(np.broadcast_to(column, (height, width, 3))).astype(np.uint8)
        return self._backdrop

    def _draw_code(self, frame: np.ndarray, url: str, s: float) -> None:
        """A white card with the QR code, sharp-edged (nearest-neighbour scaling), and the link."""
        t = self.config.menu
        card = int(330 * s)
        x0, y0 = int(1010 * s) - card // 2, int(330 * s) - card // 2
        code = self._code(url, card - int(36 * s))
        rounded_rect(frame, (x0, y0, x0 + card, y0 + card), int(18 * s), (255, 255, 255))
        offset = (card - code.shape[0]) // 2
        frame[y0 + offset:y0 + offset + code.shape[0], x0 + offset:x0 + offset + code.shape[1]] = code
        font = cv2.FONT_HERSHEY_SIMPLEX
        caption = "Scan with your phone"
        (cw, _), _ = cv2.getTextSize(caption, font, .62 * s, 1)
        cv2.putText(frame, caption, (x0 + (card - cw) // 2, y0 + card + int(36 * s)), font, .62 * s, t.text, 1,
                    cv2.LINE_AA)
        scale = min(.5 * s, .5 * s * (card + int(140 * s)) / max(1, cv2.getTextSize(url, font, .5 * s, 1)[0][0]))
        (uw, _), _ = cv2.getTextSize(url, font, scale, 1)
        cv2.putText(frame, url, (x0 + (card - uw) // 2, y0 + card + int(66 * s)), font, scale, t.accent, 1,
                    cv2.LINE_AA)

    def _code(self, url: str, size: int) -> np.ndarray:
        if self._qr is None or self._qr[:2] != (url, size):
            modules = cv2.QRCodeEncoder.create().encode(url)
            scale = max(1, size // modules.shape[0])
            image = cv2.resize(modules, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            self._qr = (url, size, cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
        return self._qr[2]
