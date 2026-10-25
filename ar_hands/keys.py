"""Turn cv2.waitKeyEx codes into key names on every HighGUI backend.

waitKeyEx reports special keys differently per backend: Windows puts the
virtual-key code in the high word (Left = 0x250000), GTK uses X11 keysyms
(Left = 0xFF51) and may add modifier bits above bit 16, Cocoa uses Unicode
private-use codes (Left = 0xF702), and Qt uses Qt::Key values (Left =
0x01000012). Masking a code with 0xFF, as simple loops do, turns GTK's Left
arrow into "Q" (0x51), which would quit the app: hence this table.
"""
from __future__ import annotations

_ARROWS = ("left", "up", "right", "down")
SPECIAL: dict[int, str] = {}
for _i, _name in enumerate(_ARROWS):
    SPECIAL[0x250000 + _i * 0x10000] = _name  # Windows virtual keys VK_LEFT..VK_DOWN.
    SPECIAL[0xFF51 + _i] = _name  # GTK / X11 keysyms.
    SPECIAL[0x01000012 + _i] = _name  # Qt.
for _i, _name in ((0, "up"), (1, "down"), (2, "left"), (3, "right")):
    SPECIAL[0xF700 + _i] = _name  # Cocoa NSUpArrowFunctionKey...
for _n in range(12):
    _name = f"f{_n + 1}"
    SPECIAL[0x700000 + _n * 0x10000] = _name  # Windows VK_F1..
    SPECIAL[0xFFBE + _n] = _name  # GTK XK_F1..
    SPECIAL[0xF704 + _n] = _name  # Cocoa NSF1FunctionKey..
    SPECIAL[0x01000030 + _n] = _name  # Qt Key_F1..
SPECIAL.update({
    0xFF09: "tab", 0xFF0D: "enter", 0xFF8D: "enter", 0xFF1B: "esc", 0xFF08: "backspace",  # GTK
    0x01000001: "tab", 0x01000004: "enter", 0x01000005: "enter", 0x01000000: "esc",  # Qt
    0x01000003: "backspace",
})
ASCII_NAMES = {8: "backspace", 9: "tab", 10: "enter", 13: "enter", 27: "esc", 32: "space"}


def decode(code: int) -> str | None:
    """A key name ("left", "f10", "tab", "esc", "a", "?") or None for no or unknown key."""
    if code is None or code < 0:
        return None
    if code in SPECIAL:
        return SPECIAL[code]
    base = code & 0xFFFF  # GTK may add modifier state above bit 16.
    if base in SPECIAL:
        return SPECIAL[base]
    if base in ASCII_NAMES:
        return ASCII_NAMES[base]
    if 32 < base < 127:
        return chr(base)
    return None
