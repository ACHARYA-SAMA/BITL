"""The shared gesture vocabulary, dependency-free so Config can name gestures."""
from enum import Enum


class Gesture(Enum):
    """Every gesture the engine reports: static poses, motions, and two-hand poses."""

    NONE = "none"
    # Single-hand poses, recognized per frame (recognizer.py).
    OPEN_PALM = "open_palm"
    FIST = "fist"
    POINT = "point"
    PEACE = "peace"
    THUMBS_UP = "thumbs_up"
    ROCK = "rock"
    THREE = "three"
    PINCH = "pinch"
    OK = "ok"
    CROSSED = "crossed"  # Index and middle up, index lying across the middle (Gojo's sign).
    # Instant motion gestures (motion.py, snap.py): STARTED then ENDED in the same frame.
    SWIPE_LEFT = "swipe_left"
    SWIPE_RIGHT = "swipe_right"
    SWIPE_UP = "swipe_up"
    SWIPE_DOWN = "swipe_down"
    SNAP = "snap"  # Thumb-middle finger snap.
    # Two-hand gestures (two_hand.py), always reported with hand="Both".
    PRAYER = "prayer"
    TWO_FISTS = "two_fists"
    TWO_PALMS = "two_palms"
    HANDS_APART = "hands_apart"
