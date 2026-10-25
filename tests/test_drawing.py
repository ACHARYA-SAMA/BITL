"""Drawing regressions with deterministic time and synthetic hand landmarks."""
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock, patch
import cv2
import numpy as np
from ar_hands.config import Config
from ar_hands.gestures.events import EventKind, GestureEvent
from ar_hands.gestures.types import Gesture
from ar_hands.smoothing import OneEuroFilter, OneEuroPointFilter
from ar_hands.canvas import AirCanvas


class SmoothingTests(unittest.TestCase):
    def test_timestamp_spacing_and_reset(self):
        fast = OneEuroFilter(1.0, 0.0, 1.0)
        slow = OneEuroFilter(1.0, 0.0, 1.0)
        self.assertEqual(fast.filter(0, 0), 0)
        slow.filter(0, 0)
        self.assertGreater(slow.filter(100, 0.2), fast.filter(100, 0.01))
        with self.assertRaises(ValueError):
            fast.filter(50, 0.01)
        fast.reset()
        self.assertEqual(fast.filter(800, 1), 800)

    def test_motion_adaptation_and_separate_axes(self):
        adaptive = OneEuroFilter(1, 0.1, 1)
        fixed = OneEuroFilter(1, 0, 1)
        adaptive.filter(0, 0)
        fixed.filter(0, 0)
        self.assertGreater(adaptive.filter(100, 0.03), fixed.filter(100, 0.03))
        pair = OneEuroPointFilter(1, 0.1, 1)
        self.assertEqual(pair.filter((100, 200), 0), (100, 200))
        x, y = pair.filter((100, 300), 0.03)
        self.assertEqual(x, 100)
        self.assertGreater(y, 200)


class CanvasTests(unittest.TestCase):
    def setUp(self):
        self.now = 10.0
        self.canvas = AirCanvas(Config(), clock=lambda: self.now)

    def stroke(self, hand="Left", start=(40, 60), end=(160, 90)):
        self.canvas.begin_stroke(hand, (255, 255, 0), 6)
        self.canvas.add_point(hand, start)
        self.canvas.add_point(hand, end)
        self.canvas.end_stroke(hand)

    def test_short_strokes_spacing_and_independent_hands(self):
        self.canvas.begin_stroke("Left", (255, 0, 255), 6)
        self.canvas.add_point("Left", (20, 20))
        self.canvas.add_point("Left", (21, 20))
        self.assertEqual(len(self.canvas.active["Left"].points), 1)
        self.canvas.end_stroke("Left")
        self.assertEqual(self.canvas.finished, [])
        for hand in ("Left", "Right"):
            self.canvas.begin_stroke(hand, (255, 0, 255), 6)
        self.canvas.add_point("Left", (20, 20))
        self.canvas.add_point("Right", (200, 200))
        self.assertNotEqual(self.canvas.active["Left"].points, self.canvas.active["Right"].points)

    def test_undo_redo_and_new_stroke_invalidate_redo(self):
        self.stroke()
        self.assertTrue(self.canvas.undo())
        self.assertFalse(self.canvas.finished)
        self.assertTrue(self.canvas.redo())
        self.assertEqual(len(self.canvas.finished), 1)
        self.canvas.undo()
        self.stroke("Right")
        self.assertFalse(self.canvas.redo())
        self.canvas.clear()
        self.assertFalse(self.canvas.active)
        self.assertFalse(self.canvas.finished)
        self.assertFalse(self.canvas.redo())

    def test_cache_and_real_transparency(self):
        self.stroke()
        shape = (180, 240, 3)
        frame = self.canvas.render(shape)
        self.assertTrue(frame.any())
        with patch.object(cv2, "GaussianBlur", wraps=cv2.GaussianBlur) as blur:
            self.assertIs(self.canvas.render(shape), frame)
            blur.assert_not_called()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "drawing.png"
            self.canvas.export_png(path)
            rgba = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            self.assertEqual(rgba.shape, (180, 240, 4))
            self.assertEqual(int(rgba[0, 0, 3]), 0)
            self.assertGreater(int(rgba[:, :, 3].max()), 0)
            self.assertTrue(np.any((rgba[:, :, 3] > 0) & (rgba[:, :, 3] < 255)))

    def test_fade_removes_finished_but_keeps_active_strokes(self):
        self.stroke()
        self.canvas.set_fade(True)
        before = self.canvas.render((180, 240, 3)).copy()
        self.now += 3
        middle = self.canvas.render((180, 240, 3))
        self.assertLess(int(middle.sum()), int(before.sum()))
        self.canvas.begin_stroke("Right", (255, 0, 255), 6)
        self.canvas.add_point("Right", (80, 90))
        self.canvas.add_point("Right", (110, 90))
        self.now += 4
        self.canvas.render((180, 240, 3))
        self.assertFalse(self.canvas.finished)
        self.assertIn("Right", self.canvas.active)

    def test_resize_preserves_strokes_and_rebuilds_cache(self):
        self.stroke()
        self.canvas.render((180, 240, 3))
        resized = self.canvas.render((360, 480, 3))
        self.assertEqual(resized.shape, (360, 480, 3))
        self.assertTrue(resized.any())

    def test_finishing_crossing_stroke_does_not_change_appearance(self):
        self.stroke(start=(30, 90), end=(210, 90))
        self.canvas.begin_stroke("Right", (255, 0, 255), 6)
        self.canvas.add_point("Right", (120, 30))
        self.canvas.add_point("Right", (120, 150))
        before = self.canvas.render((180, 240, 3)).copy()
        self.canvas.end_stroke("Right")
        after = self.canvas.render((180, 240, 3))
        np.testing.assert_array_equal(before, after)


class DrawingControlsTests(unittest.TestCase):
    def test_palette_thickness_history_export_and_toggles(self):
        from ar_hands.app import App
        with TemporaryDirectory() as directory:
            app = App(replace(Config(), base_dir=Path(directory)))
            frame = np.zeros((180, 320, 3), dtype=np.uint8)
            app.canvas.render(frame.shape)
            app.handle_key(ord("&"), frame)  # Shift+7; F7 also works.
            self.assertEqual(app.brush_color, (255, 255, 255))
            for _ in range(50):
                app.handle_key(ord("]"), frame)
            self.assertEqual(app.thickness, 30)
            for _ in range(50):
                app.handle_key(ord("["), frame)
            self.assertEqual(app.thickness, 2)
            app.canvas.begin_stroke("Left", app.brush_color, 2)
            app.canvas.add_point("Left", (30, 30))
            app.canvas.add_point("Left", (160, 100))
            app.canvas.end_stroke("Left")
            app.handle_key(ord("u"), frame)
            self.assertFalse(app.canvas.finished)
            app.handle_key(ord("y"), frame)
            self.assertEqual(len(app.canvas.finished), 1)
            app.handle_key(ord("f"), frame)
            app.handle_key(ord("k"), frame)
            self.assertTrue(app.canvas.fade_enabled)
            self.assertFalse(app.show_skeleton)
            app.handle_key(ord("p"), frame)
            exported = list((Path(directory) / "drawings").glob("*.png"))
            self.assertEqual(len(exported), 1)
            self.assertEqual(cv2.imread(str(exported[0]), -1).shape[2], 4)
            app.handle_key(ord("c"), frame)
            self.assertFalse(app.canvas.finished)


class GestureActionTests(unittest.TestCase):
    def setUp(self):
        from ar_hands.app import App
        self.directory = TemporaryDirectory()
        self.folder = Path(self.directory.name)
        self.app = App(replace(Config(), base_dir=self.folder))
        self.app.canvas.render((180, 320, 3))

    def tearDown(self):
        self.directory.cleanup()

    def send(self, kind, gesture, duration=0.0):
        self.app.bus.dispatch(GestureEvent(kind, gesture, "Right", (0.0, 0.0), duration, 0.9, 0.0))

    def test_bindings_drive_the_same_actions_as_keys(self):
        app = self.app
        # Draw mode: four STARTED bindings and the open-palm hold (draw/hover are
        # polled), plus the always-on ROCK hold that switches modes.
        self.assertEqual(len(app.bus), 8)  # Draw actions, snap, global mode switch and menu opener.
        self.send(EventKind.STARTED, Gesture.PEACE)
        self.assertEqual(app.color_index, 1)
        self.send(EventKind.STARTED, Gesture.THUMBS_UP)
        self.assertFalse((self.folder / "drawings").exists())  # Nothing drawn: nothing saved.
        app.canvas.begin_stroke("Right", app.brush_color, 6)
        app.canvas.add_point("Right", (30, 30))
        app.canvas.add_point("Right", (160, 100))
        app.canvas.end_stroke("Right")
        self.send(EventKind.STARTED, Gesture.SWIPE_LEFT)
        self.assertFalse(app.canvas.finished)
        self.send(EventKind.STARTED, Gesture.SWIPE_RIGHT)
        self.assertEqual(len(app.canvas.finished), 1)
        self.send(EventKind.STARTED, Gesture.THUMBS_UP)
        self.assertEqual(len(list((self.folder / "drawings").glob("*.png"))), 1)
        self.send(EventKind.STARTED, Gesture.OPEN_PALM)
        self.send(EventKind.HELD, Gesture.OPEN_PALM, duration=0.6)
        self.assertEqual(len(app.canvas.finished), 1)  # The hold has not completed yet.
        self.send(EventKind.HELD, Gesture.OPEN_PALM, duration=1.0)
        self.assertFalse(app.canvas.finished)

    def test_frames_draw_by_pointing_with_debug_overlay_and_hold_ring(self):
        from tests.test_gestures import make_hand
        app, tracker = self.app, MagicMock()
        app.handle_key(ord("v"), None)
        self.assertTrue(app.debug.visible)
        for frame in range(8):
            tracker.detect.return_value = [make_hand("point", offset=(-300 + 10 * frame, 0))]
            output = app.process_frame(np.zeros((720, 1280, 3), dtype=np.uint8), tracker)
        self.assertIn("Right", app.canvas.active)
        self.assertTrue(output[150:300, 900:1270].any())  # Debug panel, below the HUD lines.
        for _ in range(8):
            tracker.detect.return_value = [make_hand("open_palm")]
            app.process_frame(np.zeros((720, 1280, 3), dtype=np.uint8), tracker)
        self.assertTrue(app.gestures.is_active("Right", Gesture.OPEN_PALM))
        self.assertGreater(app.gestures.hold_progress("Right", Gesture.OPEN_PALM), 0.0)
        for _ in range(8):
            tracker.detect.return_value = [make_hand("peace")]
            app.process_frame(np.zeros((720, 1280, 3), dtype=np.uint8), tracker)
        self.assertEqual(app.color_index, 1)  # Camera frame -> engine -> bus -> action.

    def test_mode_keys_and_function_key_colors(self):
        from ar_hands.effects.manager import Mode
        app = self.app
        draw_subscriptions = len(app.bus)
        app.handle_key(ord("2"), None)
        self.assertIs(app.modes.mode, Mode.JUTSU)
        app.handle_key(ord("3"), None)
        self.assertIs(app.modes.mode, Mode.BLOCKS)
        app.handle_key(0x700000 + 2 * 0x10000, None)  # F3 from waitKeyEx on Windows.
        self.assertEqual(app.color_index, 2)
        app.handle_key(0xFFBE + 4, None)  # F5 on Linux (GTK).
        self.assertEqual(app.color_index, 4)
        app.handle_key(ord("!"), None)  # Shift+1.
        self.assertEqual(app.color_index, 0)
        app.handle_key(ord("1"), None)
        self.assertIs(app.modes.mode, Mode.DRAW)
        self.assertEqual(len(app.bus), draw_subscriptions)  # Round trip leaked nothing.

    def test_hold_to_clear_is_ignored_while_the_other_hand_draws(self):
        from tests.test_gestures import make_hand
        app, tracker, clock = self.app, MagicMock(), [100.0]

        def run(frames, *poses):
            for _ in range(frames):
                clock[0] += 1 / 30
                tracker.detect.return_value = [make_hand(pose, hand=label, offset=offset)
                                               for pose, label, offset in poses]
                app.process_frame(np.zeros((720, 1280, 3), dtype=np.uint8), tracker)

        drawing, resting = ("point", "Right", (-200, 0)), ("open_palm", "Left", (300, 0))
        with patch("ar_hands.app.time.perf_counter", lambda: clock[0]):
            for frame in range(45):  # Open palm held 1.4 s while the right hand draws.
                run(1, ("point", "Right", (-300 + 6 * frame, 0)), resting)
            self.assertIn("Right", app.canvas.active)
            self.assertIn("a hand is drawing", app._toast[0])
            run(10, ("fist", "Right", drawing[2]), resting)  # Lift the pen.
            self.assertEqual(len(app.canvas.finished), 1)
            run(8, ("fist", "Right", drawing[2]), ("fist", "Left", (300, 0)))
            run(45, ("fist", "Right", drawing[2]), resting)  # A fresh 1 s hold now clears.
        self.assertEqual(app.canvas.finished, [])


if __name__ == "__main__":
    unittest.main()
