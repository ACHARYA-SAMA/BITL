"""Pen adapter tests: stable gesture-engine decisions become clean, independent strokes."""
from dataclasses import replace
import unittest
from ar_hands.canvas import AirCanvas
from ar_hands.config import Config
from ar_hands.gestures.engine import GestureEngine
from ar_hands.gestures.types import Gesture
from ar_hands.hand_tracker import Landmark
from ar_hands.pen import PenController, PenSignal, pen_signals
from tests.test_gestures import DT, make_hand

DRAW = frozenset({Gesture.POINT})


class PenTests(unittest.TestCase):
    def setUp(self):
        self.config = Config()
        self.canvas = AirCanvas(self.config)
        self.pen = PenController(self.config)
        self.engine = GestureEngine(self.config.gestures)
        self.now = 0.0

    def step(self, *hands):
        self.now += DT
        self.engine.update(list(hands), self.now)
        return self.pen.update(list(hands), self.canvas, self.now, (255, 255, 0), 6,
                               pen_signals(self.engine, DRAW))

    @staticmethod
    def hand(x, label="Right", pose="point"):
        return make_hand(pose, hand=label, offset=(x, 0))

    def draw_line(self, frames=8, start=-300, step=8):
        for frame in range(frames):
            self.step(self.hand(start + step * frame))

    def test_pointing_draws_once_stable_and_backfills_the_start(self):
        tips = []
        for frame in range(10):
            hand = self.hand(-300 + 12 * frame)
            tips.append(hand.landmarks_px[Landmark.INDEX_TIP])
            cursors = self.step(hand)
            self.assertEqual(cursors[0].drawing, frame >= 3)
        stroke = self.canvas.active["Right"]
        # The stabilizer confirms POINT on the 4th frame, but the line still
        # starts at the 1st frame's fingertip.
        self.assertEqual(stroke.points[0], tips[0])
        self.assertEqual(len(stroke.points), 10)

    def test_both_hands_draw_independent_strokes(self):
        for frame in range(8):
            self.step(self.hand(-350 + 10 * frame, "Left"), self.hand(250 + 10 * frame, "Right"))
        self.assertEqual(set(self.canvas.active), {"Left", "Right"})
        left_xs = [x for x, _ in self.canvas.active["Left"].points]
        right_xs = [x for x, _ in self.canvas.active["Right"].points]
        self.assertLess(max(left_xs), min(right_xs))

    def test_curling_stops_ink_at_once_then_closes_the_stroke(self):
        self.draw_line()
        count = len(self.canvas.active["Right"].points)
        for frame in range(8, 20):
            self.step(self.hand(-300 + 8 * frame, pose="fist"))
            stroke = self.canvas.active.get("Right")
            if stroke is not None:
                self.assertEqual(len(stroke.points), count)  # No hook from the curling finger.
        self.assertNotIn("Right", self.canvas.active)
        self.assertEqual([len(stroke.points) for stroke in self.canvas.finished], [count])

    def test_brief_dropout_resumes_the_same_stroke_nearby(self):
        self.draw_line()
        self.step()
        self.step()
        self.assertIn("Right", self.canvas.active)
        for frame in range(10, 14):
            self.step(self.hand(-300 + 8 * frame))
        self.assertEqual(self.canvas.finished, [])
        self.assertIn("Right", self.canvas.active)

    def test_returning_far_away_after_a_dropout_never_bridges(self):
        self.draw_line()
        before = list(self.canvas.active["Right"].points)
        self.step()
        self.step()
        self.step(self.hand(200))
        self.assertEqual([stroke.points for stroke in self.canvas.finished], [before])
        self.assertEqual(len(self.canvas.active["Right"].points), 1)

    def test_far_resume_after_a_dry_pause_starts_a_new_stroke(self):
        self.draw_line()
        self.step(self.hand(-244, pose="fist"))
        self.step(self.hand(-244, pose="fist"))
        self.assertTrue(self.engine.is_active("Right", Gesture.POINT))  # Still within grace.
        self.step(self.hand(-144))
        self.assertEqual(len(self.canvas.finished), 1)
        self.assertIn("Right", self.canvas.active)

    def test_long_loss_finishes_the_stroke_and_reentry_starts_clean(self):
        self.draw_line()
        for _ in range(self.config.gestures.lost_grace_frames + 1):
            self.step()
        self.assertEqual(self.canvas.active, {})
        self.assertEqual(len(self.canvas.finished), 1)
        cursor = self.step(self.hand(250))[0]
        self.assertFalse(cursor.drawing)
        self.assertEqual(cursor.point, self.hand(250).landmarks_px[Landmark.INDEX_TIP])

    def test_one_frame_teleport_splits_the_stroke(self):
        self.draw_line()
        self.step(self.hand(300))
        self.assertEqual(len(self.canvas.finished), 1)
        self.assertEqual(len(self.canvas.active["Right"].points), 1)

    def test_missing_hand_keeps_the_pen_down_without_ink(self):
        self.draw_line(frames=6)
        self.now += DT
        self.engine.update([], self.now)
        self.assertEqual(pen_signals(self.engine, DRAW), {"Right": PenSignal(True, False)})

    def test_reset_finishes_strokes_and_drawing_resumes_as_a_new_stroke(self):
        self.draw_line()
        self.pen.reset(self.canvas)
        self.assertEqual(self.canvas.active, {})
        self.assertEqual(len(self.canvas.finished), 1)
        self.step(self.hand(-200))
        self.step(self.hand(-190))
        self.assertEqual(len(self.canvas.active["Right"].points), 2)

    def test_duplicate_labels_keep_the_confident_hand(self):
        strong, weak = self.hand(-300), replace(self.hand(300), score=0.4)
        for _ in range(6):
            cursors = self.step(strong, weak)
        self.assertEqual(len(cursors), 1)
        self.assertLess(cursors[0].point[0], 640)


if __name__ == "__main__":
    unittest.main()
