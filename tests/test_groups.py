"""Groups of friends (many hands, per-person two-hand gestures), the frame pipeline,
and the light pass (flames, bloom, spill)."""
from dataclasses import replace
import threading
import time
import unittest
import numpy as np
from ar_hands.effects.flame import Distortion, FlameField
from ar_hands.effects.lighting import LightCompositor
from ar_hands.gestures.engine import GestureEngine
from ar_hands.gestures.events import BOTH, EventKind, both_of, hand_id, is_both, pair_of, person_of
from ar_hands.gestures.types import Gesture
from ar_hands.handpose import make_hand
from ar_hands.identity import HandIdentities
from ar_hands.pipeline import FrameAnalyzer
from ar_hands.window import window_size

FRAME = (1280, 720)


def crowd(shift: float = 0.0, flip: str | None = None) -> list:
    """Three people side by side, each with a left and a right hand (mirrored view)."""
    hands = []
    for person, x in enumerate((-430, 0, 430)):
        for handedness, dx in (("Left", -70), ("Right", 70)):
            label = flip if flip and person == 1 and handedness == "Left" else handedness
            hands.append(make_hand("open_palm", scale=55, offset=(x + dx + shift, 60), hand=label, frame=FRAME))
    return hands


class HandIdTests(unittest.TestCase):
    def test_id_helpers(self):
        self.assertEqual(hand_id("Left", 1), "Left")
        self.assertEqual(hand_id("Right", 3), "Right 3")
        self.assertEqual(person_of("Left 2"), 2)
        self.assertEqual(both_of("Right 3"), "Both 3")
        self.assertEqual(both_of("Left"), BOTH)
        self.assertEqual(pair_of("Both 2"), ("Left 2", "Right 2"))
        self.assertTrue(is_both("Both 2") and is_both(BOTH) and not is_both("Left"))

    def test_six_hands_get_stable_ids_grouped_by_person(self):
        ids = HandIdentities()
        first = ids.assign(crowd(), 0.0)
        names = {hand.key for hand in first}
        self.assertEqual(names, {"Left", "Right", "Left 2", "Right 2", "Left 3", "Right 3"})
        by_x = sorted(first, key=lambda hand: hand.landmarks_px[0][0])
        self.assertEqual([person_of(hand.key) for hand in by_x], [1, 1, 2, 2, 3, 3], "neighbours pair up")
        moved = ids.assign(list(reversed(crowd(shift=25))), 0.033)
        before = {hand.key: hand.landmarks_px[0][0] for hand in first}
        for hand in moved:  # Each id is on the hand that moved 25 px (give or take rounding).
            self.assertLessEqual(abs(hand.landmarks_px[0][0] - 25 - before[hand.key]), 2, hand.key)

    def test_a_label_flip_keeps_the_id_and_a_duplicate_is_dropped(self):
        ids = HandIdentities()
        ids.assign(crowd(), 0.0)
        flipped = ids.assign(crowd(flip="Right"), 0.033)
        middle_left = min((h for h in flipped if person_of(h.key) == 2), key=lambda h: h.landmarks_px[0][0])
        self.assertEqual((middle_left.key, middle_left.handedness), ("Left 2", "Left"))
        hands = crowd()
        self.assertEqual(len(ids.assign(hands + [replace(hands[0], score=0.5)], 0.066)), 6)

    def test_a_returning_person_keeps_their_ids(self):
        ids = HandIdentities(keep_s=0.6)
        ids.assign(crowd(), 0.0)
        ids.assign(crowd()[:4], 0.2)  # Person 3 steps out of view briefly...
        back = ids.assign(crowd(), 0.4)  # ...and returns.
        self.assertEqual(len({hand.key for hand in back}), 6)
        self.assertIn("Right 3", {hand.key for hand in back})


class GroupGestureTests(unittest.TestCase):
    def test_each_person_has_their_own_two_hand_gestures(self):
        engine = GestureEngine()
        ids = HandIdentities()
        started = []
        engine.bus.subscribe(Gesture.TWO_PALMS, EventKind.STARTED, started.append)
        for frame in range(8):
            engine.update(ids.assign(crowd(), frame / 30), frame / 30)
        self.assertEqual(set(engine.tracked_pairs()), {"Both", "Both 2", "Both 3"})
        self.assertEqual({event.hand for event in started}, {"Both", "Both 2", "Both 3"})
        self.assertTrue(engine.two_hands_of("Both 3").present)
        self.assertIsNotNone(engine.palm_center("Both 2"))

    def test_both_subscriptions_hear_every_person(self):
        engine = GestureEngine()
        ids = HandIdentities()
        heard = []
        engine.bus.on_hold_for(Gesture.TWO_PALMS, 0.1, heard.append, hand=BOTH)
        for frame in range(12):
            engine.update(ids.assign(crowd(), frame / 30), frame / 30)
        self.assertEqual({event.hand for event in heard}, {"Both", "Both 2", "Both 3"})

    def test_person_two_great_fireball_is_theirs(self):
        from tests.test_effects import jutsu_scene
        from tests.test_gestures import prayer_pair
        scene = jutsu_scene()

        def person_two(hands):
            return [replace(hand, track_id=hand_id(hand.handedness, 2)) for hand in hands]

        scene.run(36, *person_two(prayer_pair()))
        self.assertIn("jutsu_seal", scene.names())
        for frame in range(12):
            gap = 40 + 14 * frame
            scene.step(*person_two([make_hand("open_palm", hand="Left", offset=(-gap, 0)),
                                    make_hand("open_palm", hand="Right", offset=(gap, 0))]))
        balls = [effect for effect in scene.manager.effects if getattr(effect, "great", False)]
        self.assertEqual([ball.owner for ball in balls], ["Both 2"])


class PipelineTests(unittest.TestCase):
    def test_analyzer_skips_stale_frames_and_reports_errors(self):
        gate = threading.Event()
        seen = []

        def analyze(frame):
            gate.wait(1)
            if frame == "bad":
                raise ValueError("broken frame")
            seen.append(frame)
            return frame

        analyzer = FrameAnalyzer(analyze).start()
        try:
            analyzer.submit("a")
            time.sleep(0.05)  # "a" is in progress...
            analyzer.submit("b")
            analyzer.submit("c")  # ...so "b" is replaced by the newer "c" before it starts.
            gate.set()
            deadline = time.monotonic() + 2
            while seen != ["a", "c"] and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertEqual(seen, ["a", "c"])
            self.assertEqual(analyzer.result(1.0), "c", "only the newest result is handed out")
            self.assertIsNone(analyzer.result(0.05))
            analyzer.submit("bad")
            with self.assertRaisesRegex(ValueError, "broken frame"):
                analyzer.result(1.0)
            analyzer.submit("d")
            self.assertEqual(analyzer.result(1.0), "d", "the worker survives an error")
        finally:
            analyzer.stop()


class LightTests(unittest.TestCase):
    def test_flame_lights_the_scene_and_stays_inside_its_patch(self):
        flames = FlameField()
        light = np.zeros((360, 640, 3), np.uint8)
        flames.draw(light, (320, 220), 40, 0.5)
        ys, xs = np.nonzero(light.max(axis=2) > 8)
        self.assertTrue(len(xs))
        self.assertLess(ys.min(), 220 - 40, "tongues rise above the ball")
        self.assertGreater(light[:, :, 2].sum(), light[:, :, 0].sum(), "fire is red-orange, not blue")
        frame = np.full((360, 640, 3), 100, np.uint8)
        compositor = LightCompositor()
        buffer = compositor.begin(frame)
        buffer[:] = light
        compositor.lit = True
        compositor.compose(frame)
        self.assertGreater(int(frame[200:240, 250:300, 2].mean()), 100, "nearby surfaces are lit")
        self.assertEqual(int(frame[5, 5].mean()), 100, "far corners are untouched")

    def test_distortion_bends_only_its_region(self):
        flames = FlameField()
        frame = np.tile(np.arange(640, dtype=np.uint8)[None, :, None], (360, 1, 3))
        before = frame.copy()
        Distortion(flames.noise).haze(frame, (320, 250), 40, 1.0)
        self.assertFalse(np.array_equal(frame, before))
        self.assertTrue(np.array_equal(frame[:, :200], before[:, :200]))

    def test_window_fits_a_portrait_frame_on_screen(self):
        self.assertEqual(window_size((1280, 720), (1920, 1080)), (1280, 720))
        width, height = window_size((720, 1280), (1920, 1080))
        self.assertLessEqual(height, 1080 * 0.85 + 1)
        self.assertAlmostEqual(width / height, 720 / 1280, places=2)


if __name__ == "__main__":
    unittest.main()
