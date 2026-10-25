"""Gesture engine tests on synthetic, anatomically plausible landmarks.

Run from the project root:  python -m unittest -v
"""
from dataclasses import replace
import math
import random
import time
import unittest
from ar_hands.config import GestureConfig
from ar_hands.gestures.engine import GestureEngine
from ar_hands.gestures.events import BOTH, EventBus, EventKind, GestureEvent
from ar_hands.gestures.features import (Finger, FingerState, extract_features, hand_scale,
                                        palm_center, palm_normal_z, pinch_ratio,
                                        pinch_strength, roll_angle)
from ar_hands.gestures.motion import MotionTracker
from ar_hands.gestures.recognizer import RULES, recognize
from ar_hands.gestures.stabilizer import GestureStabilizer, SchmittTrigger
from ar_hands.gestures.types import Gesture
from ar_hands.hand_tracker import Landmark
# The synthetic hand model lives in the package so the in-app guide can draw
# its pose icons from the same shapes these tests verify.
from ar_hands.handpose import (CURL, EXT, FRAME, PARTIAL, PINCHED, POSES, RELAXED,  # noqa: F401
                               edge_on, local_pose as _local_pose, make_hand)

EXPECTED = {pose: Gesture[pose.upper()] for pose in POSES}
CONFIG = GestureConfig()
DT = 1 / 30


def pinch_with_gap(gap, **kwargs):
    """A pinch pose whose thumb tip sits ``gap`` hand scales beside the index tip."""
    local = _local_pose("pinch")
    tip_x, tip_y = local[Landmark.INDEX_TIP]
    local[Landmark.THUMB_TIP] = (tip_x - gap, tip_y)
    return make_hand("pinch", local=local, **kwargs)


def prayer_pair(offset=(0, 0)):
    """Two open hands pressed together edge-on, as the camera sees a prayer seal."""
    return [make_hand("open_palm", local=edge_on(), hand=hand, offset=offset)
            for hand in ("Left", "Right")]


def jittered(hand, sigma_px, rng):
    """Add Gaussian landmark noise, like MediaPipe's frame-to-frame jitter."""
    return replace(hand, landmarks_px=[(round(x + rng.gauss(0, sigma_px)),
                                        round(y + rng.gauss(0, sigma_px)))
                                       for x, y in hand.landmarks_px])


def classify(hand, is_pinched=None):
    return recognize(extract_features(hand, CONFIG), is_pinched, CONFIG).gesture


def event(kind, gesture=Gesture.POINT, hand="Right", duration=0.0):
    return GestureEvent(kind, gesture, hand, (0.0, 0.0), duration, 0.9, 0.0)


class FeatureTests(unittest.TestCase):
    def test_scale_center_and_roll(self):
        hand = make_hand("open_palm", scale=137, offset=(40, -25))
        self.assertAlmostEqual(hand_scale(hand), 137, delta=1.0)
        center = palm_center(hand)
        self.assertAlmostEqual(center[0], 680, delta=1.0)
        self.assertAlmostEqual(center[1], 335, delta=1.0)
        for roll in (-150, -45, 0, 30, 120, 180):
            measured = roll_angle(make_hand("open_palm", roll=roll))
            self.assertLess(abs((measured - roll + 180) % 360 - 180), 1.0)

    def test_pinch_ratio_and_strength(self):
        self.assertLess(pinch_ratio(make_hand("pinch")), 0.1)
        self.assertGreater(pinch_ratio(make_hand("open_palm")), 1.0)
        self.assertAlmostEqual(pinch_ratio(pinch_with_gap(0.3)), 0.3, delta=0.02)
        self.assertEqual(pinch_strength(0.05, CONFIG), 1.0)
        self.assertEqual(pinch_strength(0.9, CONFIG), 0.0)
        self.assertAlmostEqual(pinch_strength(0.35, CONFIG), 0.5)

    def test_finger_states(self):
        open_hand = extract_features(make_hand("open_palm"), CONFIG)
        fist = extract_features(make_hand("fist"), CONFIG)
        for finger in Finger:
            self.assertIs(open_hand.state(finger), FingerState.EXTENDED)
            self.assertIs(fist.state(finger), FingerState.CURLED)
            self.assertGreater(open_hand.openness(finger), fist.openness(finger))
        partial = extract_features(make_hand("peace", flexions=(EXT, EXT, PARTIAL, CURL)), CONFIG)
        self.assertIs(partial.state(Finger.RING), FingerState.PARTIAL)

    def test_palm_normal_sign_follows_handedness_in_the_mirrored_view(self):
        for label in ("Left", "Right"):
            self.assertGreater(palm_normal_z(make_hand("open_palm", hand=label)), 0.5)
        # The same pixels under the other label are the back of that hand.
        back = replace(make_hand("open_palm", hand="Left"), handedness="Right")
        self.assertLess(palm_normal_z(back), -0.5)

    def test_tiny_hand_has_no_features(self):
        self.assertIsNone(extract_features(make_hand("open_palm", scale=5), CONFIG))
        self.assertIs(recognize(None).gesture, Gesture.NONE)


class RecognizerTests(unittest.TestCase):
    def test_every_pose_for_both_hands(self):
        for pose, expected in EXPECTED.items():
            for label in ("Left", "Right"):
                with self.subTest(pose=pose, hand=label):
                    result = recognize(extract_features(make_hand(pose, hand=label), CONFIG))
                    self.assertIs(result.gesture, expected)
                    self.assertGreater(result.confidence, 0.6)
                    self.assertLessEqual(result.confidence, 1.0)

    def test_scale_invariance_from_50_to_300_pixels(self):
        for scale in (50, 75, 100, 150, 220, 300):
            for pose, expected in EXPECTED.items():
                with self.subTest(pose=pose, scale=scale):
                    self.assertIs(classify(make_hand(pose, scale=scale)), expected)

    def test_roll_invariance_through_a_full_turn(self):
        for roll in range(-180, 181, 15):
            for pose, expected in EXPECTED.items():
                if pose == "thumbs_up":
                    continue
                for label in ("Left", "Right"):
                    with self.subTest(pose=pose, roll=roll, hand=label):
                        self.assertIs(classify(make_hand(pose, roll=roll, hand=label)), expected)

    def test_thumbs_up_documented_roll_range(self):
        # Thumbs-up means the thumb points UP on screen, so it cannot be
        # roll-invariant. Its natural pose tilts the thumb ~15 degrees, and the
        # limit is thumbs_up_max_tilt_deg = 45, so rolls of -40..+25 always
        # pass on both hands. Sideways and thumbs-down are rejected (not FIST).
        for roll in range(-40, 26, 5):
            for label in ("Left", "Right"):
                # Mirror the roll for the left hand: its natural tilt is mirrored too.
                signed = roll if label == "Right" else -roll
                with self.subTest(roll=signed, hand=label):
                    self.assertIs(classify(make_hand("thumbs_up", roll=signed, hand=label)),
                                  Gesture.THUMBS_UP)
        for roll in (-90, 90, 180):
            with self.subTest(roll=roll):
                self.assertIs(classify(make_hand("thumbs_up", roll=roll)), Gesture.NONE)

    def test_translation_and_noise_robustness(self):
        rng = random.Random(1234)
        for pose, expected in EXPECTED.items():
            hits = 0
            for _ in range(100):
                roll = rng.uniform(-20, 20) if pose == "thumbs_up" else rng.uniform(-180, 180)
                hand = make_hand(pose, roll=roll, offset=(rng.uniform(-400, 400), rng.uniform(-150, 150)))
                hits += classify(jittered(hand, 3.0, rng)) is expected  # 3% of hand scale.
            with self.subTest(pose=pose):
                self.assertGreaterEqual(hits, 97)

    def test_fist_never_reads_as_pinch_or_thumbs_up(self):
        fist = make_hand("fist")
        features = extract_features(fist, CONFIG)
        # A fist's tucked tips sit inside the pinch hysteresis band (0.25-0.35),
        # so even with the latch held closed, the reach guard must reject PINCH.
        self.assertLess(features.pinch_ratio, CONFIG.pinch_exit)
        self.assertIs(classify(fist, is_pinched=True), Gesture.FIST)
        for roll in range(-180, 181, 30):
            self.assertIs(classify(make_hand("fist", roll=roll)), Gesture.FIST)

    def test_partial_ring_is_neither_peace_nor_three(self):
        between = make_hand("peace", flexions=(EXT, EXT, PARTIAL, CURL))
        self.assertIs(classify(between), Gesture.NONE)

    def test_rule_order_resolves_overlaps(self):
        ok_hand = extract_features(make_hand("ok"), CONFIG)
        without_ok = [rule for rule in RULES if rule.gesture is not Gesture.OK]
        self.assertIs(recognize(ok_hand, rules=without_ok).gesture, Gesture.PINCH)
        self.assertEqual(len({rule.gesture for rule in RULES}), len(RULES))


class StabilizerTests(unittest.TestCase):
    def run_stream(self, stream, stabilizer=None):
        stabilizer = stabilizer or GestureStabilizer(CONFIG)
        changes = []
        for frame, gesture in enumerate(stream, start=1):
            transition = stabilizer.update(gesture, 0.9)
            if transition.ended or transition.started:
                changes.append((frame, transition.ended, transition.started))
        return changes, stabilizer

    def test_rejects_a_two_frame_flicker(self):
        P, X = Gesture.POINT, Gesture.PEACE
        changes, stabilizer = self.run_stream([P] * 6 + [X, X] + [P] * 6)
        self.assertEqual(changes, [(4, None, P)])
        self.assertIs(stabilizer.active, P)

    def test_entry_and_exit_take_documented_frames(self):
        P, N = Gesture.POINT, Gesture.NONE
        changes, _ = self.run_stream([P] * 10 + [N] * 10)
        # Enter: 4 winning frames. Exit: the vote flips on the 3rd NONE frame,
        # then the 3-frame grace expires on the 6th.
        self.assertEqual(changes, [(4, None, P), (16, P, None)])

    def test_direct_switch_hands_over_in_one_frame(self):
        P, F = Gesture.POINT, Gesture.FIST
        changes, _ = self.run_stream([P] * 10 + [F] * 10)
        self.assertEqual(changes, [(4, None, P), (16, P, F)])

    def test_alternating_noise_does_not_chatter(self):
        P, X = Gesture.POINT, Gesture.NONE
        changes, stabilizer = self.run_stream([P] * 6 + [X, P] * 20)
        self.assertEqual(changes, [(4, None, P)])
        self.assertIs(stabilizer.active, P)

    def test_lost_hand_ends_after_grace(self):
        _, stabilizer = self.run_stream([Gesture.FIST] * 6)
        for _ in range(CONFIG.lost_grace_frames):
            self.assertIsNone(stabilizer.update_missing().ended)
            self.assertIs(stabilizer.active, Gesture.FIST)
        self.assertIs(stabilizer.update_missing().ended, Gesture.FIST)
        self.assertTrue(stabilizer.lost)

    def test_schmitt_trigger_does_not_chatter(self):
        latch = SchmittTrigger(0.25, 0.35)
        toggles, state = 0, False
        values = [0.30, 0.34, 0.26, 0.33] * 5 + [0.20] + [0.26, 0.34, 0.30, 0.27] * 5 + [0.40]
        for value in values:
            if latch.update(value) != state:
                toggles, state = toggles + 1, not state
        self.assertEqual(toggles, 2)
        self.assertFalse(latch.state)


class EventBusTests(unittest.TestCase):
    def test_throwing_callback_is_logged_and_isolated(self):
        bus = EventBus()
        received = []

        def broken(_event):
            raise RuntimeError("boom")

        failing = bus.subscribe(None, None, broken)
        bus.subscribe(None, None, received.append)
        with self.assertLogs("ar_hands.gestures.events", level="ERROR") as logs:
            for _ in range(3):
                bus.dispatch(event(EventKind.HELD))
        self.assertEqual(len(received), 3)
        self.assertEqual(failing.errors, 3)
        self.assertEqual(len(logs.records), 1)  # Repeats are rate-limited.
        self.assertIn("boom", logs.output[0])

    def test_filters_and_unsubscribe(self):
        bus = EventBus()
        received = []
        subscription = bus.subscribe(Gesture.POINT, EventKind.STARTED, received.append, hand="Left")
        bus.dispatch(event(EventKind.STARTED, hand="Left"))
        bus.dispatch(event(EventKind.STARTED, hand="Right"))
        bus.dispatch(event(EventKind.HELD, hand="Left"))
        bus.dispatch(event(EventKind.STARTED, Gesture.FIST, hand="Left"))
        self.assertEqual(len(received), 1)
        self.assertEqual(bus.unsubscribe(subscription), 1)
        self.assertEqual(len(bus), 0)
        bus.dispatch(event(EventKind.STARTED, hand="Left"))
        self.assertEqual(len(received), 1)

    def test_unsubscribing_during_dispatch_skips_the_removed_callback(self):
        bus = EventBus()
        removed, received = [], []
        bus.subscribe(None, None, lambda _event: bus.unsubscribe(later))
        later = bus.subscribe(None, None, removed.append)
        bus.subscribe(None, None, received.append)
        bus.dispatch(event(EventKind.HELD))
        self.assertEqual(removed, [])
        self.assertEqual(len(received), 1)
        self.assertEqual(len(bus), 2)

    def test_on_hold_for_fires_once_per_activation_with_progress(self):
        bus = EventBus()
        fired = []
        bus.on_hold_for(Gesture.OPEN_PALM, 1.0, fired.append)
        for activation in range(2):
            bus.dispatch(event(EventKind.STARTED, Gesture.OPEN_PALM))
            self.assertEqual(bus.hold_progress("Right", Gesture.OPEN_PALM), 0.0)
            bus.dispatch(event(EventKind.HELD, Gesture.OPEN_PALM, duration=0.5))
            self.assertAlmostEqual(bus.hold_progress("Right", Gesture.OPEN_PALM), 0.5)
            for duration in (1.0, 1.2, 2.0):
                bus.dispatch(event(EventKind.HELD, Gesture.OPEN_PALM, duration=duration))
            self.assertEqual(bus.hold_progress("Right", Gesture.OPEN_PALM), 1.0)
            bus.dispatch(event(EventKind.ENDED, Gesture.OPEN_PALM, duration=2.0))
            self.assertEqual(len(fired), activation + 1)
        self.assertEqual(bus.hold_progress("Right", Gesture.OPEN_PALM), 0.0)

    def test_hold_needs_a_start_after_subscribing(self):
        bus = EventBus()
        fired = []
        bus.dispatch(event(EventKind.STARTED, Gesture.FIST))
        bus.on_hold_for(Gesture.FIST, 0.5, fired.append)
        bus.dispatch(event(EventKind.HELD, Gesture.FIST, duration=3.0))
        self.assertEqual(fired, [])


class MotionTests(unittest.TestCase):
    def feed(self, positions, scales=None, tracker=None):
        tracker = tracker or MotionTracker(CONFIG)
        swipes = []
        for frame, position in enumerate(positions, start=1):
            t = frame * DT
            tracker.update(t, position, scales[frame - 1] if scales else 100.0)
            swipe = tracker.detect_swipe(t)
            if swipe:
                swipes.append((t, swipe[0]))
        return swipes, tracker

    def test_swipe_directions(self):
        speed = 10 * 100 * DT  # 10 hand scales per second at scale 100.
        for (dx, dy), expected in (((1, 0), Gesture.SWIPE_RIGHT), ((-1, 0), Gesture.SWIPE_LEFT),
                                   ((0, -1), Gesture.SWIPE_UP), ((0, 1), Gesture.SWIPE_DOWN)):
            with self.subTest(expected=expected):
                swipes, _ = self.feed([(640 + dx * speed * i, 360 + dy * speed * i) for i in range(8)])
                self.assertEqual([gesture for _, gesture in swipes], [expected])

    def test_cooldown_spaces_out_swipes_and_blocks_the_return_stroke(self):
        speed = 10 * 100 * DT
        out = [(200 + speed * i, 360) for i in range(10)]
        back = [(out[-1][0] - speed * i, 360) for i in range(1, 10)]
        swipes, _ = self.feed(out + back)
        self.assertEqual([gesture for _, gesture in swipes], [Gesture.SWIPE_RIGHT])
        swipes, _ = self.feed([(100 + speed * i, 360) for i in range(45)])
        times = [t for t, _ in swipes]
        self.assertGreaterEqual(len(times), 2)
        for earlier, later in zip(times, times[1:]):
            self.assertGreaterEqual(later - earlier, CONFIG.swipe_cooldown_s - 1e-9)

    def test_slow_or_diagonal_motion_is_not_a_swipe(self):
        slow = 2 * 100 * DT
        swipes, _ = self.feed([(640 + slow * i, 360) for i in range(30)])
        self.assertEqual(swipes, [])
        fast = 10 * 100 * DT / math.sqrt(2)
        swipes, _ = self.feed([(640 + fast * i, 360 + fast * i) for i in range(10)])
        self.assertEqual(swipes, [])

    def test_single_glitch_frame_never_swipes(self):
        positions = [(640, 360)] * 10 + [(900, 360)] + [(640, 360)] * 10
        swipes, _ = self.feed(positions)
        self.assertEqual(swipes, [])

    def test_velocity_is_distance_invariant(self):
        results = []
        for scale in (50.0, 200.0):
            step = 3 * scale * DT  # 3 hand scales per second.
            _, tracker = self.feed([(640 + step * i, 360) for i in range(6)], [scale] * 6)
            results.append(tracker)
        near, far = results
        self.assertAlmostEqual(near.velocity[0], 3.0, places=3)
        self.assertAlmostEqual(far.velocity[0], 3.0, places=3)
        self.assertAlmostEqual(far.velocity_px[0] / near.velocity_px[0], 4.0, places=3)

    def test_push_amount_sign(self):
        growing = [100 + 4 * i for i in range(8)]
        _, tracker = self.feed([(640, 360)] * 8, growing)
        self.assertGreater(tracker.push_amount, 0.5)
        _, tracker = self.feed([(640, 360)] * 8, list(reversed(growing)))
        self.assertLess(tracker.push_amount, -0.5)


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.engine = GestureEngine(CONFIG)
        self.t = 0.0

    def step(self, *hands):
        self.t += DT
        return self.engine.update(list(hands), self.t)

    def test_point_lifecycle_and_lost_hand_grace(self):
        for frame in range(1, 11):
            kinds = [(e.kind, e.gesture) for e in self.step(make_hand("point"))]
            expected = ([] if frame < 4 else [(EventKind.STARTED, Gesture.POINT)] if frame == 4
                        else [(EventKind.HELD, Gesture.POINT)])
            self.assertEqual(kinds, expected, f"frame {frame}")
        self.assertTrue(self.engine.is_active("Right", Gesture.POINT))
        tip = make_hand("point").landmarks_px[Landmark.INDEX_TIP]
        self.assertEqual(self.engine.position("Right"), tip)  # POINT anchors at the fingertip.
        for missing in range(1, CONFIG.lost_grace_frames + 1):
            self.assertEqual([e.kind for e in self.step()], [EventKind.HELD])
        ended = self.step()
        self.assertEqual([(e.kind, e.gesture) for e in ended], [(EventKind.ENDED, Gesture.POINT)])
        self.assertAlmostEqual(ended[0].duration, (6 + CONFIG.lost_grace_frames + 1) * DT)
        self.assertEqual(self.engine.tracked_hands(), ())

    def test_single_frame_misdetections_never_fire(self):
        started = []
        self.engine.bus.subscribe(None, EventKind.STARTED, started.append)
        stream = (["point"] * 8 + ["peace"] + ["point"] * 5 + ["fist", "fist"]
                  + ["point"] * 5 + ["thumbs_up"] + ["point"] * 4)
        for pose in stream:
            self.step(make_hand(pose))
        self.assertEqual([(e.gesture, e.hand) for e in started], [(Gesture.POINT, "Right")])

    def test_jittered_hand_stays_one_steady_gesture(self):
        rng = random.Random(99)
        events = []
        for _ in range(150):
            events += self.step(jittered(make_hand("peace", roll=20), 4.0, rng))
        lifecycle = [(e.kind, e.gesture) for e in events if e.kind is not EventKind.HELD]
        self.assertEqual(lifecycle, [(EventKind.STARTED, Gesture.PEACE)])

    def test_pinch_hysteresis_holds_inside_the_band(self):
        for _ in range(6):
            self.step(pinch_with_gap(0.30))  # Inside the band, never latched yet.
        # With the other fingers curled, an unlatched "almost pinch" is a fist.
        self.assertIs(self.engine.active("Right"), Gesture.FIST)
        for _ in range(6):
            self.step(pinch_with_gap(0.03))
        self.assertIs(self.engine.active("Right"), Gesture.PINCH)
        for _ in range(20):
            self.step(pinch_with_gap(0.30))  # Same distance as before, but now latched.
        self.assertIs(self.engine.active("Right"), Gesture.PINCH)
        for _ in range(8):
            self.step(pinch_with_gap(0.45))  # Beyond pinch_exit: the latch opens.
        self.assertIsNot(self.engine.active("Right"), Gesture.PINCH)

    def test_swipes_fire_for_an_open_hand_but_not_while_drawing(self):
        step_px = 10.5 * 100 * DT
        for pose, expected in (("open_palm", [Gesture.SWIPE_RIGHT]), ("point", []),
                               ("rock", [])):  # Not suppressed, but not an open palm either.
            with self.subTest(pose=pose):
                engine, swipes = GestureEngine(CONFIG), []
                engine.bus.subscribe(None, EventKind.STARTED, swipes.append)
                for frame in range(1, 21):
                    engine.update([make_hand(pose, offset=(-500 + step_px * frame, 0))], frame * DT)
                self.assertEqual([e.gesture for e in swipes if e.gesture.name.startswith("SWIPE")],
                                 expected)
                swipe_events = [e for e in engine.last_events if e.gesture.name.startswith("SWIPE")]
                self.assertEqual(swipe_events, [])  # Instant: it is not still "active" later.

    def test_two_hand_prayer_and_fists_use_both(self):
        for _ in range(6):
            self.step(*prayer_pair())
        self.assertIs(self.engine.active(BOTH), Gesture.PRAYER)
        engine, started = GestureEngine(CONFIG), []
        engine.bus.subscribe(None, EventKind.STARTED, started.append)
        fists = [make_hand("fist", hand="Left", offset=(-250, 0)),
                 make_hand("fist", hand="Right", offset=(250, 0))]
        for frame in range(1, 7):
            engine.update(fists, frame * DT)
        self.assertIn((Gesture.TWO_FISTS, BOTH), [(e.gesture, e.hand) for e in started])
        self.assertAlmostEqual(engine.hands_distance, 5.0, delta=0.5)

    def test_hands_apart_fires_once_and_distance_grows(self):
        apart, distances = [], []
        self.engine.bus.subscribe(Gesture.HANDS_APART, None, apart.append)
        for frame in range(12):
            gap = 40 + 12 * frame  # Each wrist moves 3.6 scales/s, apart at 7.2.
            self.step(make_hand("open_palm", hand="Left", offset=(-gap, 0)),
                      make_hand("open_palm", hand="Right", offset=(gap, 0)))
            distances.append(self.engine.hands_distance)
        self.assertEqual([e.kind for e in apart], [EventKind.STARTED, EventKind.ENDED])
        self.assertEqual({e.hand for e in apart}, {BOTH})
        self.assertEqual(distances, sorted(distances))
        self.assertGreater(self.engine.two_hands.separation_speed, CONFIG.apart_speed)

    def test_hold_progress_and_on_hold_for(self):
        fired, progress = [], []
        self.engine.bus.on_hold_for(Gesture.OPEN_PALM, 0.5, fired.append)
        for _ in range(30):
            self.step(make_hand("open_palm"))
            progress.append(self.engine.hold_progress("Right", Gesture.OPEN_PALM))
        self.assertEqual(len(fired), 1)
        self.assertGreaterEqual(fired[0].duration, 0.5)
        self.assertEqual(progress, sorted(progress))
        self.assertEqual(progress[-1], 1.0)
        self.assertTrue(any(0.3 < value < 0.7 for value in progress))

    def test_broken_subscriber_never_crashes_the_engine(self):
        def broken(_event):
            raise ValueError("subscriber bug")

        self.engine.bus.subscribe(None, None, broken)
        with self.assertLogs("ar_hands.gestures.events", level="ERROR"):
            for _ in range(8):
                self.step(make_hand("fist"))
        self.assertIs(self.engine.active("Right"), Gesture.FIST)

    def test_reset_ends_active_gestures(self):
        for _ in range(6):
            self.step(make_hand("fist"))
        events = self.engine.reset(self.t)
        self.assertEqual([(e.kind, e.gesture) for e in events], [(EventKind.ENDED, Gesture.FIST)])
        self.assertEqual(self.engine.tracked_hands(), ())

    def test_update_costs_well_under_one_millisecond(self):
        rng = random.Random(5)
        frames = [[jittered(make_hand("point", offset=(200, 0)), 2.0, rng),
                   jittered(make_hand("open_palm", hand="Left", offset=(-200, 0)), 2.0, rng)]
                  for _ in range(60)]
        engine, samples = GestureEngine(CONFIG), []
        for frame in range(400):
            start = time.perf_counter()
            engine.update(frames[frame % len(frames)], (frame + 1) * DT)
            samples.append(time.perf_counter() - start)
        samples.sort()
        median_ms = samples[len(samples) // 2] * 1000
        self.assertLess(median_ms, 1.0, f"median {median_ms:.3f} ms for two hands")


if __name__ == "__main__":
    unittest.main()
