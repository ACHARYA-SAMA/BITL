"""Deterministic checks; no webcam or network required."""
import io
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch
from urllib.error import URLError
import zipfile

import cv2
import numpy as np

from ar_hands import app as main
from ar_hands import hand_tracker as tracking
from ar_hands.camera import fourcc_name


class TrackerTests(unittest.TestCase):
    def test_conversion_color_order_bounds_and_timestamp_contract(self):
        landmarks = [SimpleNamespace(x=0.5, y=0.25, z=-0.1) for _ in range(21)]
        landmarks[0] = SimpleNamespace(x=-0.1, y=1.1, z=0.0)
        landmarks[20] = SimpleNamespace(x=1.0, y=0.0, z=0.0)
        result = SimpleNamespace(hand_landmarks=[landmarks], handedness=[[
            SimpleNamespace(category_name="Right", score=0.97)]])
        native = MagicMock()
        native.detect_for_video.return_value = result
        with patch.object(tracking, "ensure_model", return_value=Path("model.task")), patch.object(
            tracking.vision.HandLandmarker, "create_from_options", return_value=native
        ) as create:
            with main.HandTracker() as tracker:
                frame = np.zeros((100, 200, 3), dtype=np.uint8)
                frame[:] = (10, 20, 30)
                hand = tracker.detect(frame, 1)[0]
                self.assertEqual(hand.landmarks_px[1], (100, 25))
                self.assertEqual(hand.landmarks_px[0], (0, 99))
                self.assertEqual(hand.bbox, (0, 0, 199, 99))
                self.assertEqual(hand.landmarks_norm[1], (0.5, 0.25, -0.1))
                self.assertEqual(hand.handedness, "Right")
                image = native.detect_for_video.call_args.args[0]
                self.assertEqual(tuple(image.numpy_view()[0, 0]), (30, 20, 10))
                with self.assertRaises(ValueError):
                    tracker.detect(frame, 1)
                with self.assertRaises(ValueError):
                    tracker.detect(frame, 0)
            tracker.close()
        native.close.assert_called_once()
        options = create.call_args.args[0]
        self.assertEqual(options.running_mode, tracking.vision.RunningMode.VIDEO)
        self.assertEqual(options.num_hands, 2)
        self.assertEqual(options.min_hand_presence_confidence, 0.7)

    def test_distance_and_landmark_constants(self):
        self.assertEqual(tracking.distance((0, 0), (3, 4)), 5.0)
        self.assertEqual(len(tracking.Landmark), 21)
        self.assertEqual(tracking.Landmark.INDEX_TIP, 8)

    def test_duplicate_handedness_is_resolved_by_position_or_dropped(self):
        def hand(label, x, score):
            points = [(x + i, 200 + 3 * i) for i in range(21)]
            return tracking.HandData(label, score, points, [(0.0, 0.0, 0.0)] * 21, (0, 0, 1, 1))

        resolved = tracking.resolve_handedness([hand("Right", 500, 0.8), hand("Right", 100, 0.9)], 0.25)
        self.assertEqual(sorted((h.handedness, h.landmarks_px[0][0]) for h in resolved),
                         [("Left", 100), ("Right", 500)])
        # Nearly identical landmarks are one hand detected twice: keep the confident one.
        twin = tracking.resolve_handedness([hand("Left", 102, 0.6), hand("Left", 100, 0.9)], 0.25)
        self.assertEqual([h.score for h in twin], [0.9])
        distinct = [hand("Left", 100, 0.9), hand("Right", 500, 0.8)]
        self.assertIs(tracking.resolve_handedness(distinct, 0.25), distinct)

    def test_fps_smoothing(self):
        with patch.object(main.time, "perf_counter", side_effect=[10.0, 10.1, 10.3]):
            fps = main.FPSCounter(0.9)
            self.assertEqual(fps.update(), 0.0)
            self.assertAlmostEqual(fps.update(), 10.0)
            self.assertAlmostEqual(fps.update(), 9.5)

    def test_clock_collisions_still_increase(self):
        app = main.App()
        with patch.object(main.time, "perf_counter_ns", return_value=0):
            self.assertEqual([app.next_timestamp() for _ in range(3)], [0, 1, 2])


class DownloadTests(unittest.TestCase):
    def test_complete_bundle_is_cached_atomically(self):
        payload = io.BytesIO()
        with zipfile.ZipFile(payload, "w") as archive:
            archive.writestr("model.tflite", b"test payload")
        data = payload.getvalue()
        response = MagicMock()
        response.__enter__.return_value = response
        response.headers = {"Content-Length": str(len(data))}
        response.read.side_effect = [data, b""]
        with TemporaryDirectory() as folder:
            config = replace(main.Config(), base_dir=Path(folder))
            with patch.object(tracking.urllib.request, "urlopen", return_value=response):
                path = tracking.ensure_model(config)
            self.assertEqual(path.read_bytes(), data)
            self.assertEqual(list(path.parent.iterdir()), [path])

    def test_truncated_transfer_is_not_cached(self):
        response = MagicMock()
        response.__enter__.return_value = response
        response.headers = {"Content-Length": "999"}
        response.read.side_effect = [b"truncated", b""]
        with TemporaryDirectory() as folder:
            config = replace(main.Config(), base_dir=Path(folder))
            with patch.object(tracking.urllib.request, "urlopen", return_value=response):
                with self.assertRaisesRegex(main.AppError, "incomplete"):
                    tracking.ensure_model(config)
            self.assertFalse(list((Path(folder) / "models").iterdir()))

    def test_failed_download_is_actionable_and_leaves_no_partial_file(self):
        with TemporaryDirectory() as folder:
            config = replace(main.Config(), base_dir=Path(folder))
            with patch.object(tracking.urllib.request, "urlopen", side_effect=URLError("offline")):
                with self.assertRaises(main.AppError) as raised:
                    tracking.ensure_model(config)
            self.assertIn(config.model_url, str(raised.exception))
            self.assertFalse(list((Path(folder) / "models").iterdir()))

    def test_existing_model_avoids_network(self):
        with TemporaryDirectory() as folder:
            config = replace(main.Config(), base_dir=Path(folder))
            path = config.base_dir / "models" / "hand_landmarker.task"
            path.parent.mkdir()
            path.write_bytes(b"existing model")
            with patch.object(tracking.urllib.request, "urlopen") as download:
                self.assertEqual(tracking.ensure_model(config), path)
            download.assert_not_called()


class AppTests(unittest.TestCase):
    def test_mirror_precedes_detection_and_rendering(self):
        app = main.App()
        frame = np.zeros((40, 80, 3), dtype=np.uint8)
        frame[:, 0] = (10, 20, 30)
        tracker = MagicMock()
        def detect(mirrored, timestamp):
            self.assertTrue(np.all(mirrored[:, -1] == (10, 20, 30)))
            self.assertFalse(mirrored[:, 0].any())
            return []
        tracker.detect.side_effect = detect
        with patch.object(app.renderer, "draw") as draw, patch.object(app.hud, "draw"):
            app.process_frame(frame, tracker)
        draw.assert_called_once()

    def test_window_close_exits_without_recreating_window(self):
        app = main.App()
        from ar_hands.camera import FramePacket
        stream, tracker = MagicMock(), MagicMock()
        tracker.__enter__.return_value = tracker
        stream.start.return_value = stream
        stream.read.return_value = FramePacket(np.zeros((80, 120, 3), dtype=np.uint8), 1, 0.)
        with patch.object(main, "CameraStream", return_value=stream), patch.object(
            main, "HandTracker", return_value=tracker
        ), patch.object(app, "process_frame", side_effect=lambda frame, tracker: frame), patch.object(
            cv2, "namedWindow"
        ), patch.object(cv2, "resizeWindow"), patch.object(cv2, "setMouseCallback"), patch.object(
            main, 'ensure_segmenter_model'
        ), patch.object(main, 'AudioManager', return_value=app.audio), patch.object(
            cv2, "imshow"
        ) as display, patch.object(cv2, "waitKeyEx", return_value=-1), patch.object(
            cv2, "getWindowProperty", return_value=0
        ), patch.object(cv2, "destroyAllWindows"):
            app.run()
        display.assert_called_once()
        stream.stop.assert_called_once()
        tracker.__exit__.assert_called_once()

    def test_windows_camera_fallback_releases_failed_candidates(self):
        failed, opened = MagicMock(), MagicMock()
        failed.isOpened.return_value = False
        opened.isOpened.return_value = True
        opened.get.return_value = 30
        with patch.object(main.sys, "platform", "win32"), patch.object(
            main.cv2, "VideoCapture", side_effect=[failed, opened]
        ) as capture:
            self.assertIs(main.App().open_camera(), opened)
        failed.release.assert_called_once()
        self.assertEqual(capture.call_args_list[0].args, (0, cv2.CAP_DSHOW))
        self.assertEqual(capture.call_args_list[1].args, (0,))

    def test_camera_requests_compressed_format_after_size_and_fps(self):
        camera = MagicMock()
        camera.isOpened.return_value = True
        camera.get.return_value = cv2.VideoWriter_fourcc(*"MJPG")
        with patch.object(main.sys, "platform", "linux"), patch.object(
            main.cv2, "VideoCapture", return_value=camera
        ):
            main.App().open_camera()
        # DirectShow ignores a format requested before the frame size.
        self.assertEqual([call.args[0] for call in camera.set.call_args_list],
                         [cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT,
                          cv2.CAP_PROP_FPS, cv2.CAP_PROP_FOURCC])
        self.assertEqual(fourcc_name(cv2.VideoWriter_fourcc(*"MJPG")), "MJPG")
        self.assertEqual(fourcc_name(22), "0x00000016")

    def test_camera_exhaustion(self):
        cameras = [MagicMock() for _ in range(3)]
        for camera in cameras:
            camera.isOpened.return_value = False
        with patch.object(main.sys, "platform", "linux"), patch.object(
            main.cv2, "VideoCapture", side_effect=cameras
        ) as capture:
            with self.assertRaisesRegex(main.AppError, "Camera not found"):
                main.App().open_camera()
        self.assertEqual([call.args for call in capture.call_args_list], [(0,), (1,), (2,)])
        for camera in cameras:
            camera.release.assert_called_once()

    def test_toggles_and_screenshot(self):
        with TemporaryDirectory() as folder:
            app = main.App(replace(main.Config(), base_dir=Path(folder)))
            frame = np.zeros((120, 240, 3), dtype=np.uint8)
            app.handle_key(ord("l"), frame)
            app.handle_key(ord("g"), frame)
            app.handle_key(ord("h"), frame)
            self.assertTrue(app.renderer.show_indices)
            self.assertTrue(app.renderer.glow)
            self.assertFalse(app.show_hud)
            app.handle_key(ord("s"), frame)
            app.handle_key(ord("s"), frame)
            saved = list((Path(folder) / "screenshots").glob("*.png"))
            self.assertEqual(len(saved), 2)
            self.assertEqual(cv2.imread(str(saved[0])).shape, frame.shape)
            self.assertFalse(app.handle_key(ord("q"), frame))
            self.assertFalse(app.handle_key(27, frame))

    def test_renderer_supports_every_overlay_combination(self):
        hand = tracking.HandData("Left", 0.97, [(30 + i * 4, 50 + i * 3) for i in range(21)],
                             [(0.5, 0.5, 0.0)] * 21, (10, 20, 130, 140))
        renderer = main.SkeletonRenderer(main.Config())
        for glow in (False, True):
            for labels in (False, True):
                renderer.glow, renderer.show_indices = glow, labels
                frame = np.zeros((180, 320, 3), dtype=np.uint8)
                renderer.draw(frame, [hand])
                self.assertTrue(frame.any())
                main.HUDRenderer(main.Config()).draw(frame, 29.5, 1)
                self.assertTrue(frame[:40].any())

    def test_failures_and_interrupt_always_release_resources(self):
        for interrupt in (False, True):
            with self.subTest(interrupt=interrupt):
                app = main.App(replace(main.Config(), max_frame_failures=3))
                from ar_hands.camera import FramePacket
                stream, tracker = MagicMock(), MagicMock()
                tracker.__enter__.return_value = tracker
                stream.start.return_value = stream
                stream.read.return_value = FramePacket(np.zeros((80, 120, 3), np.uint8), 1, 0.)
                error = KeyboardInterrupt if interrupt else ValueError('bad frame')
                with patch.object(main, "CameraStream", return_value=stream), patch.object(
                    main, "HandTracker", return_value=tracker
                ), patch.object(app, 'process_frame', side_effect=error), patch.object(
                    main, 'ensure_segmenter_model'
                ), patch.object(main, 'AudioManager', return_value=app.audio), patch.object(
                    cv2, 'resizeWindow'
                ), patch.object(cv2, 'setMouseCallback'), patch.object(cv2, "namedWindow"), patch.object(cv2, "imshow"), patch.object(
                    cv2, "waitKeyEx", return_value=-1
                ), patch.object(cv2, "getWindowProperty", return_value=1), patch.object(
                    cv2, "destroyAllWindows"
                ) as destroy:
                    if interrupt:
                        with self.assertRaises(KeyboardInterrupt):
                            app.run()
                    else:
                        with self.assertRaisesRegex(main.AppError, "consecutive"):
                            app.run()
                        self.assertEqual(stream.read.call_count, 3)
                stream.stop.assert_called_once()
                tracker.__exit__.assert_called_once()
                destroy.assert_called_once()


if __name__ == "__main__":
    unittest.main()
