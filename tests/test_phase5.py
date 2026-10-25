"""Regression coverage for the desktop integration and Phase 5 services."""
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import time
import unittest
import threading
import numpy as np
from ar_hands.app import App
from ar_hands.config import Config
import ar_hands.config as config_module
from ar_hands.cli import parser, build_config
from ar_hands.settings import Settings, SettingsStore, validate
from ar_hands.effects.manager import Mode
from ar_hands.handpose import make_hand
from ar_hands.camera import CameraStream, CameraThreadError
from ar_hands.profiler import Profiler
from ar_hands.recorder import Recorder
from ar_hands.window import fit
from ar_hands.effects.domains import UnlimitedVoid, MalevolentShrine
from ar_hands.effects.snap import DustEffect
from ar_hands.effects.susanoo import SusanooEffect
from ar_hands.audio import AudioManager
from ar_hands.quality import AdaptiveQuality, QualityStep
from unittest.mock import patch


class IntegrationTests(unittest.TestCase):
    def test_opening_menu_does_not_click_with_already_held_pinch(self):
        app = App()
        hand = make_hand('pinch')
        box = app.menu.layout((1280, 720))['show_hud']
        point = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        tip = hand.landmarks_px[8]
        hand = make_hand('pinch', offset=(point[0] - tip[0], point[1] - tip[1]))
        for i in range(15):
            app._update_interaction([hand], i / 30, (1280, 720))
        app.open_menu()
        app.overlay_input([hand], 1., (1280, 720))
        self.assertTrue(app.show_hud)
        app.close_desktop()

    def test_camera_resize_during_dust_keeps_pipeline_working(self):
        app = App()
        class Tracker:
            def detect(self, frame, timestamp):
                return []
        with patch.object(app.segmenter, 'mask', side_effect=lambda f, t: np.ones(f.shape[:2], np.float32)):
            app.process_frame(np.zeros((240, 320, 3), np.uint8), Tracker())
            app.snap.trigger(time.perf_counter())
            app.process_frame(np.zeros((240, 320, 3), np.uint8), Tracker())
            app.process_frame(np.zeros((360, 640, 3), np.uint8), Tracker())
        self.assertFalse(app.snap.active)
        app.close_desktop()

    def test_new_jutsu_and_domain_gestures_dispatch(self):
        for mode, pose, expected in ((Mode.JUTSU, 'ok', 'rasenshuriken'),
                                      (Mode.JJK, 'crossed', 'unlimited_void')):
            app = App()
            app.switch_mode(mode, 0)
            for i in range(50):
                app._update_interaction([make_hand(pose)], i / 30, (640, 360))
            self.assertGreater(app.effects.count(expected), 0, expected)
            app.close_desktop()

    def test_both_fists_summon_susanoo(self):
        app = App()
        app.switch_mode(Mode.JUTSU, 0)
        hands = [make_hand('fist', hand='Left', offset=(-200, 0)),
                 make_hand('fist', hand='Right', offset=(200, 0))]
        for i in range(50):
            app._update_interaction(hands, i / 30, (640, 360))
        self.assertTrue(app.susanoo.active)
        app.close_desktop()

    def test_two_palms_open_menu_in_every_mode(self):
        for mode in Mode:
            app = App()
            app.switch_mode(mode, 0)
            hands = [make_hand('open_palm', hand='Left', offset=(-250, 0)),
                     make_hand('open_palm', hand='Right', offset=(250, 0))]
            for i in range(50):
                app._update_interaction(hands, i / 30, (1280, 720))
            self.assertTrue(app.menu.visible, mode)
            self.assertTrue(app.modes.suspended)
            app.close_desktop()

    def test_two_palms_menu_does_not_clear_existing_drawing(self):
        app = App()
        app.canvas.begin_stroke('Right', app.brush_color, 6)
        app.canvas.add_point('Right', (10, 10))
        app.canvas.add_point('Right', (50, 50))
        app.canvas.end_stroke('Right')
        hands = [make_hand('open_palm', hand='Left', offset=(-250, 0)),
                 make_hand('open_palm', hand='Right', offset=(250, 0))]
        for i in range(50):
            app._update_interaction(hands, i / 30, (1280, 720))
        self.assertTrue(app.menu.visible)
        self.assertEqual(len(app.canvas.finished), 1)
        app.close_desktop()

    def test_blue_and_red_fuse_into_purple(self):
        app = App()
        app.switch_mode(Mode.JJK, 0)
        for i in range(45):
            hands = [make_hand('point', hand='Left', offset=(-220, 0)),
                     make_hand('peace', hand='Right', offset=(220, 0))]
            app._update_interaction(hands, i / 30, (1280, 720))
        self.assertEqual({orb.kind for orb in app.cursed.orbs.values()}, {'blue', 'red'})
        for i in range(45, 100):
            gap = max(0, 220 - (i - 45) * 8)
            hands = [make_hand('point', hand='Left', offset=(-gap, 0)),
                     make_hand('peace', hand='Right', offset=(gap, 0))]
            app._update_interaction(hands, i / 30, (1280, 720))
        self.assertGreater(app.effects.count('hollow_purple'), 0)
        app.close_desktop()

    def test_renderers_handle_person_mask_and_camera_resize(self):
        for factory in (lambda: UnlimitedVoid((160, 120), 0),
                        lambda: MalevolentShrine((160, 120), 0), lambda: SusanooEffect(0)):
            app = App()
            effect = factory()
            app.effects.spawn(effect)
            for width, height in ((320, 240), (640, 360)):
                camera = np.full((height, width, 3), 30, np.uint8)
                person = np.zeros((height, width), np.float32)
                person[height//3:, width//3:2*width//3] = 1
                frame = camera.copy()
                app.effects.update(.03, [], [], 1.2, (width, height), camera, person)
                app.effects.render(frame)
                self.assertGreater(np.count_nonzero(frame != camera), 1000)
            app.close_desktop()

    def test_menu_dwell_and_pinch_slider(self):
        from ar_hands.menu import Menu
        values = {'glow': False, 'volume': .7}
        menu = Menu(Config(), lambda key: values.get(key), lambda key, value: values.__setitem__(key, value))
        menu.visible = True
        box = menu.layout((1280, 720))['glow']
        point = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        menu.pointer(point, False, 0, (1280, 720))
        menu.pointer(point, False, .9, (1280, 720))
        self.assertTrue(values['glow'])
        menu.pointer(point, False, 2, (1280, 720))
        self.assertTrue(values['glow'])  # One dwell per entry, never oscillates.
        box = menu.layout((1280, 720))['volume']
        menu.pointer((box[2] - 12, box[3] - 10), True, 3, (1280, 720))
        self.assertEqual(values['volume'], 1.)

    def test_fourth_mode_is_reachable_and_has_effects(self):
        app = App()
        app.handle_key(ord('4'), None)
        self.assertEqual(app.modes.mode, Mode.JJK)
        for i in range(55):
            app._update_interaction([make_hand('point', scale=100)], i / 30, (1280, 720))
        self.assertGreater(app.effects.count('blue'), 0)
        app.modes.close()

    def test_menu_captures_drawing_and_restores_controls(self):
        app = App()
        app.handle_key(9, None)
        self.assertTrue(app.modes.suspended)
        for i in range(40):
            hand = make_hand('point', scale=100, offset=(i * 2, 0))
            app._update_interaction([hand], i / 30, (1280, 720))
        self.assertFalse(app.canvas.active)
        self.assertFalse(app.canvas.finished)
        self.assertTrue(app.handle_key(27, None))
        self.assertFalse(app.modes.suspended)
        app.modes.close()

    def test_arrow_keys_do_not_quit(self):
        app = App()
        for code in (0xFF51, 0x250000, 0xF702, 0x01000012):
            self.assertTrue(app.handle_key(code, None))

    def test_capital_f_is_fullscreen_lower_f_is_fade(self):
        app = App()
        old = app.canvas.fade_enabled
        app.handle_key(ord('F'), None)
        self.assertEqual(app.canvas.fade_enabled, old)
        app.handle_key(ord('f'), None)
        self.assertNotEqual(app.canvas.fade_enabled, old)


class ServiceTests(unittest.TestCase):
    def test_recorder_reports_final_flush_failure(self):
        class Writer:
            def isOpened(self):
                return True
            def write(self, frame):
                pass
            def release(self):
                raise OSError('final flush failed')
        with TemporaryDirectory() as temp:
            recorder = Recorder(Path(temp), writer_factory=lambda *args: Writer())
            recorder.start((16, 16), 10, 0)
            recorder.write(np.zeros((16, 16, 3), np.uint8), 0)
            with self.assertRaisesRegex(RuntimeError, 'final flush failed'):
                recorder.stop()

    def test_installed_package_uses_writable_user_resources(self):
        with TemporaryDirectory() as temp, patch.object(config_module, '__file__',
                str(Path(temp) / 'site-packages' / 'ar_hands' / 'config.py')), patch(
                'ar_hands.settings.settings_dir', return_value=Path(temp) / 'user'):
            self.assertEqual(config_module.resource_dir(), Path(temp) / 'user' / 'resources')
            self.assertEqual(config_module.output_root(), Path(temp) / 'user' / 'output')
    def test_audio_missing_or_disabled_is_silent(self):
        with patch.object(AudioManager, '_import_mixer', return_value=None):
            audio = AudioManager(Config())
            self.assertFalse(audio.available)
            self.assertFalse(audio.play('whoosh'))
            audio.on_hook('fireball_throw', {'hand': 'Left'})
            audio.close()

    def test_quality_waits_then_restores_with_headroom(self):
        state = [1]
        quality = AdaptiveQuality([QualityStep('test', lambda: state.__setitem__(0, .5),
                                  lambda: state.__setitem__(0, 1))], smoothing=0)
        quality.update(0, 100)
        quality.update(1.9, 100)
        self.assertEqual(state[0], 1)
        quality.update(2.1, 100)
        self.assertEqual(state[0], .5)
        quality.update(3, 10)
        quality.update(8.1, 10)
        self.assertEqual(state[0], 1)

    def test_camera_dimensions_are_validated(self):
        for flag, value in (('--width', '0'), ('--height', '-1'), ('--fps', '0'), ('--camera', '-2')):
            with self.subTest(flag=flag), self.assertRaises(ValueError):
                build_config(parser().parse_args([flag, value]), Settings())

    def test_recorder_stop_does_not_deadlock_after_encoder_failure(self):
        class Writer:
            def isOpened(self):
                return True
            def write(self, frame):
                raise OSError('encoder failed')
            def release(self):
                pass
        with TemporaryDirectory() as temp:
            r = Recorder(Path(temp), queue_size=1, writer_factory=lambda *args: Writer())
            r.start((16, 16), 10, 0)
            r.write(np.zeros((16, 16, 3), np.uint8), 0)
            r._thread.join(.5)
            r._queue.put_nowait((np.zeros((16, 16, 3), np.uint8), 1))
            errors = []
            def finish():
                try:
                    r.stop()
                except RuntimeError as exc:
                    errors.append(str(exc))
            stop = threading.Thread(target=finish, daemon=True)
            stop.start()
            stop.join(.5)
            self.assertFalse(stop.is_alive(), 'stop blocks on a full queue with a dead encoder')
            self.assertTrue(errors)
            self.assertIn('encoder failed', errors[0])

    def test_cli_precedence(self):
        c, options = build_config(parser().parse_args(['--no-glow', '--mode', 'jujutsu',
                                 '--detect-scale', '0.5']), Settings(glow=True, thickness=17))
        self.assertFalse(c.glow)
        self.assertEqual(c.default_thickness, 17)
        self.assertEqual(c.detect_scale, .5)
        self.assertEqual(options.mode, 'Jujutsu')

    def test_settings_validation_and_corruption_recovery(self):
        value, problems = validate({'thickness': True, 'volume': float('nan'), 'mode': 'bad'})
        self.assertEqual(value, Settings())
        self.assertEqual(len(problems), 3)
        with TemporaryDirectory() as temp:
            store = SettingsStore(Path(temp) / 'settings.json')
            store.path.write_text('{broken', encoding='utf-8')
            self.assertEqual(store.load(), Settings())
            self.assertEqual(store.path.with_suffix('.json.bak').read_text(), '{broken')
            self.assertTrue(store.save(Settings(mode='Jujutsu', volume=.4)))
            self.assertEqual(store.load().volume, .4)

    def test_profiler_measures_and_bounds_history(self):
        ticks = iter([1., 1.01, 2., 2.02])
        p = Profiler(history=1, clock=lambda: next(ticks))
        with p.stage('detect'):
            pass
        p.end_frame()
        with p.stage('detect'):
            pass
        p.end_frame()
        self.assertAlmostEqual(p.average()['detect'], 20)
        self.assertEqual(len(p.history), 1)
        self.assertIn('15.00', p.summary())

    def test_letterbox_maps_pixels_and_rejects_bars(self):
        box = fit((1280, 720), (1000, 1000))
        self.assertIsNone(box.to_frame(500, 100, (1280, 720)))
        x, y = box.to_frame(500, 500, (1280, 720))
        self.assertAlmostEqual(x, 640)
        self.assertLess(abs(y - 360), 1)

    def test_latest_frame_and_release(self):
        class Capture:
            released = False
            index = 0
            def read(self):
                time.sleep(.002)
                self.index += 1
                return True, np.full((2, 2, 3), self.index % 255, np.uint8)
            def release(self):
                self.released = True
        cap = Capture()
        with CameraStream(Config(), opener=lambda c: cap) as stream:
            first = stream.read(.5)
            time.sleep(.025)
            latest = stream.read(.5)
            self.assertGreater(latest.index, first.index + 1)
        self.assertTrue(cap.released)

    def test_camera_thread_errors_surface(self):
        class Capture:
            def read(self):
                raise ValueError('camera bug')
            def release(self):
                pass
        with CameraStream(Config(), opener=lambda c: Capture()) as stream:
            with self.assertRaisesRegex(CameraThreadError, 'camera bug'):
                stream.read(.5)

    def test_recording_resamples_wall_clock(self):
        class Writer:
            frames = 0
            released = False
            def isOpened(self):
                return True
            def write(self, image):
                self.frames += 1
            def release(self):
                self.released = True
        writer = Writer()
        with TemporaryDirectory() as temp:
            r = Recorder(Path(temp), writer_factory=lambda *args: writer)
            r.start((16, 16), 10, 0)
            frame = np.zeros((16, 16, 3), np.uint8)
            for t in (0, .1, .5, 1.):
                r.write(frame, t)
            r.stop()
        self.assertEqual(writer.frames, 11)
        self.assertTrue(writer.released)


if __name__ == '__main__':
    unittest.main()
