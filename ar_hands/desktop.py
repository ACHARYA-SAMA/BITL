"""Desktop settings, input focus, recording and optional services for App."""
from __future__ import annotations
from collections import deque
from dataclasses import replace
import logging
import time
import cv2
import numpy as np
from .audio import AudioManager
from .cli import RunOptions
from .config import Config
from .guide import GestureGuide
from .menu import Menu
from .pairing import PairingScreen
from .profiler import Profiler
from .quality import AdaptiveQuality, QualityStep
from .recorder import Recorder
from .settings import Settings, SettingsStore
from .segmentation import PersonSegmenter, BackgroundModel
from .smoothing import OneEuroPointFilter
from .effects.manager import Mode
from .gestures.types import Gesture
from .hand_tracker import Landmark

LOGGER = logging.getLogger(__name__)


class DesktopControls:
    """Shared UI actions used by keys, hands and mouse; App owns the frame pipeline."""

    def setup_desktop(self, settings: Settings | None, store: SettingsStore | None,
                      options: RunOptions | None) -> None:
        """Construct services without opening hardware; run() owns resource startup."""
        c = self.config
        self.settings = settings or Settings(glow=c.glow, show_hud=c.show_hud,
            show_skeleton=c.show_skeleton, color_index=c.default_color, thickness=c.default_thickness,
            sensitivity=c.detection_confidence, max_hands=c.num_hands, onboarded=True)
        self.store = store
        self.options = options or RunOptions()
        self.settings.max_hands = c.num_hands  # --hands wins over the saved value, as --mode does.
        self.audio = AudioManager.disabled(c)
        self.recorder = Recorder(c.output_dir / c.record_folder, c.record_queue)
        self.profiler = Profiler(c.profiler_history)
        self.profiler.visible = c.show_profiler or self.options.debug
        self.guide = GestureGuide(c)
        self.menu = Menu(c, self.menu_value, self.menu_change, self.menu_enabled)
        self.segmenter = PersonSegmenter(c)
        self.background = BackgroundModel(c.background_learn_rate)
        self.window = None
        self.stream = None  # CameraStream or PhoneCamera while run() is active.
        self.camera_source = self.options.camera_source or self.settings.camera_source
        self.pairing = PairingScreen(c)
        self._phone_live = False
        self._phone_landscape = False
        self._touches = deque()  # Phone touches waiting to be applied, one press change per frame.
        self._tracker = None
        self._quit = False
        self._frame_number = 0
        self._clean_frame = None
        self._pending_record = self.options.record
        self._pinched = False
        self._pointer_hand = None
        self._pointer_source = None
        self._overlay_release_required = True
        self._menu_filter = OneEuroPointFilter(c.menu.cursor_min_cutoff, c.menu.cursor_beta, 1.)
        self._user_glow = c.glow
        self.quality = AdaptiveQuality([
            QualityStep('skeleton glow', lambda: setattr(self.renderer, 'glow', False),
                        lambda: setattr(self.renderer, 'glow', self._user_glow), lambda: self.renderer.glow),
            QualityStep('particle budget', lambda: self.effects.particles.set_budget(.5),
                        lambda: self.effects.particles.set_budget(1.)),
            QualityStep('detection resolution', lambda: self._detect_scale(c.quality_min_detect_scale),
                        lambda: self._detect_scale(c.detect_scale),
                        lambda: c.detect_scale > c.quality_min_detect_scale),
        ], c.quality_target_fps, c.quality_down_after_s, c.quality_up_after_s,
            c.quality_headroom, enabled=self.options.adaptive and c.adaptive_quality)

    def _detect_scale(self, scale: float) -> None:
        if self._tracker is not None:
            self._tracker.detect_scale = scale

    @property
    def overlay_open(self) -> bool:
        """Whether an overlay currently owns all gesture and keyboard input."""
        return self.menu.visible or self.guide.visible

    def open_menu(self, now: float | None = None) -> None:
        """Capture input and terminate active strokes before showing settings."""
        self.modes.suspend()
        self.pen.reset(self.canvas)
        self.guide.close()
        self.menu.visible = True
        self.menu.reset_pointer()
        self._overlay_release_required = True
        self._menu_filter.reset()
        self.effects.hooks.emit('ui_click')

    def open_guide(self, onboarding: bool = False) -> None:
        """Show illustrated gestures, starting at the active mode."""
        self.modes.suspend()
        self.pen.reset(self.canvas)
        self.menu.visible = False
        self.guide.open('Welcome' if onboarding else self.modes.mode.value, onboarding)
        self._overlay_release_required = True

    def close_overlay(self) -> None:
        """Reset held gestures before returning focus so closing cannot cast an effect."""
        if self.guide.onboarding:
            self.settings.onboarded = True
        self.guide.close()
        self.menu.visible = False
        self.gestures.reset(time.perf_counter())
        self.modes.resume()
        self.save_settings()
        self._apply_tracker_settings()

    def menu_enabled(self, key: str) -> bool:
        """Show unavailable audio controls explicitly; every other control remains usable."""
        return self.audio.available if key in ('volume', 'muted') else True

    def menu_value(self, key: str) -> object:
        """Read live state so keyboard shortcuts and menu labels always agree."""
        values = {'mode': self.modes.mode.value, 'color_index': self.color_index,
                  'thickness': self.thickness, 'glow': self._user_glow,
                  'show_skeleton': self.show_skeleton, 'show_hud': self.show_hud,
                  'volume': self.audio.volume, 'muted': self.audio.muted,
                  'record': self.recorder.active or self._pending_record,
                  'adaptive_quality': self.quality.enabled, 'camera_source': self.camera_source}
        return values.get(key, getattr(self.settings, key, None))

    def menu_change(self, key: str, value: object) -> None:
        """Apply one menu choice. Model changes wait until the menu closes."""
        self.effects.hooks.emit('ui_click')
        if key == 'mode':
            self.switch_mode(Mode(value))
        elif key == 'color_index':
            self.set_color(int(value))
        elif key == 'thickness':
            self.pen.reset(self.canvas)
            self.thickness = int(value)
        elif key == 'glow':
            self.quality.reset()
            self._user_glow = self.renderer.glow = bool(value)
        elif key in ('show_skeleton', 'show_hud'):
            setattr(self, key, bool(value))
        elif key == 'volume':
            self.audio.set_volume(float(value))
        elif key == 'muted':
            self.audio.set_muted(bool(value))
        elif key == 'adaptive_quality':
            self.quality.set_enabled(bool(value))
        elif key == 'fullscreen':
            self.settings.fullscreen = bool(value)
            if self.window is not None:
                self.window.set_fullscreen(bool(value))
                self.settings.fullscreen = self.window.fullscreen
        elif key == 'record':
            self.toggle_recording()
        elif key == 'camera_source':
            self.set_camera_source(str(value))
        elif key in ('sensitivity', 'record_clean'):
            setattr(self.settings, key, value)
        elif key == 'max_hands':
            self.settings.max_hands = int(value)
        elif key == 'guide':
            self.open_guide()
        elif key == 'close':
            self.close_overlay()
        elif key == 'quit':
            self._quit = True
        elif key == 'reset':
            defaults = Settings(onboarded=True)
            for name in ('mode', 'color_index', 'thickness', 'glow', 'show_skeleton', 'show_hud',
                         'volume', 'muted', 'sensitivity', 'fullscreen', 'record_clean', 'adaptive_quality',
                         'max_hands'):
                self.menu_change(name, getattr(defaults, name))
        self.save_settings()

    def _apply_tracker_settings(self) -> None:
        """Sensitivity and max hands are fixed when MediaPipe's tracker is created:
        rebuild it (once, when the menu closes) if either changed."""
        value, hands = self.settings.sensitivity, self.settings.max_hands
        if self._tracker is None or (value, hands) == (self.config.detection_confidence, self.config.num_hands):
            return
        config = replace(self.config, detection_confidence=value, presence_confidence=value,
                         tracking_confidence=value, num_hands=hands)
        try:
            self._tracker.reconfigure(config)
            self.config = config
        except (OSError, RuntimeError, ValueError) as exc:
            LOGGER.warning('Tracker update failed: %s', exc)
            self.settings.sensitivity, self.settings.max_hands = self.config.detection_confidence, self.config.num_hands
            self.toast('Tracking settings unchanged; see log')
            self.save_settings()

    def save_settings(self) -> None:
        """Persist user choices, never temporary reductions made by adaptive quality."""
        for key in ('mode', 'color_index', 'thickness', 'glow', 'show_skeleton', 'show_hud',
                    'volume', 'muted', 'adaptive_quality', 'camera_source'):
            setattr(self.settings, key, self.menu_value(key))
        if self.store is not None:
            self.store.save(self.settings)

    def overlay_input(self, hands: list, now: float, size: tuple[int, int]) -> None:
        """Use a stable hand as pointer; mouse movement temporarily takes precedence."""
        hand = next((h for h in hands if h.handedness == self._pointer_hand), None)
        if hand is None and hands:
            hand = hands[0]
        point, pressed, source = None, False, None
        if hand is not None:
            if hand.handedness != self._pointer_hand:
                self._menu_filter.reset()
            self._pointer_hand = hand.handedness
            point = self._menu_filter.filter(hand.landmarks_px[Landmark.INDEX_TIP], now)
            # Raw thumb-index gap with hysteresis, not the stabilized PINCH: the
            # stabilizer's few frames of delay make a click feel laggy.
            features = self.gestures.features(hand.handedness)
            gestures = self.config.gestures
            threshold = gestures.pinch_exit if self._pinched else gestures.pinch_enter
            pressed = features is not None and features.pinch_ratio < threshold
            source = hand.handedness
        else:
            self._pointer_hand = None
            self._menu_filter.reset()
        mouse = self.window.mouse if self.window is not None else None
        if mouse is not None and now - mouse.moved_at < 1.5:
            point, pressed, source = mouse.position, mouse.pressed, 'mouse'
        if source != self._pointer_source:
            self.menu.reset_pointer()
            self._pointer_source = source
            self._overlay_release_required = True
        if self._overlay_release_required:
            if pressed:
                self.menu.pointer(None, False, now, size)
                self._pinched = True
                return
            self._overlay_release_required = False
        if self.guide.visible and pressed and not self._pinched:
            self.close_overlay()
        elif self.menu.visible:
            before = self.menu.hover
            self.menu.pointer(point, pressed, now, size)
            if self.menu.hover is not None and self.menu.hover != before:
                self.effects.hooks.emit('ui_hover')
        self._pinched = pressed

    def toggle_recording(self) -> None:
        """Toggle a recording; defer startup until there is a real processed frame."""
        if self.recorder.active:
            try:
                path = self.recorder.stop()
                self.toast(f'Saved {path.name}' if path else 'Recording stopped')
            except (OSError, RuntimeError, cv2.error) as exc:
                LOGGER.error('Recording could not finish: %s', exc)
                self.toast('Recording failed; see log')
            self.effects.hooks.emit('rec_stop')
        else:
            self._pending_record = not self._pending_record
            self.toast('Recording starting...' if self._pending_record else 'Recording cancelled')

    def record_frame(self, frame: np.ndarray, now: float) -> None:
        """Record the selected composition before adding the blinking REC badge."""
        try:
            if self._pending_record and self.fps.value > 0:
                self.recorder.start(frame.shape[1::-1], self.fps.value, now)
                self._pending_record = False
                self.effects.hooks.emit('rec_start')
            if self.recorder.active:
                source = self._clean_frame if self.settings.record_clean and self._clean_frame is not None else frame
                self.recorder.write(source, now)
                if int(now * 2) % 2 == 0:
                    cv2.circle(frame, (frame.shape[1] - 150, 28), 6, (40, 40, 255), -1)
                cv2.putText(frame, f'REC {self.recorder.elapsed(now):.0f}s', (frame.shape[1] - 135, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, .55, (240, 240, 255), 1, cv2.LINE_AA)
        except (OSError, RuntimeError, cv2.error) as exc:
            self._pending_record = False
            try:
                self.recorder.stop()
            except (OSError, RuntimeError, cv2.error):
                LOGGER.exception('Recording cleanup failed')
            LOGGER.error('Recording failed: %s', exc)
            self.toast('Recording failed; see log')

    def close_desktop(self) -> None:
        """Attempt every cleanup independently, even if an earlier resource fails."""
        for name, close in (('settings', self.save_settings), ('modes', self.modes.close),
                            ('recording', self.recorder.stop), ('audio', self.audio.close),
                            ('segmentation', self.segmenter.close)):
            try:
                close()
            except Exception:
                LOGGER.exception('Could not close %s', name)
        LOGGER.info('%s', self.profiler.summary())
