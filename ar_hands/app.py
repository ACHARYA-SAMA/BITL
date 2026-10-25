"""Application orchestration and desktop controls."""
from __future__ import annotations
from datetime import datetime
import logging
import sys
import time
from typing import Any
import cv2
import numpy as np
from . import AppError
from .actions import Action
from .config import Config
from .camera import open_camera, CameraStream, CameraState
from .desktop import DesktopControls
from .audio import AudioManager
from .cli import parser, build_config, RunOptions
from .settings import Settings, SettingsStore
from .keys import decode
from .window import Window
from .logging_setup import setup_logging
from .segmentation import ensure_segmenter_model
from .hand_tracker import HandData, HandTracker
from .skeleton import SkeletonRenderer
from .hud import FPSCounter, HUDRenderer
from .canvas import AirCanvas
from .pen import Cursor, PenController, pen_signals
from .debug_overlay import GestureDebugOverlay
from .draw_mode import DrawMode
from .effects.base import EffectContext
from .effects.blocks import BlockBuilder, BlockWorld
from .effects.fireball import FireballJutsu
from .effects.lightning import Chidori
from .effects.manager import EffectManager, Mode, ModeManager, ModeSwitcher, MenuOpener
from .effects.rasenshuriken import RasenshurikenJutsu
from .effects.susanoo import Susanoo
from .effects.snap import SnapVanish
from .effects.jjk import CursedTechniques
from .effects.domains import DomainCaster
from .effects.sprites import SpriteLibrary
from .gestures.engine import GestureEngine
from .gestures.events import BOTH, EventBus
from .gestures.types import Gesture

# cv2.waitKeyEx codes for F1..F7 on Windows, GTK (Linux), Cocoa (macOS), and Qt builds.
FUNCTION_KEYS = {code: index for index in range(7) for code in
                 (0x700000 + index * 0x10000, 0xFFBE + index, 0xF704 + index, 0x01000030 + index)}
SHIFTED_DIGITS = "!@#$%^&"  # Shift+1..7 on a US layout, the fallback for F1..F7.
MODE_KEYS = {ord("1"): Mode.DRAW, ord("2"): Mode.JUTSU, ord("3"): Mode.BLOCKS, ord("4"): Mode.JJK}
MODE_HINTS = {
    Mode.DRAW: "Point: draw | Peace: color | Thumbs-up: save | Hold palm: clear | "
               "Palm swipe: undo/redo | Hold rock: next mode",
    Mode.JUTSU: "Palm: fireball | OK: Rasenshuriken | Both fists: Susanoo | Fist then palm: Chidori | ? Guide",
    Mode.BLOCKS: "Pinch: new block / grab | Both pinch, touch, pull apart: beam | Point 0.35 s: "
                 "break | Peace: type | Hold thumbs-up: clear",
    Mode.JJK: "Point: Blue | Peace: Red | Bring orbs together: Purple | Crossed: Gojo | Prayer: Sukuna | ? Guide",
}
HOOK_TOASTS = {"blocks_clear": "Blocks cleared", "beam_place": "Beam placed",
               "jutsu_seal": "Seal formed: pull your hands apart!"}


class App(DesktopControls):
    """Coordinate capture, tracking, gestures, modes, effects, rendering, and cleanup."""

    def __init__(self, config: Config | None = None, settings: Settings | None = None,
                 store: SettingsStore | None = None, options: RunOptions | None = None) -> None:
        """Initialize this component and its independent runtime state."""
        self.config = config or Config()
        c = self.config
        self.renderer = SkeletonRenderer(c)
        self.hud = HUDRenderer(c)
        self.fps = FPSCounter(c.fps_smoothing)
        self.show_hud = c.show_hud
        self.show_skeleton = c.show_skeleton
        self.canvas = AirCanvas(c)
        self.pen = PenController(c)
        self.bus = EventBus()
        self.gestures = GestureEngine(c.gestures, self.bus)
        self.debug = GestureDebugOverlay(c)
        self.color_index = c.default_color
        self.thickness = c.default_thickness
        self._frame_shape: tuple[int, ...] | None = None
        self._timestamp = -1
        self._last_time: float | None = None
        self._toast: tuple[str, float] | None = None
        # Phase 4: effects, the persistent block world, and one controller set per mode.
        self.sprites = SpriteLibrary(c.effects, c.base_dir / c.sprite_folder)
        self.effects = EffectManager(c.effects, self.gestures, self.sprites)
        self.world = BlockWorld(c.effects, self.sprites, self.effects.particles, self.effects.hooks)
        self.effects.spawn(self.world)
        self.draw_mode = DrawMode(self.bus, c.gesture_actions, self.perform)
        chidori = Chidori(self.bus, self.gestures, self.effects, c.effects)
        self.rasen = RasenshurikenJutsu(self.bus, self.gestures, self.effects, c.effects,
                                       lambda hand: chidori.is_busy(hand) or hand in self.jutsu.held)
        self.jutsu = FireballJutsu(self.bus, self.gestures, self.effects, c.effects,
                                  lambda hand: chidori.is_busy(hand) or self.rasen.holding(hand))
        self.susanoo = Susanoo(self.bus, self.gestures, self.effects, c.effects)
        self.snap = SnapVanish(self.bus, self.effects, c.effects)
        self.cursed = CursedTechniques(self.bus, self.gestures, self.effects, c.effects)
        self.domains = DomainCaster(self.bus, self.gestures, self.effects, c.effects)
        self.builder = BlockBuilder(self.bus, self.gestures, self.effects, self.world, c.effects)
        switcher = ModeSwitcher(self.bus, c.effects.mode_hold_seconds, lambda now: self.switch_mode(None, now))
        opener = MenuOpener(self.bus, 1., self.open_menu)
        self.modes = ModeManager({Mode.DRAW: (self.draw_mode, self.snap),
                                  Mode.JUTSU: (self.jutsu, chidori, self.rasen, self.susanoo),
                                  Mode.BLOCKS: (self.builder,), Mode.JJK: (self.cursed, self.domains)},
                                 (switcher, opener), self.effects.hooks)
        self._draw_gestures = self.draw_mode.draw_gestures
        self._ctx: EffectContext | None = None
        self.effects.hooks.subscribe(self._on_hook)
        self.setup_desktop(settings, store, options)
        self.switch_mode(Mode(self.options.mode or self.settings.mode))

    @property
    def brush_color(self) -> tuple[int, int, int]:
        """Return the current shared brush color in OpenCV's BGR order."""
        return self.config.palette[self.color_index]

    def _on_hook(self, name: str, payload: dict[str, Any]) -> None:
        """Toast the effect events that change state the user cannot see directly."""
        if name == "block_select":
            self.toast(f"Block: {payload['kind'].replace('_', ' ').title()}")
        elif name in HOOK_TOASTS:
            self.toast(HOOK_TOASTS[name])

    def _drawing_now(self) -> bool:
        """True while any hand holds the draw gesture (in Draw mode)."""
        return self.modes.mode is Mode.DRAW and any(
            self.gestures.active(hand) in self._draw_gestures for hand in self.gestures.tracked_hands())

    def toast(self, text: str) -> None:
        """Show a short confirmation, because gesture actions have no key press to feel."""
        self._toast = (text, time.perf_counter() + self.config.toast_seconds)

    def switch_mode(self, mode: Mode | None, now: float | None = None) -> None:
        """Switch to ``mode`` (None = next). Strokes end; the block world stays."""
        now = time.perf_counter() if now is None else now
        previous = self.modes.mode
        if mode is None:
            self.modes.next(now)
        else:
            self.modes.switch(mode, now)
        if self.modes.mode is not previous:
            self.pen.reset(self.canvas)

    def perform(self, action: Action, from_gesture: bool = False) -> None:
        """Run one action; keys and gesture bindings share this single code path."""
        if action is Action.CLEAR:
            if from_gesture and self.gestures.active(BOTH) is Gesture.TWO_PALMS:
                return  # Two palms belong to the menu, never the single-palm clear action.
            if from_gesture and self._drawing_now():
                # One hand drawing while the other rests open is normal; it must
                # not wipe the canvas. Keys are deliberate, so they still clear.
                self.toast("Canvas kept: a hand is drawing")
                return
            self.pen.reset(self.canvas)
            self.canvas.clear()
            self.toast("Canvas cleared")
        elif action is Action.NEXT_COLOR:
            self.set_color((self.color_index + 1) % len(self.config.palette))
        elif action is Action.SAVE_PNG:
            if from_gesture and not (self.canvas.finished or self.canvas.active):
                self.toast("Nothing to save yet")
            else:
                self.export_drawing()
        elif action in (Action.UNDO, Action.REDO):
            self.pen.reset(self.canvas)
            undo = action is Action.UNDO
            changed = self.canvas.undo() if undo else self.canvas.redo()
            self.toast(("Undo" if undo else "Redo") if changed else
                       ("Nothing to undo" if undo else "Nothing to redo"))
        # DRAW and HOVER are states, not one-shot actions: pen_signals() polls them.

    def set_color(self, index: int) -> None:
        """Change the brush color; open strokes end so they keep their own color."""
        self.pen.reset(self.canvas)
        self.color_index = index
        self.toast(f"Color: {self.config.palette_names[index]}")

    def open_camera(self) -> cv2.VideoCapture:
        """Open a camera using the shared backend and index fallback policy."""
        return open_camera(self.config)

    def next_timestamp(self) -> int:
        """Return strictly increasing milliseconds, even within one clock tick."""
        # Wall-clock time can jump; perf_counter cannot. The +1 guard also
        # prevents duplicate milliseconds, which VIDEO tracking cannot accept.
        self._timestamp = max(self._timestamp + 1, time.perf_counter_ns() // 1_000_000)
        return self._timestamp

    def _update_interaction(self, hands: list[HandData], now: float,
                            size: tuple[int, int] | None = None, camera: np.ndarray | None = None,
                            person: np.ndarray | None = None) -> list[Cursor]:
        """Gestures (whose actions run during dispatch), effects and modes, then the pen."""
        dt = 0.0 if self._last_time is None else now - self._last_time
        self._last_time = now
        # Effects dispatched by gesture callbacks need THIS frame's anchors/mask.
        self.effects.context(now, dt, hands, size=size, camera=camera, person=person, background=self.background)
        with self.profiler.stage('gestures'):
            events = self.gestures.update(hands, now)
        self.debug.record(events)
        with self.profiler.stage('effects'):
            self._ctx = self.effects.update(dt, hands, events, now, size, camera, person, self.background)
            self.modes.update(self._ctx)
        if self.overlay_open:
            self.pen.reset(self.canvas)
            return []
        drawing = self.modes.mode is Mode.DRAW and not self.snap.active
        signals = pen_signals(self.gestures, self._draw_gestures) if drawing else {}
        return self.pen.update(hands if drawing else [], self.canvas, now, self.brush_color, self.thickness, signals)

    def process_frame(self, frame: np.ndarray, tracker: HandTracker) -> np.ndarray:
        """Mirror, detect, recognize gestures, run effects and pens, then composite."""
        # Flip BEFORE detection: MediaPipe handedness assumes a mirrored selfie
        # view. This also keeps Left/Right labels and rendered coordinates aligned.
        cv2.flip(frame, 1, dst=frame)
        timestamp = self.next_timestamp()
        with self.profiler.stage('detect'):
            hands = tracker.detect(frame, timestamp)
        now = time.perf_counter()
        if self._frame_shape is not None and self._frame_shape != frame.shape:
            self.pen.reset(self.canvas)
            self.background.reset()
            if self.snap.active:
                # The snap captured a full-resolution ghost. A reconnected
                # camera may have a different size; restore the live person.
                self.snap.effect.kill()
                self.effects.particles.dust.clear()
        self._frame_shape = frame.shape
        self._frame_number += 1
        person = None
        needs_mask = self.effects.wants_person or (self.modes.mode is Mode.DRAW and
                     self._frame_number % self.config.background_every_frames == 0)
        with self.profiler.stage('segment'):
            if needs_mask:
                person = self.segmenter.mask(frame, timestamp)
                if person is not None and not self.snap.active:
                    self.background.update(frame, person)
        camera = frame.copy() if person is not None or self.effects.wants_person else None
        cursors = self._update_interaction(hands, now, frame.shape[1::-1], camera, person)
        if self.overlay_open:
            self.overlay_input(hands, now, frame.shape[1::-1])
        # Additive light leaves the camera visible through the soft neon halo.
        with self.profiler.stage('canvas'):
            cv2.add(frame, self.canvas.render(frame.shape), dst=frame)
        with self.profiler.stage('effects'):
            self.effects.render(frame)
            self.modes.render(frame, self._ctx)
        render_start = time.perf_counter()
        if self.show_skeleton and not self.snap.active:
            self.renderer.draw(frame, hands)
        if self.modes.mode is Mode.DRAW and not self.snap.active and not self.overlay_open:
            self.hud.draw_cursors(frame, cursors, self.brush_color)
        self.effects.post(frame)
        self._clean_frame = frame.copy() if self.settings.record_clean else None
        self._draw_hold_rings(frame)
        if self.debug.visible:
            self.debug.draw(frame, self.gestures, self.effects.stats())
        fps = self.fps.update()
        if self.show_hud:
            self.hud.show_indices = self.renderer.show_indices
            self.hud.glow = self.renderer.glow
            self._draw_hud(frame, fps, len(hands))
        self.hud.draw_banner(frame, self.modes.mode.value.upper(),
                             (now - self.modes.changed_at) / self.config.effects.mode_banner_s)
        if self._toast is not None:
            if now < self._toast[1]:
                self.hud.draw_toast(frame, self._toast[0])
            else:
                self._toast = None
        self.profiler.draw(frame)
        self.menu.render(frame)
        self.guide.render(frame)
        self.record_frame(frame, now)
        self.profiler.add('render', (time.perf_counter() - render_start) * 1000)
        return frame

    def _draw_hud(self, frame: np.ndarray, fps: float, hand_count: int) -> None:
        mode = self.modes.mode
        status, icon = None, None
        gestures = self._gesture_status()
        if mode is Mode.JUTSU:
            status = (f"Fireballs {self.effects.count('fireball')}  |  "
                      f"{gestures or 'Hold an open palm to summon a fireball'}")
        elif mode is Mode.BLOCKS:
            kind = self.builder.kind
            icon = self.sprites.icon(kind, 28)
            status = (f"      {kind.replace('_', ' ').title()}  |  Blocks "
                      f"{len(self.world.live_blocks)}/{self.config.effects.max_blocks}  Beams "
                      f"{len(self.world.beams)}  |  {gestures or 'Pinch to place a block'}")
        self.hud.draw(frame, fps, hand_count, self.color_index, self.thickness,
                      self.canvas.fade_enabled, len(self.canvas.finished), gestures,
                      mode.value, status, icon, MODE_HINTS[mode])

    def _gesture_status(self) -> str:
        parts = [f"{hand} {self.gestures.active(hand).name.replace('_', ' ').lower()}"
                 for hand in sorted(self.gestures.tracked_hands(), reverse=True)
                 if self.gestures.active(hand) is not Gesture.NONE]
        if self.gestures.active(BOTH) is not Gesture.NONE:
            parts.append(f"both {self.gestures.active(BOTH).name.replace('_', ' ').lower()}")
        return "  ".join(parts)

    def _draw_hold_rings(self, frame: np.ndarray) -> None:
        """Show how long is left before a HOLD binding fires (clear, mode switch, seal...)."""
        if self._drawing_now():
            return  # Holds are ignored while drawing, so do not promise one.
        for gesture, seconds in self.modes.hold_rings():
            for hand in (*self.gestures.tracked_hands(), BOTH):
                if not self.gestures.is_active(hand, gesture):
                    continue
                progress = self.gestures.hold_progress(hand, gesture, seconds)
                center = self.gestures.palm_center(hand)
                if center is not None and 0.0 < progress < 1.0:
                    self.hud.draw_hold_ring(frame, center,
                                            self.gestures.hand_scale(hand) * self.config.hold_ring_ratio,
                                            progress, self.brush_color)

    def export_drawing(self) -> None:
        """Export a transparent drawing without the webcam, skeleton, or HUD."""
        folder = self.config.output_dir / "drawings"
        stem = f"drawing_{datetime.now():%Y%m%d_%H%M%S}"
        path = folder / f"{stem}.png"
        suffix = 1
        while path.exists():
            path = folder / f"{stem}_{suffix:02d}.png"
            suffix += 1
        try:
            self.canvas.export_png(path)
            logging.getLogger(__name__).info('Drawing exported: %s', path)
            self.toast(f"Saved {path.name}")
        except (OSError, ValueError, cv2.error) as exc:
            logging.getLogger(__name__).warning('Drawing could not be exported: %s', exc)
            self.toast("Save failed (see console)")

    def save_screenshot(self, frame: np.ndarray) -> None:
        """Save exactly the displayed frame without overwriting same-second shots."""
        try:
            folder = self.config.output_dir / "screenshots"
            folder.mkdir(parents=True, exist_ok=True)
            stem = f"skeleton_{datetime.now():%Y%m%d_%H%M%S}"
            path = folder / f"{stem}.png"
            suffix = 1
            while path.exists():
                path = folder / f"{stem}_{suffix:02d}.png"
                suffix += 1
            # imencode + Python file I/O also handles Unicode paths on Windows.
            success, encoded = cv2.imencode(".png", frame)
            if not success:
                raise OSError("OpenCV could not encode the screenshot.")
            with path.open("xb") as output:
                output.write(encoded.tobytes())
            logging.getLogger(__name__).info('Screenshot saved: %s', path)
        except (OSError, cv2.error) as exc:
            logging.getLogger(__name__).warning('Screenshot could not be saved: %s', exc)

    def handle_key(self, key: int, frame: np.ndarray | None) -> bool:
        """Apply controls (cv2.waitKeyEx codes) and return False when the user requests exit."""
        name = decode(key)
        if name is None:
            return not self._quit
        if self.guide.visible:
            if self.guide.onboarding or name in ('esc', '?', 'tab'):
                self.close_overlay()
            elif name in ('left', 'a'):
                self.guide.turn(-1)
            elif name in ('right', 'd'):
                self.guide.turn(1)
            return True
        if self.menu.visible:
            if name in ('esc', 'tab'):
                self.close_overlay()
            else:
                self.menu.key(name)
            return not self._quit
        if name == 'tab':
            self.open_menu()
            return True
        if name == '?':
            self.open_guide()
            return True
        if name == 'f10':
            self.profiler.visible = not self.profiler.visible
            return True
        if name == 'F':
            self.menu_change('fullscreen', not self.settings.fullscreen)
            return True
        if name.lower() == 'r':
            self.toggle_recording()
            return True
        if name.lower() == 'm':
            self.audio.toggle_mute()
            self.save_settings()
            return True
        if name.lower() == 'b':
            self.background.reset()
            self.toast('Background reset: step aside briefly to learn the empty scene')
            return True
        if name.lower() == 'n' and self.modes.mode is Mode.DRAW:
            self.snap.trigger(time.perf_counter())
            return True
        color = FUNCTION_KEYS.get(key)
        if color is None and 0 < key < 128 and chr(key) in SHIFTED_DIGITS:
            color = SHIFTED_DIGITS.index(chr(key))
        if color is not None:
            self.set_color(color)
            return True
        # Never truncate special keys: X11 Left (0xFF51) otherwise becomes Q.
        if len(name) != 1 and name != 'esc':
            return True
        key = 27 if name == 'esc' else ord(name)
        if key in (ord("q"), ord("Q"), 27):
            return False
        if key in MODE_KEYS:
            self.switch_mode(MODE_KEYS[key])
        elif key in (ord("l"), ord("L")):
            self.renderer.show_indices = not self.renderer.show_indices
        elif key in (ord("g"), ord("G")):
            self.menu_change('glow', not self._user_glow)
        elif key in (ord("h"), ord("H")):
            self.show_hud = not self.show_hud
        elif key in (ord("s"), ord("S")) and frame is not None:
            self.save_screenshot(frame)
        elif key in (ord("["), ord("]")):
            self.pen.reset(self.canvas)
            change = self.config.thickness_step * (1 if key == ord("]") else -1)
            self.thickness = max(self.config.min_thickness,
                                 min(self.config.max_thickness, self.thickness + change))
        elif key in (ord("c"), ord("C")):
            self.perform(Action.CLEAR)
        elif key in (ord("u"), ord("U")):
            self.perform(Action.UNDO)
        elif key in (ord("y"), ord("Y")):
            self.perform(Action.REDO)
        elif key == ord("f"):
            self.canvas.set_fade(not self.canvas.fade_enabled)
        elif key in (ord("k"), ord("K")):
            self.show_skeleton = not self.show_skeleton
        elif key in (ord("p"), ord("P")):
            self.perform(Action.SAVE_PNG)
        elif key in (ord("v"), ord("V")):
            self.debug.visible = not self.debug.visible
        self.save_settings()
        return not self._quit

    def _window_open(self) -> bool:
        try:
            return cv2.getWindowProperty(self.config.window_name, cv2.WND_PROP_VISIBLE) >= 1
        except cv2.error:
            return False  # Some backends raise after the X button destroys the window.

    def run(self) -> None:
        """Keep the UI responsive during slow capture, camera loss and recoverable frame errors."""
        logger = logging.getLogger(__name__)
        stream = None
        try:
            with HandTracker(self.config) as tracker:
                self._tracker = tracker
                try:
                    ensure_segmenter_model(self.config)
                except (AppError, OSError, RuntimeError) as exc:
                    logger.warning('Person effects will use overlays: %s', exc)
                self.audio = AudioManager(self.config, self.settings.volume, self.settings.muted,
                                          enabled=self.options.audio)
                self.effects.hooks.subscribe(self.audio.on_hook)
                self.window = Window(self.config.window_name, (self.config.width, self.config.height),
                                     self.settings.fullscreen)
                if not self.settings.onboarded:
                    self.open_guide(onboarding=True)
                stream = CameraStream(self.config, opener=lambda c: self.open_camera()).start()
                failures = 0
                last_frame: np.ndarray | None = None
                while not self._quit and self.window.is_open():
                    with self.profiler.stage('capture'):
                        packet = stream.read(timeout=.015)
                    if packet is not None:
                        try:
                            last_frame = self.process_frame(packet.frame, tracker)
                            failures = 0
                        except Exception:
                            logger.exception('Frame skipped after a processing error')
                            failures += 1
                            if failures >= self.config.max_frame_failures:
                                raise AppError(f'Stopped after {failures} consecutive frame errors; see logs/ar_hands.log.')
                            self.pen.reset(self.canvas)
                            self.profiler.end_frame()
                            if not self.handle_key(cv2.waitKeyEx(1), last_frame):
                                break
                            continue
                    elif stream.state is not CameraState.LIVE:
                        self._update_interaction([], time.perf_counter())
                        last_frame = np.zeros((self.config.height, self.config.width, 3), np.uint8)
                        self.hud.draw_toast(last_frame, 'Camera lost, reconnecting... TAB settings / Q quit')
                        self.menu.render(last_frame)
                        self.guide.render(last_frame)
                    with self.profiler.stage('imshow'):
                        if last_frame is not None:
                            self.window.present(last_frame)
                        key = cv2.waitKeyEx(1)
                    if packet is not None:
                        self.profiler.end_frame()
                        message = self.quality.update(time.perf_counter(), self.profiler.busy_ms())
                        if message:
                            self.toast(message)
                    if not self.window.is_open() or not self.handle_key(key, last_frame):
                        break
        finally:
            try:
                if stream is not None:
                    stream.stop()
            finally:
                self.close_desktop()
                self._tracker = None
                try:
                    cv2.destroyAllWindows()
                except cv2.error as exc:
                    logger.warning('Window cleanup warning: %s', exc)


def main(argv: list[str] | None = None) -> int:
    """Launch with friendly expected-error reporting and quiet Ctrl+C handling."""
    args = parser().parse_args(argv)
    setup_logging(Config().output_dir / 'logs', args.debug)
    try:
        if args.selftest:
            from .selftest import run_selftest
            config, _ = build_config(args, Settings())
            return run_selftest(config)
        store = SettingsStore()
        settings = store.reset() if args.reset_settings else store.load()
        config, options = build_config(args, settings)
        App(config, settings, store, options).run()
        return 0
    except KeyboardInterrupt:
        return 0
    except (AppError, cv2.error, OSError, RuntimeError, ValueError) as exc:
        logging.getLogger(__name__).error('AR Hands: %s', exc)
        return 1
