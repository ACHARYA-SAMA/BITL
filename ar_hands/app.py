"""Application orchestration and desktop controls."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
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
from .phone import PhoneCamera
from .identity import HandIdentities
from .pipeline import Analysis, FrameAnalyzer, fast_timer
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
from .gestures.events import EventBus
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
PHONE_ACTIONS = {"color": Action.NEXT_COLOR, "undo": Action.UNDO, "redo": Action.REDO,
                 "clear": Action.CLEAR, "save": Action.SAVE_PNG}
# Phone buttons that work on top of the menu or guide; any other button closes it first.
OVERLAY_COMMANDS = frozenset({"menu", "guide", "record", "mute"})


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
        self.identities = HandIdentities(duplicate_ratio=c.duplicate_hand_ratio)
        self._cursors: list[Cursor] = []
        self._want_mask = False  # Set by the renderer, read by the analyzer thread.
        self._reset_identities = False
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
            if from_gesture and any(self.gestures.active(both) is Gesture.TWO_PALMS
                                    for both in self.gestures.tracked_pairs()):
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
                            person: np.ndarray | None = None, fresh: bool = True) -> list[Cursor]:
        """Gestures (whose actions run during dispatch), effects and modes, then the pen.

        ``fresh`` is False when re-rendering the last camera frame between camera
        frames: effects keep animating, but gestures and pens only advance on new
        detections (their timing counts frames, and a hand seen twice has not moved)."""
        dt = 0.0 if self._last_time is None else now - self._last_time
        self._last_time = now
        # Effects dispatched by gesture callbacks need THIS frame's anchors/mask.
        self.effects.context(now, dt, hands, size=size, camera=camera, person=person, background=self.background)
        events = []
        if fresh:
            with self.profiler.stage('gestures'):
                events = self.gestures.update(hands, now)
            self.debug.record(events)
        with self.profiler.stage('effects'):
            self._ctx = self.effects.update(dt, hands, events, now, size, camera, person, self.background)
            self.modes.update(self._ctx)
        if self.overlay_open:
            self.pen.reset(self.canvas)
            self._cursors = []
            return []
        if not fresh:
            return self._cursors
        drawing = self.modes.mode is Mode.DRAW and not self.snap.active
        signals = pen_signals(self.gestures, self._draw_gestures) if drawing else {}
        self._cursors = self.pen.update(hands if drawing else [], self.canvas, now, self.brush_color,
                                        self.thickness, signals)
        return self._cursors

    def analyze(self, frame: np.ndarray, tracker: HandTracker,
                segmenting: ThreadPoolExecutor | None = None) -> Analysis:
        """Mirror, detect and identify hands, and segment the person when effects need it.

        Runs on the analyzer thread in run(); the segmenter works in parallel
        with the hand tracker when a pool is given. Only state that this thread
        owns is written here (timestamps, hand ids)."""
        # Flip BEFORE detection: MediaPipe handedness assumes a mirrored selfie
        # view. This also keeps Left/Right labels and rendered coordinates aligned.
        cv2.flip(frame, 1, dst=frame)
        timestamp = self.next_timestamp()
        timings: dict[str, float] = {}
        mask_job = None
        start = time.perf_counter()
        if self._want_mask and segmenting is not None:
            mask_job = segmenting.submit(self.segmenter.mask, frame, timestamp)
        hands = tracker.detect(frame, timestamp)
        timings['detect'] = (time.perf_counter() - start) * 1000
        if self._reset_identities:
            self._reset_identities = False
            self.identities.reset()
        hands = self.identities.assign(hands, time.perf_counter())
        person = None
        if mask_job is not None:
            person = mask_job.result()
        elif self._want_mask:
            start = time.perf_counter()
            person = self.segmenter.mask(frame, timestamp)
            timings['segment'] = (time.perf_counter() - start) * 1000
        return Analysis(frame, hands, person, timestamp, timings)

    def process_frame(self, frame: np.ndarray, tracker: HandTracker) -> np.ndarray:
        """Mirror, detect, recognize gestures, run effects and pens, then composite."""
        self._want_mask = self._needs_mask(self._frame_number + 1)
        return self.compose(self.analyze(frame, tracker))

    def _needs_mask(self, frame_number: int) -> bool:
        """Person mask wanted: effects use it, or Draw mode is learning the background."""
        return self.effects.wants_person or (self.modes.mode is Mode.DRAW and
                                             frame_number % self.config.background_every_frames == 0)

    def compose(self, analysis: Analysis, fresh: bool = True) -> np.ndarray:
        """Run interaction and draw everything onto a copy of the analyzed frame.

        With ``fresh`` False the same camera frame is drawn again: effects move
        on, so the picture stays smooth at the display rate even when the camera
        delivers fewer frames."""
        frame = analysis.frame.copy()
        hands, person = analysis.hands, analysis.person
        now = time.perf_counter()
        if fresh:
            for stage, ms in analysis.timings.items():
                self.profiler.add(stage, ms)
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
            if person is not None and not self.snap.active:
                self.background.update(analysis.frame, person)
            self._want_mask = self._needs_mask(self._frame_number + 1)
        camera = analysis.frame if person is not None or self.effects.wants_person else None
        cursors = self._update_interaction(hands, now, frame.shape[1::-1], camera, person, fresh)
        if self.overlay_open:
            self.overlay_input(hands, now, frame.shape[1::-1])
        # The neon drawing is light: it joins the effects' light buffer, so strokes
        # bloom and tint the hand and face with their color (effects/lighting.py).
        with self.profiler.stage('canvas'):
            neon = self.canvas.render(frame.shape) if self.canvas.finished or self.canvas.active else None
        with self.profiler.stage('effects'):
            self.effects.render(frame, neon)
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
            self._draw_hud(frame, fps, hands)
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

    def _draw_hud(self, frame: np.ndarray, fps: float, hands: list[HandData]) -> None:
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
        people = len({hand.key.rpartition(" ")[2] for hand in hands})
        camera_fps = self.stream.fps if self.stream is not None else 0.0
        self.hud.draw(frame, fps, len(hands), self.color_index, self.thickness,
                      self.canvas.fade_enabled, len(self.canvas.finished), gestures,
                      mode.value, status, icon, MODE_HINTS[mode], camera_fps, people)

    def _gesture_status(self) -> str:
        parts = [f"{hand} {self.gestures.active(hand).name.replace('_', ' ').lower()}"
                 for hand in sorted(self.gestures.tracked_hands(), reverse=True)
                 if self.gestures.active(hand) is not Gesture.NONE]
        for both in self.gestures.tracked_pairs():
            if self.gestures.active(both) is not Gesture.NONE:
                parts.append(f"{both.lower()} {self.gestures.active(both).name.replace('_', ' ').lower()}")
        return "  ".join(parts)

    def _draw_hold_rings(self, frame: np.ndarray) -> None:
        """Show how long is left before a HOLD binding fires (clear, mode switch, seal...)."""
        if self._drawing_now():
            return  # Holds are ignored while drawing, so do not promise one.
        for gesture, seconds in self.modes.hold_rings():
            for hand in (*self.gestures.tracked_hands(), *self.gestures.tracked_pairs()):
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
        if name.lower() == 'o':
            self.menu_change('camera_source', 'Webcam' if self.camera_source == 'Phone' else 'Phone')
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

    def _open_stream(self) -> CameraStream | PhoneCamera:
        """Start the selected camera: the phone page, or the webcam with its fallbacks."""
        if self.camera_source == 'Phone':
            return PhoneCamera(self.config).start()
        return CameraStream(self.config, opener=lambda c: self.open_camera()).start()

    def set_camera_source(self, source: str) -> None:
        """Switch between the webcam and the phone; while running, the stream swaps now."""
        if source == self.camera_source:
            return
        self.camera_source = source
        self._phone_live = False
        self._auto_fullscreen(False)
        if self.stream is not None:
            self.stream.stop()
            self.stream = self._open_stream()
        self._reset_identities = True
        self.toast('Phone camera: scan the QR code with your phone' if source == 'Phone' else 'Laptop webcam')

    def phone_command(self, name: str, value: str | None = None) -> None:
        """A button on the phone page; it runs through the same paths as keys and menu items."""
        if self.overlay_open and name not in OVERLAY_COMMANDS:
            self.close_overlay()
        if name == 'mode' and value in {mode.value for mode in Mode}:
            self.menu_change('mode', value)
        elif name in PHONE_ACTIONS:
            self.perform(PHONE_ACTIONS[name])
        elif name == 'snap' and self.modes.mode is Mode.DRAW:
            self.snap.trigger(time.perf_counter())
        elif name == 'record':
            self.toggle_recording()
        elif name == 'mute':
            self.audio.toggle_mute()
        elif name == 'background':
            self.background.reset()
            self.toast('Background reset: step aside briefly to learn the empty scene')
        elif name in ('menu', 'guide'):
            showing = self.menu.visible if name == 'menu' else self.guide.visible
            if showing:
                self.close_overlay()
            elif name == 'menu':
                self.open_menu()
            else:
                self.open_guide()
        self.save_settings()

    def _phone_input(self, stream: CameraStream | PhoneCamera) -> None:
        """Apply the phone page's buttons and its touches on the menu or guide."""
        if not isinstance(stream, PhoneCamera):
            return
        live = stream.state is CameraState.LIVE
        if live and not self._phone_live and stream.size is not None:
            self.toast(f'Phone camera connected ({stream.size[0]}x{stream.size[1]})')
        self._phone_live = live
        self._auto_fullscreen(live and stream.size is not None and stream.size[0] >= stream.size[1])
        for command in stream.commands():
            self.phone_command(command.name, command.value)
        self._touches.extend(stream.pointer_events())
        if self.window is None or self._frame_shape is None:
            self._touches.clear()
            return
        height, width = self._frame_shape[:2]
        mouse = self.window.mouse
        while self._touches:
            touch = self._touches.popleft()
            changed = touch.down != mouse.pressed
            mouse.position = (touch.x * (width - 1), touch.y * (height - 1))
            mouse.pressed, mouse.moved_at = touch.down, time.perf_counter()
            if changed:
                break  # The menu samples the button once per frame: a quick tap needs two frames.

    def _auto_fullscreen(self, landscape: bool) -> None:
        """A phone held sideways fills the laptop screen; turning it upright (or the
        phone leaving) returns to the saved choice. Only these changes toggle it, so
        Shift+F still works while the phone stays sideways."""
        if self.window is None or not self.config.phone_fullscreen or landscape == self._phone_landscape:
            return
        self._phone_landscape = landscape
        wanted = landscape or self.settings.fullscreen
        if wanted != self.window.fullscreen:
            self.window.set_fullscreen(wanted)

    def _phone_state(self) -> dict[str, object]:
        """What the phone page shows natively: its mode tabs, REC, sound and color buttons."""
        blue, green, red = self.brush_color
        return {'mode': self.modes.mode.value, 'recording': self.recorder.active or self._pending_record,
                'muted': self.audio.muted, 'audio': self.audio.available,
                'overlay': 'menu' if self.menu.visible else 'guide' if self.guide.visible else '',
                'color': f'#{red:02x}{green:02x}{blue:02x}', 'fps': round(self.fps.value)}

    def _waiting_frame(self, stream: CameraStream | PhoneCamera) -> np.ndarray:
        """No camera picture: the pairing screen for the phone, a reconnect notice for the webcam."""
        frame = np.zeros((self.config.height, self.config.width, 3), np.uint8)
        if isinstance(stream, PhoneCamera):
            self.pairing.render(frame, stream, time.perf_counter())
        else:
            self.hud.draw_toast(frame, 'Camera lost, reconnecting... O phone camera / TAB settings / Q quit')
        self.menu.render(frame)
        self.guide.render(frame)
        return frame

    def _window_open(self) -> bool:
        try:
            return cv2.getWindowProperty(self.config.window_name, cv2.WND_PROP_VISIBLE) >= 1
        except cv2.error:
            return False  # Some backends raise after the X button destroys the window.

    def run(self) -> None:
        """Keep the UI responsive during slow capture, camera loss and recoverable frame errors."""
        with fast_timer(), ThreadPoolExecutor(1, thread_name_prefix='segment') as segmenting:
            self._run(segmenting)

    def _run(self, segmenting: ThreadPoolExecutor) -> None:
        logger = logging.getLogger(__name__)
        analyzer: FrameAnalyzer | None = None
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
                self.stream = self._open_stream()
                analyzer = FrameAnalyzer(lambda frame: self.analyze(frame, tracker, segmenting)).start()
                failures = 0
                last: Analysis | None = None
                last_frame: np.ndarray | None = None
                last_render = 0.0
                tick = 1.0 / self.config.display_fps
                while not self._quit and self.window.is_open():
                    stream = self.stream  # The menu, O key or phone may swap it during this pass.
                    packet = stream.read(timeout=0)
                    if packet is not None:
                        analyzer.submit(packet.frame)
                    self._phone_input(stream)
                    rendered, fresh = False, False
                    try:
                        with self.profiler.stage('capture'):
                            result = analyzer.result(timeout=.004)
                        now = time.perf_counter()
                        if result is not None:
                            last, fresh = result, True
                            last_frame, rendered = self.compose(result), True
                            failures = 0
                        elif stream.state is not CameraState.LIVE and analyzer.idle:
                            if now - last_render >= tick:
                                last = None
                                self._update_interaction([], now)
                                last_frame, rendered = self._waiting_frame(stream), True
                        elif last is not None and now - last_render >= tick and not analyzer.busy:
                            # Between camera frames: effects keep moving at the display rate.
                            last_frame, rendered = self.compose(last, fresh=False), True
                    except Exception:
                        logger.exception('Frame skipped after a processing error')
                        failures += 1
                        if failures >= self.config.max_frame_failures:
                            raise AppError(f'Stopped after {failures} consecutive frame errors; see logs/ar_hands.log.')
                        self.pen.reset(self.canvas)
                        last = None
                    with self.profiler.stage('imshow'):
                        if rendered and last_frame is not None:
                            self.window.present(last_frame)
                            last_render = time.perf_counter()
                        key = cv2.waitKeyEx(1)
                    if rendered:
                        if isinstance(stream, PhoneCamera) and stream.state is CameraState.LIVE:
                            stream.publish(last_frame, self._phone_state())
                        self.profiler.end_frame()
                        if fresh:
                            message = self.quality.update(time.perf_counter(), self.profiler.busy_ms())
                            if message:
                                self.toast(message)
                    if not self.window.is_open() or not self.handle_key(key, last_frame):
                        break
        finally:
            if analyzer is not None:
                analyzer.stop()
            try:
                if self.stream is not None:
                    self.stream.stop()
                    self.stream = None
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
